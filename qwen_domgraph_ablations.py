"""qwen_domgraph_ablations.py

Runs Qwen2.5-VL-7B-Instruct on Multimodal-Mind2Web using a filtered DOM graph
as the structural context modality instead of raw HTML or AXTree.

WHAT THIS DOES
--------------
For each datapoint we:
  1. Parse cleaned_html → full DOM graph (via dom_graph.build_dom_graph)
  2. Match each candidate action string to its node(s) in the graph
  3. Extract ONLY the candidate nodes + their immediate parents/siblings
     → a compact "candidate context" snippet instead of the whole graph
  4. Build a prompt with: task, screenshot, candidate context, candidate list
  5. Ask the model to output a ranked JSON object (same schema as
     intern_axtree_ablations.py) so eval_next_action.py works unchanged.

TWO BASELINES (mirrors intern_axtree_ablations.py)
---------------------------------------------------
  qwen_domgraph       — image + task + candidate graph context, no CoT
  qwen_domgraph_cot   — same + chain-of-thought reasoning field

OUTPUT JSONL (eval_next_action.py-compatible)
---------------------------------------------
Every record written to preds_<baseline>_<split>.jsonl has exactly the fields
that eval_next_action.evaluate_file() expects:
  split, baseline, annotation_id, action_uid,
  gt_action_index, gt_action, gt_action_repr,
  pred_action_index, pred_action_indices, pred_action, pred_action_repr,
  pred_target_element, reasoning, parse_error, raw_output, candidates

USAGE
-----
  python qwen_domgraph_ablations.py --splits test_website

  # smoke test (first 50 examples, no-CoT only)
  python qwen_domgraph_ablations.py --splits test_task --limit 50 \
      --baselines qwen_domgraph

  # then evaluate
  python eval_next_action.py \
      --input inference_outputs/qwen_domgraph \
      --pattern "preds_*.jsonl" \
      --out inference_outputs/qwen_domgraph/metrics_summary.json

REQUIRES
--------
  pip install transformers>=4.40 datasets bitsandbytes pillow tqdm \
              beautifulsoup4 lxml python-dotenv
  # optional for constrained decoding:
  pip install lm-format-enforcer pydantic
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

# dom_graph.py and mind2web_dataloader.py must live in the same directory.
from dom_graph import build_dom_graph
from mind2web_dataloader import MultimodalMind2WebDataset

load_dotenv()

# ── Model config (from multimodal_qwen_domgraph.py / multimodal_qwen_instruct.py) ──
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
MIN_PIXELS = 256  * 28 * 28
HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

DEFAULT_SPLITS = ["test_website"]

# ── Baselines (mirrors intern_axtree_ablations.py) ────────────────────────────
BASELINES: List[Dict[str, Any]] = [
    {
        "name": "qwen_domgraph",
        "cot":  False,
        # JSON object is ~50-80 tokens; 256 gives comfortable headroom.
        "max_new_tokens": 256,
    },
]

# ── JSON output schemas shown to the model (identical to intern script) ────────
_JSON_SCHEMA = '''\
Respond with a single JSON object and nothing else. Do NOT include any text outside the JSON.
Choose the candidate index that best matches the next action for the given task.
Rank your top-3 best candidate indices from most to least confident.
If only one candidate clearly matches, you may list fewer.
IMPORTANT: all JSON values must be properly quoted strings or arrays of integers.
{
  "top3_action_indices": [<integer: best index>, <integer: 2nd best>, <integer: 3rd best>],
  "action_type": "<action verb for the best candidate: CLICK, TYPE, SELECT, HOVER, SCROLL>",
  "target_element": "<brief description of the element being acted on>"
}'''

# ── Optional constrained decoding ─────────────────────────────────────────────
try:
    from lm_format_enforcer import JsonSchemaParser
    from lm_format_enforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )
    _LM_FORMAT_ENFORCER_AVAILABLE = True
except ImportError:
    _LM_FORMAT_ENFORCER_AVAILABLE = False

try:
    from pydantic import BaseModel as _BaseModel

    class _Pred(_BaseModel):
        top3_action_indices: List[int]
        action_type: str
        target_element: str

    _SCHEMA = _Pred.model_json_schema()
except ImportError:
    _SCHEMA = {
        "type": "object",
        "properties": {
            "top3_action_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
            "action_type":    {"type": "string"},
            "target_element": {"type": "string"},
        },
        "required": ["top3_action_indices", "action_type", "target_element"],
        "additionalProperties": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DOM graph helpers — candidate-filtered context
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_action_repr(action_repr: str) -> Dict[str, str]:
    """
    Converts an action_repr string like "[button]  Submit -> CLICK"
    into {"tag": "button", "text": "Submit"} for node matching.
    """
    m = re.match(r"^\[([^\]]+)\]\s*(.*?)\s*(?:->\s*\w+)?\s*$", action_repr or "")
    if m:
        return {"tag": m.group(1).lower().strip(), "text": m.group(2).strip()}
    return {"tag": "", "text": (action_repr or "").strip()}


def _match_node_by_repr(action_repr: str, nodes: List[Dict[str, Any]]) -> Optional[int]:
    """
    Tries to find the node_id in the DOM graph that best matches a candidate
    action_repr string. Falls back gracefully if no match is found.
    """
    parsed = _parse_action_repr(action_repr)
    tag    = parsed["tag"]
    text   = parsed["text"][:60]

    # Strict: tag + text substring match
    if tag and text:
        for node in nodes:
            if node["tag"] == tag and text in (node["text"] or ""):
                return node["node_id"]
    # Looser: tag + text appears in label
    if tag and text:
        for node in nodes:
            if node["tag"] == tag and text[:20] in (node["label"] or ""):
                return node["node_id"]
    return None


def build_candidate_context(
    html_str: str,
    action_reprs: List[str],
    max_nodes: int = 150,
    max_chars: int = 6000,
) -> str:
    """
    Builds a compact DOM graph text containing ONLY the nodes relevant to the
    candidate actions — specifically:
      - The candidate node itself
      - Its parent (one level up, for structural context)
      - Its direct siblings (other children of the same parent)

    This is the key idea: instead of dumping the whole page graph into the
    prompt, the model gets a focused view of just the elements it must choose
    between, with enough surrounding structure to understand their context.

    Args:
        html_str:     cleaned_html field from the dataset row
        action_reprs: list of candidate action strings (the numbered list)
        max_nodes:    passed to build_dom_graph for initial pruning
        max_chars:    hard cap on the returned string length

    Returns:
        A multi-line string ready to inject into the prompt.
    """
    if not html_str or not html_str.strip():
        return "[No HTML available]"

    try:
        nodes, _ = build_dom_graph(html_str)
    except Exception as exc:
        return f"[DOM parse failed: {exc}]"

    if not nodes:
        return "[Empty DOM]"

    node_lookup = {n["node_id"]: n for n in nodes}

    # ── 1. Match each candidate to a DOM node ─────────────────────────────
    # candidate_node_map: candidate_index -> node_id (None if unmatched)
    candidate_node_map: Dict[int, Optional[int]] = {}
    for i, repr_str in enumerate(action_reprs):
        candidate_node_map[i] = _match_node_by_repr(repr_str, nodes)

    # ── 2. Collect node_ids to include ────────────────────────────────────
    # For each matched candidate node we also pull its parent and siblings.
    include_ids: set = set()
    for cand_idx, node_id in candidate_node_map.items():
        if node_id is None:
            continue
        include_ids.add(node_id)
        node = node_lookup.get(node_id)
        if node is None:
            continue
        # Parent
        if node["parent_id"] is not None:
            include_ids.add(node["parent_id"])
            parent = node_lookup.get(node["parent_id"])
            # Siblings (other children of the same parent)
            if parent:
                for sib_id in parent.get("children_ids", []):
                    include_ids.add(sib_id)

    # ── 3. Render the filtered graph as text ──────────────────────────────
    node_to_cand: Dict[int, int] = {
        nid: cidx for cidx, nid in candidate_node_map.items() if nid is not None
    }

    lines = [
        f"=== CANDIDATE DOM CONTEXT ({len(include_ids)} nodes) ===",
        "FORMAT: Node [role, depth]: label | Parent: X | Children: [Y] | Siblings: [Z]",
        "",
    ]

    # Output in DFS order (node_id is assigned in DFS traversal order).
    for node in nodes:
        nid = node["node_id"]
        if nid not in include_ids:
            continue

        cand_marker  = ""
        if nid in node_to_cand:
            cand_marker = f" *** CANDIDATE (index {node_to_cand[nid]}) ***"

        label        = node["label"] or node["tag"]
        depth_indent = "  " * min(node["depth"], 6)
        children_shown = [c for c in node.get("children_ids", []) if c in include_ids]
        siblings = []
        if node["parent_id"] is not None:
            parent = node_lookup.get(node["parent_id"])
            if parent:
                siblings = [
                    c for c in parent.get("children_ids", [])
                    if c != nid and c in include_ids
                ]

        parts = [f"Node {nid} [{node['role']}, depth={node['depth']}]: {label}{cand_marker}"]
        if node["parent_id"] is not None and node["parent_id"] in include_ids:
            parts.append(f"Parent: {node['parent_id']}")
        if children_shown:
            parts.append(f"Children: {children_shown}")
        if siblings:
            parts.append(f"Siblings: {siblings[:5]}")

        lines.append(depth_indent + " | ".join(parts))

    # ── 4. Note unmatched candidates ──────────────────────────────────────
    unmatched = [i for i, nid in candidate_node_map.items() if nid is None]
    if unmatched:
        lines.append("")
        lines.append(
            f"NOTE: Candidates {unmatched} could not be matched to DOM nodes "
            "(may be dynamically injected or use non-standard attributes)."
        )

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[context truncated]"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════════════

def _candidate_list_text(action_reprs: List[str]) -> str:
    return "\n".join(f"{i}: {c}" for i, c in enumerate(action_reprs))


def build_prompt(
    row: Dict[str, Any],
    max_dom_chars: int,
    max_dom_nodes: int,
) -> str:
    task        = row.get("confirmed_task", "")
    html_str    = row.get("html", "") or row.get("cleaned_html", "") or ""
    candidates  = row.get("action_reprs") or []

    candidate_context = build_candidate_context(
        html_str,
        candidates,
        max_nodes=max_dom_nodes,
        max_chars=max_dom_chars,
    )

    return (
        "You are predicting the next web action for a browser agent.\n"
        "Given the task, screenshot, candidate DOM context, and candidate actions, "
        "choose the correct action index.\n\n"
        f"Task:\n{task}\n\n"
        "Candidate DOM Context:\n"
        "The following shows only the DOM nodes that correspond to the candidate "
        "actions. Nodes marked *** CANDIDATE (index N) *** are the elements you "
        "must choose between.\n"
        f"{candidate_context}\n\n"
        f"Candidate actions:\n{_candidate_list_text(candidates)}\n\n"
        f"{_JSON_SCHEMA}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_qwen_model(model_name: str, dtype: torch.dtype):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Qwen2.5-VL inference but torch.cuda.is_available() "
            "returned False."
        )
    print(f"Loading {model_name}  (4-bit NF4, dtype={dtype})")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM free: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model.eval()
    print(f"Model ready on: {next(model.parameters()).device}\n")
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def to_pil_image(image_field: Any) -> Optional[Image.Image]:
    import io as _io
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, str):
        return Image.open(image_field).convert("RGB")
    if isinstance(image_field, dict):
        path = image_field.get("path")
        if path:
            return Image.open(path).convert("RGB")
        raw = image_field.get("bytes")
        if raw is not None:
            return Image.open(_io.BytesIO(raw)).convert("RGB")
    return None


def run_qwen_chat(
    model: Any,
    processor: Any,
    image: Optional[Image.Image],
    prompt: str,
    generation_config: Dict[str, Any],
) -> str:
    """Single Qwen2.5-VL forward pass → raw text output."""
    content: List[Dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})

    messages  = [{"role": "user", "content": content}]
    text_tmpl = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    proc_kwargs: Dict[str, Any] = dict(
        text=[text_tmpl], padding=True, return_tensors="pt"
    )
    if image is not None:
        proc_kwargs["images"] = [image]

    inputs    = processor(**proc_kwargs)
    inputs    = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_config)

    generated = output_ids[:, input_len:]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Output parsing (identical to intern_axtree_ablations.py)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_model_output(raw_output: str, num_candidates: int) -> Dict[str, Any]:
    """
    Parse the model's JSON response into a structured dict.

    Tries full JSON first, then falls back to targeted-regex on partial outputs.
    Mirrors intern_axtree_ablations.parse_model_output exactly so eval behaviour
    is consistent across baselines.
    """
    result: Dict[str, Any] = {
        "pred_action_indices": [],
        "action_type":    None,
        "target_element": None,
        "parse_error":    None,
    }
    if not raw_output:
        result["parse_error"] = "empty output"
        return result

    # 1. Full JSON parse (preferred) — strip markdown fences first.
    json_text  = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.S)
    json_match = re.search(r"\{.*\}", json_text, flags=re.S)
    if json_match:
        try:
            parsed      = json.loads(json_match.group(0))
            raw_indices = parsed.get("top3_action_indices") or []
            if isinstance(raw_indices, int):
                raw_indices = [raw_indices]
            valid_indices = [
                int(i) for i in raw_indices
                if isinstance(i, (int, float)) and 0 <= int(i) < num_candidates
            ][:3]
            result["pred_action_indices"] = valid_indices
            if not valid_indices:
                result["parse_error"] = (
                    f"top3_action_indices {raw_indices!r} had no valid indices "
                    f"in [0, {num_candidates})"
                )
            atype = parsed.get("action_type")
            result["action_type"]    = str(atype).upper() if atype else None
            result["target_element"] = parsed.get("target_element") or None
            return result
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"JSON decode error: {exc}"

    # 2. Partial-JSON / targeted-regex fallback.
    idx_match = re.search(r'"top3_action_indices"\s*:\s*\[([^\]]+)\]', raw_output)
    if idx_match:
        raw_list = idx_match.group(1)
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_list)]
        valid    = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + (
            " (partial-JSON: indices OOR)" if not valid else " (partial-JSON fallback)"
        )
    else:
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_output)]
        valid    = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + (
            " (regex fallback)" if valid else " (no valid index found)"
        )

    # 2b. Recover action_type from partial text.
    if result["action_type"] is None:
        m = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw_output, re.I)
        if m:
            result["action_type"] = m.group(1).strip().upper()
        else:
            bare = re.search(
                r'\b(CLICK|TYPE|SELECT(?:_OPTION)?|HOVER|SCROLL|PRESS|ENTER)\b',
                raw_output, re.I,
            )
            if bare:
                result["action_type"] = bare.group(1).upper()

    return result


def normalize_gt_operation(operation_field: Any) -> Optional[str]:
    """
    Extract the action verb from the Mind2Web operation field.

    Handles dict / JSON-string / plain-string forms.
    Uses UPPERCASE keys (OP, ORIGINAL_OP) which is what Multimodal-Mind2Web
    actually stores — the old lowercase-only lookup always returned None,
    making ActionAcc trivially 0 (bug documented in eval_next_action.py).
    """
    if operation_field is None:
        return None
    if isinstance(operation_field, str):
        s = operation_field.strip()
        if not s:
            return None
        if s.startswith("{"):
            try:
                operation_field = json.loads(s)
            except json.JSONDecodeError:
                return None
        else:
            return s.upper()
    if isinstance(operation_field, dict):
        for key in ("OP", "ORIGINAL_OP", "op", "operation", "action", "type", "name"):
            v = operation_field.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
    return None


def extract_operation(action_repr: str) -> Optional[str]:
    """Pull the action verb from the start of a candidate action_repr string."""
    if not action_repr:
        return None
    m = re.match(r"\s*([A-Za-z_]+)", action_repr)
    return m.group(1).upper() if m else None


def make_prediction_record(
    row: Dict[str, Any],
    split: str,
    baseline_name: str,
    parsed: Dict[str, Any],
    raw_output: str,
) -> Dict[str, Any]:
    """
    Assemble the JSONL record that eval_next_action.py expects.
    Schema is identical to intern_axtree_ablations.make_prediction_record.
    """
    candidates   = row.get("action_reprs") or []
    gt_idx       = int(row.get("target_action_index") or -1)
    pred_indices: List[int] = parsed.get("pred_action_indices") or []
    pred_idx     = pred_indices[0] if pred_indices else None

    pred_action_repr = (
        candidates[pred_idx]
        if pred_idx is not None and 0 <= pred_idx < len(candidates)
        else None
    )
    # Prefer the explicitly named action_type; fall back to reading the repr string.
    pred_action = parsed.get("action_type") or extract_operation(pred_action_repr or "")

    return {
        # identifiers
        "split":         split,
        "baseline":      baseline_name,
        "annotation_id": row.get("annotation_id"),
        "action_uid":    row.get("action_uid"),
        # ground truth
        "gt_action_index": gt_idx,
        "gt_action":       normalize_gt_operation(row.get("operation")),
        "gt_action_repr":  row.get("target_action_reprs"),
        # prediction
        "pred_action_index":   pred_idx,
        "pred_action_indices": pred_indices,
        "pred_action":         pred_action,
        "pred_action_repr":    pred_action_repr,
        "pred_target_element": parsed.get("target_element"),
        # diagnostics
        "parse_error": parsed.get("parse_error"),
        "raw_output":  raw_output,
        "candidates":  candidates,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Optional constrained decoding
# ═══════════════════════════════════════════════════════════════════════════════

def build_json_prefix_fn(processor: Any) -> Optional[Any]:
    if not _LM_FORMAT_ENFORCER_AVAILABLE:
        warnings.warn(
            "--json_constrained requested but lm-format-enforcer is not installed. "
            "Falling back to unconstrained decoding. "
            "Install with: pip install lm-format-enforcer pydantic",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    parser = JsonSchemaParser(_SCHEMA)
    return build_transformers_prefix_allowed_tokens_fn(processor.tokenizer, parser)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-baseline runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_baseline(
    model: Any,
    processor: Any,
    dataset,
    split: str,
    baseline: Dict[str, Any],
    args: argparse.Namespace,
) -> int:
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else baseline["max_new_tokens"]
    )
    do_sample = args.temperature > 0.0
    generation_config: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample":      do_sample,
    }
    if do_sample:
        generation_config["temperature"] = args.temperature
    if args.json_constrained:
        prefix_fn = build_json_prefix_fn(processor)
        if prefix_fn is not None:
            generation_config["prefix_allowed_tokens_fn"] = prefix_fn

    total    = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_dir  = Path(args.output_dir)
    out_file = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count already-written rows and skip them.
    start_idx = 0
    if args.resume and out_file.exists():
        with out_file.open("r", encoding="utf8") as fh:
            start_idx = sum(1 for _ in fh)
        if start_idx >= total:
            print(
                f"  Skipping {baseline['name']} / {split}: "
                f"already complete ({start_idx}/{total})"
            )
            return 0
        print(f"  Resuming {baseline['name']} / {split} from row {start_idx}")

    written = 0
    with out_file.open("a", encoding="utf8") as fh:
        for idx in tqdm(
            range(start_idx, total),
            desc=f"{baseline['name']} | {split}",
            total=total - start_idx,
        ):
            row   = dataset[idx]
            image = to_pil_image(row.get("screenshot"))

            prompt = build_prompt(
                row=row,
                max_dom_chars=args.max_dom_chars,
                max_dom_nodes=args.max_dom_nodes,
            )

            try:
                raw_output = run_qwen_chat(model, processor, image, prompt, generation_config)
            except Exception as exc:
                raw_output = f"[inference error: {exc}]"

            num_candidates = len(row.get("action_reprs") or [])
            parsed  = parse_model_output(raw_output, num_candidates=num_candidates)
            record  = make_prediction_record(row, split, baseline["name"], parsed, raw_output)

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1

            torch.cuda.empty_cache()

    return written


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B DOM-graph ablations for Multimodal-Mind2Web"
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument(
        "--splits", nargs="+", default=DEFAULT_SPLITS,
        choices=["test_task", "test_website", "test_domain"],
    )
    parser.add_argument(
        "--data_dir", default="data/mind2web",
        help="Local directory produced by download_mind2web.py (one sub-folder per split).",
    )
    parser.add_argument(
        "--output_dir", default="inference_outputs/qwen_domgraph",
    )
    parser.add_argument(
        "--max_dom_chars", type=int, default=6000,
        help=(
            "Max characters of candidate DOM context passed in the prompt. "
            "Because we only include candidate nodes + parents/siblings (not the "
            "whole graph), 6000 chars is usually sufficient and leaves ample room "
            "for image tokens + task + candidates in Qwen's 32k context."
        ),
    )
    parser.add_argument(
        "--max_dom_nodes", type=int, default=150,
        help="Max nodes for the initial full-DOM parse before candidate filtering.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override default max_new_tokens (256).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0.0 = greedy decoding (recommended for deterministic eval).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max examples per split. Useful for smoke-testing.",
    )
    parser.add_argument(
        "--json_constrained", action="store_true",
        help="Constrained JSON decoding via lm-format-enforcer (requires pip install).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip rows already written to the output JSONL.",
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(name, torch.float32)


def main() -> None:
    args  = parse_args()
    dtype = _resolve_dtype(args.dtype)

    model, processor = load_qwen_model(args.model_name, dtype)

    for split in args.splits:
        print(f"\nLoading split '{split}' from {args.data_dir} ...")
        dataset = MultimodalMind2WebDataset(
            split=split,
            local_dir=args.data_dir,
            html_field="cleaned_html",
        )
        print(f"  {len(dataset)} samples\n")

        for baseline in BASELINES:
            print(f"Running: {baseline['name']} on {split}")
            out_file = Path(args.output_dir) / f"preds_{baseline['name']}_{split}.jsonl"
            n = run_single_baseline(
                model, processor, dataset, split, baseline, args
            )
            print(f"  → {out_file}  ({n} new rows written)\n")


if __name__ == "__main__":
    main()