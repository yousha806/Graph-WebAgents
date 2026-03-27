"""Run Qwen3-8B-VL baselines for Mind2Web next-action prediction.

This script implements exactly two baselines on the Qwen3-8B-VL model:
1) multimodal: screenshot + HTML DOM + task + candidates
2) text_only:  HTML DOM + task + candidates (no image)

No chain-of-thought in either baseline.

It evaluates by default on all three test splits:
- test_task
- test_website
- test_domain

Expected input data:
- Mind2Web multimodal splits saved with datasets.save_to_disk under
  data/mind2web/<split>

Metrics collected per record (sufficient for):
  - Element Accuracy (Top-1)        : gt_action_index, pred_action_index
  - Action Accuracy                 : gt_action, pred_action
  - Exact Match (element + action)  : both above
  - Parse Failure Rate              : parse_error
  - Top-K Element Accuracy (Top-3)  : pred_action_indices (ranked list)
  - MRR                             : pred_action_indices + gt_action_index
  - Task Success Rate               : annotation_id (group steps per task)

Failure analysis output (written to failure_log_<baseline>_<split>.jsonl):
  One JSON record per incorrect prediction, with the schema:
  {
    "id":           annotation_id of the example,
    "model":        HF model name,
    "split":        dataset split name,
    "input_task":   confirmed_task string,
    "html_snippet": first 500 chars of dom_content,
    "screenshot":   path to screenshot file, or null if bytes-only,
    "candidates":   list of "idx: repr" strings,
    "gt_index":     ground-truth candidate index,
    "gt_element":   "idx: repr" string for the gt candidate,
    "gt_action":    normalised ground-truth action verb,
    "gt_value":     value field from operation dict (or null),
    "pred_text":    raw model output string,
    "pred_idx":     top-1 predicted index (or null on parse failure),
    "pred_element": "idx: repr" string for predicted candidate (or null),
    "pred_action":  predicted action verb (or null),
    "pred_value":   value parsed from predicted repr (or null),
    "task_success": false (always false in the failure log),
    "value_states": null (reserved for future hidden-state logging),
    "timestamp":    ISO-8601 UTC timestamp of the inference call
  }
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration
    _QWEN_VL_AVAILABLE = True
except ImportError:
    _QWEN_VL_AVAILABLE = False

# qwen_vl_utils provides process_vision_info for building pixel_values + image_grid_thw
try:
    from qwen_vl_utils import process_vision_info
    _QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    _QWEN_VL_UTILS_AVAILABLE = False

# Optional: lm-format-enforcer for hard JSON-schema-constrained decoding.
# Install with: pip install lm-format-enforcer
# Without it the script still runs; --json_constrained will warn and be skipped.
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

    class _PredSchema(_BaseModel):
        top3_action_indices: List[int]   # ranked best→worst, length 1-3
        action_type: str
        target_element: str

    _SCHEMA = _PredSchema.model_json_schema()
except ImportError:
    # Pydantic unavailable: fall back to hand-written JSON Schema dict.
    _SCHEMA = {
        "type": "object",
        "properties": {
            "top3_action_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 3,
            },
            "action_type":    {"type": "string"},
            "target_element": {"type": "string"},
        },
        "required": ["top3_action_indices", "action_type", "target_element"],
        "additionalProperties": False,
    }

DEFAULT_SPLITS = ["test_website"]

BASELINES = [
    {
        "name": "qwen3_multimodal_allinputs",
        "multimodal": True,
        # JSON object is ~50-80 tokens; 256 gives comfortable headroom.
        "max_new_tokens": 256,
    },
    {
        "name": "qwen3_text_only_allinputs",
        "multimodal": False,
        # Same budget — no image, but prompt structure is identical otherwise.
        "max_new_tokens": 256,
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-8B-VL baselines (multimodal vs text-only) for Mind2Web"
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3-8B-VL",
        help="Hugging Face model id for Qwen3-8B-VL checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        default="data/mind2web",
        help="Directory containing Mind2Web splits saved with datasets.save_to_disk",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=["test_task", "test_website", "test_domain"],
        help="Which test splits to run",
    )
    parser.add_argument(
        "--output_dir",
        default="inference_outputs/qwen3_baselines",
        help="Where to write JSONL predictions and failure logs",
    )
    parser.add_argument(
        "--max_dom_chars", type=int, default=12000,
        help="Max characters of HTML DOM to include in the prompt.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override the per-baseline max_new_tokens. Rarely needed.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. 0.0 = greedy decoding (default, recommended for "
             "deterministic evaluation). Values > 0 enable sampling.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional max examples per split",
    )
    parser.add_argument(
        "--json_constrained",
        action="store_true",
        help=(
            "Enforce JSON-schema-valid output via constrained decoding (lm-format-enforcer). "
            "Requires: pip install lm-format-enforcer pydantic"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSONL outputs by skipping already written rows",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Optional bitsandbytes quantization mode",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# dtype / quantization helpers
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def build_quantization_config(mode: str, dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
    if mode == "none":
        return None
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode: {mode}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen_model(model_name: str, dtype: torch.dtype, quantization: str):
    """Load Qwen3-8B-VL model and processor onto CUDA."""
    if not _QWEN_VL_AVAILABLE:
        raise ImportError(
            "Qwen3-8B-VL model class not found. "
            "Install with: pip install transformers>=4.51.0"
        )
    if not _QWEN_VL_UTILS_AVAILABLE:
        raise ImportError(
            "qwen_vl_utils not found. "
            "Install with: pip install qwen-vl-utils"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for Qwen3-8B-VL inference but torch.cuda.is_available() returned False.\n"
            f"  torch version    : {torch.__version__}\n"
            f"  torch CUDA build : {torch.version.cuda}\n"
        )

    quantization_config = build_quantization_config(quantization, dtype)

    gpu_cc = torch.cuda.get_device_capability(0)
    print(f"Loading Qwen3-8B-VL model: {model_name} (quantization={quantization}, dtype={dtype})")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  "
        f"compute={gpu_cc[0]}.{gpu_cc[1]}  |  "
        f"VRAM free: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB"
    )

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # min_pixels / max_pixels control Qwen's dynamic-resolution tiling.
    # 256*28*28 → 200,704 px (≈448²) minimum; 1280*28*28 → ~1M px maximum.
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    print(f"Model loaded on: {next(model.parameters()).device}")
    return model, processor


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_split_dataset(data_dir: str, split: str):
    split_path = Path(data_dir) / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split not found: {split_path}. "
            "Ensure the Mind2Web multimodal dataset has been saved to disk at this path."
        )
    return load_from_disk(str(split_path))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_pil_image(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, str):
        return Image.open(image_field).convert("RGB")
    if isinstance(image_field, dict):
        path = image_field.get("path")
        if path:
            return Image.open(path).convert("RGB")
        if image_field.get("bytes") is not None:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
    raise ValueError("Unsupported screenshot format from dataset row")


def screenshot_path_from_field(image_field: Any) -> Optional[str]:
    """Return a file path string for the screenshot field, or None if bytes-only."""
    if isinstance(image_field, str):
        return image_field
    if isinstance(image_field, dict):
        return image_field.get("path") or None
    return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int, label: str = "text") -> str:
    text = text or ""
    if len(text) > max_chars:
        logging.warning(
            "build_prompt: %s (%d chars) exceeds max_chars=%d and will be clipped.",
            label,
            len(text),
            max_chars,
        )
        return text[:max_chars]
    return text


def candidate_lines(action_reprs: List[str]) -> str:
    return "\n".join(f"{idx}: {candidate}" for idx, candidate in enumerate(action_reprs))


# ---------------------------------------------------------------------------
# Prompt schema (no CoT in either baseline)
# ---------------------------------------------------------------------------

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


def build_qwen_messages(
    row: Dict[str, Any],
    multimodal: bool,
    max_dom_chars: int,
) -> List[Dict[str, Any]]:
    """Build the Qwen3 chat-template messages list for one example.

    For the multimodal baseline the user turn contains an image content block
    followed by all text modalities. For text-only the image block is omitted.

    Modalities included:
        confirmed_task  — the high-level task description          (both baselines)
        screenshot      — PIL image                                (multimodal only)
        dom_content     — raw HTML DOM of the page                 (both baselines)
        action_reprs    — the candidate action list for this step  (both baselines)
    """
    task    = row.get("confirmed_task", "")
    dom     = truncate(
        row.get("cleaned_html", "") or row.get("raw_html", "") or "",
        max_dom_chars,
        "cleaned_html",
    )
    actions = row.get("action_reprs", [])

    text_body = (
        "You are predicting the next web action for a browser agent.\n"
        "Given the task and page context, choose the correct candidate action index.\n\n"
        f"Task:\n{task}\n\n"
    )

    if dom:
        text_body += f"HTML DOM:\n{dom}\n\n"

    text_body += f"Candidate actions:\n{candidate_lines(actions)}\n\n{_JSON_SCHEMA}"

    if multimodal:
        image = to_pil_image(row["screenshot"])
        content = [
            {
                "type": "image",
                # Pass PIL image directly; process_vision_info handles encoding.
                "image": image,
            },
            {
                "type": "text",
                "text": text_body,
            },
        ]
    else:
        content = [{"type": "text", "text": text_body}]

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_qwen_inference(
    model: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    multimodal: bool,
) -> str:
    """Run one forward pass through Qwen3-8B-VL and return the decoded string."""
    # apply_chat_template produces the full prompt string with special tokens.
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if multimodal:
        # process_vision_info extracts PIL images / video frames from the
        # messages content list and returns pixel tensors + grid metadata.
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_config)

    # Trim the input tokens; only decode the newly generated tokens.
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_model_output(raw_output: str, num_candidates: int) -> Dict[str, Any]:
    """Parse the model's JSON response into a structured dict.

    Tries JSON first (the instructed format), then falls back to regex so that
    partial or malformed responses still yield usable indices.

    Returns a dict with keys:
        pred_action_indices : List[int]    — ranked top-1..3 candidate indices (empty if unparseable)
        action_type         : str | None   — e.g. "CLICK" (for top-1)
        target_element      : str | None   — element description from model
        reasoning           : str | None   — always None (no CoT in these baselines)
        parse_error         : str | None   — description of any parse failure
    """
    result: Dict[str, Any] = {
        "pred_action_indices": [],
        "action_type": None,
        "target_element": None,
        "reasoning": None,   # kept for schema compatibility; always None here
        "parse_error": None,
    }
    if not raw_output:
        result["parse_error"] = "empty output"
        return result

    # --- 1. JSON parse (preferred) ---
    # Strip markdown code fences the model sometimes wraps around JSON.
    json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.S)
    # Also handle output that has prose before/after the JSON object.
    json_match = re.search(r"\{.*\}", json_text, flags=re.S)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            raw_indices = parsed.get("top3_action_indices") or []
            # Accept either the list form or the legacy single-int form.
            if isinstance(raw_indices, int):
                raw_indices = [raw_indices]
            valid_indices = [
                int(i) for i in raw_indices
                if isinstance(i, (int, float)) and 0 <= int(i) < num_candidates
            ][:3]
            result["pred_action_indices"] = valid_indices
            if not valid_indices:
                result["parse_error"] = (
                    f"top3_action_indices {raw_indices!r} contained no valid indices "
                    f"in [0, {num_candidates})"
                )
            atype = parsed.get("action_type")
            result["action_type"] = str(atype).upper() if atype else None
            result["target_element"] = parsed.get("target_element") or None
            return result
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"JSON decode error: {exc}"

    # --- 2. Partial-JSON / targeted-regex fallback ---
    # The model sometimes truncates the JSON mid-string. Use targeted key-value
    # regexes rather than scanning all integers.

    # 2a. Recover top3_action_indices from the partial key match.
    idx_match = re.search(r'"top3_action_indices"\s*:\s*\[([^\]]+)\]', raw_output)
    if idx_match:
        raw_list = idx_match.group(1)
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_list)]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        if not valid:
            result["parse_error"] = (
                (result["parse_error"] or "") + " (partial-JSON: indices out of range)"
            )
        else:
            result["parse_error"] = (
                (result["parse_error"] or "") + " (used partial-JSON fallback)"
            )
    else:
        # Last-resort: scrape all integers from the whole output.
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_output)]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + (
            " (used regex fallback)" if valid else " (no valid index found)"
        )

    # 2b. Recover action_type from key-value pair in partial text.
    if result["action_type"] is None:
        atype_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw_output, re.I)
        if atype_match:
            result["action_type"] = atype_match.group(1).strip().upper()
        else:
            bare_match = re.search(
                r'\b(CLICK|TYPE|SELECT(?:_OPTION)?|HOVER|SCROLL|PRESS|ENTER)\b',
                raw_output, re.I,
            )
            if bare_match:
                result["action_type"] = bare_match.group(1).upper()

    return result


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def extract_operation(action_repr: str) -> Optional[str]:
    """Pull the leading action verb from a repr string like 'CLICK [button]'."""
    if not action_repr:
        return None
    match = re.match(r"\s*([A-Za-z_]+)", action_repr)
    if not match:
        return None
    return match.group(1).upper()


def extract_value(action_repr: str) -> Optional[str]:
    """Pull the typed/selected value from a repr string like "TYPE 'hello'"."""
    if not action_repr:
        return None
    match = re.search(r"['\"](.+?)['\"]", action_repr)
    return match.group(1) if match else None


def normalize_gt_operation(operation_field: Any) -> Optional[str]:
    """Extract and normalise the action verb from the Mind2Web operation field.

    The ``operation`` column in Multimodal-Mind2Web can arrive in several forms:
    * A dict   : {"OP": "CLICK", "ORIGINAL_OP": "CLICK", "VALUE": ""}
    * A JSON string of the above (HF sometimes serialises nested dicts to str)
    * A plain string like "CLICK"
    """
    if operation_field is None:
        return None

    if isinstance(operation_field, str):
        stripped = operation_field.strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            try:
                operation_field = json.loads(stripped)
                # Fall through to the dict branch below.
            except json.JSONDecodeError:
                return stripped.upper()
        else:
            return stripped.upper()

    if isinstance(operation_field, dict):
        for key in ("OP", "ORIGINAL_OP", "op", "operation", "action", "type", "name"):
            value = operation_field.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()

    return None


def extract_gt_value(operation_field: Any) -> Optional[str]:
    """Extract the VALUE field from the Mind2Web operation dict, if present."""
    if isinstance(operation_field, dict):
        val = operation_field.get("VALUE") or operation_field.get("value")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None
    if isinstance(operation_field, str) and operation_field.strip().startswith("{"):
        try:
            d = json.loads(operation_field)
            val = d.get("VALUE") or d.get("value")
            if isinstance(val, str) and val.strip():
                return val.strip()
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------

def make_prediction_record(
    row: Dict[str, Any],
    split: str,
    baseline_name: str,
    parsed: Dict[str, Any],
    raw_output: str,
) -> Dict[str, Any]:
    candidates = row.get("action_reprs", [])
    raw_gt = row.get("target_action_index")
    gt_idx = int(raw_gt) if raw_gt is not None else -1

    pred_indices: List[int] = parsed.get("pred_action_indices") or []
    pred_idx = pred_indices[0] if pred_indices else None   # top-1
    pred_action_repr = (
        candidates[pred_idx]
        if pred_idx is not None and 0 <= pred_idx < len(candidates)
        else None
    )
    # Prefer the action type the model explicitly named; fall back to parsing
    # the action repr string (e.g. "CLICK [button 'Submit']" → "CLICK").
    pred_action = parsed.get("action_type") or extract_operation(pred_action_repr or "")

    return {
        # ── identifiers ──────────────────────────────────────────────────────
        "split": split,
        "baseline": baseline_name,
        "annotation_id": row.get("annotation_id"),
        "action_uid": row.get("action_uid"),
        # ── ground truth ─────────────────────────────────────────────────────
        "gt_action_index": gt_idx,
        "gt_action": normalize_gt_operation(row.get("operation")),
        "gt_action_repr": row.get("target_action_reprs"),
        # ── model prediction ─────────────────────────────────────────────────
        "pred_action_index": pred_idx,          # top-1  → ElemAcc / StepAcc
        "pred_action_indices": pred_indices,    # ranked → Top-K ElemAcc / MRR
        "pred_action": pred_action,
        "pred_action_repr": pred_action_repr,
        "pred_target_element": parsed.get("target_element"),
        "reasoning": parsed.get("reasoning"),  # always None; kept for schema parity
        # ── diagnostics ──────────────────────────────────────────────────────
        "parse_error": parsed.get("parse_error"),
        "raw_output": raw_output,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Failure logging
# ---------------------------------------------------------------------------

def make_failure_record(
    row: Dict[str, Any],
    split: str,
    model_name: str,
    parsed: Dict[str, Any],
    raw_output: str,
    timestamp: str,
) -> Dict[str, Any]:
    """Build a failure-analysis record matching the project schema exactly.

    Schema fields:
        id, model, split, input_task, html_snippet, screenshot,
        candidates, gt_index, gt_element, gt_action, gt_value,
        pred_text, pred_idx, pred_element, pred_action, pred_value,
        task_success, value_states, timestamp
    """
    candidates: List[str] = row.get("action_reprs", [])
    raw_gt = row.get("target_action_index")
    gt_idx = int(raw_gt) if raw_gt is not None else -1

    # Ground-truth element string in "idx: repr" form to match the candidates list format
    gt_element: Optional[str] = (
        f"{gt_idx}: {candidates[gt_idx]}"
        if 0 <= gt_idx < len(candidates)
        else None
    )
    gt_action = normalize_gt_operation(row.get("operation"))
    gt_value  = extract_gt_value(row.get("operation"))

    pred_indices: List[int] = parsed.get("pred_action_indices") or []
    pred_idx = pred_indices[0] if pred_indices else None
    pred_element: Optional[str] = (
        f"{pred_idx}: {candidates[pred_idx]}"
        if pred_idx is not None and 0 <= pred_idx < len(candidates)
        else None
    )
    pred_action_repr = (
        candidates[pred_idx]
        if pred_idx is not None and 0 <= pred_idx < len(candidates)
        else None
    )
    pred_action = parsed.get("action_type") or extract_operation(pred_action_repr or "")
    pred_value  = extract_value(pred_action_repr or "")

    # html_snippet: first 500 chars of the DOM for quick inspection
    dom_full = row.get("cleaned_html", "") or row.get("raw_html", "") or ""
    html_snippet = dom_full[:500] if dom_full else None

    # Screenshot path (null when the dataset stores raw bytes without a path)
    screenshot = screenshot_path_from_field(row.get("screenshot"))

    # Candidate strings as "idx: repr" to match the schema format
    candidate_strs = [f"{i}: {r}" for i, r in enumerate(candidates)]

    return {
        "id":           row.get("annotation_id"),
        "model":        model_name,
        "split":        split,
        "input_task":   row.get("confirmed_task", ""),
        "html_snippet": html_snippet,
        "screenshot":   screenshot,
        "candidates":   candidate_strs,
        "gt_index":     gt_idx,
        "gt_element":   gt_element,
        "gt_action":    gt_action,
        "gt_value":     gt_value,
        "pred_text":    raw_output,
        "pred_idx":     pred_idx,
        "pred_element": pred_element,
        "pred_action":  pred_action,
        "pred_value":   pred_value,
        "task_success": False,   # always False — this record is in the failure log
        "value_states": None,    # reserved for future hidden-state logging
        "timestamp":    timestamp,
    }


def is_correct_prediction(gt_idx: int, pred_idx: Optional[int]) -> bool:
    """Return True iff the top-1 prediction matches the ground-truth index."""
    return pred_idx is not None and pred_idx == gt_idx


# ---------------------------------------------------------------------------
# Constrained decoding helper
# ---------------------------------------------------------------------------

def build_json_prefix_fn(processor: Any) -> Optional[Any]:
    """Return a prefix_allowed_tokens_fn for lm-format-enforcer, or None."""
    if not _LM_FORMAT_ENFORCER_AVAILABLE:
        import warnings
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


# ---------------------------------------------------------------------------
# Single-baseline runner
# ---------------------------------------------------------------------------

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
        else baseline.get("max_new_tokens", 256)
    )
    do_sample = args.temperature > 0
    generation_config: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_config["temperature"] = args.temperature

    if args.json_constrained:
        prefix_fn = build_json_prefix_fn(processor)
        if prefix_fn is not None:
            generation_config["prefix_allowed_tokens_fn"] = prefix_fn

    total_examples = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_dir = Path(args.output_dir)
    out_file     = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
    failure_file = out_dir / f"failure_log_{baseline['name']}_{split}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if args.resume and out_file.exists():
        with out_file.open("r", encoding="utf8") as handle:
            start_idx = sum(1 for _ in handle)
        if start_idx >= total_examples:
            print(
                f"Skipping {baseline['name']} on {split}: "
                f"already complete ({start_idx}/{total_examples})"
            )
            return 0

    iterator = range(start_idx, total_examples)
    written = 0
    multimodal = baseline["multimodal"]

    with (
        out_file.open("a", encoding="utf8") as pred_handle,
        failure_file.open("a", encoding="utf8") as fail_handle,
    ):
        for idx in tqdm(iterator, desc=f"{baseline['name']} | {split}"):
            row = dataset[idx]
            # Capture UTC timestamp before inference so it reflects the call time.
            timestamp = datetime.now(timezone.utc).isoformat()

            messages = build_qwen_messages(
                row=row,
                multimodal=multimodal,
                max_dom_chars=args.max_dom_chars,
            )

            raw_output = run_qwen_inference(
                model=model,
                processor=processor,
                messages=messages,
                generation_config=generation_config,
                multimodal=multimodal,
            )

            parsed = parse_model_output(
                raw_output,
                num_candidates=len(row.get("action_reprs", [])),
            )

            # ── Main prediction record ────────────────────────────────────────
            record = make_prediction_record(row, split, baseline["name"], parsed, raw_output)
            pred_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_handle.flush()
            written += 1

            # ── Failure log: write only when top-1 prediction is wrong ────────
            raw_gt = row.get("target_action_index")
            gt_idx = int(raw_gt) if raw_gt is not None else -1
            pred_indices: List[int] = parsed.get("pred_action_indices") or []
            pred_idx = pred_indices[0] if pred_indices else None

            if not is_correct_prediction(gt_idx, pred_idx):
                failure_record = make_failure_record(
                    row=row,
                    split=split,
                    model_name=args.model_name,
                    parsed=parsed,
                    raw_output=raw_output,
                    timestamp=timestamp,
                )
                fail_handle.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
                fail_handle.flush()

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    model, processor = load_qwen_model(args.model_name, dtype, args.quantization)

    out_dir = Path(args.output_dir)
    for split in args.splits:
        print(f"\nLoading split: {split}")
        dataset = load_split_dataset(args.data_dir, split)

        for baseline in BASELINES:
            print(f"Running baseline '{baseline['name']}' on {split}")
            out_file     = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
            failure_file = out_dir / f"failure_log_{baseline['name']}_{split}.jsonl"
            num_written = run_single_baseline(
                model, processor, dataset, split, baseline, args
            )
            print(f"Predictions : {out_file} (new rows written: {num_written})")
            print(f"Failure log : {failure_file}")


if __name__ == "__main__":
    main()
