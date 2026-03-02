"""Run InternVL2 AXTree ablations for Mind2Web next-action prediction.

This script implements exactly two ablations on the Intern model:
1) image + all inputs + AXTree
2) image + all inputs + AXTree + chain-of-thought

It evaluates by default on all three test splits:
- test_task
- test_website
- test_domain

Expected input data:
- Precomputed AXTree splits under data/mind2web_axtree/<split>
  (produced by precompute_axtree.py)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

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

    class _PredNoCoT(_BaseModel):
        top3_action_indices: List[int]   # ranked best→worst, length 1-3
        action_type: str
        target_element: str

    class _PredCoT(_BaseModel):
        reasoning: str
        top3_action_indices: List[int]   # ranked best→worst, length 1-3
        action_type: str
        target_element: str

    _SCHEMA_NO_COT = _PredNoCoT.model_json_schema()
    _SCHEMA_COT    = _PredCoT.model_json_schema()
except ImportError:
    # Pydantic unavailable: fall back to hand-written JSON Schema dicts.
    _SCHEMA_NO_COT = {
        "type": "object",
        "properties": {
            "top3_action_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 3,
            },
            "action_type":     {"type": "string"},
            "target_element":  {"type": "string"},
        },
        "required": ["top3_action_indices", "action_type", "target_element"],
        "additionalProperties": False,
    }
    _SCHEMA_COT = {
        "type": "object",
        "properties": {
            "reasoning":       {"type": "string"},
            "top3_action_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 3,
            },
            "action_type":     {"type": "string"},
            "target_element":  {"type": "string"},
        },
        "required": ["reasoning", "top3_action_indices", "action_type", "target_element"],
        "additionalProperties": False,
    }

# InternVL2 image preprocessing constants (ImageNet normalisation, 448px tiles)
_INTERN_INPUT_SIZE = 448
_INTERN_MEAN = (0.485, 0.456, 0.406)
_INTERN_STD  = (0.229, 0.224, 0.225)

DEFAULT_SPLITS = ["test_website"]

BASELINES = [
    {
        "name": "intern_image_allinputs_axtree",
        "cot": False,
        # Non-CoT output is a single integer (e.g. "3") — 32 tokens is generous.
        "max_new_tokens": 32,
    },
    {
        "name": "intern_image_allinputs_axtree_cot",
        "cot": True,
        # CoT needs a full reasoning chain before FINAL_ANSWER: <idx>.
        # 512 tokens fits ~350 words of reasoning + the answer tag.
        "max_new_tokens": 512,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL2 AXTree ablations for Mind2Web")
    parser.add_argument(
        "--model_name",
        default="OpenGVLab/InternVL2-8B",
        help="Hugging Face model id for InternVL2 checkpoint",
    )
    parser.add_argument(
        "--axtree_dir",
        default="data/mind2web_axtree",
        help="Directory containing AXTree-precomputed splits saved with datasets.save_to_disk",
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
        default="inference_outputs/intern_axtree_ablations",
        help="Where to write JSONL predictions",
    )
    parser.add_argument(
        "--max_axtree_chars", type=int, default=12000,
        help=(
            "Max characters of axtree_text to include in the prompt. "
            "Matches the precompute_axtree.py default (max_chars=12000) so nothing is wasted. "
            "\n\nToken budget for InternVL2-8B (32768-token context, 1-tile / 256 image tokens):\n"
            "  image=256, boilerplate+task+candidates≈900, max_new_tokens≤512, margin=500\n"
            "  → ~30,600 tokens / ~107,000 chars available for axtree.\n"
            "Current practical ceiling = 12,000 chars (what precompute stores). "
            "To go higher, rerun precompute_axtree.py with --max_chars <N> first."
        ),
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override the per-baseline max_new_tokens (32 non-CoT, 512 CoT). "
             "Rarely needed; set only to override the baseline defaults.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. 0.0 = greedy decoding (default, recommended for "
             "deterministic evaluation). Values > 0 enable sampling.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per split")
    parser.add_argument(
        "--json_constrained",
        action="store_true",
        help=(
            "Enforce JSON-schema-valid output via constrained decoding (lm-format-enforcer). "
            "Guarantees the model emits valid JSON — no regex fallback needed. "
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


def _patch_internvl2_meta_device_issue() -> None:
    """Patch InternVL2's cached model code to avoid .item() on meta tensors.

    InternVL2's InternVisionEncoder.__init__ calls:
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, ...)]

    In transformers >= 4.45, model __init__ runs inside accelerate's meta-device
    context even when low_cpu_mem_usage=False, causing:
        RuntimeError: Tensor.item() cannot be called on meta tensors

    This patch replaces that line with pure-Python arithmetic — equivalent in
    value, but creates no tensors, so it is safe in any device context.
    """
    BUGGY = (
        "dpr = [x.item() for x in torch.linspace("
        "0, config.drop_path_rate, config.num_hidden_layers)]"
    )
    FIXED = (
        "n = config.num_hidden_layers\n"
        "        dpr = ([0.0] + [config.drop_path_rate * i / (n - 1) for i in range(1, n)]"
        " if n > 1 else [0.0])"
    )
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    search_dir = hf_home / "modules" / "transformers_modules"
    if not search_dir.is_dir():
        return
    for fpath in search_dir.rglob("modeling_intern_vit.py"):
        content = fpath.read_text(encoding="utf-8")
        if BUGGY in content:
            fpath.write_text(content.replace(BUGGY, FIXED), encoding="utf-8")
            print(f"[patch] Applied meta-device fix to: {fpath}")


def load_intern_model(model_name: str, dtype: torch.dtype, quantization: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for InternVL2-8B inference but torch.cuda.is_available() returned False.\n"
            f"  torch version    : {torch.__version__}\n"
            f"  torch CUDA build : {torch.version.cuda}\n"
            f"  Likely cause     : driver version too old for this CUDA build, or kernel module\n"
            f"                     not loaded (reboot required after driver install).\n"
            f"  cu121 needs driver >= 520; cu124 needs >= 550; cu128 needs >= 570."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    quantization_config = build_quantization_config(quantization, dtype)

    use_flash_attn = False  # flash-attn disabled; using standard attention

    gpu_cc = torch.cuda.get_device_capability(0)
    print(f"Loading Intern model: {model_name} (quantization={quantization}, dtype={dtype})")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"compute={gpu_cc[0]}.{gpu_cc[1]}  |  "
          f"VRAM free: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB  |  "
          f"flash_attn=False")

    # Patch InternVL2's cached model file to fix the .item() on meta tensor crash.
    # InternVisionEncoder.__init__ calls torch.linspace(...).item() which fails
    # when transformers initialises the model on a meta device (all transformers
    # >= 4.37 do this by default). The patch replaces it with pure-Python arithmetic.
    _patch_internvl2_meta_device_issue()

    if quantization_config is not None:

        # Use {"": 0} (dict form, not "auto") — "auto" triggers accelerate's
        # meta-device dispatch which breaks InternVL2's __init__ (.item() on linspace).
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "device_map": {"": 0},
            "quantization_config": quantization_config,
            "use_flash_attn": use_flash_attn,
        }
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    else:
        # Non-quantized path: load to CPU, then move to GPU.
        # Do NOT use device_map here — newer transformers (>= 4.46) runs
        # caching_allocator_warmup() when device_map is set, which calls
        # model.all_tied_weights_keys (missing on InternVLChatModel → AttributeError).
        # The meta-device .item() crash in InternVisionEncoder.__init__ is handled
        # by _patch_internvl2_meta_device_issue() called above.
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=False,
            use_flash_attn=use_flash_attn,
        )
        model = model.to("cuda")

    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")

    if not hasattr(model, "chat"):
        raise RuntimeError(
            f"Model '{model_name}' does not expose chat(). "
            "Use an InternVL2 checkpoint with trust_remote_code chat API."
        )
    return model, tokenizer


def load_split_dataset(axtree_dir: str, split: str):
    split_path = Path(axtree_dir) / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"AXTree split not found: {split_path}. "
            "Run precompute_axtree.py for this split first."
        )
    return load_from_disk(str(split_path))


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


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) > max_chars:
        import logging
        logging.warning(
            "build_prompt: axtree_text (%d chars) exceeds max_axtree_chars=%d "
            "and will be clipped mid-line. Consider increasing --max_axtree_chars.",
            len(text),
            max_chars,
        )
        return text[:max_chars]
    return text


def candidate_lines(action_reprs: List[str]) -> str:
    return "\n".join(f"{idx}: {candidate}" for idx, candidate in enumerate(action_reprs))


# JSON schema the model is asked to emit, templated per baseline type.
_JSON_SCHEMA_NO_COT = '''\
Respond with a single JSON object and nothing else.
Rank your top-3 best candidate indices from most to least confident.
If only one candidate clearly matches, you may list fewer.
{
  "top3_action_indices": [<best index>, <2nd best>, <3rd best>],
  "action_type": <action verb for the best candidate, e.g. "CLICK", "TYPE", "SELECT">,
  "target_element": <brief description of the element being acted on>
}'''

_JSON_SCHEMA_COT = '''\
Respond with a single JSON object and nothing else.
"reasoning" must appear first so you can think before committing to indices.
Rank your top-3 best candidate indices from most to least confident.
If only one candidate clearly matches, you may list fewer.
{
  "reasoning": <step-by-step explanation of which candidates match the task>,
  "top3_action_indices": [<best index>, <2nd best>, <3rd best>],
  "action_type": <action verb for the best candidate, e.g. "CLICK", "TYPE", "SELECT">,
  "target_element": <brief description of the element being acted on>
}'''


def build_prompt(row: Dict[str, Any], use_cot: bool, max_axtree_chars: int) -> str:
    task = row.get("confirmed_task", "")
    axtree = truncate(row.get("axtree_text", ""), max_axtree_chars)
    actions = row.get("action_reprs", [])

    schema = _JSON_SCHEMA_COT if use_cot else _JSON_SCHEMA_NO_COT

    # Place <image> explicitly after the task so the model can cross-attend
    # between the screenshot and the surrounding text context. If <image> is
    # absent from the prompt string, InternVL2's chat() prepends it at position
    # 0 (before everything), which severs that cross-attention relationship.
    return (
        "You are predicting the next web action for a browser agent.\n"
        "Given task, screenshot, AXTree, and candidate actions, choose the correct action index.\n\n"
        f"Task:\n{task}\n\n"
        "Screenshot:\n<image>\n\n"
        f"AXTree:\n{axtree}\n\n"
        f"Candidate actions:\n{candidate_lines(actions)}\n\n"
        f"{schema}"
    )


def parse_model_output(raw_output: str, num_candidates: int) -> Dict[str, Any]:
    """Parse the model's JSON response into a structured dict.

    Tries JSON first (the instructed format), then falls back to regex so that
    partial or malformed responses still yield usable indices.

    Returns a dict with keys:
        pred_action_indices : List[int]    — ranked top-1..3 candidate indices (empty if unparseable)
        action_type         : str | None   — e.g. "CLICK" (for top-1)
        target_element      : str | None   — element description from model
        reasoning           : str | None   — CoT chain (None for non-CoT responses)
        parse_error         : str | None   — description of any parse failure
    """
    result: Dict[str, Any] = {
        "pred_action_indices": [],
        "action_type": None,
        "target_element": None,
        "reasoning": None,
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
            # Accept either the new list form or the legacy single-int form.
            if isinstance(raw_indices, int):
                raw_indices = [raw_indices]
            valid_indices = [
                int(i) for i in raw_indices
                if isinstance(i, (int, float)) and 0 <= int(i) < num_candidates
            ][:3]  # cap at 3, preserve order
            result["pred_action_indices"] = valid_indices
            if not valid_indices:
                result["parse_error"] = f"top3_action_indices {raw_indices!r} contained no valid indices in [0, {num_candidates})"
            atype = parsed.get("action_type")
            result["action_type"] = str(atype).upper() if atype else None
            result["target_element"] = parsed.get("target_element") or None
            result["reasoning"] = parsed.get("reasoning") or None
            return result
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"JSON decode error: {exc}"

    # --- 2. Regex fallback ---
    # Scrape all integers that fall in range; treat them as an implicit ranked list.
    all_ints = [int(m) for m in re.findall(r"-?\d+", raw_output)]
    valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
    if valid:
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + " (used regex fallback)"
    else:
        result["parse_error"] = (result["parse_error"] or "") + " (no valid index found)"
    return result


def extract_operation(action_repr: str) -> Optional[str]:
    if not action_repr:
        return None
    match = re.match(r"\s*([A-Za-z_]+)", action_repr)
    if not match:
        return None
    return match.group(1).upper()


def normalize_gt_operation(operation_field: Any) -> Optional[str]:
    if operation_field is None:
        return None

    if isinstance(operation_field, str):
        return operation_field.strip().upper() if operation_field.strip() else None

    if isinstance(operation_field, dict):
        for key in ("op", "operation", "action", "type", "name"):
            value = operation_field.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()

    return None


def pil_to_pixel_values(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    """Preprocess a PIL image into the InternVL2 pixel_values tensor.

    InternVL2's chat() expects a CUDA tensor of shape (1, 3, 448, 448) with
    ImageNet normalisation, not a raw PIL image.
    """
    transform = transforms.Compose([
        transforms.Resize(
            (_INTERN_INPUT_SIZE, _INTERN_INPUT_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_INTERN_MEAN, std=_INTERN_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(dtype=dtype, device="cuda")


def build_json_prefix_fn(
    tokenizer: Any,
    use_cot: bool,
) -> Optional[Any]:
    """Return a prefix_allowed_tokens_fn for lm-format-enforcer, or None.

    When injected into generation_config, transformers' generate() calls this
    function at every decoding step to mask tokens that would violate the
    JSON schema — making invalid JSON structurally impossible to emit.

    Works by passing the fn as ``prefix_allowed_tokens_fn`` in the dict that
    InternVL2's chat() unpacks into model.generate(**generation_config).
    """
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
    schema = _SCHEMA_COT if use_cot else _SCHEMA_NO_COT
    parser = JsonSchemaParser(schema)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def run_intern_chat(
    model: Any,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt: str,
    generation_config: Dict[str, Any],
) -> str:
    """Call InternVL2 chat() with a preprocessed pixel_values tensor.

    InternVL2's chat signature:
        chat(tokenizer, pixel_values, question, generation_config) -> str

    If pixel_values is None the model treats it as text-only.
    The model prepends '<image>\\n' to the question automatically when
    pixel_values is provided and the token is absent from the prompt.
    """
    output = model.chat(tokenizer, pixel_values, prompt, generation_config)
    if isinstance(output, tuple):
        return str(output[0])
    return str(output)


def make_prediction_record(
    row: Dict[str, Any],
    split: str,
    baseline_name: str,
    parsed: Dict[str, Any],
    raw_output: str,
) -> Dict[str, Any]:
    candidates = row.get("action_reprs", [])
    gt_idx = int(row.get("target_action_index", -1))
    pred_indices: List[int] = parsed.get("pred_action_indices") or []
    pred_idx = pred_indices[0] if pred_indices else None  # top-1
    pred_action_repr = (
        candidates[pred_idx]
        if pred_idx is not None and 0 <= pred_idx < len(candidates)
        else None
    )
    # Use the action_type the model explicitly named when available;
    # fall back to extracting it from the candidate repr string.
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
        "pred_action_index": pred_idx,            # top-1 for ElemAcc / StepAcc
        "pred_action_indices": pred_indices,      # ranked list for Top3Elem / MRR
        "pred_action": pred_action,
        "pred_action_repr": pred_action_repr,
        "pred_target_element": parsed.get("target_element"),
        "reasoning": parsed.get("reasoning"),
        # ── diagnostics ──────────────────────────────────────────────────────
        "parse_error": parsed.get("parse_error"),
        "raw_output": raw_output,
        "candidates": candidates,
    }


def run_single_baseline(
    model: Any,
    tokenizer: Any,
    dataset,
    split: str,
    baseline: Dict[str, Any],
    args: argparse.Namespace,
    model_dtype: torch.dtype,
) -> int:
    # Use the baseline's own token budget, or the CLI override if supplied.
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else baseline.get("max_new_tokens", 64)
    )
    do_sample = args.temperature > 0
    generation_config: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    # Only pass temperature when actually sampling — passing it during greedy
    # decoding triggers a transformers UserWarning and is meaningless.
    if do_sample:
        generation_config["temperature"] = args.temperature
    # Constrained decoding: inject the JSON-schema prefix fn so that
    # model.generate() can only emit tokens consistent with the schema.
    if args.json_constrained:
        prefix_fn = build_json_prefix_fn(tokenizer, use_cot=baseline["cot"])
        if prefix_fn is not None:
            generation_config["prefix_allowed_tokens_fn"] = prefix_fn

    total_examples = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_dir = Path(args.output_dir)
    out_file = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if args.resume and out_file.exists():
        with out_file.open("r", encoding="utf8") as handle:
            start_idx = sum(1 for _ in handle)
        if start_idx >= total_examples:
            print(f"Skipping {baseline['name']} on {split}: already complete ({start_idx}/{total_examples})")
            return 0

    iterator = range(start_idx, total_examples)
    written = 0

    with out_file.open("a", encoding="utf8") as handle:
        for idx in tqdm(iterator, desc=f"{baseline['name']} | {split}"):
            row = dataset[idx]
            image = to_pil_image(row["screenshot"])
            pixel_values = pil_to_pixel_values(image, model_dtype)
            prompt = build_prompt(
                row=row,
                use_cot=baseline["cot"],
                max_axtree_chars=args.max_axtree_chars,
            )

            with torch.inference_mode():
                raw_output = run_intern_chat(model, tokenizer, pixel_values, prompt, generation_config)

            parsed = parse_model_output(raw_output, num_candidates=len(row.get("action_reprs", [])))
            record = make_prediction_record(row, split, baseline["name"], parsed, raw_output)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1

    return written


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    model, tokenizer = load_intern_model(args.model_name, dtype, args.quantization)

    out_dir = Path(args.output_dir)
    for split in args.splits:
        print(f"Loading split: {split}")
        dataset = load_split_dataset(args.axtree_dir, split)

        for baseline in BASELINES:
            print(f"Running baseline '{baseline['name']}' on {split}")
            out_file = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
            num_written = run_single_baseline(model, tokenizer, dataset, split, baseline, args, dtype)
            print(f"Updated predictions: {out_file} (new rows written: {num_written})")


if __name__ == "__main__":
    main()
