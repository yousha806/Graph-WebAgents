"""Run InternVL2-8B image-only (vision) baseline for Mind2Web next-action prediction.

Baseline #1: Screenshot-Only (Vision Baseline)

This script implements a vision baseline that uses:
  - Screenshot image
  - Task description (confirmed_task)
  - Candidate action list (action_reprs)

NO DOM, HTML, or AXTree text is provided.  The model must ground its
prediction purely from the visual content of the screenshot.
Candidates are still presented so the model can pick an index (standard
Mind2Web evaluation protocol), but no page-structure text aids it.

This establishes how much element grounding is possible from visual
perception alone, and is the weakest expected baseline — useful for
showing the ceiling of pure vision approaches.

Dataset: osunlp/Multimodal-Mind2Web

It evaluates by default on all three test splits:
  - test_task
  - test_website
  - test_domain

Expected input data:
  - Multimodal Mind2Web splits under data/mind2web/<split>
    (produced by download_mind2web.py)

Output:
  - JSONL files compatible with eval_next_action.py
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

# Some Mind2Web screenshots are very large; raise PIL's limit to avoid
# DecompressionBombWarning / DecompressionBombError.
Image.MAX_IMAGE_PIXELS = 300_000_000

# Optional: lm-format-enforcer for hard JSON-schema-constrained decoding.
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
        top3_action_indices: List[int]
        action_type: str
        target_element: str

    _JSON_SCHEMA = _PredSchema.model_json_schema()
except ImportError:
    _JSON_SCHEMA = {
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

# InternVL2 image preprocessing constants (ImageNet normalisation, 448px tiles)
_INTERN_INPUT_SIZE = 448
_INTERN_MEAN = (0.485, 0.456, 0.406)
_INTERN_STD  = (0.229, 0.224, 0.225)

DEFAULT_SPLITS = ["test_website"]

BASELINES = [
    {
        "name": "intern_image_only",
        "max_new_tokens": 256,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="InternVL2 image-only (vision) baseline for Mind2Web"
    )
    parser.add_argument(
        "--model_name",
        default="OpenGVLab/InternVL2-8B",
        help="Hugging Face model id for InternVL2 checkpoint",
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
        default="inference_outputs/intern_image_only_baseline",
        help="Where to write JSONL predictions",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=None,
        help="Override the per-baseline max_new_tokens.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature. 0.0 = greedy decoding (default).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per split")
    parser.add_argument(
        "--json_constrained",
        action="store_true",
        help="Enforce JSON-schema-valid output via constrained decoding (lm-format-enforcer).",
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
    """Patch InternVL2's cached model code to avoid .item() on meta tensors."""
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

    use_flash_attn = False

    gpu_cc = torch.cuda.get_device_capability(0)
    print(f"Loading Intern model: {model_name} (quantization={quantization}, dtype={dtype})")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"compute={gpu_cc[0]}.{gpu_cc[1]}  |  "
          f"VRAM free: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB  |  "
          f"flash_attn=False")

    _patch_internvl2_meta_device_issue()

    if quantization_config is not None:
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "device_map": {"": 0},
            "quantization_config": quantization_config,
            "use_flash_attn": use_flash_attn,
        }
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
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


def load_split_dataset(data_dir: str, split: str):
    split_path = Path(data_dir) / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"Mind2Web split not found: {split_path}. "
            "Run download_mind2web.py for this split first."
        )
    return load_from_disk(str(split_path))


def to_pil_image(image_field: Any) -> Optional[Image.Image]:
    """Convert a dataset image field to a PIL Image, or return None if unsupported."""
    if image_field is None:
        return None

    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, bytes):
        return Image.open(io.BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, str):
        if image_field.strip():
            return Image.open(image_field).convert("RGB")
        return None

    if isinstance(image_field, dict):
        path = image_field.get("path")
        if path:
            return Image.open(path).convert("RGB")
        raw_bytes = image_field.get("bytes")
        if raw_bytes is not None:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    return None


def candidate_lines(action_reprs: List[str]) -> str:
    return "\n".join(f"{idx}: {candidate}" for idx, candidate in enumerate(action_reprs))


# ── Prompt template (Screenshot + Task + Candidates — NO DOM/HTML/AXTree) ────

_JSON_INSTRUCTION = '''\
Respond with a single JSON object and nothing else. Do NOT include any text outside the JSON.
Choose the candidate index that best matches the next action for the given task.
Use ONLY the screenshot to understand the page — no HTML or DOM text is available.
Rank your top-3 best candidate indices from most to least confident.
If only one candidate clearly matches, you may list fewer.
IMPORTANT: all JSON values must be properly quoted strings or arrays of integers.
{
  "top3_action_indices": [<integer: best index>, <integer: 2nd best>, <integer: 3rd best>],
  "action_type": "<action verb for the best candidate: CLICK, TYPE, SELECT, HOVER, SCROLL>",
  "target_element": "<brief description of the element being acted on>"
}'''


def build_prompt(row: Dict[str, Any]) -> str:
    """Build prompt using image + task + candidates (NO DOM/HTML/AXTree).

    The model must rely purely on visual perception of the screenshot to
    determine which candidate action is correct.
    """
    task = row.get("confirmed_task", "")
    actions = row.get("action_reprs", [])

    return (
        "You are predicting the next web action for a browser agent.\n"
        "You are given the task description, a screenshot of the current webpage, "
        "and a list of candidate actions. Choose the correct action index.\n"
        "You must rely ONLY on the screenshot to understand the page layout and "
        "content — no HTML or DOM text is provided.\n\n"
        f"Task:\n{task}\n\n"
        "Screenshot:\n<image>\n\n"
        f"Candidate actions:\n{candidate_lines(actions)}\n\n"
        f"{_JSON_INSTRUCTION}"
    )


def parse_model_output(raw_output: str, num_candidates: int) -> Dict[str, Any]:
    """Parse the model's JSON response into a structured dict.

    Tries JSON first, then falls back to regex.

    Returns a dict with keys:
        pred_action_indices : List[int]
        action_type         : str | None
        target_element      : str | None
        parse_error         : str | None
    """
    result: Dict[str, Any] = {
        "pred_action_indices": [],
        "action_type": None,
        "target_element": None,
        "parse_error": None,
    }
    if not raw_output:
        result["parse_error"] = "empty output"
        return result

    # --- 1. JSON parse (preferred) ---
    json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.S)
    json_match = re.search(r"\{.*\}", json_text, flags=re.S)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            raw_indices = parsed.get("top3_action_indices") or []
            if isinstance(raw_indices, int):
                raw_indices = [raw_indices]
            valid_indices = [
                int(i) for i in raw_indices
                if isinstance(i, (int, float)) and 0 <= int(i) < num_candidates
            ][:3]
            result["pred_action_indices"] = valid_indices
            if not valid_indices:
                result["parse_error"] = f"top3_action_indices {raw_indices!r} contained no valid indices in [0, {num_candidates})"
            atype = parsed.get("action_type")
            result["action_type"] = str(atype).upper() if atype else None
            result["target_element"] = parsed.get("target_element") or None
            return result
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"JSON decode error: {exc}"

    # --- 2. Regex fallback ---
    idx_match = re.search(r'"top3_action_indices"\s*:\s*\[([^\]]+)\]', raw_output)
    if idx_match:
        raw_list = idx_match.group(1)
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_list)]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        if not valid:
            result["parse_error"] = (result["parse_error"] or "") + " (partial-JSON: indices out of range)"
        else:
            result["parse_error"] = (result["parse_error"] or "") + " (used partial-JSON fallback)"
    else:
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_output)]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + (
            " (used regex fallback)" if valid else " (no valid index found)"
        )

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


def extract_operation(action_repr: str) -> Optional[str]:
    if not action_repr:
        return None
    match = re.match(r"\s*([A-Za-z_]+)", action_repr)
    if not match:
        return None
    return match.group(1).upper()


def normalize_gt_operation(operation_field: Any) -> Optional[str]:
    """Extract and normalise the action verb from the Mind2Web operation field."""
    if operation_field is None:
        return None

    if isinstance(operation_field, str):
        stripped = operation_field.strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            try:
                operation_field = json.loads(stripped)
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


def pil_to_pixel_values(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    """Preprocess a PIL image into the InternVL2 pixel_values tensor."""
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
) -> Optional[Any]:
    """Return a prefix_allowed_tokens_fn for lm-format-enforcer, or None."""
    if not _LM_FORMAT_ENFORCER_AVAILABLE:
        import warnings
        warnings.warn(
            "--json_constrained requested but lm-format-enforcer is not installed. "
            "Falling back to unconstrained decoding.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    parser = JsonSchemaParser(_JSON_SCHEMA)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def run_intern_chat(
    model: Any,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt: str,
    generation_config: Dict[str, Any],
) -> str:
    """Call InternVL2 chat() with a preprocessed pixel_values tensor."""
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
    pred_idx = pred_indices[0] if pred_indices else None
    pred_action_repr = (
        candidates[pred_idx]
        if pred_idx is not None and 0 <= pred_idx < len(candidates)
        else None
    )
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
        "pred_action_index": pred_idx,
        "pred_action_indices": pred_indices,
        "pred_action": pred_action,
        "pred_action_repr": pred_action_repr,
        "pred_target_element": parsed.get("target_element"),
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
    if do_sample:
        generation_config["temperature"] = args.temperature
    if args.json_constrained:
        prefix_fn = build_json_prefix_fn(tokenizer)
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
            image = to_pil_image(row.get("screenshot"))
            if image is None:
                tqdm.write(f"[WARN] Skipping idx={idx}: screenshot is None or unsupported format")
                continue
            try:
                pixel_values = pil_to_pixel_values(image, model_dtype)
            except Exception as exc:
                tqdm.write(f"[WARN] Skipping idx={idx}: image preprocessing failed: {exc}")
                continue
            prompt = build_prompt(row=row)

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
        dataset = load_split_dataset(args.data_dir, split)

        for baseline in BASELINES:
            print(f"Running baseline '{baseline['name']}' on {split}")
            out_file = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
            num_written = run_single_baseline(model, tokenizer, dataset, split, baseline, args, dtype)
            print(f"Updated predictions: {out_file} (new rows written: {num_written})")


if __name__ == "__main__":
    main()
