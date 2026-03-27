"""Run Qwen3-VL-8B-Instruct CoT ablations for Mind2Web next-action prediction.

Runs all 6 combinations of:
  temperatures   : 0.1, 0.3, 0.5
  max_new_tokens : 256, 512

Uses vLLM for fast batched inference. Single-pass: the model generates a CoT
JSON response containing "reasoning" and "top3_action_indices". The top-3
ordering is the model's own self-ranking from its reasoning chain.

Each combination writes its own JSONL file under --output_dir:
  preds_qwen_cot_t<temp>_n<tokens>_<split>.jsonl

Usage:
  python inference/qwen_cot_ablations.py \
      --data_dir data/mind2web \
      --splits test_website \
      --output_dir inference_outputs/qwen_cot_ablations
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_from_disk
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SPLITS = ["test_website"]

VARIATIONS = [
    {"temperature": t, "max_new_tokens": n}
    for t in [0.1, 0.3, 0.5]
    for n in [256, 512]
]

_JSON_SCHEMA_COT = '''\
Respond with a single JSON object and nothing else. Do NOT include any text outside the JSON.
"reasoning" must appear first so you can think before committing to indices.
Choose the candidate index that best matches the next action for the given task.
IMPORTANT: all JSON values must be properly quoted strings or arrays of integers.
{
  "reasoning": "<step-by-step explanation of which candidates match the task>",
  "top3_action_indices": [<integer: best index>, <integer: 2nd best>, <integer: 3rd best>],
  "action_type": "<action verb for the best candidate: CLICK, TYPE, SELECT, HOVER, SCROLL>",
  "target_element": "<brief description of the element being acted on>"
}'''


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B CoT ablations via vLLM (temperature x max_new_tokens)"
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--data_dir", default="data/mind2web",
                        help="Directory produced by download_mind2web.py")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                        choices=["train", "test_task", "test_website", "test_domain"])
    parser.add_argument("--output_dir", default="inference_outputs/qwen_cot_ablations")
    parser.add_argument("--max_html_chars", type=int, default=15000)
    parser.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="Max examples per split")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-written rows in existing JSONL files")
    return parser.parse_args()


def load_vllm_model(model_name: str, dtype: str, gpu_memory_utilization: float, max_model_len: int):
    from vllm import LLM
    from transformers import AutoProcessor

    print(f"Loading {model_name} with vLLM (dtype={dtype}, max_model_len={max_model_len})...")
    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return llm, processor


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_messages(row: Dict[str, Any], max_html_chars: int) -> List[Dict]:
    task = row.get("confirmed_task", "")
    html = (row.get("cleaned_html") or "")[:max_html_chars]
    candidates = row.get("action_reprs", [])
    candidate_text = "\n".join(f"{i}: {c}" for i, c in enumerate(candidates))

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["screenshot"]},
                {
                    "type": "text",
                    "text": (
                        "You are predicting the next web action for a browser agent.\n"
                        "Given the task, screenshot, HTML, and candidate actions, "
                        "choose the correct action index.\n\n"
                        f"Task:\n{task}\n\n"
                        f"HTML:\n{html}\n\n"
                        f"Candidate actions:\n{candidate_text}\n\n"
                        f"{_JSON_SCHEMA_COT}"
                    ),
                },
            ],
        }
    ]


def build_vllm_input(row: Dict[str, Any], processor, max_html_chars: int) -> Dict:
    """Build a single vLLM input dict with prompt text and image data."""
    messages = build_messages(row, max_html_chars)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    image = row["screenshot"]
    return {"prompt": text, "multi_modal_data": {"image": image}}


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_cot_output(raw_output: str, num_candidates: int) -> Dict[str, Any]:
    """Parse the CoT JSON. top3_action_indices from the model is used directly
    as the ranking — this is the model's own self-ranked ordering."""
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

    json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.S)
    json_match = re.search(r"\{.*\}", json_text, flags=re.S)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            raw_indices = parsed.get("top3_action_indices") or []
            if isinstance(raw_indices, int):
                raw_indices = [raw_indices]
            valid = [
                int(i) for i in raw_indices
                if isinstance(i, (int, float)) and 0 <= int(i) < num_candidates
            ][:3]
            result["pred_action_indices"] = valid
            if not valid:
                result["parse_error"] = f"top3_action_indices {raw_indices!r} had no valid indices"
            atype = parsed.get("action_type")
            result["action_type"] = str(atype).upper() if atype else None
            result["target_element"] = parsed.get("target_element") or None
            result["reasoning"] = parsed.get("reasoning") or None
            return result
        except json.JSONDecodeError as exc:
            result["parse_error"] = f"JSON decode error: {exc}"

    # Fallback: recover indices and action_type from partial output
    idx_match = re.search(r'"top3_action_indices"\s*:\s*\[([^\]]+)\]', raw_output)
    if idx_match:
        all_ints = [int(m) for m in re.findall(r"-?\d+", idx_match.group(1))]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + " (partial-JSON fallback)"
    else:
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_output)]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        result["pred_action_indices"] = valid
        result["parse_error"] = (result["parse_error"] or "") + (
            " (regex fallback)" if valid else " (no valid index found)"
        )

    atype_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw_output, re.I)
    if atype_match:
        result["action_type"] = atype_match.group(1).strip().upper()
    else:
        bare = re.search(r'\b(CLICK|TYPE|SELECT(?:_OPTION)?|HOVER|SCROLL|PRESS|ENTER)\b', raw_output, re.I)
        if bare:
            result["action_type"] = bare.group(1).upper()

    return result


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def normalize_gt_operation(operation_field: Any) -> Optional[str]:
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


def extract_operation(action_repr: str) -> Optional[str]:
    if not action_repr:
        return None
    match = re.match(r"\s*([A-Za-z_]+)", action_repr)
    return match.group(1).upper() if match else None


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
        "split": split,
        "baseline": baseline_name,
        "annotation_id": row.get("annotation_id"),
        "action_uid": row.get("action_uid"),
        "gt_action_index": gt_idx,
        "gt_action": normalize_gt_operation(row.get("operation")),
        "gt_action_repr": row.get("target_action_reprs"),
        "pred_action_index": pred_idx,
        "pred_action_indices": pred_indices,
        "pred_action": pred_action,
        "pred_action_repr": pred_action_repr,
        "pred_target_element": parsed.get("target_element"),
        "reasoning": parsed.get("reasoning"),
        "parse_error": parsed.get("parse_error"),
        "raw_output": raw_output,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_variation(
    llm,
    processor,
    dataset,
    split: str,
    variation: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    from vllm import SamplingParams

    temperature = variation["temperature"]
    max_new_tokens = variation["max_new_tokens"]
    t_str = str(temperature).replace(".", "")
    name = f"qwen_cot_t{t_str}_n{max_new_tokens}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"preds_{name}_{split}.jsonl"

    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    start_idx = 0
    if args.resume and out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= total:
            print(f"  Skipping {name} on {split}: already complete ({start_idx}/{total})")
            return

    print(f"  Building inputs for {name} on {split} [{start_idx}/{total}]...")
    rows = []
    vllm_inputs = []
    for idx in tqdm(range(start_idx, total), desc="building prompts"):
        row = dataset[idx]
        rows.append(row)
        try:
            vllm_inputs.append(build_vllm_input(row, processor, args.max_html_chars))
        except Exception as exc:
            print(f"    [ERROR] build idx={idx}: {exc}")
            vllm_inputs.append(None)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    print(f"  Generating {len(vllm_inputs)} samples with vLLM...")
    valid_pairs = [(i, inp) for i, inp in enumerate(vllm_inputs) if inp is not None]
    raw_outputs = [""] * len(rows)

    if valid_pairs:
        indices, inputs = zip(*valid_pairs)
        outputs = llm.generate(list(inputs), sampling_params)
        for i, output in zip(indices, outputs):
            raw_outputs[i] = output.outputs[0].text

    print(f"  Writing results to {out_file}...")
    with out_file.open("a", encoding="utf-8") as handle:
        for row, raw_output in zip(rows, raw_outputs):
            num_candidates = len(row.get("action_reprs", []))
            parsed = parse_cot_output(raw_output, num_candidates)
            record = make_prediction_record(row, split, name, parsed, raw_output)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Saved: {out_file}")


def main() -> None:
    args = parse_args()
    llm, processor = load_vllm_model(
        args.model_name, args.dtype, args.gpu_memory_utilization, args.max_model_len
    )

    for split in args.splits:
        split_path = Path(args.data_dir) / split
        print(f"\nLoading split: {split_path}")
        dataset = load_from_disk(str(split_path))
        for variation in VARIATIONS:
            run_variation(llm, processor, dataset, split, variation, args)

    print("\nAll variations complete.")


if __name__ == "__main__":
    main()
