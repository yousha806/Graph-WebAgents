"""Evaluate next-action prediction outputs from intern_axtree_ablations.py.

Expected prediction JSONL schema (one JSON object per line):
{
  "split": "test_task",
  "baseline": "intern_image_allinputs_axtree",
  "annotation_id": "...",
  "action_uid": "...",
  "gt_action_index": 3,
  "pred_action_index": 2,
  "gt_action": "CLICK",
  "pred_action": "TYPE",
  ...
}

Examples:
  python inference/eval_next_action.py \
    --input inference_outputs/intern_axtree_ablations/preds_intern_image_allinputs_axtree_test_task.jsonl

  python inference/eval_next_action.py \
    --input inference_outputs/intern_axtree_ablations \
    --pattern "preds_*.jsonl" \
    --out inference_outputs/intern_axtree_ablations/metrics_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate next-action prediction JSONL outputs")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL file OR directory containing JSONL files",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern when --input is a directory",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON summary",
    )
    return parser.parse_args()


def safe_load(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except Exception:
        return None


def action_index_accuracy(records: List[Dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for row in records:
        gt_idx = row.get("gt_action_index")
        pred_idx = row.get("pred_action_index")
        if gt_idx is None or pred_idx is None:
            continue
        total += 1
        if int(gt_idx) == int(pred_idx):
            correct += 1
    return correct / total if total else 0.0


def action_label_accuracy(records: List[Dict[str, Any]]) -> float:
    total = 0
    correct = 0
    for row in records:
        gt_action = row.get("gt_action")
        pred_action = row.get("pred_action")
        if gt_action is None or pred_action is None:
            continue
        total += 1
        if str(gt_action).upper() == str(pred_action).upper():
            correct += 1
    return correct / total if total else 0.0


def parse_failure_rate(records: List[Dict[str, Any]]) -> float:
    total = len(records)
    if total == 0:
        return 0.0
    failures = sum(1 for row in records if row.get("pred_action_index") is None)
    return failures / total


def top3_element_accuracy(records: List[Dict[str, Any]]) -> float:
    """Element index exact match (Top-3 collapses to Top-1 for single-output models)."""
    total = 0
    correct = 0
    for row in records:
        gt_idx = row.get("gt_action_index")
        pred_idx = row.get("pred_action_index")
        if gt_idx is None or pred_idx is None:
            continue
        total += 1
        if int(gt_idx) == int(pred_idx):
            correct += 1
    return correct / total if total else 0.0


def mean_reciprocal_rank(records: List[Dict[str, Any]]) -> float:
    """MRR over steps: rank=1 if prediction correct, else worst-case rank=len(candidates)."""
    total = 0
    rr_sum = 0.0
    for row in records:
        gt_idx = row.get("gt_action_index")
        pred_idx = row.get("pred_action_index")
        n = len(row.get("candidates") or [1])
        if gt_idx is None:
            continue
        total += 1
        if pred_idx is not None and int(gt_idx) == int(pred_idx):
            rr_sum += 1.0
        else:
            rr_sum += 1.0 / max(n, 1)
    return rr_sum / total if total else 0.0


def task_success_rate(records: List[Dict[str, Any]]) -> float:
    """Fraction of tasks (annotation_ids) where every step is predicted correctly."""
    from collections import defaultdict
    tasks: Dict[str, List[bool]] = defaultdict(list)
    for row in records:
        ann_id = row.get("annotation_id")
        gt_idx = row.get("gt_action_index")
        if ann_id is None or gt_idx is None:
            continue
        pred_idx = row.get("pred_action_index")
        tasks[ann_id].append(pred_idx is not None and int(pred_idx) == int(gt_idx))
    if not tasks:
        return 0.0
    return sum(all(steps) for steps in tasks.values()) / len(tasks)


def step_accuracy(records: List[Dict[str, Any]]) -> float:
    """Fraction of steps where BOTH element index AND action type are correct."""
    total = 0
    correct = 0
    for row in records:
        gt_idx = row.get("gt_action_index")
        gt_action = row.get("gt_action")
        if gt_idx is None or gt_action is None:
            continue
        total += 1
        pred_idx = row.get("pred_action_index")
        pred_action = row.get("pred_action")
        elem_ok = pred_idx is not None and int(pred_idx) == int(gt_idx)
        action_ok = pred_action is not None and str(gt_action).upper() == str(pred_action).upper()
        if elem_ok and action_ok:
            correct += 1
    return correct / total if total else 0.0


def per_action_prf(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    labels = set()
    for row in records:
        gt = row.get("gt_action")
        pred = row.get("pred_action")
        if gt:
            labels.add(str(gt).upper())
        if pred:
            labels.add(str(pred).upper())

    results: Dict[str, Dict[str, float]] = {}
    for label in sorted(labels):
        tp = fp = fn = 0
        for row in records:
            gt = str(row.get("gt_action") or "").upper() or None
            pred = str(row.get("pred_action") or "").upper() or None
            if pred == label and gt == label:
                tp += 1
            if pred == label and gt != label:
                fp += 1
            if gt == label and pred != label:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    return results


def evaluate_file(path: Path) -> Dict[str, Any]:
    raw_lines = []
    parsed = []

    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            raw_lines.append(line)
            item = safe_load(line)
            if item is not None:
                parsed.append(item)

    baselines = sorted({row.get("baseline") for row in parsed if row.get("baseline")})
    splits = sorted({row.get("split") for row in parsed if row.get("split")})

    metrics = {
        "file": str(path),
        "num_lines": len(raw_lines),
        "num_parsed": len(parsed),
        "json_parse_failure_rate": 1.0 - (len(parsed) / len(raw_lines)) if raw_lines else 0.0,
        "prediction_parse_failure_rate": parse_failure_rate(parsed),
        "ActionAcc": action_label_accuracy(parsed),
        "Top3Elem": top3_element_accuracy(parsed),
        "MRR": mean_reciprocal_rank(parsed),
        "TaskSuccess": task_success_rate(parsed),
        "StepAcc": step_accuracy(parsed),
        "per_action": per_action_prf(parsed),
        "baselines": baselines,
        "splits": splits,
    }
    return metrics


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_list:
        return {
            "num_files": 0,
            "macro_ActionAcc": 0.0,
            "macro_Top3Elem": 0.0,
            "macro_MRR": 0.0,
            "macro_TaskSuccess": 0.0,
            "macro_StepAcc": 0.0,
            "macro_prediction_parse_failure_rate": 0.0,
        }

    return {
        "num_files": len(metrics_list),
        "macro_ActionAcc": mean([m["ActionAcc"] for m in metrics_list]),
        "macro_Top3Elem": mean([m["Top3Elem"] for m in metrics_list]),
        "macro_MRR": mean([m["MRR"] for m in metrics_list]),
        "macro_TaskSuccess": mean([m["TaskSuccess"] for m in metrics_list]),
        "macro_StepAcc": mean([m["StepAcc"] for m in metrics_list]),
        "macro_prediction_parse_failure_rate": mean([m["prediction_parse_failure_rate"] for m in metrics_list]),
    }


def resolve_input_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        matches = sorted(Path(p) for p in glob.glob(str(input_path / pattern)))
        return [p for p in matches if p.is_file()]
    raise FileNotFoundError(f"Input not found: {input_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    files = resolve_input_files(input_path, args.pattern)

    if not files:
        raise FileNotFoundError(f"No JSONL files found under: {input_path} with pattern '{args.pattern}'")

    per_file = [evaluate_file(path) for path in files]
    summary = {
        "aggregate": aggregate(per_file),
        "files": per_file,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote metrics to {out_path}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
