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
import logging
import re
import warnings
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


def _recover_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a patched copy of a prediction record for legacy JSONL files.

    Two problems can exist in older outputs:
    1. pred_action is None even though raw_output contains "action_type": "CLICK"
       — JSON parse failed (truncated output), old code left action_type=None.
    2. pred_action_indices stored in the record don't match what raw_output
       actually contains — caused by an earlier ordering bug in the regex fallback.

    We re-derive both fields from raw_output using the same targeted-regex
    approach now used by parse_model_output, so eval is consistent with what
    new inference runs would produce.  The original row dict is never mutated.
    """
    raw = row.get("raw_output") or ""
    num_candidates = len(row.get("candidates") or [])
    if not raw or num_candidates == 0:
        return row

    patched: Dict[str, Any] = {}

    # --- re-derive pred_action_indices from raw_output ---
    # Only patch when the stored list looks wrong (differs from raw_output parse).
    idx_match = re.search(r'"top3_action_indices"\s*:\s*\[([^\]]+)\]', raw)
    if idx_match:
        raw_list = idx_match.group(1)
        all_ints = [int(m) for m in re.findall(r"-?\d+", raw_list)]
        valid = list(dict.fromkeys(v for v in all_ints if 0 <= v < num_candidates))[:3]
        stored = row.get("pred_action_indices") or []
        if valid and valid != list(stored):
            patched["pred_action_indices"] = valid
            patched["pred_action_index"] = valid[0]
            # Re-derive pred_action_repr from the corrected top-1 index.
            candidates = row.get("candidates") or []
            if 0 <= valid[0] < len(candidates):
                patched["pred_action_repr"] = candidates[valid[0]]

    # --- re-derive pred_action when it is None but raw_output has "action_type" ---
    # Use a targeted key-value regex: match the key name explicitly so we only
    # capture the action_type value, not stray words elsewhere in the output.
    # The pattern stops at the first closing quote after the value, which is
    # correct for both complete JSON and truncated-mid-target_element outputs.
    if row.get("pred_action") is None:
        recovered_action: Optional[str] = None

        # 1. Targeted key-value regex on raw_output (most reliable).
        atype_match = re.search(r'"action_type"\s*:\s*"([A-Za-z_]+)"', raw)
        if atype_match:
            recovered_action = atype_match.group(1).strip().upper()

        # 2. Extract from stored pred_action_repr (e.g. "[span]  Reggae -> CLICK").
        if recovered_action is None:
            repr_str = (
                patched.get("pred_action_repr")
                or row.get("pred_action_repr")
                or ""
            )
            if repr_str:
                m = re.search(r"->\s*([A-Za-z_]+)", repr_str)
                if m:
                    recovered_action = m.group(1).strip().upper()

        # 3. Extract from candidates[pred_action_index] — always available.
        if recovered_action is None:
            candidates = row.get("candidates") or []
            top1_idx = (
                patched.get("pred_action_index")
                if "pred_action_index" in patched
                else row.get("pred_action_index")
            )
            if top1_idx is not None and 0 <= int(top1_idx) < len(candidates):
                m = re.search(r"->\s*([A-Za-z_]+)", candidates[int(top1_idx)])
                if m:
                    recovered_action = m.group(1).strip().upper()

        # 4. Bare-word scan as absolute last resort.
        if recovered_action is None:
            bare = re.search(
                r'\b(CLICK|TYPE|SELECT(?:_OPTION)?|HOVER|SCROLL|PRESS|ENTER)\b',
                raw, re.I,
            )
            if bare:
                recovered_action = bare.group(1).upper()

        if recovered_action is not None:
            patched["pred_action"] = recovered_action

    if not patched:
        return row
    return {**row, **patched}


def _resolve_gt_action(value: Any) -> Optional[str]:
    """Return a plain uppercase action verb from a gt_action field.

    Mind2Web stores the ``operation`` dict as JSONL, so older inference runs
    (before the normalize_gt_operation fix) may have written gt_action as the
    full JSON string ``'{"OP": "CLICK", ...}'`` instead of just ``'CLICK'``.
    This helper handles both forms so that eval_next_action can still compute
    correct ActionAcc / StepAcc on those legacy output files.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            try:
                d = json.loads(stripped)
                for key in ("OP", "ORIGINAL_OP", "op", "operation", "action", "type"):
                    v = d.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip().upper()
            except json.JSONDecodeError:
                pass
            return None  # unrecognised JSON shape
        return stripped.upper()
    if isinstance(value, dict):
        for key in ("OP", "ORIGINAL_OP", "op", "operation", "action", "type"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
    return None


def action_index_accuracy(records: List[Dict[str, Any]]) -> float:
    """Top-1 element accuracy: pred_action_indices[0] == gt_action_index."""
    total = 0
    correct = 0
    for row in records:
        gt_idx = row.get("gt_action_index")
        if gt_idx is None:
            continue
        total += 1
        pred_indices = row.get("pred_action_indices") or []
        # Fall back to legacy scalar field for backwards-compatibility.
        if not pred_indices:
            pi = row.get("pred_action_index")
            pred_indices = [pi] if pi is not None else []
        if pred_indices and int(pred_indices[0]) == int(gt_idx):
            correct += 1
    return correct / total if total else 0.0


def action_label_accuracy(records: List[Dict[str, Any]]) -> float:
    """Fraction of steps where the predicted action type matches ground truth.

    Parse failures (pred_action is None) count as wrong, not skipped.
    Only rows with a known gt_action contribute to the denominator.
    """
    total = 0
    correct = 0
    for row in records:
        gt_action = _resolve_gt_action(row.get("gt_action"))
        if gt_action is None:
            continue
        total += 1
        pred_action = row.get("pred_action")
        if pred_action is not None and gt_action == str(pred_action).upper():
            correct += 1
    if total == 0:
        warnings.warn(
            "action_label_accuracy: no rows with a valid gt_action found — "
            "returning 0.0 but the denominator is 0. "
            "Check that 'gt_action' is populated in the JSONL records "
            "(normalize_gt_operation may have failed to find the operation key).",
            RuntimeWarning,
            stacklevel=2,
        )
    return correct / total if total else 0.0


def parse_failure_rate(records: List[Dict[str, Any]]) -> float:
    total = len(records)
    if total == 0:
        return 0.0
    def _no_prediction(row: Dict[str, Any]) -> bool:
        indices = row.get("pred_action_indices")
        if indices is not None:
            return len(indices) == 0
        return row.get("pred_action_index") is None
    failures = sum(1 for row in records if _no_prediction(row))
    return failures / total


def action_type_parse_failure_rate(records: List[Dict[str, Any]]) -> float:
    """Fraction of steps where pred_action is None, among rows with a valid gt_action.

    A non-zero value here is the diagnostic for why ActionAcc can be 0.0 even
    when prediction_parse_failure_rate (index parsing) is 0.0: the model output
    valid element indices (via JSON or regex fallback) but the action type was
    never extracted — most likely because JSON parsing failed and the regex
    fallback only recovered the index, leaving action_type=None.
    """
    eligible = [r for r in records if _resolve_gt_action(r.get("gt_action")) is not None]
    total = len(eligible)
    if total == 0:
        return 0.0
    failures = sum(1 for r in eligible if r.get("pred_action") is None)
    return failures / total


def top3_element_accuracy(records: List[Dict[str, Any]]) -> float:
    """Fraction of steps where gt_action_index appears anywhere in the top-3 ranked predictions."""
    total = 0
    correct = 0
    for row in records:
        gt_idx = row.get("gt_action_index")
        if gt_idx is None:
            continue
        total += 1
        pred_indices = row.get("pred_action_indices") or []
        # Backwards-compat: if only the scalar field is present, treat as top-1 list.
        if not pred_indices:
            pi = row.get("pred_action_index")
            pred_indices = [pi] if pi is not None else []
        if int(gt_idx) in [int(i) for i in pred_indices]:
            correct += 1
    return correct / total if total else 0.0


def mean_reciprocal_rank(records: List[Dict[str, Any]]) -> float:
    """MRR over steps using the ranked top-3 list.

    Reciprocal rank = 1 / position of gt in pred_action_indices (1-based).
    If gt is not in the list (or list is empty), reciprocal rank = 0.
    """
    total = 0
    rr_sum = 0.0
    for row in records:
        gt_idx = row.get("gt_action_index")
        if gt_idx is None:
            continue
        total += 1
        pred_indices = row.get("pred_action_indices") or []
        if not pred_indices:
            pi = row.get("pred_action_index")
            pred_indices = [pi] if pi is not None else []
        try:
            rank = [int(i) for i in pred_indices].index(int(gt_idx)) + 1  # 1-based
            rr_sum += 1.0 / rank
        except ValueError:
            pass  # gt not in list → reciprocal rank = 0
    return rr_sum / total if total else 0.0


def task_success_rate(records: List[Dict[str, Any]]) -> float:
    """Fraction of tasks (annotation_ids) where every step is predicted correctly.

    Mind2Web schema:
        annotation_id  — unique task identifier (one task = multiple steps)
        action_uid     — unique step identifier within a task
        gt_action_index — correct candidate index in action_reprs for this step
        pred_action_indices — model's ranked predictions (top-1 used here)

    A task counts as successful only if the top-1 prediction is correct for
    ALL of its steps. Any parse failure or wrong prediction fails the whole task.
    Step order within a task does not matter for this metric (all() is order-invariant).
    """
    from collections import defaultdict
    tasks: Dict[str, List[bool]] = defaultdict(list)
    for row in records:
        ann_id = row.get("annotation_id")
        gt_idx = row.get("gt_action_index")
        if ann_id is None or gt_idx is None:
            continue
        # Prefer the ranked list; fall back to legacy scalar for older JSONL files.
        pred_indices = row.get("pred_action_indices") or []
        if not pred_indices:
            pi = row.get("pred_action_index")
            pred_indices = [pi] if pi is not None else []
        top1_correct = bool(pred_indices) and int(pred_indices[0]) == int(gt_idx)
        tasks[ann_id].append(top1_correct)
    if not tasks:
        return 0.0
    return sum(all(steps) for steps in tasks.values()) / len(tasks)


def step_accuracy(records: List[Dict[str, Any]]) -> float:
    """Fraction of steps where BOTH element index (top-1) AND action type are correct."""
    total = 0
    correct = 0
    for row in records:
        gt_idx = row.get("gt_action_index")
        gt_action = _resolve_gt_action(row.get("gt_action"))
        if gt_idx is None or gt_action is None:
            continue
        total += 1
        pred_indices = row.get("pred_action_indices") or []
        if not pred_indices:
            pi = row.get("pred_action_index")
            pred_indices = [pi] if pi is not None else []
        pred_action = row.get("pred_action")
        elem_ok = bool(pred_indices) and int(pred_indices[0]) == int(gt_idx)
        action_ok = pred_action is not None and gt_action == str(pred_action).upper()
        if elem_ok and action_ok:
            correct += 1
    if total == 0:
        warnings.warn(
            "step_accuracy: no rows with both gt_action_index and gt_action found — "
            "returning 0.0 but the denominator is 0. "
            "Is 'gt_action' always None? Check normalize_gt_operation and the operation field.",
            RuntimeWarning,
            stacklevel=2,
        )
    return correct / total if total else 0.0


def per_action_prf(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    labels = set()
    for row in records:
        gt = _resolve_gt_action(row.get("gt_action"))
        pred = row.get("pred_action")
        if gt:
            labels.add(gt)
        if pred:
            labels.add(str(pred).upper())

    results: Dict[str, Dict[str, float]] = {}
    for label in sorted(labels):
        tp = fp = fn = 0
        for row in records:
            gt = _resolve_gt_action(row.get("gt_action"))
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

    # Patch records whose pred_action / pred_action_indices were truncated or
    # mis-ordered in an older inference run (re-derived from raw_output).
    parsed = [_recover_row(r) for r in parsed]

    baselines = sorted({row.get("baseline") for row in parsed if row.get("baseline")})
    splits = sorted({row.get("split") for row in parsed if row.get("split")})

    elem_parse_fail = parse_failure_rate(parsed)
    atype_parse_fail = action_type_parse_failure_rate(parsed)
    action_acc = action_label_accuracy(parsed)
    step_acc = step_accuracy(parsed)

    # Emit a clear diagnostic when ActionAcc is 0 but it looks like parsing
    # succeeded (elem_parse_fail==0): the likely cause is that JSON parsing
    # failed for most rows and the regex fallback recovered indices but not
    # the action type, leaving pred_action=None for all rows.
    if action_acc == 0.0 and elem_parse_fail == 0.0 and atype_parse_fail > 0.0:
        logging.warning(
            "%s: ActionAcc=0.0 but prediction_parse_failure_rate=0.0. "
            "action_type_parse_failure_rate=%.3f — pred_action is None for %.0f%% of rows "
            "with a gt_action. JSON parsing likely fell back to regex (which recovers "
            "indices but not action_type). Check raw_output fields in the JSONL.",
            path.name,
            atype_parse_fail,
            atype_parse_fail * 100,
        )

    metrics = {
        "file": str(path),
        "num_lines": len(raw_lines),
        "num_parsed": len(parsed),
        "json_parse_failure_rate": 1.0 - (len(parsed) / len(raw_lines)) if raw_lines else 0.0,
        "prediction_parse_failure_rate": elem_parse_fail,
        "action_type_parse_failure_rate": atype_parse_fail,
        "ElemAcc": action_index_accuracy(parsed),
        "ActionAcc": action_acc,
        "Top3Elem": top3_element_accuracy(parsed),  # == ElemAcc for single-prediction models
        "MRR": mean_reciprocal_rank(parsed),
        "TaskSuccess": task_success_rate(parsed),
        "StepAcc": step_acc,
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
            "macro_ElemAcc": 0.0,
            "macro_ActionAcc": 0.0,
            "macro_Top3Elem": 0.0,
            "macro_MRR": 0.0,
            "macro_TaskSuccess": 0.0,
            "macro_StepAcc": 0.0,
            "macro_prediction_parse_failure_rate": 0.0,
            "macro_action_type_parse_failure_rate": 0.0,
        }

    return {
        "num_files": len(metrics_list),
        "macro_ElemAcc": mean([m["ElemAcc"] for m in metrics_list]),
        "macro_ActionAcc": mean([m["ActionAcc"] for m in metrics_list]),
        "macro_Top3Elem": mean([m["Top3Elem"] for m in metrics_list]),
        "macro_MRR": mean([m["MRR"] for m in metrics_list]),
        "macro_TaskSuccess": mean([m["TaskSuccess"] for m in metrics_list]),
        "macro_StepAcc": mean([m["StepAcc"] for m in metrics_list]),
        "macro_prediction_parse_failure_rate": mean([m["prediction_parse_failure_rate"] for m in metrics_list]),
        "macro_action_type_parse_failure_rate": mean([m.get("action_type_parse_failure_rate", 0.0) for m in metrics_list]),
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
