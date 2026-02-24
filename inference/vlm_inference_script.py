"""Utility script: generate final checklist and compute evaluation metrics.

  - Evaluate predictions (JSONL lines with fields below):
      python vlm_inference_script.py --eval preds.jsonl --out metrics.json

Predictions JSONL format (one JSON object per line):
  {
    "gt_element": "element_id",
    "gt_action": "CLICK",
    "gt_value": "optional",
    "pred_element": "element_id",
    "pred_action": "CLICK",
    "pred_value": "optional",
    "candidates": ["elem1","elem2",...]
  }

The script computes:
  - Element Accuracy (Top-1)
  - Action Accuracy
  - Exact Match (element+action)
  - Parse Failure Rate (unreadable lines)
  - Top-K Element Accuracy (Top-3 by default)
  - MRR (Mean Reciprocal Rank)
  - Task Success Rate (fraction of successful tasks)

This is intended for quick project-level evaluation and checklist generation.
"""
import argparse
import json
from collections import defaultdict, Counter
from typing import List, Dict

FINAL_CHECKLIST = [
    "Element Accuracy (Top-1 grounding)",
    "Action Accuracy (classification)",
    "Exact Match (element AND action)",
    "Parse Failure Rate (invalid outputs)",
    "Top-K Element Accuracy (Top-3/Top-5)",
    "MRR (Mean Reciprocal Rank)",
    "Per-action Precision/Recall/F1",
    "Ablation table for modality combinations",
    "Grouped breakdowns by DOM depth / site / complexity",
    "Optional: Full Episode Success Rate (task completion)",
]

def write_checklist(path: str):
    with open(path, "w", encoding="utf8") as f:
        f.write("# Final Checklist for Mind2Web Evaluation\n\n")
        for i, item in enumerate(FINAL_CHECKLIST, 1):
            f.write(f"{i}. {item}\n")
    print(f"Wrote checklist to {path}")

def safe_load_line(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None

def element_accuracy(records: List[Dict]) -> float:
    total = 0
    correct = 0
    for r in records:
        if r.get("pred_element") is None or r.get("gt_element") is None:
            continue
        total += 1
        if r.get("pred_element") == r.get("gt_element"):
            correct += 1
    return correct / total if total else 0.0

def action_accuracy(records: List[Dict]) -> float:
    total = 0
    correct = 0
    for r in records:
        if r.get("pred_action") is None or r.get("gt_action") is None:
            continue
        total += 1
        if r.get("pred_action") == r.get("gt_action"):
            correct += 1
    return correct / total if total else 0.0

def exact_match(records: List[Dict]) -> float:
    total = 0
    correct = 0
    for r in records:
        if r.get("pred_action") is None or r.get("gt_action") is None:
            continue
        if r.get("pred_element") is None or r.get("gt_element") is None:
            continue
        total += 1
        if r.get("pred_action") == r.get("gt_action") and r.get("pred_element") == r.get("gt_element"):
            correct += 1
    return correct / total if total else 0.0

def parse_failure_rate(raw_lines: List[str], records: List[Dict]) -> float:
    return 1.0 - (len(records) / len(raw_lines)) if raw_lines else 0.0

def top_k_element_accuracy(records: List[Dict], k: int = 3) -> float:
    total = 0
    correct = 0
    for r in records:
        cands = r.get("candidates") or []
        gt = r.get("gt_element")
        if gt is None:
            continue
        total += 1
        if gt in cands[:k]:
            correct += 1
    return correct / total if total else 0.0

def mrr(records: List[Dict]) -> float:
    rr_sum = 0.0
    total = 0
    for r in records:
        cands = r.get("candidates") or []
        gt = r.get("gt_element")
        if gt is None:
            continue
        total += 1
        try:
            rank = cands.index(gt) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            rr_sum += 0.0
    return rr_sum / total if total else 0.0

def per_action_prf(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    actions = set()
    for r in records:
        if r.get("gt_action"): actions.add(r["gt_action"])
        if r.get("pred_action"): actions.add(r["pred_action"])
    results = {}
    for a in actions:
        tp = fp = fn = 0
        for r in records:
            pred = r.get("pred_action")
            gt = r.get("gt_action")
            if pred == a and gt == a:
                tp += 1
            if pred == a and gt != a:
                fp += 1
            if gt == a and pred != a:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results[a] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return results

def task_success_rate(records: List[Dict]) -> float:
    """Compute the fraction of records where 'task_success' is True."""
    total = 0
    success = 0
    for r in records:
        if "task_success" in r:
            total += 1
            if r["task_success"]:
                success += 1
    return success / total if total else None

def compute_metrics_from_file(path: str, topk: int = 3) -> Dict:
    raw_lines = []
    records = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            raw_lines.append(line)
            j = safe_load_line(line)
            if j is not None:
                records.append(j)

    metrics = {
        "num_lines": len(raw_lines),
        "num_parsed": len(records),
        "parse_failure_rate": parse_failure_rate(raw_lines, records),
        "element_accuracy": element_accuracy(records),
        "action_accuracy": action_accuracy(records),
        "exact_match": exact_match(records),
        "top_{}_element_accuracy".format(topk): top_k_element_accuracy(records, k=topk),
        "mrr": mrr(records),
        "per_action": per_action_prf(records),
    }
    tsr = task_success_rate(records)
    if tsr is not None:
        metrics["task_success_rate"] = tsr
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-checklist", help="Path to write checklist markdown")
    p.add_argument("--eval", help="Path to predictions JSONL file")
    p.add_argument("--out", help="Output JSON path for metrics")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()
    if args.write_checklist:
        write_checklist(args.write_checklist)
    if args.eval:
        metrics = compute_metrics_from_file(args.eval, topk=args.topk)
        if args.out:
            with open(args.out, "w", encoding="utf8") as fo:
                json.dump(metrics, fo, indent=2)
            print(f"Wrote metrics to {args.out}")
        else:
            print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()
