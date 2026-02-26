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
import os
import random
import time
try:
    from baselines.models.multimodal_internvl2 import run as internvl2_run
except Exception:
    internvl2_run = None

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


# --- Baseline runner / value-state collection utilities ---
def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            j = safe_load_line(line)
            if j is not None:
                out.append(j)
    return out

def save_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def simulate_model_response(example: Dict, model_name: str, state_dim: int = 16) -> Dict:
    """Simulate a model response and return predicted fields and a numeric 'value_states' vector.

    This is used as a fallback for baseline testing when the actual model bindings
    are not available. Replace the body of this function with real inference calls.
    """
    # deterministic pseudo-randomness for reproducibility across runs
    seed = hash(str(example.get("id", example.get("image", example.get("instruction", time.time()))))) & 0xFFFFFFFF
    rnd = random.Random(seed)
    # Simulated prediction pieces
    pred_value = example.get("gt_value") or f"SIM_VAL_{example.get('id', rnd.randint(0,999999))}"
    pred_action = rnd.choice(["CLICK", "TYPE", "SCROLL"]) if rnd.random() < 0.9 else "NOOP"
    pred_element = None
    cands = example.get("candidates") or []
    if cands:
        pred_element = cands[0]

    # Simulated numeric state vector (value_states)
    value_states = [rnd.random() for _ in range(state_dim)]

    return {
        "pred_value": pred_value,
        "pred_action": pred_action,
        "pred_element": pred_element,
        "value_states": value_states,
    }

def run_baseline_on_dataset(dataset_path: str, model_name: str, out_preds_path: str, simulate: bool = True, state_dim: int = 16):
    """Run baseline inference across a JSONL dataset and save predictions JSONL including value_states.

    Currently supports a `simulate` mode. To integrate a real model, implement a
    wrapper that calls the model client and returns the same keys produced by
    `simulate_model_response`.
    """
    examples = load_jsonl(dataset_path)
    preds = []
    for ex in examples:
        if simulate:
            out = simulate_model_response(ex, model_name, state_dim=state_dim)
        else:
            # Placeholder: user should implement actual model inference here.
            raise RuntimeError("Non-simulated model inference not implemented. Replace with real model calls.")

        # Build a prediction record compatible with the evaluator
        rec = {
            "id": ex.get("id"),
            "gt_element": ex.get("gt_element"),
            "gt_action": ex.get("gt_action"),
            "gt_value": ex.get("gt_value"),
            "pred_element": out.get("pred_element"),
            "pred_action": out.get("pred_action"),
            "pred_value": out.get("pred_value"),
            "value_states": out.get("value_states"),
            "candidates": ex.get("candidates") or [],
        }
        preds.append(rec)

    os.makedirs(os.path.dirname(out_preds_path) or ".", exist_ok=True)
    save_jsonl(out_preds_path, preds)
    print(f"Wrote simulated predictions (model={model_name}) to {out_preds_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--write-checklist", help="Path to write checklist markdown")
    p.add_argument("--eval", help="Path to predictions JSONL file")
    p.add_argument("--out", help="Output JSON path for metrics")
    p.add_argument("--topk", type=int, default=3)
    # Baseline / inference options
    p.add_argument("--run-baselines", action="store_true", help="Run baseline inference on a dataset and save preds JSONL")
    p.add_argument("--model", choices=["intern_vl2_8b", "image_only"], default="intern_vl2_8b", help="Which baseline model to run")
    p.add_argument("--dataset", help="Path to input dataset JSONL for running baselines")
    p.add_argument("--dataset-split", help="Hugging Face dataset split name (e.g. test_website) when running HF runner")
    p.add_argument("--preds-out", help="Path to write baseline predictions JSONL (includes value_states)")
    p.add_argument("--simulate-baselines", action="store_true", help="Use simulated model outputs instead of real model calls")
    p.add_argument("--state-dim", type=int, default=16, help="Dimensionality of simulated value_states vectors")
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
    if args.run_baselines:
        if not args.dataset or not args.preds_out:
            print("--run-baselines requires --dataset and --preds-out")
            return
        # If simulate flag is set, use the lightweight simulator
        if args.simulate_baselines:
            run_baseline_on_dataset(args.dataset, args.model, args.preds_out, simulate=True, state_dim=args.state_dim)
            return

        # Non-simulated: route to model-specific runner when available
        if args.model == "intern_vl2_8b":
            if internvl2_run is None:
                print("Intern VL2 runner not available (can't import baselines.models.multimodal_internvl2).")
                print("Ensure baselines/models/multimodal_internvl2.py is present and importable.")
                return
            # args.dataset may be a HF split name; fall back to 'test_website' if not provided
            dataset_split = args.dataset_split or args.dataset or "test_website"
            # Call the intern vl2 runner which will write preds to --preds-out
            internvl2_run(dataset_split=dataset_split, preds_out=args.preds_out, extract_states=True)
            return

        # Fallback: if no specialized runner, try file-based runner
        run_baseline_on_dataset(args.dataset, args.model, args.preds_out, simulate=False, state_dim=args.state_dim)
if __name__ == "__main__":
    main()
