"""
extract_failures.py

Reads a predictions JSONL file and writes only the incorrect predictions
to a separate failures JSONL file.

A prediction is incorrect when pred_action_index != gt_action_index.

USAGE:
    python extract_failures.py \
        --preds preds_qwen_domgraph_test_website.jsonl \
        --out failures_qwen_domgraph.jsonl

    # Optionally print a summary without writing a file:
    python extract_failures.py --preds preds_qwen_domgraph_test_website.jsonl --summary-only
"""

import json
import argparse
from collections import Counter


def extract_failures(preds_path, out_path=None, summary_only=False):

    total    = 0
    correct  = 0
    failures = []

    with open(preds_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] Skipping malformed line: {e}")
                continue

            total += 1
            gt_idx   = record.get("gt_action_index")
            pred_idx = record.get("pred_action_index")

            if gt_idx == pred_idx:
                correct += 1
            else:
                # Label the failure mode for easier analysis
                if record.get("parse_error"):
                    record["failure_mode"] = "PARSE_ERROR"
                elif pred_idx is None:
                    record["failure_mode"] = "NO_PREDICTION"
                else:
                    record["failure_mode"] = "WRONG_ELEMENT"
                failures.append(record)

    # ── Summary ─────────────────────────────────────────────────────────────
    failed = len(failures)
    print(f"\n=== Failure Extraction Summary ===")
    print(f"  Total samples : {total}")
    print(f"  Correct       : {correct}  ({correct/total:.2%})")
    print(f"  Failed        : {failed}  ({failed/total:.2%})")

    mode_counts = Counter(r["failure_mode"] for r in failures)
    print(f"\n  Failure modes:")
    for mode, count in mode_counts.most_common():
        print(f"    {mode:<20} {count:>5}  ({count/total:.2%} of all samples)")

    # ── Action type breakdown for failures ───────────────────────────────────
    # Tells you whether errors are concentrated on a particular action type
    action_counts = Counter(r.get("gt_action", "UNKNOWN") for r in failures)
    print(f"\n  Failures by ground truth action type:")
    for action, count in action_counts.most_common():
        print(f"    {action:<10} {count:>5}")

    # ── Write output ─────────────────────────────────────────────────────────
    if not summary_only and out_path:
        with open(out_path, "w", encoding="utf8") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n  → Wrote {len(failures)} failure records to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preds",        required=True, help="Path to predictions JSONL")
    p.add_argument("--out",          default="failures.jsonl", help="Path to write failures JSONL")
    p.add_argument("--summary-only", action="store_true", help="Print summary without writing a file")
    args = p.parse_args()

    extract_failures(
        preds_path=args.preds,
        out_path=args.out,
        summary_only=args.summary_only,
    )
