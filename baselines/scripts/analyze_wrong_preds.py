import argparse
import json
import csv
from collections import defaultdict


def detect_issue(rec):
    gt_action = (rec.get('gt_action') or '').upper() if rec.get('gt_action') is not None else ''
    pred_action = (rec.get('pred_action') or '').upper() if rec.get('pred_action') is not None else ''
    gt_value = rec.get('gt_value')
    pred_value = rec.get('pred_value')

    reasons = []
    # Missing value when GT expects a value
    if gt_value not in (None, '', []) and (pred_value in (None, '', []) or pred_value is None):
        reasons.append('missing_pred_value')

    # Action mismatch
    if gt_action and pred_action and gt_action != pred_action:
        reasons.append('action_mismatch')

    # GT TYPE but pred not TYPE
    if 'TYPE' in (gt_action or '') and 'TYPE' not in (pred_action or ''):
        reasons.append('type_to_non_type')

    # Both TYPE but different values
    if 'TYPE' in (gt_action or '') and 'TYPE' in (pred_action or '') and gt_value not in (None, '') and pred_value not in (None, '') and str(gt_value).strip() != str(pred_value).strip():
        reasons.append('different_values')

    if not reasons:
        reasons.append('other')

    return reasons


def main():
    parser = argparse.ArgumentParser(description='Analyze wrong predictions JSONL')
    parser.add_argument('--wrong-file', required=True, help='Path to wrong JSONL file')
    parser.add_argument('--out-csv', default='wrong_analysis.csv', help='CSV output path')
    parser.add_argument('--samples-per-type', type=int, default=5, help='How many example rows per issue type to save')
    args = parser.parse_args()

    counts_by_gt = defaultdict(int)
    counts_by_pred = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    issue_counts = defaultdict(int)
    samples = defaultdict(list)
    total = 0

    with open(args.wrong_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            gt_action = (rec.get('gt_action') or 'NONE')
            pred_action = (rec.get('pred_action') or 'NONE')
            counts_by_gt[gt_action] += 1
            counts_by_pred[pred_action] += 1
            confusion[gt_action][pred_action] += 1

            reasons = detect_issue(rec)
            for r in reasons:
                issue_counts[r] += 1
                if len(samples[r]) < args.samples_per_type:
                    samples[r].append(rec)

    # Write CSV summarizing each sample (first write samples grouped by issue)
    with open(args.out_csv, 'w', newline='', encoding='utf8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['issue_type', 'id', 'gt_action', 'pred_action', 'gt_value', 'pred_value', 'gt_element', 'pred_element', 'candidates'])
        for issue, recs in samples.items():
            for r in recs:
                writer.writerow([
                    issue,
                    r.get('id'),
                    r.get('gt_action'),
                    r.get('pred_action'),
                    r.get('gt_value'),
                    r.get('pred_value'),
                    r.get('gt_element'),
                    r.get('pred_element'),
                    '|'.join(r.get('candidates') or [])
                ])

    # Print summary
    print(f'Total wrong records processed: {total}')
    print('\nTop issue counts:')
    for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')

    print('\nTop GT actions:')
    for k, v in sorted(counts_by_gt.items(), key=lambda x: -x[1])[:20]:
        print(f'  {k}: {v}')

    print('\nTop Pred actions:')
    for k, v in sorted(counts_by_pred.items(), key=lambda x: -x[1])[:20]:
        print(f'  {k}: {v}')

    print('\nConfusion (GT -> Pred) (top 10 GTs):')
    for gt, preds in sorted(confusion.items(), key=lambda x: -sum(x[1].values()))[:10]:
        top_preds = sorted(preds.items(), key=lambda x: -x[1])[:5]
        print(f'  {gt}: ' + ', '.join([f'{p}:{c}' for p, c in top_preds]))

    print(f'Wrote sample examples to {args.out_csv}')


if __name__ == '__main__':
    main()
