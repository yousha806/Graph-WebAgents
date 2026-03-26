import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--file1', required=True)
parser.add_argument('--file2', required=True)
parser.add_argument('--out-csv', default='baselines/models/compare_wrong_preds.csv')
parser.add_argument('--max-samples', type=int, default=20)
args = parser.parse_args()

def load_by_id(path):
    d = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            r = json.loads(line)
            d[r['id']] = r
    return d

f1 = load_by_id(args.file1)
f2 = load_by_id(args.file2)

ids1 = set(f1.keys())
ids2 = set(f2.keys())

common = sorted(list(ids1 & ids2))
only1 = sorted(list(ids1 - ids2))
only2 = sorted(list(ids2 - ids1))

print(f'file1: {args.file1}  records: {len(ids1)}')
print(f'file2: {args.file2}  records: {len(ids2)}')
print(f'common ids: {len(common)}')
print(f'only in file1: {len(only1)}')
print(f'only in file2: {len(only2)}')

# write side-by-side CSV for common ids (up to max_samples)
import csv
with open(args.out_csv, 'w', newline='', encoding='utf8') as csvf:
    w = csv.writer(csvf)
    w.writerow(['id', 'model1', 'pred_idx1', 'pred_text1', 'task_success1', 'model2', 'pred_idx2', 'pred_text2', 'task_success2', 'gt_index', 'gt_element'])
    for i, cid in enumerate(common[:args.max_samples]):
        r1 = f1[cid]
        r2 = f2[cid]
        w.writerow([
            cid,
            r1.get('model'), r1.get('pred_idx'), r1.get('pred_text'), r1.get('task_success'),
            r2.get('model'), r2.get('pred_idx'), r2.get('pred_text'), r2.get('task_success'),
            r1.get('gt_index') or r2.get('gt_index'), r1.get('gt_element') or r2.get('gt_element')
        ])

print(f'Wrote comparison CSV to {args.out_csv} (samples: {min(len(common), args.max_samples)})')
