import json,collections,sys
files=['baselines/models/out_preds_internvl2.jsonl','baselines/models/out_preds_internvl2_hp.jsonl']
for p in files:
    cnt=collections.Counter()
    idxcnt=collections.Counter()
    total=0
    try:
        with open(p,'r',encoding='utf8') as f:
            for line in f:
                try:
                    r=json.loads(line)
                except:
                    continue
                total+=1
                txt=(r.get('pred_text') or '').strip()
                cnt[txt]+=1
                idx=r.get('pred_idx')
                idxcnt[str(idx)]+=1
    except FileNotFoundError:
        print('\nFILE:',p,'-- MISSING')
        continue
    print('\nFILE:',p,'total',total)
    print('Top pred_text:')
    for k,v in cnt.most_common(10):
        print(f'  {repr(k)}: {v}')
    print('Top pred_idx:')
    for k,v in idxcnt.most_common(10):
        print(f'  {k}: {v}')
