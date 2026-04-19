import json
import statistics as st
from datasets import load_dataset
from bs4 import BeautifulSoup

INTER = {'button','input','a','select','textarea','option','label','form'}

def bid(c):
    if isinstance(c, dict):
        return str(c.get('backend_node_id', ''))
    if isinstance(c, str):
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return str(d.get('backend_node_id', ''))
            return c
        except Exception:
            return c
    return ''

def kept_ids(html, max_nodes):
    soup = BeautifulSoup(html or '<html></html>', 'html.parser')
    all_els = soup.find_all(True)
    interactive = [e for e in all_els if e.name in INTER]
    other = [e for e in all_els if e.name not in INTER]
    elements = (interactive + other)[:max_nodes]
    return [str((e.attrs or {}).get('backend_node_id', '')) for e in elements], len(all_els), len(interactive)

ds = load_dataset('osunlp/Multimodal-Mind2Web', split='train[:2000]')
rows = 0
node_counts = []
interactive_counts = []
cover128 = []
cover256 = []

for s in ds:
    html = s.get('cleaned_html') or s.get('raw_html') or ''
    if not html:
        continue
    pos = s.get('pos_candidates') or []
    pos_set = {bid(c) for c in pos if c}
    ids128, n, ni = kept_ids(html, 128)
    ids256, _, _ = kept_ids(html, 256)

    set128 = set(ids128)
    set256 = set(ids256)

    node_counts.append(n)
    interactive_counts.append(ni)

    if pos_set:
        k128 = sum(1 for b in pos_set if b in set128)
        k256 = sum(1 for b in pos_set if b in set256)
        cover128.append(k128 / len(pos_set))
        cover256.append(k256 / len(pos_set))

    rows += 1

print('rows', rows)
print('dom_nodes median p90 p95 max', int(st.median(node_counts)), int(st.quantiles(node_counts, n=10)[8]), int(st.quantiles(node_counts, n=20)[18]), max(node_counts))
print('interactive_nodes median p90 max', int(st.median(interactive_counts)), int(st.quantiles(interactive_counts, n=10)[8]), max(interactive_counts))
if cover128:
    print('pos_bid recall@128 mean median >=1hit', round(sum(cover128)/len(cover128),4), round(st.median(cover128),4), round(sum(1 for x in cover128 if x > 0)/len(cover128),4))
    print('pos_bid recall@256 mean median >=1hit', round(sum(cover256)/len(cover256),4), round(st.median(cover256),4), round(sum(1 for x in cover256 if x > 0)/len(cover256),4))
