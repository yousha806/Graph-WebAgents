# =========================
# 0. INSTALL (Colab cell)
# =========================
# !pip install -q datasets transformers sentence-transformers torch tqdm beautifulsoup4

# =========================
# 0b. MOUNT GOOGLE DRIVE (Colab cell — run before training)
# =========================
# from google.colab import drive
# drive.mount('/content/drive')

import os, math, json, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR   = "/content/drive/MyDrive/gat_pretrain"
LOCAL_DATA = "/content/mind2web_train"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# =========================
# 1. DATASET
# =========================
if Path(LOCAL_DATA).exists():
    dataset = load_from_disk(LOCAL_DATA)
else:
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split="train")
    dataset.save_to_disk(LOCAL_DATA)
print(f"Loaded {len(dataset)} samples")

# =========================
# 2. CONSTANTS  ← must match multimodal_GNN_Cross.py exactly
# =========================
TAG_VOCAB = {
    "a": 0, "abbr": 1, "address": 2, "area": 3, "article": 4, "aside": 5,
    "audio": 6, "b": 7, "base": 8, "bdi": 9, "bdo": 10, "blockquote": 11,
    "body": 12, "br": 13, "button": 14, "canvas": 15, "caption": 16,
    "cite": 17, "code": 18, "col": 19, "colgroup": 20, "data": 21,
    "datalist": 22, "dd": 23, "del": 24, "details": 25, "dfn": 26,
    "dialog": 27, "div": 28, "dl": 29, "dt": 30, "em": 31, "embed": 32,
    "fieldset": 33, "figcaption": 34, "figure": 35, "footer": 36,
    "form": 37, "h1": 38, "h2": 39, "h3": 40, "h4": 41, "h5": 42,
    "h6": 43, "head": 44, "header": 45, "hgroup": 46, "hr": 47,
    "html": 48, "i": 49, "iframe": 50, "img": 51, "input": 52,
    "ins": 53, "kbd": 54, "label": 55, "legend": 56, "li": 57,
    "link": 58, "main": 59, "map": 60, "mark": 61, "menu": 62,
    "meta": 63, "meter": 64, "nav": 65, "noscript": 66, "object": 67,
    "ol": 68, "optgroup": 69, "option": 70, "output": 71, "p": 72,
    "picture": 73, "pre": 74, "progress": 75, "q": 76, "rp": 77,
    "rt": 78, "ruby": 79, "s": 80, "samp": 81, "script": 82,
    "search": 83, "section": 84, "select": 85, "slot": 86, "small": 87,
    "source": 88, "span": 89, "strong": 90, "style": 91, "sub": 92,
    "summary": 93, "sup": 94, "table": 95, "tbody": 96, "td": 97,
    "template": 98, "textarea": 99, "tfoot": 100, "th": 101,
    "thead": 102, "time": 103, "title": 104, "tr": 105, "track": 106,
    "u": 107, "ul": 108, "var": 109, "video": 110, "wbr": 111,
    "svg": 112, "path": 113, "g": 114, "rect": 115, "circle": 116,
    "ellipse": 117, "line": 118, "polyline": 119, "polygon": 120,
    "defs": 121, "use": 122, "symbol": 123, "tspan": 124,
    "font": 125, "center": 126, "strike": 127, "frame": 128,
    "frameset": 129, "noframes": 130, "tt": 131, "big": 132,
    "[UNK]": 133,
}
TAG_VOCAB_SIZE   = 134
NODE_FEAT_DIM    = 523   # 384 (sbert) + 134 (tag one-hot) + 5 (structural)
SBERT_DIM        = 384
GAT_LAYERS       = 3

# ── GPU config ── set GPU_TYPE to "A100" or "H100" ───────────
GPU_TYPE = "A100"   # change to "H100" for H100 runs

if GPU_TYPE == "H100":
    MAX_NODES        = 256   # H100 80 GB handles larger DOM
    GRAPH_HIDDEN_DIM = 512
    GRAPH_OUT_DIM    = 1024
    GAT_HEADS        = 8
    GRAD_ACCUM       = 32
    LEARNING_RATE    = 5e-4
else:                        # A100 (40 GB default)
    MAX_NODES        = 128
    GRAPH_HIDDEN_DIM = 256
    GRAPH_OUT_DIM    = 512
    GAT_HEADS        = 4
    GRAD_ACCUM       = 16
    LEARNING_RATE    = 3e-4
_INTERACTIVE     = {"button", "input", "a", "select", "textarea", "option", "label", "form"}

ACTION_TYPES = ["CLICK", "TYPE", "SCROLL", "NOOP", "HOVER", "SELECT", "SUBMIT", "PRESS", "LOAD"]
ACT2IDX      = {a: i for i, a in enumerate(ACTION_TYPES)}
NUM_ACTIONS  = len(ACTION_TYPES)

def _bid(c):
    if isinstance(c, dict):
        return str(c.get("backend_node_id", ""))
    if isinstance(c, str):
        try:
            d = json.loads(c)
            return str(d.get("backend_node_id", "")) if isinstance(d, dict) else c
        except Exception:
            return c
    return ""

# =========================
# 3. TEXT ENCODER (frozen sentence-transformer)
# =========================
sbert = SentenceTransformer("all-MiniLM-L6-v2")
sbert.eval()
for p in sbert.parameters():
    p.requires_grad = False

@torch.no_grad()
def encode_texts(texts: list[str]) -> torch.Tensor:
    return sbert.encode(texts, convert_to_tensor=True,
                        show_progress_bar=False, device="cpu").float()  # (N, 384)

# =========================
# 4. DOM → GRAPH  (identical logic to multimodal_GNN_Cross.dom_to_graph)
#    Extra return value: backend_ids per node for label extraction.
# =========================
def dom_to_graph(dom_html: str):
    """
    Returns:
        node_feats  : (N, NODE_FEAT_DIM=523)
        edge_index  : (2, E) long
        backend_ids : list[str] of length N  — backend_node_id attribute per element
    """
    soup        = BeautifulSoup(dom_html or "<html></html>", "html.parser")
    all_els     = soup.find_all(True)
    interactive = [e for e in all_els if e.name in _INTERACTIVE]
    other       = [e for e in all_els if e.name not in _INTERACTIVE]
    elements    = (interactive + other)[:MAX_NODES]

    if not elements:
        return (torch.zeros(1, NODE_FEAT_DIM),
                torch.tensor([[0], [0]], dtype=torch.long),
                [""])

    node_id_map = {id(el): idx for idx, el in enumerate(elements)}

    depths    = [len(list(el.parents)) for el in elements]
    max_depth = max(depths) if depths else 1

    sibling_ranks = []
    for el in elements:
        if el.parent is None:
            sibling_ranks.append(0.0)
        else:
            sibs = [c for c in el.parent.children if hasattr(c, "name") and c.name]
            pos  = next((i for i, s in enumerate(sibs) if s is el), 0)
            sibling_ranks.append(pos / max(1.0, len(sibs) - 1))

    raw_texts = []
    for el in elements:
        tag   = el.name or ""
        attrs = " ".join(f"{k}={v}" for k, v in (el.attrs or {}).items())
        text  = (el.get_text(separator=" ", strip=True) or "")[:128]
        raw_texts.append(f"{tag} {attrs} {text}".strip())

    text_embs  = encode_texts(raw_texts)   # (N, 384) on cpu
    tag_ids    = [TAG_VOCAB.get(el.name, TAG_VOCAB["[UNK]"]) for el in elements]
    tag_onehot = F.one_hot(torch.tensor(tag_ids, dtype=torch.long),
                           num_classes=TAG_VOCAB_SIZE).float()  # (N, 134)

    struct_rows = []
    for i, el in enumerate(elements):
        num_ch = len([c for c in el.children if hasattr(c, "name") and c.name])
        struct_rows.append(torch.tensor([
            depths[i] / max(float(max_depth), 1.0),
            float(el.name in _INTERACTIVE),
            math.log1p(num_ch) / math.log1p(100),
            sibling_ranks[i],
            float("id" in (el.attrs or {})),
        ], dtype=torch.float))
    struct_feats = torch.stack(struct_rows)   # (N, 5)

    node_feats = torch.cat([text_embs, tag_onehot, struct_feats], dim=-1)  # (N, 523)

    src_list, dst_list = [], []
    for el in elements:
        for child in el.children:
            if hasattr(child, "name") and child.name:
                if id(el) in node_id_map and id(child) in node_id_map:
                    p, c = node_id_map[id(el)], node_id_map[id(child)]
                    src_list += [p, c]
                    dst_list += [c, p]

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        n = len(elements)
        edge_index = torch.tensor([list(range(n)), list(range(n))], dtype=torch.long)

    backend_ids = [str((el.attrs or {}).get("backend_node_id", "")) for el in elements]
    return node_feats, edge_index, backend_ids

# =========================
# 5. GAT LAYER  (identical to multimodal_GNN_Cross.GATLayer)
# =========================
class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads    = heads
        self.out_dim  = out_dim
        self.head_dim = out_dim // heads
        self.W          = nn.Linear(in_dim, out_dim, bias=False)
        self.a          = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N  = x.size(0)
        Wx = self.W(x).view(N, self.heads, self.head_dim)   # (N, H, D/H)
        src, dst = edge_index[0], edge_index[1]

        # Per-edge attention scores (E, H)
        e = self.leaky_relu(
            self.a(torch.cat([Wx[src], Wx[dst]], dim=-1))
        ).squeeze(-1)

        # Dense N×N attention matrix — avoids scatter, safe for MAX_NODES=64
        # dtype follows e so bfloat16 stays consistent end-to-end
        attn = torch.full((N, N, self.heads), float("-inf"),
                          device=x.device, dtype=e.dtype)
        attn[dst, src] = e          # fancy-index preserves dtype, no scatter needed
        alpha = self.dropout(
            torch.softmax(attn, dim=1).nan_to_num(0.0)   # isolated nodes → 0
        )  # (N, N, H)

        # out[i,h,d] = Σ_j alpha[i,j,h] * Wx[j,h,d]
        out = torch.einsum("ijh,jhd->ihd", alpha, Wx)    # (N, H, D/H)
        return F.elu(out.reshape(N, self.out_dim))

# =========================
# 6. DOM GRAPH TRANSFORMER  (identical to multimodal_GNN_Cross.DOMGraphTransformer)
# =========================
class DOMGraphTransformer(nn.Module):
    def __init__(self, in_dim=NODE_FEAT_DIM, hidden_dim=GRAPH_HIDDEN_DIM,
                 out_dim=GRAPH_OUT_DIM, heads=GAT_HEADS,
                 num_layers=GAT_LAYERS, dropout=0.1):
        super().__init__()
        dims             = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers      = nn.ModuleList([GATLayer(dims[i], dims[i+1], heads, dropout)
                                          for i in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(num_layers)])
        self.dropout     = nn.Dropout(dropout)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = node_feats
        for layer, norm in zip(self.layers, self.norm_layers):
            x = self.dropout(norm(layer(x, edge_index)))
        return x   # (N, out_dim)

# =========================
# 7. PRETRAINER  (GAT + task-conditioned heads)
# =========================
class GATPretrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = DOMGraphTransformer()
        # Project SBERT task embedding to graph space for FiLM-style conditioning
        self.task_proj   = nn.Linear(SBERT_DIM, GRAPH_OUT_DIM)
        # Binary node-relevance head: (node_emb ⊙ task_proj) → logit
        self.node_head   = nn.Linear(GRAPH_OUT_DIM, 2)
        # Action-type head over mean-pooled graph
        self.action_head = nn.Linear(GRAPH_OUT_DIM, NUM_ACTIONS)

    def forward(self, node_feats: torch.Tensor,
                edge_index: torch.Tensor,
                task_emb: torch.Tensor):
        node_embs     = self.gat(node_feats, edge_index)           # (N, 512)
        task_proj     = self.task_proj(task_emb)                   # (1, 512)
        node_logits   = self.node_head(node_embs * task_proj)      # (N, 2)
        action_logits = self.action_head(node_embs.mean(0))        # (NUM_ACTIONS,)
        return node_logits, action_logits

# =========================
# 8. INIT
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

model     = GATPretrainer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

if DEVICE == "cuda":
    compile_kwargs = {"dynamic": True}
    if GPU_TYPE == "H100":
        compile_kwargs["mode"] = "max-autotune"
    model = torch.compile(model, **compile_kwargs)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"GPU_TYPE={GPU_TYPE}  MAX_NODES={MAX_NODES}  GRAD_ACCUM={GRAD_ACCUM}  LR={LEARNING_RATE}")

# =========================
# 8b. PREPROCESSING CACHE
#     Runs once; subsequent epochs read from disk.
#     Eliminates SBERT + BeautifulSoup overhead (~20 ms/sample) during training.
# =========================
CACHE_PATH = f"{SAVE_DIR}/dom_cache_n{MAX_NODES}.pkl"

if Path(CACHE_PATH).exists():
    cache = pickle.load(open(CACHE_PATH, "rb"))
    print(f"Loaded cache: {len(cache)} samples from {CACHE_PATH}")
else:
    cache = {}
    print("Building preprocessing cache (runs once)...")
    for i, sample in enumerate(tqdm(dataset, desc="Caching", dynamic_ncols=True)):
        html = sample.get("cleaned_html") or sample.get("raw_html") or ""
        if not html:
            continue
        try:
            nf, ei, bids = dom_to_graph(html)
            te = encode_texts([sample["confirmed_task"]])  # (1, 384)

            pos_cands   = sample.get("pos_candidates") or []
            pos_bid_set = {_bid(c) for c in pos_cands if c}
            node_labels = torch.tensor(
                [1 if b in pos_bid_set else 0 for b in bids], dtype=torch.long
            )

            op     = sample.get("operation") or {}
            op_str = (op.get("op", "NOOP") if isinstance(op, dict) else "NOOP").upper()
            act    = ACT2IDX.get(op_str, ACT2IDX["NOOP"])

            cache[i] = (nf, ei, node_labels, te, act)
        except Exception:
            continue

        if i % 500 == 0 and i > 0:
            pickle.dump(cache, open(CACHE_PATH, "wb"))

    pickle.dump(cache, open(CACHE_PATH, "wb"))
    print(f"Cache built: {len(cache)} samples → {CACHE_PATH}")

# =========================
# 9. TRAINING LOOP
# =========================
EPOCHS     = 5
LOG_EVERY  = 100
SAVE_EVERY = 1000

_AMP_DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
_AMP_DEVICE = "cuda"          if DEVICE == "cuda" else "cpu"

cached_indices = sorted(cache.keys())
step, running_loss = 0, 0.0
optimizer.zero_grad()

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(cached_indices, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)

    for i in pbar:
        node_feats, edge_index, node_labels, task_emb, act = cache[i]
        N = node_feats.size(0)

        node_feats  = node_feats.to(DEVICE)
        edge_index  = edge_index.to(DEVICE)
        node_labels = node_labels.to(DEVICE)
        act_label   = torch.tensor(act, device=DEVICE)
        task_emb    = task_emb.to(DEVICE)

        with torch.amp.autocast(_AMP_DEVICE, dtype=_AMP_DTYPE):
            node_logits, action_logits = model(node_feats, edge_index, task_emb)

            n_pos = max(node_labels.sum().item(), 1)
            n_neg = N - n_pos
            cls_w = torch.tensor([1.0, n_neg / n_pos], device=DEVICE)
            node_loss   = F.cross_entropy(node_logits, node_labels, weight=cls_w)
            action_loss = F.cross_entropy(action_logits.unsqueeze(0), act_label.unsqueeze(0))
            loss = (node_loss + 0.5 * action_loss) / GRAD_ACCUM

        loss.backward()
        step         += 1
        running_loss += loss.item() * GRAD_ACCUM

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            pbar.set_postfix(loss=f"{running_loss / LOG_EVERY:.4f}", step=step)
            running_loss = 0.0

        if step % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch, "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, f"{SAVE_DIR}/ckpt_step{step}.pt")

    print(f"Epoch {epoch+1} done  (step={step})")

# =========================
# 10. SAVE GAT WEIGHTS  (load these into DOMGraphTransformer in multimodal_GNN_Cross.py)
# =========================
torch.save({
    "gat": model.gat.state_dict(),
    "config": {
        "in_dim":      NODE_FEAT_DIM,
        "hidden_dim":  GRAPH_HIDDEN_DIM,
        "out_dim":     GRAPH_OUT_DIM,
        "heads":       GAT_HEADS,
        "num_layers":  GAT_LAYERS,
    },
}, f"{SAVE_DIR}/gat_pretrained.pt")
print("Saved GAT weights →", f"{SAVE_DIR}/gat_pretrained.pt")
