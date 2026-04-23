"""
Graph Transformer + LoRA Fine-Tuning: Qwen3-VL-8B-Instruct on Multimodal-Mind2Web
─────────────────────────────────────────────────────────────────────────────────
Architecture:
    DOM Tree (HTML nodes)
        ↓
    DOMGraphTransformer  [TRAINABLE]   — GAT over MiniLM semantic node features
        ↓
    GraphCrossAttentionAdapter [TRAINABLE]
        — Residual cross-attention hook injected at GRAPH_INJECT_LAYER.
          Sequence length is NEVER modified → no mask or causal-attn bugs.
        ↓
    Qwen3-VL (frozen base) + LoRA adapters [TRAINABLE]
        ↓
    Predicted top-3 ranked action list

Fixes vs previous version:
  FIX 1: GraphCrossAttentionAdapter — xavier_uniform init with gain=0.1,
          clamp nodes AND hidden states before MHA, clamp correction before
          gate multiply. Prevents softmax overflow → NaN cascade.
  FIX 2: Mind2WebDataset label construction — append target tokens AFTER
          the last real token, expanding the sequence if needed. Guarantees
          at least one non-(-100) label token per sample so loss is never
          computed on an empty set (which returns NaN).
  FIX 3: Training loop — added per-batch NaN guard on loss *before* any
          backward call; skipped batches are counted and reported so you
          can spot if a whole epoch is poisoned.
  FIX 4: collate_fn — labels tensor uses the expanded max_len that accounts
          for appended target tokens, not just the input_ids length.

NEW — dom_to_graph node selection:
  Instead of heuristic scoring, candidate nodes are selected by computing
  cosine similarity between each node's MiniLM text embedding and the
  combined task+candidate query embedding, then keeping the top-K most
  similar nodes. Interactive elements receive a fixed similarity bonus so
  they are never unfairly deprioritised.

NEW — Evaluation metrics (computed every validation epoch):
  ElemAcc   — fraction of steps where the top-1 predicted element ID matches
               the ground-truth element ID.
  Top3Elem  — fraction of steps where the ground-truth element ID appears in
               the top-3 predicted element IDs.
  MRR       — Mean Reciprocal Rank of the ground-truth element in the
               ranked prediction list (capped at rank 3).
  ActionAcc — fraction of steps where the predicted action type (CLICK, TYPE,
               …) matches the ground-truth action type decoded from labels.
  StepAcc   — fraction of steps where BOTH the element ID and action type are
               correct simultaneously (i.e. the full step is correct).

Install:
    pip install torch transformers datasets accelerate bitsandbytes peft \\
                qwen-vl-utils beautifulsoup4 sentence-transformers
Run:
    python train_lora_mind2web_merged.py
"""

import io
import os
import re
import json
import math
import time
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from bs4 import BeautifulSoup
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _gpu_mem() -> str:
    if not torch.cuda.is_available():
        return ""
    return (f" [GPU {torch.cuda.memory_allocated()/1e9:.1f}/"
            f"{torch.cuda.memory_reserved()/1e9:.1f} GB]")


def _log_stage(name: str):
    bar = "─" * 60
    logger.info(f"\n{bar}\n▶  {name}{_gpu_mem()}\n{bar}")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR       = "qwen_lora_mind2web_merged"
NUM_EPOCHS       = 3
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE    = 3e-4
MAX_TARGET_LEN   = 80          # tokens for the ranked-list answer

# Qwen3-VL pixel budget: each 28x28 patch = 1 vision token.
# 256*28*28 = 200704 pixels -> ~256 vision tokens max. Keeps GPU memory stable.
SCREENSHOT_MIN_PIXELS = 4 * 28 * 28    # 3136
SCREENSHOT_MAX_PIXELS = 256 * 28 * 28  # 200704

LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# ── MiniLM node features (dim=384) + tag one-hot (134) + struct (5) = 523
NODE_FEAT_DIM    = 523
GRAPH_HIDDEN_DIM = 256
GRAPH_OUT_DIM    = 512
GAT_HEADS        = 4
GAT_LAYERS       = 3
MAX_NODES        = 128

# ── Similarity-based node selection
# Bonus added to cosine similarity for interactive elements so they are never
# buried by purely textual relevance. Value is in [-1, 1] cosine sim units.
INTERACTIVE_SIM_BONUS = 0.15

QWEN_HIDDEN_DIM  = 4096        # Qwen3-VL-8B hidden size
PROJ_ATTN_HEADS  = 8
GRAPH_INJECT_LAYER = 20        # which transformer layer gets the residual hook

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def save_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_action_from_text(s: str):
    """Return the action-type keyword from a free-form action string."""
    if not s or not isinstance(s, str):
        return None
    for a in ["CLICK", "TYPE", "SCROLL", "NOOP", "HOVER", "SELECT", "SUBMIT"]:
        if a in s.upper().split():
            return a
    toks = s.strip().split()
    if toks:
        t0 = toks[0].upper().strip(':,')
        if len(t0) <= 10 and t0.isalpha():
            return t0
    return None


def _screenshot_to_pil(screenshot) -> Image.Image:
    if screenshot is None:
        return Image.new("RGB", (1280, 720), color=(255, 255, 255))
    if isinstance(screenshot, Image.Image):
        return screenshot.convert("RGB")
    if isinstance(screenshot, bytes):
        return Image.open(io.BytesIO(screenshot)).convert("RGB")
    if isinstance(screenshot, str):
        return Image.open(screenshot).convert("RGB")
    try:
        return Image.fromarray(screenshot).convert("RGB")
    except Exception:
        return Image.new("RGB", (1280, 720), color=(255, 255, 255))


def _tokenize_text(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


# ─────────────────────────────────────────────
# Sentence-transformer singleton
# ─────────────────────────────────────────────
_TEXT_ENCODER = None


def _get_text_encoder():
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        _log_stage("Loading sentence-transformers (all-MiniLM-L6-v2)")
        from sentence_transformers import SentenceTransformer
        _TEXT_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("sentence-transformers ready")
    return _TEXT_ENCODER


# ─────────────────────────────────────────────
# STEP 0: DOM → Graph
# ─────────────────────────────────────────────

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
TAG_VOCAB_SIZE = len(TAG_VOCAB)
_INTERACTIVE = {"button", "input", "a", "select", "textarea", "option", "label", "form"}


def dom_to_graph(dom_html: str, task_text: str = "",
                 candidate_reprs: list[str] | None = None):
    """
    Build a graph over DOM nodes with MiniLM semantic features.

    Node selection strategy (NEW — similarity-based top-K):
      When the DOM has more than MAX_NODES elements, we:
        1. Encode ALL node text representations with MiniLM (batched, on CPU).
        2. Encode the combined query (task + candidate reprs) with MiniLM.
        3. Compute cosine similarity of every node against the query.
        4. Add INTERACTIVE_SIM_BONUS to interactive-element scores so that
           buttons/inputs are never unfairly dropped.
        5. Keep the top-MAX_NODES nodes by similarity score.

      This is semantically richer than the old token-overlap heuristic and
      naturally adapts to paraphrased or long-form task descriptions.

    Returns:
        node_feats  : (N, 523) float tensor
        edge_index  : (2, E)   long tensor
    """
    soup    = BeautifulSoup(dom_html or "<html></html>", "html.parser")
    all_els = soup.find_all(True)

    encoder = _get_text_encoder()

    if len(all_els) <= MAX_NODES:
        elements = list(all_els)
    else:
        # ── Build raw text for every node (needed for both selection and feats)
        all_raw_texts = []
        for el in all_els:
            tag   = el.name or ""
            attrs = " ".join(f"{k}={v}" for k, v in (el.attrs or {}).items())
            text  = (el.get_text(separator=" ", strip=True) or "")[:128]
            all_raw_texts.append(f"{tag} {attrs} {text}".strip())

        # ── Encode all nodes (batched — single GPU/CPU call)
        with torch.no_grad():
            node_embs_all = encoder.encode(
                all_raw_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device="cpu",
            ).float()   # (N_all, 384)

        # ── Encode the query
        candidate_reprs = candidate_reprs or []
        query_text = f"{task_text} {' '.join(candidate_reprs[:20])}".strip()
        if not query_text:
            query_text = "web page interaction"
        with torch.no_grad():
            query_emb = encoder.encode(
                [query_text],
                convert_to_tensor=True,
                show_progress_bar=False,
                device="cpu",
            ).float()   # (1, 384)

        # ── Cosine similarity between every node and the query
        node_norm  = F.normalize(node_embs_all, dim=-1)   # (N_all, 384)
        query_norm = F.normalize(query_emb,     dim=-1)   # (1, 384)
        sim_scores = (node_norm @ query_norm.T).squeeze(-1)  # (N_all,)

        # ── Bonus for interactive elements (keeps them competitive)
        for idx, el in enumerate(all_els):
            if el.name in _INTERACTIVE:
                sim_scores[idx] = sim_scores[idx] + INTERACTIVE_SIM_BONUS

        # ── Top-K by similarity
        top_k_indices = torch.topk(sim_scores, k=MAX_NODES, largest=True).indices
        top_k_set     = set(top_k_indices.tolist())
        elements      = [el for i, el in enumerate(all_els) if i in top_k_set]

        logger.debug(
            f"dom_to_graph: {len(all_els)} → {len(elements)} nodes via cosine-sim top-K"
        )

    if not elements:
        return (torch.zeros(1, NODE_FEAT_DIM),
                torch.tensor([[0], [0]], dtype=torch.long))

    # ── Structural features ───────────────────────────────────────────────
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

    # ── Text representations for the selected nodes ───────────────────────
    raw_texts = []
    for el in elements:
        tag   = el.name or ""
        attrs = " ".join(f"{k}={v}" for k, v in (el.attrs or {}).items())
        text  = (el.get_text(separator=" ", strip=True) or "")[:128]
        raw_texts.append(f"{tag} {attrs} {text}".strip())

    with torch.no_grad():
        text_embs = encoder.encode(
            raw_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device="cpu",
        ).float()   # (N, 384)

    # ── Tag one-hot ───────────────────────────────────────────────────────
    tag_ids    = [TAG_VOCAB.get(el.name, TAG_VOCAB["[UNK]"]) for el in elements]
    tag_onehot = F.one_hot(
        torch.tensor(tag_ids, dtype=torch.long), num_classes=TAG_VOCAB_SIZE
    ).float()       # (N, 134)

    # ── Structural feature vectors ────────────────────────────────────────
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
    struct_feats = torch.stack(struct_rows)  # (N, 5)

    node_feats = torch.cat([text_embs, tag_onehot, struct_feats], dim=-1)  # (N, 523)

    # ── Edge index (parent ↔ child, bidirectional) ────────────────────────
    node_id_map = {id(el): idx for idx, el in enumerate(elements)}
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

    return node_feats, edge_index


# ─────────────────────────────────────────────
# STEP 1: Graph Transformer (GAT)
# ─────────────────────────────────────────────

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
        Wx = self.W(x)
        Wx = Wx.view(N, self.heads, self.head_dim)
        src, dst = edge_index[0], edge_index[1]

        Wx_src = Wx[src]
        Wx_dst = Wx[dst]
        e = self.leaky_relu(
            self.a(torch.cat([Wx_src, Wx_dst], dim=-1))
        ).squeeze(-1)

        e_exp = e.exp()
        denom = torch.zeros(N, self.heads, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (denom[dst] + 1e-8)
        alpha = self.dropout(alpha)

        weighted = alpha.unsqueeze(-1) * Wx_src
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand_as(weighted),
            weighted,
        )
        return F.elu(out.view(N, self.out_dim))


class DOMGraphTransformer(nn.Module):
    def __init__(self, in_dim=NODE_FEAT_DIM, hidden_dim=GRAPH_HIDDEN_DIM,
                 out_dim=GRAPH_OUT_DIM, heads=GAT_HEADS, num_layers=GAT_LAYERS,
                 dropout=0.1):
        super().__init__()
        dims             = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers      = nn.ModuleList(
            [GATLayer(dims[i], dims[i + 1], heads, dropout) for i in range(num_layers)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = node_feats
        for layer, norm in zip(self.layers, self.norm_layers):
            x = self.dropout(norm(layer(x, edge_index)))
        return x   # (N, GRAPH_OUT_DIM)


# ─────────────────────────────────────────────
# STEP 2: Graph → Qwen adapter
# ─────────────────────────────────────────────

class GraphCrossAttentionAdapter(nn.Module):
    """
    Projects graph node embeddings into Qwen hidden space as a residual
    cross-attention.

    FIX 1 (NaN cascade):
      - xavier_uniform init with gain=0.1 on node_proj → small initial logits
      - Clamp node projections to [-10, 10] before MHA → softmax can't overflow
      - Clamp hidden states to [-100, 100] before MHA (bfloat16 spikes are common
        in early training when LoRA weights are near-zero and the frozen base
        produces large activations)
      - Clamp correction to [-50, 50] before the gate multiply → no single
        batch can blow up the residual stream
      - All ops stay in fp32; cast back to orig_dtype only at the very end
    """

    def __init__(self, graph_dim=GRAPH_OUT_DIM, qwen_dim=QWEN_HIDDEN_DIM,
                 attn_heads=PROJ_ATTN_HEADS, dropout=0.1):
        super().__init__()
        self.node_proj  = nn.Linear(graph_dim, qwen_dim).float()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=qwen_dim, num_heads=attn_heads,
            dropout=dropout, batch_first=True,
        ).float()
        self.norm = nn.LayerNorm(qwen_dim).float()
        self.gate = nn.Parameter(torch.zeros(1))  # tanh(0)=0 → identity at init

        # FIX 1a: small init prevents large attention logits on step 0
        nn.init.xavier_uniform_(self.node_proj.weight, gain=0.1)
        nn.init.zeros_(self.node_proj.bias)

    def forward(self, hidden_states: torch.Tensor,
                node_embs: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        B = hidden_states.size(0)

        hs    = hidden_states.float()
        nodes = node_embs.float()

        # FIX 1c: clamp hidden states — bfloat16 can produce ~65504 spikes
        hs    = torch.clamp(hs,    -100.0, 100.0)
        # FIX 1d: project then clamp nodes so attention logits stay small
        nodes = self.node_proj(nodes)
        nodes = torch.clamp(nodes, -10.0,  10.0)
        nodes = nodes.unsqueeze(0).expand(B, -1, -1)

        correction, _ = self.cross_attn(hs, nodes, nodes)
        correction = self.norm(correction)
        # FIX 1e: clamp correction before scaling
        correction = torch.clamp(correction, -50.0, 50.0)

        out = hidden_states + self.gate.tanh() * correction.to(orig_dtype)
        return out.to(torch.bfloat16)


# ─────────────────────────────────────────────
# STEP 3: Load + LoRA
# ─────────────────────────────────────────────

def load_model():
    _log_stage("Loading Qwen3-VL-8B-Instruct in 4-bit nf4")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info(f"Qwen loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B{_gpu_mem()}")
    return model, processor


def _disable_checkpointing_on_inject_layers(model, inject_layer):
    path_candidates = [
        'model.language_model.model.layers',
        'model.model.layers',
        'model.layers',
        'language_model.model.layers',
        'base_model.model.language_model.model.layers',
        'base_model.model.model.layers',
        'base_model.model.layers',
    ]
    decoder_layers = None
    for path in path_candidates:
        cur = model
        for part in path.split('.'):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if isinstance(cur, (nn.ModuleList, list)) and len(cur) > inject_layer:
            decoder_layers = cur
            break
    if decoder_layers is None:
        logger.warning('Could not find decoder layers to selectively disable checkpointing.')
        model.gradient_checkpointing_disable()
        return
    disabled = []
    for idx in [inject_layer - 1, inject_layer, inject_layer + 1]:
        if 0 <= idx < len(decoder_layers):
            layer = decoder_layers[idx]
            if hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = False
            if hasattr(layer, '_gradient_checkpointing_func'):
                layer._gradient_checkpointing_func = None
            disabled.append(idx)
    logger.info(f'Gradient checkpointing disabled on layers {disabled}.')


def apply_lora(model):
    _log_stage(f"Applying LoRA (focused on layers {GRAPH_INJECT_LAYER-1}–{GRAPH_INJECT_LAYER+1})")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False

    _disable_checkpointing_on_inject_layers(model, GRAPH_INJECT_LAYER)

    target_proj_names = {"q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"}
    target_layer_indices = {GRAPH_INJECT_LAYER - 1, GRAPH_INJECT_LAYER,
                            GRAPH_INJECT_LAYER + 1}
    eligible_modules = []
    try:
        import bitsandbytes as bnb
        for name, module in model.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue
            if name.split(".")[-1] not in target_proj_names:
                continue
            m = re.search(r'layers\.(\d+)\.', name)
            if m and int(m.group(1)) in target_layer_indices:
                eligible_modules.append(name)
    except Exception:
        pass

    if not eligible_modules:
        logger.warning("4-bit scan found no modules; falling back to generic target names")
        eligible_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    logger.info(f"LoRA target modules ({len(eligible_modules)}): {eligible_modules[:6]} ...")
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=eligible_modules,
        lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# STEP 4: Combined model wrapper
# ─────────────────────────────────────────────

class QwenWithDOMGraph(nn.Module):
    def __init__(self, qwen_model, graph_transformer, adapter,
                 inject_layer: int = GRAPH_INJECT_LAYER):
        super().__init__()
        self.qwen         = qwen_model
        self.graph_enc    = graph_transformer
        self.adapter      = adapter
        self.inject_layer = inject_layer
        self._node_embs_cache: torch.Tensor | None = None
        self._vis_device = None

    def _get_decoder_layers(self):
        path_candidates = [
            "model.language_model.model.layers",
            "model.model.layers",
            "model.layers",
            "language_model.model.layers",
            "base_model.model.language_model.model.layers",
            "base_model.model.model.layers",
            "base_model.model.layers",
        ]
        for path in path_candidates:
            cur = self.qwen
            for part in path.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    break
            if isinstance(cur, (nn.ModuleList, list)) and len(cur) > self.inject_layer:
                return cur

        best = None
        for name, module in self.qwen.named_modules():
            if (isinstance(module, nn.ModuleList) and
                    len(module) > self.inject_layer and
                    "vision" not in name.lower()):
                if best is None or len(module) > len(best):
                    best = module
        if best is not None:
            return best
        raise AttributeError("Could not find decoder layers for hook injection.")

    def _build_inject_hook(self):
        def hook(module, args, kwargs):
            node_embs = self._node_embs_cache
            if node_embs is None:
                return args, kwargs
            hidden_states = args[0]
            new_hs = self.adapter(hidden_states, node_embs)
            return (new_hs,) + args[1:], kwargs
        return hook

    @staticmethod
    def _find_visual_device(qwen_model) -> torch.device:
        for attr in ["visual", "model.visual", "model.model.visual",
                     "base_model.model.visual",
                     "base_model.model.model.visual"]:
            obj = qwen_model
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    return next(obj.parameters()).device
                except StopIteration:
                    pass
        for name, mod in qwen_model.named_modules():
            if "patch_embed" in name:
                try:
                    return next(mod.parameters()).device
                except StopIteration:
                    pass
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                labels, node_feats, edge_index, mm_token_type_ids=None):

        graph_device = next(self.graph_enc.parameters()).device
        node_embs    = self.graph_enc(
            node_feats.to(graph_device), edge_index.to(graph_device)
        )

        qwen_device           = self.qwen.get_input_embeddings().weight.device
        self._node_embs_cache = node_embs.to(device=qwen_device, dtype=torch.bfloat16)

        input_ids      = input_ids.to(qwen_device)
        attention_mask = attention_mask.to(qwen_device)
        if labels is not None:
            labels = labels.to(qwen_device)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(qwen_device)

        if self._vis_device is None:
            self._vis_device = self._find_visual_device(self.qwen)
            logger.info(f'Visual encoder device (cached): {self._vis_device}')
        pixel_values   = pixel_values.to(device=self._vis_device, dtype=torch.float32)
        image_grid_thw = image_grid_thw.to(device=self._vis_device)

        handle = self._get_decoder_layers()[self.inject_layer].register_forward_pre_hook(
            self._build_inject_hook(), with_kwargs=True
        )
        try:
            outputs = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                mm_token_type_ids=mm_token_type_ids,
            )
        finally:
            handle.remove()
            # NOTE: do NOT clear _node_embs_cache here — gradient checkpointing
            # recomputes forward() during backward and needs the cache.
            # It is cleared by the training loop after loss.backward().

        return outputs


# ─────────────────────────────────────────────
# STEP 5: Dataset
# ─────────────────────────────────────────────

class Mind2WebDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.data      = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example    = self.data[idx]
        task       = example.get("confirmed_task", "")
        candidates = example.get("action_reprs", [])
        target_idx = int(example.get("target_action_index", 0))
        screenshot = example.get("screenshot")
        dom_html   = (example.get("cleaned_html") or
                      example.get("dom_tree") or "")

        screenshot = _screenshot_to_pil(screenshot)

        node_feats, edge_index = dom_to_graph(
            dom_html, task_text=task, candidate_reprs=candidates
        )

        candidate_text = "\n".join([f"[{i}] {c}" for i, c in enumerate(candidates)])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": (
                    f"Task: {task}\n\n"
                    f"Candidate actions:\n{candidate_text}\n\n"
                    f"Based on the screenshot and page structure, rank the top 3 "
                    f"most appropriate actions. Reply in this exact format:\n"
                    f"1. [ID] action text\n2. [ID] action text\n3. [ID] action text"
                )},
            ],
        }]

        text_prompt     = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs          = self.processor(
            text=[text_prompt],
            images=image_inputs,
            return_tensors="pt",
            min_pixels=SCREENSHOT_MIN_PIXELS,
            max_pixels=SCREENSHOT_MAX_PIXELS,
        )

        input_ids_row = inputs["input_ids"].squeeze(0)
        attn_mask_row = inputs["attention_mask"].squeeze(0)
        mm_ttids_row  = (inputs["mm_token_type_ids"].squeeze(0)
                         if "mm_token_type_ids" in inputs else None)

        # ── FIX 2: label construction ─────────────────────────────────────
        neg_cands  = [(i, c) for i, c in enumerate(candidates) if i != target_idx]
        random.shuffle(neg_cands)
        top3       = [(target_idx, candidates[target_idx])] + neg_cands[:2]
        target_str = "\n".join([f"{r+1}. [{i}] {c}" for r, (i, c) in enumerate(top3)])
        target_tids = self.processor.tokenizer(
            target_str, add_special_tokens=False
        ).input_ids

        pad_id = self.processor.tokenizer.pad_token_id or 0
        S      = input_ids_row.shape[0]

        non_pad    = (input_ids_row != pad_id).nonzero(as_tuple=True)[0]
        insert_at  = non_pad[-1].item() + 1 if len(non_pad) > 0 else S

        target_tids = target_tids[:MAX_TARGET_LEN]
        n_tgt       = len(target_tids)

        overflow = max(0, insert_at + n_tgt - S)
        if overflow > 0:
            pad_ext        = torch.full((overflow,), pad_id, dtype=input_ids_row.dtype)
            mask_ext       = torch.zeros(overflow, dtype=attn_mask_row.dtype)
            input_ids_row  = torch.cat([input_ids_row,  pad_ext],  dim=0)
            attn_mask_row  = torch.cat([attn_mask_row,  mask_ext], dim=0)
            if mm_ttids_row is not None:
                ttids_ext  = torch.zeros(overflow, dtype=mm_ttids_row.dtype)
                mm_ttids_row = torch.cat([mm_ttids_row, ttids_ext], dim=0)

        S_new  = input_ids_row.shape[0]
        labels = torch.full((S_new,), -100, dtype=torch.long)
        for j, tid in enumerate(target_tids):
            pos = insert_at + j
            if pos < S_new:
                labels[pos] = tid

        assert (labels != -100).any(), (
            f"Sample {idx}: all labels are -100 after construction! "
            f"insert_at={insert_at}, n_tgt={n_tgt}, S_new={S_new}"
        )

        return {
            "input_ids":         input_ids_row,
            "attention_mask":    attn_mask_row,
            "pixel_values":      inputs["pixel_values"].squeeze(0),
            "mm_token_type_ids": mm_ttids_row,
            "image_grid_thw":    inputs["image_grid_thw"].squeeze(0),
            "node_feats":        node_feats,
            "edge_index":        edge_index,
            "labels":            labels,
            "num_candidates":    len(candidates),
            # Ground-truth element index and action type stored for metrics
            "target_idx":        target_idx,
            "target_action":     extract_action_from_text(
                candidates[target_idx] if candidates else ""
            ) or "UNKNOWN",
        }


def build_dataloaders(processor):
    _log_stage("Loading Multimodal-Mind2Web dataset")
    raw         = load_dataset("osunlp/Multimodal-Mind2Web")
    full_train  = raw["train"]
    split       = full_train.train_test_split(test_size=0.1, seed=42)
    train_split = split["train"]
    val_split   = split["test"]
    logger.info(f"Train: {len(train_split)} | Val: {len(val_split)}")

    train_ds = Mind2WebDataset(train_split, processor)
    val_ds   = Mind2WebDataset(val_split,   processor)

    def collate_fn(batch):
        # FIX 4: use the actual per-sample lengths (which may now be longer than
        # the raw input_ids length due to the label-overflow extension in FIX 2)
        max_len = max(b["input_ids"].shape[0] for b in batch)
        pad_id  = processor.tokenizer.pad_token_id or 0

        def pad1d(t, pv):
            diff = max_len - t.shape[0]
            return F.pad(t, (0, diff), value=pv) if diff > 0 else t

        mm_present = [b for b in batch if b["mm_token_type_ids"] is not None]
        result = {
            "input_ids":       torch.stack([pad1d(b["input_ids"],      pad_id) for b in batch]),
            "attention_mask":  torch.stack([pad1d(b["attention_mask"],  0)     for b in batch]),
            "labels":          torch.stack([pad1d(b["labels"],        -100)    for b in batch]),
            "pixel_values":    torch.stack([b["pixel_values"]   for b in batch]),
            "image_grid_thw":  torch.stack([b["image_grid_thw"] for b in batch]),
            "node_feats":      batch[0]["node_feats"],
            "edge_index":      batch[0]["edge_index"],
            "num_candidates":  [b["num_candidates"]  for b in batch],
            # Per-sample ground-truth info for metrics
            "target_idx":      [b["target_idx"]      for b in batch],
            "target_action":   [b["target_action"]   for b in batch],
        }
        if mm_present:
            result["mm_token_type_ids"] = torch.stack(
                [pad1d(b["mm_token_type_ids"], 0) for b in batch
                 if b["mm_token_type_ids"] is not None]
            )
        return result

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


# ─────────────────────────────────────────────
# STEP 5b: Metric helpers
# ─────────────────────────────────────────────

def _decode_ranked_ids(gen_text: str, num_candidates: int) -> list[int]:
    """
    Parse the model's ranked output and return a list of element IDs
    in predicted rank order (deduplicated, clamped to valid range).

    Expected format: "1. [3] click search  2. [1] type query  3. [0] ..."
    Falls back gracefully to any [N] pattern found.
    """
    matches = re.findall(r'\[(\d+)\]', gen_text)
    ranked  = []
    for m in matches:
        idx = int(m)
        if 0 <= idx < num_candidates and idx not in ranked:
            ranked.append(idx)
    return ranked


_ACTION_WHITELIST = frozenset(
    ["CLICK", "TYPE", "SCROLL", "NOOP", "HOVER", "SELECT", "SUBMIT"]
)


def _decode_action_type(gen_text: str) -> str | None:
    """
    Return the first recognised action-type keyword found in *gen_text*,
    or None if no known action type appears.

    Uses a strict whitelist — the generic fallback in extract_action_from_text
    is intentionally NOT used here so that spurious words like "nothing" or
    "rank" are never treated as valid action types in metric calculations.
    """
    if not gen_text:
        return None
    tokens = re.findall(r"[A-Za-z]+", gen_text.upper())
    for tok in tokens:
        if tok in _ACTION_WHITELIST:
            return tok
    return None


class EvalMetrics:
    """
    Accumulates per-step predictions and computes the five evaluation metrics
    at the end of a validation epoch.

    Definitions
    ───────────
    ElemAcc   — Top-1 element accuracy: predicted_rank[0] == ground_truth_elem
    Top3Elem  — Ground-truth element appears anywhere in predicted_rank[:3]
    MRR       — Mean Reciprocal Rank: 1/(rank+1) if found in top-3, else 0
                 (rank is 0-indexed, so rank 0 → 1.0, rank 1 → 0.5, rank 2 → 0.33)
    ActionAcc — Predicted action type == ground-truth action type
    StepAcc   — ElemAcc AND ActionAcc simultaneously (full step correct)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._elem_acc_hits   = 0
        self._top3_elem_hits  = 0
        self._mrr_sum         = 0.0
        self._action_acc_hits = 0
        self._step_acc_hits   = 0
        self._total           = 0

    def update(self,
               pred_ranked_ids: list[int],
               pred_action: str | None,
               true_elem_id: int,
               true_action: str | None):
        """
        Call once per validation step.

        Args:
            pred_ranked_ids : list of predicted element IDs in rank order
            pred_action     : predicted action type string (e.g. "CLICK")
            true_elem_id    : ground-truth element ID (int)
            true_action     : ground-truth action type string
        """
        self._total += 1

        # ── Element-level metrics ────────────────────────────────────────
        top1    = pred_ranked_ids[0] if pred_ranked_ids else -1
        top3    = pred_ranked_ids[:3]

        elem_hit = int(top1 == true_elem_id)
        self._elem_acc_hits  += elem_hit

        if true_elem_id in top3:
            self._top3_elem_hits += 1
            rank = top3.index(true_elem_id)           # 0-indexed
            self._mrr_sum += 1.0 / (rank + 1)
        # MRR contribution is 0 when not in top-3 (no else branch needed)

        # ── Action-level metric ──────────────────────────────────────────
        norm_pred   = (pred_action   or "").upper().strip()
        norm_true   = (true_action   or "").upper().strip()
        action_hit  = int(bool(norm_pred) and norm_pred == norm_true)
        self._action_acc_hits += action_hit

        # ── Step-level metric ────────────────────────────────────────────
        self._step_acc_hits += int(elem_hit and action_hit)

    def compute(self) -> dict:
        """Return a dict of metric_name → float value (percentages where noted)."""
        n = max(self._total, 1)
        return {
            "ElemAcc(%)":   100.0 * self._elem_acc_hits   / n,
            "Top3Elem(%)":  100.0 * self._top3_elem_hits  / n,
            "MRR":                  self._mrr_sum          / n,
            "ActionAcc(%)": 100.0 * self._action_acc_hits / n,
            "StepAcc(%)":   100.0 * self._step_acc_hits   / n,
            "_total":               self._total,
        }

    def log(self, epoch: int, prefix: str = "Val"):
        metrics = self.compute()
        lines = [
            f"\n{'='*60}",
            f"  {prefix} Metrics — Epoch {epoch}  (n={metrics['_total']})",
            f"{'─'*60}",
            f"  ElemAcc   : {metrics['ElemAcc(%)']:>7.2f}%   "
            f"(top-1 element correct)",
            f"  Top3Elem  : {metrics['Top3Elem(%)']:>7.2f}%   "
            f"(GT element in top-3 predictions)",
            f"  MRR       : {metrics['MRR']:>8.4f}    "
            f"(mean reciprocal rank, top-3 window)",
            f"  ActionAcc : {metrics['ActionAcc(%)']:>7.2f}%   "
            f"(action type correct)",
            f"  StepAcc   : {metrics['StepAcc(%)']:>7.2f}%   "
            f"(element + action both correct)",
            f"{'='*60}",
        ]
        logger.info("\n".join(lines))
        return metrics


def _generate_prediction(combined_model, processor, batch,
                          graph_device: torch.device,
                          input_device: torch.device,
                          vis_device:   torch.device) -> list[str]:
    """
    Run greedy generation for a single validation batch and return
    the decoded text strings (one per item in the batch).
    Used only for metric computation — not for loss.
    """
    node_feats  = batch["node_feats"].to(graph_device)
    edge_index  = batch["edge_index"].to(graph_device)
    input_ids   = batch["input_ids"].to(input_device)
    attn_mask   = batch["attention_mask"].to(input_device)
    pix_vals    = batch["pixel_values"].to(device=vis_device, dtype=torch.float32)
    grid_thw    = batch["image_grid_thw"].to(device=vis_device)

    with torch.no_grad():
        node_embs = combined_model.graph_enc(node_feats, edge_index)
        combined_model._node_embs_cache = node_embs.to(
            device=input_device, dtype=torch.bfloat16
        )
        handle = (
            combined_model._get_decoder_layers()[combined_model.inject_layer]
            .register_forward_pre_hook(combined_model._build_inject_hook(),
                                       with_kwargs=True)
        )
        try:
            generated = combined_model.qwen.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pix_vals,
                image_grid_thw=grid_thw,
                max_new_tokens=MAX_TARGET_LEN,
                do_sample=False,
            )
        finally:
            handle.remove()
            combined_model._node_embs_cache = None

    prompt_len = input_ids.shape[1]
    decoded = []
    for i in range(generated.shape[0]):
        text = processor.decode(
            generated[i][prompt_len:], skip_special_tokens=True
        )
        decoded.append(text)
    return decoded


# ─────────────────────────────────────────────
# STEP 6: Training loop
# ─────────────────────────────────────────────

def train(combined_model, train_loader, val_loader, processor):
    _log_stage("Starting training")
    trainable   = [p for p in combined_model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    optimizer   = AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    scheduler   = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    global_step   = 0

    # Device references (cached once)
    graph_device  = next(combined_model.graph_enc.parameters()).device
    input_device  = combined_model.qwen.get_input_embeddings().weight.device
    vis_device    = QwenWithDOMGraph._find_visual_device(combined_model.qwen)

    for epoch in range(NUM_EPOCHS):
        _log_stage(f"Epoch {epoch+1}/{NUM_EPOCHS} — TRAIN")
        combined_model.train()
        running_loss  = 0.0
        nan_skipped   = 0
        optimizer.zero_grad()
        epoch_t0 = time.time()

        for step, batch in enumerate(train_loader):
            step_t0      = time.time()
            gpu_batch    = {k: v for k, v in batch.items()
                            if k not in ("node_feats", "edge_index",
                                         "num_candidates", "target_idx",
                                         "target_action")}
            node_feats   = batch["node_feats"].to(graph_device)
            edge_index   = batch["edge_index"].to(graph_device)

            logger.info(
                f"Step {step} | data loaded ({time.time()-step_t0:.1f}s) | "
                f"nodes={node_feats.shape[0]} ids={batch['input_ids'].shape} "
                f"pix={batch['pixel_values'].shape}{_gpu_mem()}"
            )

            t_fwd = time.time()
            outputs = combined_model(
                **gpu_batch, node_feats=node_feats, edge_index=edge_index
            )
            fwd_loss_val = outputs.loss.item()
            logger.info(
                f"Step {step} | forward done ({time.time()-t_fwd:.1f}s) "
                f"loss={fwd_loss_val:.4f}{_gpu_mem()}"
            )

            # FIX 3: guard before backward
            if not torch.isfinite(outputs.loss):
                nan_skipped += 1
                logger.warning(
                    f"Step {step} | NaN/Inf loss — skipping "
                    f"(total skipped this window: {nan_skipped})"
                )
                combined_model._node_embs_cache = None
                optimizer.zero_grad()
                continue

            loss  = outputs.loss / GRAD_ACCUM_STEPS
            t_bwd = time.time()
            loss.backward()
            logger.info(
                f"Step {step} | backward done ({time.time()-t_bwd:.1f}s){_gpu_mem()}"
            )

            combined_model._node_embs_cache = None
            running_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                logger.info(
                    f"Step {step} | optimizer stepped | "
                    f"global_step={global_step}{_gpu_mem()}"
                )

                if global_step % 50 == 0:
                    denom   = 50 * GRAD_ACCUM_STEPS - nan_skipped
                    avg     = running_loss / max(denom, 1)
                    elapsed = time.time() - epoch_t0
                    steps_done  = step + 1
                    steps_total = len(train_loader)
                    eta = (elapsed / steps_done) * (steps_total - steps_done)
                    logger.info(
                        f"Epoch {epoch+1} | Step {global_step} | Loss {avg:.4f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e} | "
                        f"NaN-skipped {nan_skipped} | "
                        f"ETA {eta/60:.1f}min{_gpu_mem()}"
                    )
                    running_loss = 0.0
                    nan_skipped  = 0

            total_step_time = time.time() - step_t0
            logger.info(
                f"Step {step} | TOTAL {total_step_time:.1f}s | "
                f"est epoch time {total_step_time * len(train_loader) / 60:.0f}min"
            )

        # ── Validate ──────────────────────────────────────────────────────
        _log_stage(f"Epoch {epoch+1}/{NUM_EPOCHS} — VALIDATE")
        combined_model.eval()
        val_loss    = 0.0
        val_batches = 0
        metrics     = EvalMetrics()

        with torch.no_grad():
            for batch in val_loader:
                gpu_batch  = {k: v for k, v in batch.items()
                              if k not in ("node_feats", "edge_index",
                                           "num_candidates", "target_idx",
                                           "target_action")}
                node_feats = batch["node_feats"].to(graph_device)
                edge_index = batch["edge_index"].to(graph_device)

                # ── Forward pass for loss ─────────────────────────────────
                outputs = combined_model(
                    **gpu_batch, node_feats=node_feats, edge_index=edge_index
                )

                if not torch.isfinite(outputs.loss):
                    logger.warning("Val batch has non-finite loss — skipping.")
                    combined_model._node_embs_cache = None
                    continue

                val_loss    += outputs.loss.item()
                val_batches += 1
                combined_model._node_embs_cache = None

                # ── Generation for metrics ────────────────────────────────
                # Re-uses the same batch; generates autoregressively.
                gen_texts = _generate_prediction(
                    combined_model, processor, batch,
                    graph_device, input_device, vis_device,
                )

                B_val = len(gen_texts)
                for b in range(B_val):
                    n_cands    = batch["num_candidates"][b]
                    true_elem  = batch["target_idx"][b]
                    true_act   = batch["target_action"][b]

                    # Parse predictions from generated text
                    pred_ranked = _decode_ranked_ids(gen_texts[b], n_cands)
                    pred_action = _decode_action_type(gen_texts[b])

                    # If model produced no parseable IDs, fall back to [0]
                    if not pred_ranked:
                        pred_ranked = list(range(min(3, n_cands)))

                    metrics.update(pred_ranked, pred_action, true_elem, true_act)

        avg_val_loss = val_loss / max(val_batches, 1)
        metric_dict  = metrics.log(epoch + 1, prefix="Val")

        logger.info(
            f"\n{'='*60}\nEpoch {epoch+1}/{NUM_EPOCHS}\n"
            f"  Val Loss  : {avg_val_loss:.4f}\n"
            f"  Epoch time: {(time.time()-epoch_t0)/60:.1f}min\n{'='*60}\n"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path     = os.path.join(OUTPUT_DIR, "best")
            os.makedirs(save_path, exist_ok=True)
            combined_model.qwen.save_pretrained(os.path.join(save_path, "lora"))
            torch.save(combined_model.graph_enc.state_dict(),
                       os.path.join(save_path, "graph_enc.pt"))
            torch.save(combined_model.adapter.state_dict(),
                       os.path.join(save_path, "adapter.pt"))
            # Save metric snapshot alongside checkpoint
            with open(os.path.join(save_path, "best_metrics.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "val_loss": avg_val_loss,
                           **metric_dict}, f, indent=2)
            logger.info(f"Checkpoint saved to {save_path}")

    final_path = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    combined_model.qwen.save_pretrained(os.path.join(final_path, "lora"))
    torch.save(combined_model.graph_enc.state_dict(),
               os.path.join(final_path, "graph_enc.pt"))
    torch.save(combined_model.adapter.state_dict(),
               os.path.join(final_path, "adapter.pt"))
    logger.info("Training complete.")


# ─────────────────────────────────────────────
# STEP 7: Inference
# ─────────────────────────────────────────────

def load_for_inference(checkpoint_path: str):
    _log_stage(f"Loading model for inference from {checkpoint_path}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    qwen_model = PeftModel.from_pretrained(
        base_model, os.path.join(checkpoint_path, "lora")
    )
    graph_enc  = DOMGraphTransformer()
    adapter    = GraphCrossAttentionAdapter()
    graph_enc.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "graph_enc.pt"),
                   weights_only=True)
    )
    adapter.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "adapter.pt"),
                   weights_only=True)
    )
    combined = QwenWithDOMGraph(qwen_model, graph_enc, adapter)
    combined.eval()
    logger.info("Inference model ready.")
    return combined


def predict(combined_model, processor, task, candidates, screenshot,
            dom_html="", top_k=1):
    graph_device = next(combined_model.graph_enc.parameters()).device
    node_feats, edge_index = dom_to_graph(
        dom_html, task_text=task, candidate_reprs=candidates
    )
    candidate_text = "\n".join([f"[{i}] {c}" for i, c in enumerate(candidates)])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": screenshot.convert("RGB")},
        {"type": "text",  "text": (
            f"Task: {task}\n\nCandidate actions:\n{candidate_text}\n\n"
            f"Rank the top 3 most appropriate actions:\n"
            f"1. [ID] action text\n2. [ID] action text\n3. [ID] action text"
        )},
    ]}]

    text_prompt     = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs          = processor(
        text=[text_prompt], images=image_inputs, return_tensors="pt",
        min_pixels=SCREENSHOT_MIN_PIXELS, max_pixels=SCREENSHOT_MAX_PIXELS,
    )

    input_device = combined_model.qwen.get_input_embeddings().weight.device
    vis_device   = QwenWithDOMGraph._find_visual_device(combined_model.qwen)

    with torch.no_grad():
        node_embs = combined_model.graph_enc(
            node_feats.to(graph_device), edge_index.to(graph_device)
        )
        combined_model._node_embs_cache = node_embs.to(
            device=input_device, dtype=torch.bfloat16
        )
        handle = (
            combined_model._get_decoder_layers()[combined_model.inject_layer]
            .register_forward_pre_hook(combined_model._build_inject_hook(),
                                       with_kwargs=True)
        )
        try:
            generated = combined_model.qwen.generate(
                input_ids=inputs["input_ids"].to(input_device),
                attention_mask=inputs["attention_mask"].to(input_device),
                pixel_values=inputs.get("pixel_values").to(
                    device=vis_device, dtype=torch.float32
                ),
                image_grid_thw=inputs.get("image_grid_thw").to(device=vis_device),
                max_new_tokens=MAX_TARGET_LEN,
                do_sample=False,
            )
        finally:
            handle.remove()
            combined_model._node_embs_cache = None

    prompt_len = inputs["input_ids"].shape[1]
    gen_text   = processor.decode(
        generated[0][prompt_len:], skip_special_tokens=True
    )
    ranked = _decode_ranked_ids(gen_text, len(candidates))
    if not ranked:
        ranked = list(range(min(3, len(candidates))))
    return ranked[0] if top_k == 1 else ranked[:top_k]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    _log_stage("MAIN — script start")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _get_text_encoder()

    qwen_model, processor = load_model()

    _log_stage("Visual encoder device audit")
    for name, mod in qwen_model.named_modules():
        if any(k in name for k in ["patch_embed", "visual.merger", "visual.blocks"]):
            try:
                dev = next(mod.parameters()).device
                logger.info(f"  {name}: {dev}")
                break
            except StopIteration:
                pass
    vis_dev = QwenWithDOMGraph._find_visual_device(qwen_model)
    logger.info(f"  → pixel_values will be sent to: {vis_dev}")

    qwen_model = apply_lora(qwen_model)

    GRAPH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph_enc    = DOMGraphTransformer().to(GRAPH_DEVICE)
    adapter      = GraphCrossAttentionAdapter().to(device=GRAPH_DEVICE)
    logger.info(
        f"Graph encoder params: {sum(p.numel() for p in graph_enc.parameters())/1e6:.2f}M"
    )
    logger.info(
        f"Adapter params:       {sum(p.numel() for p in adapter.parameters())/1e6:.2f}M"
    )

    combined_model = QwenWithDOMGraph(
        qwen_model, graph_enc, adapter, inject_layer=GRAPH_INJECT_LAYER
    )
    logger.info(f"Combined model assembled.{_gpu_mem()}")

    train_loader, val_loader = build_dataloaders(processor)

    train(combined_model, train_loader, val_loader, processor)

    _log_stage("Inference smoke test")
    inf_model        = load_for_inference(os.path.join(OUTPUT_DIR, "best"))
    dummy_screenshot = Image.new("RGB", (1280, 720), color=(200, 200, 200))
    dummy_dom        = (
        "<html><body>"
        "<button id='search'>Search</button>"
        "<input type='text'/>"
        "</body></html>"
    )
    pred = predict(
        inf_model, processor,
        task="Book a flight from New York to London",
        candidates=["Click search button", "Type 'New York' in origin", "Select date"],
        screenshot=dummy_screenshot, dom_html=dummy_dom, top_k=3,
    )
    logger.info(f"Smoke test top-3: {pred}")
    _log_stage("ALL DONE")