"""
Graph Transformer + LoRA Fine-Tuning: Qwen3-VL-8B-Instruct on Multimodal-Mind2Web
─────────────────────────────────────────────────────────────────────────────────
Architecture (Additive Cross-Attention variant):
    DOM Tree (HTML nodes)
        ↓
    DOMGraphTransformer  [TRAINABLE]   — GAT over tag/attr/text node features
        ↓
    GraphCrossAttentionAdapter [TRAINABLE]
        — Fires as a pre-hook at GRAPH_INJECT_LAYER.
          Every existing text/image token attends over ALL projected DOM nodes
          and the result is added as a residual to the hidden states.
          Sequence length is NEVER modified, so the 4-D causal mask and LoRA
          adapters at every subsequent layer work correctly.
        ↓
    Qwen3-VL (frozen base) + LoRA adapters at layers 19-21 [TRAINABLE]
        ↓
    Generated top-3 ranked list: "1. [ID] action  2. [ID] action  3. [ID] action"

Why NOT prefix tokens:
    Qwen's model loop computes one causal_mask = (B,1,S,S) before iterating
    over layers.  Prepending K tokens at layer-20 via a pre-hook makes layer
    20 see (B,K+S,D) hidden states, but layers 21-28 still get the original
    (B,1,S,S) mask applied to (B,K+S,D) states → shape mismatch / silent
    corruption.  The additive approach avoids this entirely.

Only the Graph Transformer, adapter, and LoRA adapters are trained.
Qwen base weights are fully frozen.

Install:
    pip install torch transformers datasets accelerate bitsandbytes peft \\
                qwen-vl-utils torch-geometric beautifulsoup4 sentence-transformers
Run:
    python multimodal_GNN_Cross.py
"""
import io
import os
import re
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from bs4 import BeautifulSoup
from transformers import (
    Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration,
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
import math
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR       = "qwen_lora_mind2web_cross"
NUM_EPOCHS       = 3
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE    = 3e-4
MAX_INPUT_LEN    = 1024
MAX_TARGET_LEN   = 80   # generative top-3 ranked list output
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# Graph Transformer settings
NODE_FEAT_DIM    = 523     # 384 (sentence-transformer) + 134 (tag one-hot) + 5 (structural)
GRAPH_HIDDEN_DIM = 256     # hidden dim inside the GAT layers
GRAPH_OUT_DIM    = 512     # output dim of the graph transformer
GAT_HEADS        = 4       # multi-head attention heads in GAT
GAT_LAYERS       = 3       # number of GAT message-passing layers
MAX_NODES        = 64      # truncate DOM to this many nodes (memory budget)
MAX_NEIGHBORS    = None    # max incoming edges each GAT node attends to; None = all
QWEN_HIDDEN_DIM  = 4096    # Qwen3-VL-8B hidden size
PROJ_ATTN_HEADS  = 8       # attention heads in the cross-attention adapter
GRAPH_INJECT_LAYER = 20    # Qwen transformer layer at which graph is fused (0-indexed, out of 28)

# Pretrained GAT checkpoint (from GAT_Pretrain.py training run)
GAT_PRETRAINED_PATH = "/content/drive/MyDrive/gat_pretrain/best_model.pt"  # set to None to skip
GAT_FROZEN          = True   # freeze GAT weights during cross-attention fine-tuning


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def save_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_action_from_text(s: str):
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


def _screenshot_to_pil(screenshot):
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


# ─────────────────────────────────────────────
# STEP 0: DOM → Graph utilities
# ─────────────────────────────────────────────

# Full HTML5 tag vocabulary + common inline SVG + deprecated HTML4
# [UNK] catches custom/framework elements (e.g. Angular/React components)
TAG_VOCAB = {
    # ── HTML5 (0–111) ────────────────────────────────────────────────────
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
    # ── Inline SVG (112–124) ─────────────────────────────────────────────
    "svg": 112, "path": 113, "g": 114, "rect": 115, "circle": 116,
    "ellipse": 117, "line": 118, "polyline": 119, "polygon": 120,
    "defs": 121, "use": 122, "symbol": 123, "tspan": 124,
    # ── Deprecated HTML4 (125–132) ───────────────────────────────────────
    "font": 125, "center": 126, "strike": 127, "frame": 128,
    "frameset": 129, "noframes": 130, "tt": 131, "big": 132,
    # ── Fallback ─────────────────────────────────────────────────────────
    "[UNK]": 133,
}
TAG_VOCAB_SIZE = len(TAG_VOCAB)  # 134

_TEXT_ENCODER = None

def _get_text_encoder():
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _TEXT_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _TEXT_ENCODER


def dom_to_graph(dom_html: str):
    """
    Parses HTML into a graph with rich node features.
        node_feats : (N, NODE_FEAT_DIM=523) = 384 (sentence-transformer)
                                             + 134 (tag one-hot, HTML5+SVG+deprecated)
                                             +   5 (structural: depth, is_interactive,
                                                    log_children, sibling_rank, has_id)
        edge_index : (2, E)    — bidirectional parent↔child edges
    Interactive elements are prioritised so the node budget is not wasted on <head>/<meta>.
    """
    soup = BeautifulSoup(dom_html or "<html></html>", "html.parser")
    _INTERACTIVE = {"button", "input", "a", "select", "textarea", "option", "label", "form"}
    all_els     = soup.find_all(True)
    interactive = [e for e in all_els if e.name in _INTERACTIVE]
    other       = [e for e in all_els if e.name not in _INTERACTIVE]
    elements    = (interactive + other)[:MAX_NODES]

    if not elements:
        return torch.zeros(1, NODE_FEAT_DIM), torch.tensor([[0], [0]], dtype=torch.long)

    node_id_map = {id(el): idx for idx, el in enumerate(elements)}

    # ── Depths ────────────────────────────────────────────────────────────
    depths    = [len(list(el.parents)) for el in elements]
    max_depth = max(depths) if depths else 1

    # ── Sibling ranks ──────────────────────────────────────────────────────
    sibling_ranks = []
    for el in elements:
        if el.parent is None:
            sibling_ranks.append(0.0)
        else:
            siblings = [c for c in el.parent.children if hasattr(c, "name") and c.name]
            pos = next((i for i, s in enumerate(siblings) if s is el), 0)
            sibling_ranks.append(pos / max(1.0, len(siblings) - 1))

    # ── Raw text strings for batch encoding ───────────────────────────────
    raw_texts = []
    for el in elements:
        tag   = el.name or ""
        attrs = " ".join(f"{k}={v}" for k, v in (el.attrs or {}).items())
        text  = (el.get_text(separator=" ", strip=True) or "")[:128]
        raw_texts.append(f"{tag} {attrs} {text}".strip())

    # ── Text embeddings (384-dim) ──────────────────────────────────────────
    encoder = _get_text_encoder()
    with torch.no_grad():
        text_embs = encoder.encode(
            raw_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device="cpu",
        ).float()   # (N, 384)

    # ── Tag one-hot (TAG_VOCAB_SIZE=134 dim) ──────────────────────────────
    tag_ids    = [TAG_VOCAB.get(el.name, TAG_VOCAB["[UNK]"]) for el in elements]
    tag_onehot = F.one_hot(
        torch.tensor(tag_ids, dtype=torch.long), num_classes=TAG_VOCAB_SIZE
    ).float()   # (N, 134)

    # ── Structural features (5-dim) ────────────────────────────────────────
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

    node_feats = torch.cat([text_embs, tag_onehot, struct_feats], dim=-1)  # (N, NODE_FEAT_DIM=523)

    # ── Edges (bidirectional: parent→child + child→parent) ────────────────
    src_list, dst_list = [], []
    for el in elements:
        for child in el.children:
            if hasattr(child, "name") and child.name:
                if id(el) in node_id_map and id(child) in node_id_map:
                    p, c = node_id_map[id(el)], node_id_map[id(child)]
                    src_list.append(p); dst_list.append(c)  # parent → child
                    src_list.append(c); dst_list.append(p)  # child  → parent

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
    """
    Single Graph Attention Network layer.
    Implements: h_i' = σ( Σ_j  α_ij · W · h_j )
    Numerically stable softmax via per-destination max subtraction.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, max_neighbors: int = None):
        super().__init__()
        self.heads         = heads
        self.out_dim       = out_dim
        self.head_dim      = out_dim // heads
        self.max_neighbors = max_neighbors
        self.W             = nn.Linear(in_dim, out_dim, bias=False)
        self.a             = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout       = nn.Dropout(dropout)
        self.leaky_relu    = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N  = x.size(0)
        Wx = self.W(x).view(N, self.heads, self.head_dim)   # (N, H, D/H)

        src, dst = edge_index[0], edge_index[1]
        Wx_src = Wx[src]
        Wx_dst = Wx[dst]
        e = self.leaky_relu(
            self.a(torch.cat([Wx_src, Wx_dst], dim=-1))
        ).squeeze(-1)  # (E, H)

        # Optionally restrict each node to its top max_neighbors incoming edges
        if self.max_neighbors is not None:
            score_avg = e.mean(dim=1)                                  # (E,) head-averaged score for selection
            keep = torch.zeros(e.size(0), dtype=torch.bool, device=x.device)
            for d in range(N):
                nbr_idx = (dst == d).nonzero(as_tuple=True)[0]
                if len(nbr_idx) == 0:
                    continue
                if len(nbr_idx) <= self.max_neighbors:
                    keep[nbr_idx] = True
                else:
                    topk = score_avg[nbr_idx].topk(self.max_neighbors).indices
                    keep[nbr_idx[topk]] = True
            e = e.masked_fill(~keep.unsqueeze(1), float('-inf'))

        # Numerically stable softmax over incoming edges per destination node
        idx   = dst.unsqueeze(1).expand_as(e)
        e_max = torch.full((N, self.heads), float('-inf'), device=x.device)
        e_max.scatter_reduce_(0, idx, e, reduce='amax', include_self=True)
        e_exp = (e - e_max[dst]).exp()
        denom = torch.zeros(N, self.heads, device=x.device)
        denom.scatter_add_(0, idx, e_exp)
        alpha = self.dropout(e_exp / (denom[dst] + 1e-8))  # (E, H)

        weighted = alpha.unsqueeze(-1) * Wx_src             # (E, H, D/H)
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(weighted), weighted)
        return F.elu(out.view(N, self.out_dim))


class DOMGraphTransformer(nn.Module):
    """Stacked GAT layers over the DOM tree. Returns per-node embeddings (N, out_dim)."""

    def __init__(
        self,
        in_dim:        int = NODE_FEAT_DIM,
        hidden_dim:    int = GRAPH_HIDDEN_DIM,
        out_dim:       int = GRAPH_OUT_DIM,
        heads:         int = GAT_HEADS,
        num_layers:    int = GAT_LAYERS,
        dropout:       float = 0.1,
        max_neighbors: int = None,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers      = nn.ModuleList([GATLayer(dims[i], dims[i+1], heads, dropout, max_neighbors) for i in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(num_layers)])
        self.dropout     = nn.Dropout(dropout)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = node_feats
        for layer, norm in zip(self.layers, self.norm_layers):
            x = self.dropout(norm(layer(x, edge_index)))
        return x  # (N, out_dim)


# ─────────────────────────────────────────────
# STEP 2: Graph → Qwen Adapter (Additive Cross-Attention)
# ─────────────────────────────────────────────

class GraphCrossAttentionAdapter(nn.Module):
    """
    Fuses GAT node embeddings into Qwen's hidden states via cross-attention.

    Instead of prepending K prefix tokens (which changes the sequence length
    mid-layer and breaks the 4-D causal mask for all subsequent layers), this
    adapter lets every existing token attend over the projected DOM nodes and
    adds the result as a gated residual.  Sequence length is always preserved.

    As a bonus, the queries are the ACTUAL layer-(inject-1) hidden states
    (rich contextual representations of text + image), not raw embedding-table
    lookups — so the graph signal is task-conditioned without a separate
    task-embedding hack.

    Architecture:
        hidden_states  (B, S, D)      ─── Q   (rich text+image context)
        node_embs      (N, graph_dim) ─► node_proj ─── K / V  (B, N, D)
        cross-attention(Q, K, V)      ─► correction  (B, S, D)
        LayerNorm + gated residual    ─► (B, S, D)

    Gate is a learnable scalar initialised to 0 so tanh(gate)=0 at init,
    meaning the adapter starts as identity and opens gradually during training.
    """

    def __init__(
        self,
        graph_dim:  int = GRAPH_OUT_DIM,
        qwen_dim:   int = QWEN_HIDDEN_DIM,
        attn_heads: int = PROJ_ATTN_HEADS,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.node_proj  = nn.Linear(graph_dim, qwen_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=qwen_dim, num_heads=attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(qwen_dim)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh(0) = 0 at init → identity

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, S, D)
        node_embs:     torch.Tensor,   # (N, graph_dim)
    ) -> torch.Tensor:
        """Returns (B, S, D) — same shape as hidden_states."""
        B = hidden_states.size(0)
        nodes = self.node_proj(node_embs).unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        correction, _ = self.cross_attn(hidden_states, nodes, nodes)      # (B, S, D)
        return hidden_states + self.gate.tanh() * self.norm(correction)


# ─────────────────────────────────────────────
# STEP 3: Load Qwen + apply LoRA
# ─────────────────────────────────────────────

def load_model():
    logger.info("Loading Qwen3-VL in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.info(f"Qwen loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, processor


def apply_lora(model):
    logger.info("Freezing Qwen base weights and applying LoRA...")
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        # q/k/v/o: self-attention; gate/up/down: FFN — adapting FFN helps model
        # route new graph context into its knowledge retrieval pathways
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # layer inject-1: produces Q vectors that attend to DOM nodes
        # layer inject:   receives cross-attention output
        # layer inject+1: first layer to integrate the fused representation
        layers_to_transform=[
            GRAPH_INJECT_LAYER - 1,
            GRAPH_INJECT_LAYER,
            GRAPH_INJECT_LAYER + 1,
        ],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# STEP 4: Combined model wrapper
# ─────────────────────────────────────────────

class QwenWithDOMGraph(nn.Module):
    """
    Wraps Qwen3-VL + DOMGraphTransformer + GraphCrossAttentionAdapter.

    Graph information is fused at GRAPH_INJECT_LAYER via additive cross-attention
    so Qwen has already built rich contextual representations of text and image
    before the DOM graph is incorporated.  Sequence length is never modified,
    so the 4-D causal mask and LoRA adapters at layers 20-21 work correctly.

    Forward pass:
        1. GAT encodes DOM graph → node_embs (N, D_graph)
        2. embed_tokens → token_embeds (B, S, D)
        3. Qwen layers 0 … inject_layer-1 run on token_embeds normally
        4. Pre-hook at inject_layer calls adapter(hidden_states, node_embs):
               every token cross-attends over all graph nodes → additive residual
               hidden_states shape stays (B, S, D) — mask is untouched
        5. Qwen layers inject_layer … N-1 run on graph-augmented hidden states
        6. Normal CE loss on original labels (no sequence-length offset)

    Backprop chain:
        loss → lm_head → layers 28..20 (LoRA at 20,21) → hook → adapter → GAT
    """

    def __init__(self, qwen_model, graph_transformer, adapter,
                 inject_layer: int = GRAPH_INJECT_LAYER):
        super().__init__()
        self.qwen          = qwen_model
        self.graph_enc     = graph_transformer
        self.adapter       = adapter
        self.inject_layer  = inject_layer
        self._pending_node_embs: torch.Tensor | None = None

    def _build_inject_hook(self):
        """Pre-hook for inject_layer: fuses graph via additive cross-attention.
        Sequence length is never modified, so attention masks stay valid."""
        def hook(module, args, kwargs):
            node_embs = self._pending_node_embs
            if node_embs is None:
                return args, kwargs
            hidden_states = args[0]                          # (B, S, D)
            new_hs = self.adapter(hidden_states, node_embs)  # (B, S, D) — same shape
            self._pending_node_embs = None                   # inject only once
            return (new_hs,) + args[1:], kwargs              # mask/pos_ids unchanged
        return hook

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, S)
        attention_mask: torch.Tensor,   # (B, S)
        pixel_values:   torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels:         torch.Tensor,   # (B, S)
        node_feats:     torch.Tensor,   # (N, NODE_FEAT_DIM)
        edge_index:     torch.Tensor,   # (2, E)
    ):
        device = input_ids.device

        # ── 1. Encode DOM graph ───────────────────────────────────────────
        node_embs = self.graph_enc(
            node_feats.to(device), edge_index.to(device),
        )                                                    # (N, GRAPH_OUT_DIM)

        # ── 2. Token embeddings ───────────────────────────────────────────
        token_embeds = self.qwen.model.embed_tokens(input_ids)   # (B, S, D)

        # ── 3. Register mid-layer cross-attention hook ────────────────────
        self._pending_node_embs = node_embs
        handle = self.qwen.model.layers[self.inject_layer].register_forward_pre_hook(
            self._build_inject_hook(), with_kwargs=True
        )

        # ── 4. Qwen forward — hook fuses graph at layer inject_layer ─────
        try:
            outputs = self.qwen(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,               # no offset: sequence length unchanged
            )
        finally:
            handle.remove()
            self._pending_node_embs = None

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
        target_idx = example.get("target_action_index", 0)
        screenshot = example.get("screenshot")
        # FIX 1: Mind2Web uses cleaned_html / raw_html, not dom_tree
        dom_html   = example.get("cleaned_html") or example.get("raw_html") or ""

        # ── Screenshot → PIL RGB ──────────────────────────────────────────
        if screenshot is None:
            screenshot = Image.new("RGB", (1280, 720), color=(255, 255, 255))
        elif isinstance(screenshot, bytes):
            screenshot = Image.open(io.BytesIO(screenshot)).convert("RGB")
        elif isinstance(screenshot, str):
            screenshot = Image.open(screenshot).convert("RGB")
        elif not isinstance(screenshot, Image.Image):
            screenshot = Image.fromarray(screenshot).convert("RGB")
        else:
            screenshot = screenshot.convert("RGB")

        # ── DOM HTML → graph tensors ──────────────────────────────────────
        node_feats, edge_index = dom_to_graph(dom_html)

        # ── Build Qwen multimodal message ─────────────────────────────────
        candidate_text = "\n".join([f"[{i}] {c}" for i, c in enumerate(candidates)])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": (
                    f"Task: {task}\n\n"
                    f"Candidate actions:\n{candidate_text}\n\n"
                    f"Based on the screenshot and page structure, rank the top 3 "
                    f"most appropriate actions from most to least recommended. "
                    f"Reply in this exact format:\n"
                    f"1. [ID] action text\n2. [ID] action text\n3. [ID] action text"
                )},
            ],
        }]

        text_prompt      = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _  = process_vision_info(messages)
        inputs           = self.processor(
            text=[text_prompt], images=image_inputs,
            padding="max_length", truncation=True,
            max_length=MAX_INPUT_LEN, return_tensors="pt",
        )

        # ── Target label: ranked top-3 list ───────────────────────────────
        # Rank 1 = ground-truth candidate; ranks 2-3 = randomly sampled negatives.
        # The model learns to generate the correct candidate first (most important
        # supervision signal) while producing a plausible ranked list overall.
        neg_cands = [(i, c) for i, c in enumerate(candidates) if i != int(target_idx)]
        random.shuffle(neg_cands)
        top3 = [(int(target_idx), candidates[int(target_idx)])] + neg_cands[:2]
        target_str    = "\n".join([f"{r+1}. [{i}] {c}" for r, (i, c) in enumerate(top3)])
        target_tokens = self.processor.tokenizer(
            target_str, add_special_tokens=False,
        ).input_ids

        labels        = torch.full((MAX_INPUT_LEN,), -100, dtype=torch.long)
        pad_id        = self.processor.tokenizer.pad_token_id
        input_ids_row = inputs["input_ids"].squeeze(0)
        non_pad       = (input_ids_row != pad_id).nonzero(as_tuple=True)[0]
        if len(non_pad) > 0:
            insert_at = non_pad[-1].item() + 1
            for j, tid in enumerate(target_tokens):
                if insert_at + j < MAX_INPUT_LEN:
                    labels[insert_at + j] = tid

        return {
            "input_ids":       inputs["input_ids"].squeeze(0),
            "attention_mask":  inputs["attention_mask"].squeeze(0),
            "pixel_values":    inputs["pixel_values"].squeeze(0),
            "image_grid_thw":  inputs["image_grid_thw"].squeeze(0),
            "node_feats":      node_feats,
            "edge_index":      edge_index,
            "labels":          labels,
            "num_candidates":  len(candidates),
        }


def build_dataloaders(processor):
    logger.info("Loading Mind2Web dataset...")
    raw        = load_dataset("osunlp/Multimodal-Mind2Web")
    full_train = raw["train"]
    split      = full_train.train_test_split(test_size=0.1, seed=42)
    train_split, val_split = split["train"], split["test"]
    logger.info(f"Split: {len(train_split)} train / {len(val_split)} val")

    train_ds = Mind2WebDataset(train_split, processor)
    val_ds   = Mind2WebDataset(val_split,   processor)

    def collate_fn(batch):
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
            "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch]),
            "node_feats":     batch[0]["node_feats"],
            "edge_index":     batch[0]["edge_index"],
            "labels":         torch.stack([b["labels"]         for b in batch]),
            "num_candidates": [b["num_candidates"]             for b in batch],
        }

    _nw = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=_nw, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=_nw, collate_fn=collate_fn)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# STEP 6: Training loop
# ─────────────────────────────────────────────

def train(combined_model, train_loader, val_loader, processor):
    device   = next(combined_model.parameters()).device
    trainable = [p for p in combined_model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")

    optimizer   = AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    scheduler   = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(NUM_EPOCHS):

        # ── Train ─────────────────────────────────────────────────────────
        combined_model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            gpu_batch  = {k: v.to(device) for k, v in batch.items()
                          if k not in ("node_feats", "edge_index", "num_candidates")}
            node_feats = batch["node_feats"].to(device)
            edge_index = batch["edge_index"].to(device)

            outputs = combined_model(**gpu_batch, node_feats=node_feats, edge_index=edge_index)
            loss    = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = running_loss / (50 * GRAD_ACCUM_STEPS)
                    logger.info(f"Epoch {epoch+1} | Step {global_step} | Loss {avg:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

        # ── Validate ──────────────────────────────────────────────────────
        combined_model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for batch in val_loader:
                gpu_batch  = {k: v.to(device) for k, v in batch.items()
                              if k not in ("node_feats", "edge_index", "num_candidates")}
                node_feats = batch["node_feats"].to(device)
                edge_index = batch["edge_index"].to(device)

                outputs  = combined_model(**gpu_batch, node_feats=node_feats, edge_index=edge_index)
                val_loss += outputs.loss.item()

                # FIX 3: length-normalised log-prob scoring over all candidates
                # Handles multi-token targets (indices 10-49 tokenise as 2 tokens)
                labels_cpu   = batch["labels"]               # (B, S)
                num_cands    = batch["num_candidates"]        # list[int]
                B_val        = labels_cpu.size(0)
                S            = labels_cpu.size(1)

                for b in range(B_val):
                    # Locate where the target answer starts in labels
                    tgt_mask = labels_cpu[b] != -100         # (S,)
                    if not tgt_mask.any():
                        continue
                    first_t = tgt_mask.float().argmax().item()
                    if first_t == 0:
                        continue                              # no preceding logit position

                    # Decode the ground-truth target string from labels
                    tgt_tids = []
                    pos = first_t
                    while pos < S and labels_cpu[b, pos] != -100:
                        tgt_tids.append(labels_cpu[b, pos].item())
                        pos += 1
                    if not tgt_tids:
                        continue
                    try:
                        true_idx = int(processor.tokenizer.decode(tgt_tids).strip())
                    except ValueError:
                        continue

                    # Score every valid candidate index by length-normalised log-prob
                    log_p = F.log_softmax(outputs.logits[b], dim=-1)  # (S, vocab)
                    n_c   = num_cands[b]
                    best_idx, best_score = 0, float('-inf')
                    for ci in range(n_c):
                        tids = processor.tokenizer(
                            str(ci), add_special_tokens=False
                        ).input_ids
                        if not tids:
                            continue
                        # log-prob of first token at position (first_t - 1)
                        sc = log_p[first_t - 1, tids[0]].item()
                        if len(tids) >= 2 and first_t < S:
                            # conditional log-prob of second token
                            sc += log_p[first_t, tids[1]].item()
                        sc /= len(tids)                      # length normalisation
                        if sc > best_score:
                            best_score, best_idx = sc, ci

                    if best_idx == true_idx:
                        correct += 1
                    total += 1

        avg_val_loss = val_loss / len(val_loader)
        accuracy     = correct / total if total > 0 else 0.0
        logger.info(f"\n{'='*50}\nEpoch {epoch+1}/{NUM_EPOCHS}\n  Val Loss : {avg_val_loss:.4f}\n  Accuracy : {accuracy:.2%}\n{'='*50}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUT_DIR, "best")
            os.makedirs(save_path, exist_ok=True)
            combined_model.qwen.save_pretrained(os.path.join(save_path, "lora"))
            torch.save(combined_model.graph_enc.state_dict(), os.path.join(save_path, "graph_enc.pt"))
            torch.save(combined_model.adapter.state_dict(),   os.path.join(save_path, "adapter.pt"))
            logger.info(f"  Best model saved to {save_path}")

    final_path = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    combined_model.qwen.save_pretrained(os.path.join(final_path, "lora"))
    torch.save(combined_model.graph_enc.state_dict(), os.path.join(final_path, "graph_enc.pt"))
    torch.save(combined_model.adapter.state_dict(),   os.path.join(final_path, "adapter.pt"))
    logger.info(f"Training done. Saved to {final_path}")


# ─────────────────────────────────────────────
# STEP 7: Inference
# ─────────────────────────────────────────────

def load_for_inference(checkpoint_path: str):
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
    )
    qwen_model = PeftModel.from_pretrained(base_model, os.path.join(checkpoint_path, "lora"))

    graph_enc = DOMGraphTransformer()
    adapter   = GraphCrossAttentionAdapter()
    graph_enc.load_state_dict(torch.load(os.path.join(checkpoint_path, "graph_enc.pt"), weights_only=True))
    adapter.load_state_dict(torch.load(os.path.join(checkpoint_path, "adapter.pt"),     weights_only=True))

    combined = QwenWithDOMGraph(qwen_model, graph_enc, adapter, inject_layer=GRAPH_INJECT_LAYER)
    combined.eval()
    return combined


def predict(
    combined_model,
    processor,
    task:       str,
    candidates: list[str],
    screenshot: Image.Image,
    dom_html:   str = "",
    top_k:      int = 1,
) -> int | list[int]:
    """
    Full multimodal + graph inference.

    Generates a ranked top-3 list of candidate action indices using autoregressive
    decoding, then parses the [ID] tokens from the output text.

    Returns the top-1 index (int) when top_k=1, or a ranked list of top_k
    indices when top_k>1.
    """
    device = next(combined_model.parameters()).device

    node_feats, edge_index = dom_to_graph(dom_html)

    candidate_text = "\n".join([f"[{i}] {c}" for i, c in enumerate(candidates)])
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": screenshot.convert("RGB")},
            {"type": "text", "text": (
                f"Task: {task}\n\n"
                f"Candidate actions:\n{candidate_text}\n\n"
                f"Based on the screenshot and page structure, rank the top 3 "
                f"most appropriate actions from most to least recommended. "
                f"Reply in this exact format:\n"
                f"1. [ID] action text\n2. [ID] action text\n3. [ID] action text"
            )},
        ],
    }]

    text_prompt      = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _  = process_vision_info(messages)
    inputs           = processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(device)

    with torch.no_grad():
        node_embs    = combined_model.graph_enc(node_feats.to(device), edge_index.to(device))
        token_embeds = combined_model.qwen.model.embed_tokens(inputs["input_ids"])

        combined_model._pending_node_embs = node_embs
        handle = combined_model.qwen.model.layers[combined_model.inject_layer].register_forward_pre_hook(
            combined_model._build_inject_hook(), with_kwargs=True
        )
        try:
            generated = combined_model.qwen.generate(
                inputs_embeds=token_embeds,
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=MAX_TARGET_LEN,
                do_sample=False,
            )
        finally:
            handle.remove()
            combined_model._pending_node_embs = None

    # Decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    gen_text = processor.decode(generated[0][prompt_len:], skip_special_tokens=True)

    # Parse "[ID]" tokens in order: "1. [2] click search\n2. [0] type query\n3. [5] scroll"
    matches = re.findall(r'\[(\d+)\]', gen_text)
    ranked = []
    for m in matches:
        idx = int(m)
        if 0 <= idx < len(candidates) and idx not in ranked:
            ranked.append(idx)

    # Fallback: if generation was unparseable, return first candidates
    if not ranked:
        ranked = list(range(min(max(top_k, 3), len(candidates))))

    if top_k == 1:
        return ranked[0] if ranked else 0
    return ranked[:top_k]


def evaluate_split(
    combined_model,
    processor,
    dataset_split: str = "test_website",
    preds_out: str = "out_preds_cross.jsonl",
    top_k: int = 3,
):
    """Run full-dataset inference and emit evaluator-compatible records."""
    combined_model.eval()
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=dataset_split)
    results, correct_top1, correct_topk, total = [], 0, 0, 0

    for row in tqdm(dataset, desc=f"Eval {dataset_split}"):
        task       = row.get("confirmed_task", "")
        candidates = row.get("action_reprs") or []
        target     = int(row["target_action_index"]) if row.get("target_action_index") is not None else None
        # FIX 1: use cleaned_html
        dom_html   = row.get("cleaned_html") or row.get("raw_html") or ""
        screenshot = _screenshot_to_pil(row.get("screenshot"))

        pred_ranked  = predict(combined_model, processor, task=task, candidates=candidates,
                               screenshot=screenshot, dom_html=dom_html, top_k=top_k)
        pred_idx     = pred_ranked[0] if isinstance(pred_ranked, list) else pred_ranked
        pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None
        pred_action  = extract_action_from_text(pred_element) if pred_element else "CLICK"
        gt_element   = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
        gt_action    = extract_action_from_text(gt_element) if gt_element else None

        in_topk = (target in pred_ranked) if isinstance(pred_ranked, list) else (pred_idx == target)

        results.append({
            "id": row.get("annotation_id") or row.get("id"),
            "gt_element": gt_element, "gt_action": gt_action,
            "gt_value": row.get("gt_value"),
            "pred_element": pred_element, "pred_action": pred_action,
            "pred_value": None, "candidates": candidates,
            "pred_top1": pred_idx,
            "pred_topk": pred_ranked if isinstance(pred_ranked, list) else [pred_idx],
            "task_success_top1": (pred_idx == target) if (pred_idx is not None and target is not None) else None,
            "task_success_topk": in_topk if target is not None else None,
        })

        if pred_idx is not None and target is not None:
            if pred_idx == target:
                correct_top1 += 1
            if in_topk:
                correct_topk += 1
        total += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_jsonl(preds_out, results)
    acc1 = correct_top1 / total if total else 0.0
    acck = correct_topk / total if total else 0.0
    logger.info(f"Wrote {len(results)} records to {preds_out}")
    logger.info(f"Top-1 accuracy on {dataset_split}: {acc1:.2%}")
    logger.info(f"Top-{top_k} accuracy on {dataset_split}: {acck:.2%}")
    return acc1, acck


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    qwen_model, processor = load_model()
    qwen_model            = apply_lora(qwen_model)

    graph_enc = DOMGraphTransformer(
        in_dim=NODE_FEAT_DIM, hidden_dim=GRAPH_HIDDEN_DIM,
        out_dim=GRAPH_OUT_DIM, heads=GAT_HEADS, num_layers=GAT_LAYERS,
        max_neighbors=MAX_NEIGHBORS,
    )

    # Load pretrained GAT weights from GAT_Pretrain.py checkpoint.
    # The checkpoint stores GATPretrainer state; the DOMGraphTransformer
    # sub-module is keyed under "gat.*".
    from pathlib import Path
    if GAT_PRETRAINED_PATH and Path(GAT_PRETRAINED_PATH).exists():
        ckpt = torch.load(GAT_PRETRAINED_PATH, map_location="cpu")
        raw_state = ckpt.get("model_state_dict", ckpt)
        gat_state = {k[len("gat."):]: v for k, v in raw_state.items() if k.startswith("gat.")}
        missing, unexpected = graph_enc.load_state_dict(gat_state, strict=False)
        logger.info(f"Loaded pretrained GAT from {GAT_PRETRAINED_PATH} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        logger.warning(f"GAT_PRETRAINED_PATH not found ({GAT_PRETRAINED_PATH}), using random init.")

    if GAT_FROZEN:
        for p in graph_enc.parameters():
            p.requires_grad_(False)
        logger.info("GAT weights frozen.")

    adapter = GraphCrossAttentionAdapter(
        graph_dim=GRAPH_OUT_DIM, qwen_dim=QWEN_HIDDEN_DIM, attn_heads=PROJ_ATTN_HEADS,
    )

    combined_model = QwenWithDOMGraph(qwen_model, graph_enc, adapter, inject_layer=GRAPH_INJECT_LAYER)

    train_loader, val_loader = build_dataloaders(processor)
    train(combined_model, train_loader, val_loader, processor)

    logger.info("\n--- Inference smoke test ---")
    inf_model        = load_for_inference(os.path.join(OUTPUT_DIR, "best"))
    dummy_screenshot = Image.new("RGB", (1280, 720), color=(200, 200, 200))
    dummy_dom        = "<html><body><button id='search'>Search</button><input type='text'/></body></html>"

    pred = predict(
        inf_model, processor,
        task="Book a flight from New York to London",
        candidates=["Click search button", "Type 'New York' in origin field", "Select departure date"],
        screenshot=dummy_screenshot,
        dom_html=dummy_dom,
        top_k=3,
    )
    logger.info(f"Top-3 predicted action indices: {pred}")
