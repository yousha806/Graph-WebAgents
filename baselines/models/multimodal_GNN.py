"""
Graph Transformer + LoRA Fine-Tuning: Qwen2-VL-7B-Instruct on Multimodal-Mind2Web
─────────────────────────────────────────────────────────────────────────────────
Architecture:
    DOM Tree (HTML nodes)
        ↓
    DOMGraphTransformer  [TRAINABLE]   — GAT over tag/attr/text node features
        ↓
    GraphProjector       [TRAINABLE]   — linear: graph_dim → qwen_hidden_dim
        ↓
    Prefix tokens prepended to [screenshot tokens + text tokens]
        ↓
    Qwen2-VL (frozen base) + LoRA adapters [TRAINABLE]
        ↓
    Predicted action index

Only the Graph Transformer, projector, and LoRA adapters are trained.
Qwen base weights are fully frozen.

Install:
    pip install torch transformers datasets accelerate bitsandbytes peft \\
                qwen-vl-utils torch-geometric beautifulsoup4
Run:
    python train_lora_mind2web.py
"""
import io
import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from bs4 import BeautifulSoup
from transformers import (
    Qwen2VLForConditionalGeneration,
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
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR       = "qwen_lora_mind2web"
NUM_EPOCHS       = 3
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE    = 3e-4
MAX_INPUT_LEN    = 1024
MAX_TARGET_LEN   = 8
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# Graph Transformer settings
NODE_FEAT_DIM    = 128     # dimension of each node's text embedding
GRAPH_HIDDEN_DIM = 256     # hidden dim inside the GAT layers
GRAPH_OUT_DIM    = 512     # output dim of the graph transformer
GAT_HEADS        = 4       # multi-head attention heads in GAT
GAT_LAYERS       = 3       # number of GAT message-passing layers
MAX_NODES        = 64      # truncate DOM to this many nodes (memory budget)
NUM_GRAPH_TOKENS = 8       # how many prefix tokens the graph produces
QWEN_HIDDEN_DIM  = 3584    # Qwen3-VL-7B hidden size (verify per model config)


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def save_jsonl(path: str, records: list[dict]):
    """Write a list of dicts to JSONL."""
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_action_from_text(s: str):
    """Heuristically extract an action label from a candidate string."""
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
    """Normalize screenshot to a PIL RGB image."""
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

# Vocabulary of common HTML tags mapped to integer IDs
TAG_VOCAB = {
    "div": 0, "span": 1, "a": 2, "button": 3, "input": 4, "select": 5,
    "option": 6, "form": 7, "img": 8, "p": 9, "h1": 10, "h2": 11,
    "h3": 12, "li": 13, "ul": 14, "ol": 15, "table": 16, "tr": 17,
    "td": 18, "th": 19, "label": 20, "textarea": 21, "nav": 22,
    "header": 23, "footer": 24, "main": 25, "section": 26, "article": 27,
    "[UNK]": 28,
}
TAG_VOCAB_SIZE = len(TAG_VOCAB)

# Simple character-level hash to embed arbitrary text into a fixed vector
def text_to_vector(text: str, dim: int = NODE_FEAT_DIM) -> torch.Tensor:
    """
    Converts a raw text string into a fixed-size float vector using
    a deterministic character-hash projection.  No learned embeddings
    needed here — the Graph Transformer will learn to transform these.
    """
    vec = torch.zeros(dim)
    if not text:
        return vec
    text = text[:256]   # cap length
    for i, ch in enumerate(text):
        vec[i % dim] += ord(ch) / 1000.0
    # L2-normalise so scale doesn't dominate
    norm = vec.norm()
    return vec / (norm + 1e-8)


def dom_to_graph(dom_html: str):
    """
    Parses an HTML string into a graph:
        node_feats : (N, NODE_FEAT_DIM)  — one row per DOM node
        edge_index : (2, E)              — parent→child directed edges
    Node features = character-hash of  "<tag> id=... class=... inner_text..."
    """
    soup = BeautifulSoup(dom_html or "<html></html>", "html.parser")
    elements = soup.find_all(True)[:MAX_NODES]   # cap nodes

    node_feats = []
    node_id_map = {}   # maps BeautifulSoup element id → integer index

    for idx, el in enumerate(elements):
        node_id_map[id(el)] = idx
        tag   = el.name or ""
        attrs = " ".join(f"{k}={v}" for k, v in (el.attrs or {}).items())
        text  = (el.get_text(separator=" ", strip=True) or "")[:128]
        raw   = f"{tag} {attrs} {text}"
        node_feats.append(text_to_vector(raw))

    node_feats = torch.stack(node_feats)   # (N, NODE_FEAT_DIM)

    # Build parent→child edges from the DOM tree structure
    src_list, dst_list = [], []
    for el in elements:
        child_els = [c for c in el.children if hasattr(c, "name") and c.name]
        for child in child_els:
            if id(el) in node_id_map and id(child) in node_id_map:
                src_list.append(node_id_map[id(el)])
                dst_list.append(node_id_map[id(child)])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        # No edges → self-loops as fallback so GAT doesn't crash
        n = len(elements)
        idx_range = list(range(n))
        edge_index = torch.tensor([idx_range, idx_range], dtype=torch.long)

    return node_feats, edge_index   # (N, D), (2, E)


# ─────────────────────────────────────────────
# STEP 1: Graph Transformer (GAT)
# ─────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single Graph Attention Network layer.
    Implements: h_i' = σ( Σ_j  α_ij · W · h_j )
    where α_ij are learned attention coefficients over neighbours.
    We implement this manually so there's no torch-geometric dependency.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads   = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads

        # Shared linear for all neighbours
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention vector: [h_i || h_j] → scalar per head
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, in_dim)
        edge_index : (2, E)   src → dst
        returns    : (N, out_dim)
        """
        N = x.size(0)
        Wx = self.W(x)                          # (N, out_dim)
        Wx = Wx.view(N, self.heads, self.head_dim)   # (N, H, D/H)

        src, dst = edge_index[0], edge_index[1]

        # Attention: for each edge compute e_ij then softmax over neighbours
        Wx_src = Wx[src]    # (E, H, D/H)
        Wx_dst = Wx[dst]    # (E, H, D/H)
        e = self.leaky_relu(
            self.a(torch.cat([Wx_src, Wx_dst], dim=-1))  # (E, H, 1)
        ).squeeze(-1)       # (E, H)

        # Softmax over incoming edges per destination node
        # We use scatter_softmax manually via exp + scatter_add
        e_exp = e.exp()     # (E, H)
        denom = torch.zeros(N, self.heads, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand_as(e_exp), e_exp)
        alpha = e_exp / (denom[dst] + 1e-8)    # (E, H)
        alpha = self.dropout(alpha)

        # Aggregate: h_i' = Σ_j α_ij · Wx_j
        weighted = (alpha.unsqueeze(-1) * Wx_src)   # (E, H, D/H)
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand_as(weighted),
            weighted,
        )
        out = out.view(N, self.out_dim)   # (N, out_dim)
        return F.elu(out)


class DOMGraphTransformer(nn.Module):
    """
    Stacked GAT layers over the DOM tree.
    Input  : raw node features (N, NODE_FEAT_DIM)
    Output : graph-level embedding (GRAPH_OUT_DIM,) via mean pooling
             + per-node embeddings (N, GRAPH_OUT_DIM) for prefix token selection
    """

    def __init__(
        self,
        in_dim:     int = NODE_FEAT_DIM,
        hidden_dim: int = GRAPH_HIDDEN_DIM,
        out_dim:    int = GRAPH_OUT_DIM,
        heads:      int = GAT_HEADS,
        num_layers: int = GAT_LAYERS,
        dropout:    float = 0.1,
    ):
        super().__init__()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GATLayer(dims[i], dims[i + 1], heads=heads, dropout=dropout)
            for i in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feats:  torch.Tensor,   # (N, in_dim)
        edge_index:  torch.Tensor,   # (2, E)
    ) -> torch.Tensor:
        """Returns node embeddings (N, out_dim)."""
        x = node_feats
        for layer, norm in zip(self.layers, self.norm_layers):
            x = norm(layer(x, edge_index))
            x = self.dropout(x)
        return x   # (N, out_dim)


class GraphProjector(nn.Module):
    """
    Takes the top-K node embeddings from the GAT and projects them
    into Qwen's token embedding space as NUM_GRAPH_TOKENS prefix tokens.

    Strategy: pick the K nodes with highest L2 norm (most "activated" by the
    GAT), project each to QWEN_HIDDEN_DIM, treat them as soft tokens.
    """

    def __init__(
        self,
        graph_dim:       int = GRAPH_OUT_DIM,
        qwen_dim:        int = QWEN_HIDDEN_DIM,
        num_tokens:      int = NUM_GRAPH_TOKENS,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(graph_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        node_embeddings : (N, graph_dim)
        returns         : (num_tokens, qwen_dim)  ← prefix token embeddings
        """
        N = node_embeddings.size(0)
        k = min(self.num_tokens, N)

        # Select top-k nodes by activation magnitude
        scores    = node_embeddings.norm(dim=-1)    # (N,)
        topk_idx  = scores.topk(k).indices          # (k,)
        selected  = node_embeddings[topk_idx]       # (k, graph_dim)

        prefix = self.proj(selected)                # (k, qwen_dim)

        # Pad to exactly num_tokens if DOM was smaller than k
        if k < self.num_tokens:
            pad = torch.zeros(
                self.num_tokens - k, prefix.size(-1),
                device=prefix.device, dtype=prefix.dtype
            )
            prefix = torch.cat([prefix, pad], dim=0)

        return prefix   # (num_tokens, qwen_dim)


# ─────────────────────────────────────────────
# STEP 2: Load Qwen + apply LoRA
# ─────────────────────────────────────────────

def load_model():
    """
    Loads Qwen2-VL in 4-bit NF4 quantization.
    Returns model + processor.
    """
    logger.info("Loading Qwen2-VL in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logger.info(f"Qwen loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, processor


def apply_lora(model):
    """
    Freezes ALL Qwen base weights, then injects small trainable LoRA
    matrices into the attention projection layers.
    """
    logger.info("Freezing Qwen base weights and applying LoRA...")

    model.gradient_checkpointing_enable()

    # Hard freeze — nothing in Qwen trains except LoRA deltas
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# STEP 3: Combined model wrapper
# ─────────────────────────────────────────────

class QwenWithDOMGraph(nn.Module):
    """
    Wraps Qwen2-VL + DOMGraphTransformer + GraphProjector into one module.

    Forward pass:
        1. GAT encodes the DOM graph → node embeddings
        2. Projector selects top-K nodes → prefix token embeddings (B, K, D)
        3. Qwen's token embedding layer encodes input_ids → token embeddings (B, S, D)
        4. Graph prefix is prepended: [graph_prefix | token_embeddings]
        5. Qwen runs its transformer layers on the combined sequence
        6. Loss is computed on the label tokens as usual
    """

    def __init__(self, qwen_model, graph_transformer, projector):
        super().__init__()
        self.qwen        = qwen_model
        self.graph_enc   = graph_transformer
        self.projector   = projector

    def forward(
        self,
        input_ids:      torch.Tensor,         # (B, S)
        attention_mask: torch.Tensor,         # (B, S)
        pixel_values:   torch.Tensor,         # (B, C, H, W)
        image_grid_thw: torch.Tensor,         # (B, 3)
        labels:         torch.Tensor,         # (B, T)
        node_feats:     torch.Tensor,         # (N, NODE_FEAT_DIM)  — whole batch shares graph
        edge_index:     torch.Tensor,         # (2, E)
    ):
        B = input_ids.size(0)
        device = input_ids.device

        # ── 1. Encode DOM graph ───────────────────────────────────────────
        node_embs = self.graph_enc(
            node_feats.to(device),
            edge_index.to(device),
        )                                     # (N, GRAPH_OUT_DIM)

        # ── 2. Project to prefix tokens ───────────────────────────────────
        prefix = self.projector(node_embs)    # (NUM_GRAPH_TOKENS, QWEN_HIDDEN_DIM)
        prefix = prefix.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        # ── 3. Get Qwen token embeddings for the text+image tokens ────────
        # Qwen's embedding layer lives at model.model.embed_tokens
        token_embeds = self.qwen.model.embed_tokens(input_ids)  # (B, S, D)

        # ── 4. Prepend graph prefix to token sequence ─────────────────────
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)  # (B, K+S, D)

        # Extend attention mask to cover the K new prefix tokens
        prefix_mask = torch.ones(B, NUM_GRAPH_TOKENS, dtype=attention_mask.dtype, device=device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, K+S)

        # Extend labels: prefix positions are masked out with -100
        prefix_labels = torch.full((B, NUM_GRAPH_TOKENS), -100, dtype=labels.dtype, device=device)
        extended_labels = torch.cat([prefix_labels, labels], dim=1)   # (B, K+T) — note: labels shorter than S

        # ── 5. Forward through Qwen using inputs_embeds ───────────────────
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=extended_labels,
        )

        return outputs


# ─────────────────────────────────────────────
# STEP 4: Dataset
# ─────────────────────────────────────────────

class Mind2WebDataset(Dataset):
    """
    Multimodal Mind2Web dataset.
    Each item returns:
        - Qwen processor outputs (input_ids, attention_mask, pixel_values, image_grid_thw)
        - DOM graph tensors (node_feats, edge_index)
        - labels (target action index as token id)
    """

    def __init__(self, hf_dataset, processor):
        self.data      = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        task       = example.get("confirmed_task", "")
        candidates = example.get("action_reprs", [])
        target_idx = example.get("target_action_index", 0)
        screenshot = example.get("screenshot")
        dom_html   = example.get("dom_tree", "")   # raw HTML string

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
        candidate_text = "\n".join([f"{i}: {c}" for i, c in enumerate(candidates)])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {
                        "type": "text",
                        "text": (
                            f"Task: {task}\n\n"
                            f"Candidate actions:\n{candidate_text}\n\n"
                            f"Look at the screenshot and the page structure, "
                            f"then select the correct action index. "
                            f"Answer with ONLY the number."
                        ),
                    },
                ],
            }
        ]

        text_prompt  = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_tensors="pt",
        )

        # ── Target label ──────────────────────────────────────────────────
        target_enc = self.processor.tokenizer(
            str(int(target_idx)),
            return_tensors="pt",
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "node_feats":     node_feats,    # (N, NODE_FEAT_DIM)  — variable N
            "edge_index":     edge_index,    # (2, E)              — variable E
            "labels":         labels,
        }


def build_dataloaders(processor):
    logger.info("Loading Mind2Web dataset...")

    raw         = load_dataset("osunlp/Multimodal-Mind2Web")
    train_split = raw["train"]
    val_split   = raw.get("test", raw["train"].select(range(200)))

    train_ds = Mind2WebDataset(train_split, processor)
    val_ds   = Mind2WebDataset(val_split,   processor)

    def collate_fn(batch):
        """
        Most tensors are padded to fixed length so stacking is safe.
        node_feats and edge_index vary per sample — we pass only the
        first sample's graph when batch_size=1 (the recommended setting).
        For batch_size > 1 you would need to batch graphs (e.g. with
        torch_geometric's Batch.from_data_list).
        """
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
            "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch]),
            "node_feats":     batch[0]["node_feats"],    # (N, D) — first sample
            "edge_index":     batch[0]["edge_index"],    # (2, E)
            "labels":         torch.stack([b["labels"]   for b in batch]),
        }

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# STEP 5: Training loop
# ─────────────────────────────────────────────

def train(combined_model, train_loader, val_loader):
    """
    Trains ONLY the graph transformer, projector, and LoRA adapters.
    Qwen base weights stay frozen throughout.
    """
    device = next(combined_model.parameters()).device

    # Only parameters with requires_grad=True (graph encoder + projector + LoRA)
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
            # Move fixed-shape tensors to GPU
            gpu_batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ("node_feats", "edge_index")
            }
            # Graph tensors moved separately (variable shape)
            node_feats = batch["node_feats"].to(device)
            edge_index = batch["edge_index"].to(device)

            outputs = combined_model(
                **gpu_batch,
                node_feats=node_feats,
                edge_index=edge_index,
            )

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg = running_loss / 50
                    logger.info(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"Loss {avg:.4f} | LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0

        # ── Validate ──────────────────────────────────────────────────────
        combined_model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for batch in val_loader:
                gpu_batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k not in ("node_feats", "edge_index")
                }
                node_feats = batch["node_feats"].to(device)
                edge_index = batch["edge_index"].to(device)

                outputs = combined_model(
                    **gpu_batch,
                    node_feats=node_feats,
                    edge_index=edge_index,
                )
                val_loss += outputs.loss.item()

                logits      = outputs.logits[:, -1, :]
                pred_token  = logits.argmax(dim=-1)
                label_token = batch["labels"].to(device)[:, 0]
                correct += (pred_token == label_token).sum().item()
                total   += label_token.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy     = correct / total if total > 0 else 0.0

        logger.info(
            f"\n{'='*50}\n"
            f"Epoch {epoch+1}/{NUM_EPOCHS} complete\n"
            f"  Val Loss : {avg_val_loss:.4f}\n"
            f"  Accuracy : {accuracy:.2%}\n"
            f"{'='*50}\n"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUT_DIR, "best")
            os.makedirs(save_path, exist_ok=True)
            # Save LoRA adapters
            combined_model.qwen.save_pretrained(os.path.join(save_path, "lora"))
            # Save graph components
            torch.save(combined_model.graph_enc.state_dict(), os.path.join(save_path, "graph_enc.pt"))
            torch.save(combined_model.projector.state_dict(), os.path.join(save_path, "projector.pt"))
            logger.info(f"  ✅ Best model saved to {save_path}")

    # Save final
    final_path = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    combined_model.qwen.save_pretrained(os.path.join(final_path, "lora"))
    torch.save(combined_model.graph_enc.state_dict(), os.path.join(final_path, "graph_enc.pt"))
    torch.save(combined_model.projector.state_dict(), os.path.join(final_path, "projector.pt"))
    logger.info(f"Training done. Saved to {final_path}")


# ─────────────────────────────────────────────
# STEP 6: Inference
# ─────────────────────────────────────────────

def load_for_inference(checkpoint_path: str):
    """Loads base Qwen + LoRA + graph components from a saved checkpoint."""
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    qwen_model = PeftModel.from_pretrained(base_model, os.path.join(checkpoint_path, "lora"))

    graph_enc  = DOMGraphTransformer()
    projector  = GraphProjector()
    graph_enc.load_state_dict(torch.load(os.path.join(checkpoint_path, "graph_enc.pt")))
    projector.load_state_dict(torch.load(os.path.join(checkpoint_path, "projector.pt")))

    combined = QwenWithDOMGraph(qwen_model, graph_enc, projector)
    combined.eval()
    return combined


def predict(
    combined_model,
    processor,
    task:       str,
    candidates: list[str],
    screenshot: Image.Image,
    dom_html:   str = "",
) -> int:
    """Full multimodal + graph inference."""
    device = next(combined_model.parameters()).device

    node_feats, edge_index = dom_to_graph(dom_html)

    candidate_text = "\n".join([f"{i}: {c}" for i, c in enumerate(candidates)])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot.convert("RGB")},
                {
                    "type": "text",
                    "text": (
                        f"Task: {task}\n\n"
                        f"Candidate actions:\n{candidate_text}\n\n"
                        f"Look at the screenshot and the page structure, "
                        f"then select the correct action index. "
                        f"Answer with ONLY the number."
                    ),
                },
            ],
        }
    ]

    text_prompt  = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text_prompt], images=image_inputs, return_tensors="pt").to(device)

    # Build prefix from graph
    with torch.no_grad():
        node_embs = combined_model.graph_enc(node_feats.to(device), edge_index.to(device))
        prefix    = combined_model.projector(node_embs).unsqueeze(0)   # (1, K, D)

        token_embeds  = combined_model.qwen.model.embed_tokens(inputs["input_ids"])
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask   = torch.ones(1, NUM_GRAPH_TOKENS, dtype=torch.long, device=device)
        extended_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)

        output_ids = combined_model.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            max_new_tokens=4,
            do_sample=False,
        )

    generated = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract just the number from the generated text
    match = re.search(r"\d+", generated)
    try:
        return int(match.group()) if match else 0
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse output: '{generated}', defaulting to 0")
        return 0


def evaluate_split(
    combined_model,
    processor,
    dataset_split: str = "test_website",
    preds_out: str = "out_preds.jsonl",
):
    """Run full-dataset inference and emit evaluator-compatible records."""
    combined_model.eval()

    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=dataset_split)
    results = []
    correct = 0
    total = 0

    for row in tqdm(dataset, desc=f"Eval {dataset_split}"):
        task = row.get("confirmed_task", "")
        candidates = row.get("action_reprs") or []
        target = int(row["target_action_index"]) if row.get("target_action_index") is not None else None
        dom_html = row.get("cleaned_html") or row.get("dom_tree") or ""
        screenshot = _screenshot_to_pil(row.get("screenshot"))

        pred_idx = predict(
            combined_model,
            processor,
            task=task,
            candidates=candidates,
            screenshot=screenshot,
            dom_html=dom_html,
        )

        pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None
        pred_action = extract_action_from_text(pred_element) if pred_element else None
        if pred_action is None:
            pred_action = "CLICK"

        gt_element = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
        gt_action = extract_action_from_text(gt_element) if gt_element else None
        gt_value = row.get("gt_value") if row.get("gt_value") is not None else None

        rec = {
            "id": row.get("annotation_id") or row.get("id"),
            "gt_element": gt_element,
            "gt_action": gt_action,
            "gt_value": gt_value,
            "pred_element": pred_element,
            "pred_action": pred_action,
            "pred_value": None,
            "candidates": candidates,
            "task_success": (pred_idx == target) if (pred_idx is not None and target is not None) else None,
        }
        results.append(rec)

        if pred_idx is not None and target is not None and pred_idx == target:
            correct += 1
        total += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_jsonl(preds_out, results)
    accuracy = correct / total if total else 0.0
    logger.info(f"Wrote {len(results)} records to {preds_out}")
    logger.info(f"Accuracy on {dataset_split}: {accuracy:.2%}")
    return accuracy


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Qwen + apply LoRA
    qwen_model, processor = load_model()
    qwen_model            = apply_lora(qwen_model)

    # Build graph components (these are fully trainable)
    graph_enc = DOMGraphTransformer(
        in_dim=NODE_FEAT_DIM,
        hidden_dim=GRAPH_HIDDEN_DIM,
        out_dim=GRAPH_OUT_DIM,
        heads=GAT_HEADS,
        num_layers=GAT_LAYERS,
    )
    projector = GraphProjector(
        graph_dim=GRAPH_OUT_DIM,
        qwen_dim=QWEN_HIDDEN_DIM,
        num_tokens=NUM_GRAPH_TOKENS,
    )

    # Combine everything
    combined_model = QwenWithDOMGraph(qwen_model, graph_enc, projector)

    # Dataloaders
    train_loader, val_loader = build_dataloaders(processor)

    # Train
    train(combined_model, train_loader, val_loader)

    # Smoke test
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
    )
    logger.info(f"Predicted action index: {pred}")