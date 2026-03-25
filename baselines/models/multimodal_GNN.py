import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data

load_dotenv()

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

# Setup Quantization Config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def save_jsonl(path, records):
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

# ----------------- Graph Transformer -----------------
class DOMGraphTransformer(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(node_feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_emb = global_mean_pool(x, batch)
        return x, graph_emb

# ----------------- DOM to Graph -----------------
def html_to_dom_graph(html, max_nodes=512):
    soup = BeautifulSoup(html, "html.parser")
    nodes, edges, features = [], [], []

    def traverse(node, parent_idx=None):
        if len(nodes) >= max_nodes:
            return
        idx = len(nodes)
        nodes.append(node)
        # node feature: simple tag + text length + number of attributes
        feat = torch.tensor([
            hash(node.name) % 1000 / 1000.0 if node.name else 0.0,
            len(node.get_text(strip=True)) / 100.0,
            len(node.attrs) / 10.0
        ], dtype=torch.float32)
        features.append(feat)
        if parent_idx is not None:
            edges.append([parent_idx, idx])
        for child in node.children:
            if hasattr(child, 'name'):
                traverse(child, idx)

    traverse(soup)
    if not nodes:
        return None  # fallback
    x = torch.stack(features)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data

# ----------------- Main Run -----------------
def run(dataset_split: str = "test_website", preds_out: str = "out_preds.jsonl"):
    print(f"Loading model {MODEL_NAME}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    # Graph Transformer (small hidden dim for efficiency)
    gnn = DOMGraphTransformer(node_feat_dim=3, hidden_dim=32).to(device)
    gnn.eval()

    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=dataset_split)

    results, correct, total = [], 0, 0

    for row in tqdm(dataset):
        task = row.get("confirmed_task", "")
        html = row.get("cleaned_html", "")[:15000]
        candidates = row.get("action_reprs") or []
        target = int(row["target_action_index"]) if row.get("target_action_index") else None
        screenshot = row.get("screenshot")

        # ---------------- Graph Embedding ----------------
        graph_data = html_to_dom_graph(html)
        if graph_data is not None:
            graph_data = graph_data.to(device)
            with torch.no_grad():
                node_embs, graph_emb = gnn(graph_data.x, graph_data.edge_index, batch=None)
            # Convert graph embedding to a string representation (for Qwen text input)
            graph_emb_text = "GraphEmbedding: " + ", ".join([f"{v.item():.4f}" for v in graph_emb[:64]])
        else:
            graph_emb_text = ""

        candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": f"Task: {task}\nDOM Graph Info: {graph_emb_text}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number."},
                ],
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input],
            images=[screenshot],
            padding=True,
            return_tensors="pt",
            process_condition_type="test",
            min_pixels=256 * 28 * 28,
            max_pixels=MAX_PIXELS
        ).to(device)

        pred_text = None
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=10)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ] if hasattr(inputs, 'input_ids') else output_ids
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception:
            pred_text = "0"

        # Parse predicted index
        pred_idx = None
        try:
            clean_pred = "".join(filter(str.isdigit, pred_text.strip()))
            pred_idx = int(clean_pred) if clean_pred else None
        except Exception:
            pred_idx = None

        # Map to pred_element and pred_action
        pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None
        pred_action = extract_action_from_text(pred_element) if pred_element else "CLICK"

        gt_element = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
        gt_action = extract_action_from_text(gt_element) if gt_element else None
        gt_value = row.get("gt_value")

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
        torch.cuda.empty_cache()

    save_jsonl(preds_out, results)
    print(f"Wrote {len(results)} records to {preds_out}")
    if total:
        print(f"Final Accuracy: {correct/total:.2%}")