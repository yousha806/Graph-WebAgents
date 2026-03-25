import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv

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
    """Heuristically extract action from candidate string."""
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

def run(dataset_split: str = "test_website", preds_out: str = "out_preds.jsonl", extract_states: bool = True):
    print(f"Loading model {MODEL_NAME}...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,  # VERY IMPORTANT for Qwen3
)
    processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True)

# Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=dataset_split)

    results = []
    correct = 0
    total = 0

    for row in tqdm(dataset):
        task = row.get("confirmed_task", "")
        html = row.get("cleaned_html", "")[:15000]
        candidates = row.get("action_reprs") or []
        target = int(row["target_action_index"]) if row.get("target_action_index") is not None else None
        screenshot = row.get("screenshot")

        candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": f"Task: {task}\nHTML: {html}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number."},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[screenshot],
            padding=True,
            return_tensors="pt",
            process_condition_type="test",
            min_pixels=256 * 28 * 28,
            max_pixels=MAX_PIXELS
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        pred_text = None
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=10)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ] if hasattr(inputs, 'input_ids') else output_ids
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception as e:
            pred_text = str(output_ids[0]) if output_ids is not None else "0"

        # Parse predicted index
        pred_idx = None
        try:
            clean_pred = "".join(filter(str.isdigit, pred_text.strip()))
            pred_idx = int(clean_pred) if clean_pred else None
        except Exception:
            pred_idx = None

        # Map to pred_element and pred_action
        pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None
        pred_action = extract_action_from_text(pred_element) if pred_element else None
        if pred_action is None:
            pred_action = "CLICK"  # default fallback

        # Extract ground truth
        gt_element = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
        gt_action = extract_action_from_text(gt_element) if gt_element else None
        if gt_action is None and target is not None and 0 <= target < len(candidates):
            gt_action = extract_action_from_text(candidates[target])
        gt_value = row.get("gt_value") if row.get("gt_value") is not None else None

        # Build evaluator-compatible record
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

    # Save predictions JSONL
    save_jsonl(preds_out, results)
    print(f"Wrote {len(results)} prediction records to {preds_out}")
    if total:
        print(f"Final Accuracy: {correct / total:.2%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_website", help="HF dataset split name")
    p.add_argument("--preds-out", required=True, help="Path to write predictions JSONL")
    p.add_argument("--no-states", dest="extract_states", action="store_false", help="Disable extraction of hidden-state value vectors")
    args = p.parse_args()
    run(dataset_split=args.split, preds_out=args.preds_out, extract_states=args.extract_states)
