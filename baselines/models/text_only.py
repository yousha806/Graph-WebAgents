import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
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
    if not s or not isinstance(s, str): return None
    for a in ["CLICK", "TYPE", "SCROLL", "NOOP", "HOVER", "SELECT", "SUBMIT"]:
        if a in s.upper().split(): return a
    toks = s.strip().split()
    if toks:
        t0 = toks[0].upper().strip(':,')
        if len(t0) <= 10 and t0.isalpha(): return t0
    return None

def run(dataset_split: str = "test_website", preds_out: str = "out_preds_blind.jsonl"):
    print(f"Loading model {MODEL_NAME} for BLIND baseline...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=dataset_split)

    results = []
    correct, total = 0, 0

    for row in tqdm(dataset):
        task = row.get("confirmed_task", "")
        # Blind baseline: we can afford slightly more HTML context (up to 20k)
        html = row.get("cleaned_html", "")[:20000] 
        candidates = row.get("action_reprs") or []
        target = int(row["target_action_index"]) if row.get("target_action_index") is not None else None

        candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])

        # THE BLIND PROMPT: Notice no "type": "image" block here
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            f"You are a web agent viewing a page's source code only. "
                            f"Identify the correct action to fulfill the task.\n\n"
                            f"Task: {task}\n"
                            f"HTML: {html}\n"
                            f"Actions:\n{candidate_text}\n"
                            f"Select the correct action index. Answer with ONLY the number."
                        )
                    },
                ],
            }
        ]

        # Apply template (no images=True internally when image content is missing)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process Text Only
        inputs = processor(
            text=[text],
            images=None, # Explicitly no images
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=10)
            
            # Extract only the generated part
            input_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_len:]
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            pred_text = "0"

        # Parsing Logic
        clean_pred = "".join(filter(str.isdigit, pred_text.strip()))
        pred_idx = int(clean_pred) if clean_pred else None

        pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None
        pred_action = extract_action_from_text(pred_element) or "CLICK"

        gt_element = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
        gt_action = extract_action_from_text(gt_element)

        rec = {
            "id": row.get("annotation_id") or row.get("id"),
            "gt_element": gt_element,
            "gt_action": gt_action,
            "pred_element": pred_element,
            "pred_action": pred_action,
            "task_success": (pred_idx == target) if (pred_idx is not None and target is not None) else False,
        }
        results.append(rec)

        if pred_idx == target and target is not None:
            correct += 1
        total += 1
        
        torch.cuda.empty_cache()

    save_jsonl(preds_out, results)
    print(f"Final Blind Accuracy: {correct / total:.2%}" if total > 0 else "No records processed.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_website")
    p.add_argument("--preds-out", default="out_preds_blind.jsonl")
    args = p.parse_args()
    run(dataset_split=args.split, preds_out=args.preds_out)