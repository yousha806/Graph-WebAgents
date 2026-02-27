import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import os
from dotenv import load_dotenv


load_dotenv()
MODEL_NAME = "OpenGVLab/InternVL2-8B"
MAX_PIXELS = 1024 * 28 * 28
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN not found in environment. If the model is gated, authentication will fail.")

# Optional 4-bit quantization config (requires bitsandbytes installed)
bnb_available = True
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
except Exception:
    bnb_available = False
    bnb_config = None


def save_jsonl(path, records):
    with open(path, "w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(dataset_split: str = "test_website", preds_out: str = "out_preds.jsonl", extract_states: bool = True):
    print(f"Loading model {MODEL_NAME}...")
    if bnb_available:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_auth_token=HF_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_auth_token=HF_TOKEN,
        )

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, use_auth_token=HF_TOKEN)

    # Load dataset
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=dataset_split)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    correct = 0
    total = 0

    for row in tqdm(dataset):
        task = row.get("confirmed_task")
        html = row.get("cleaned_html", "")[:15000]
        candidates = row.get("action_reprs") or []
        target = int(row["target_action_index"]) if row.get("target_action_index") is not None else None

        candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row.get("screenshot")},
                    {
                        "type": "text",
                        "text": f"Task: {task}\nHTML: {html}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number.",
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[row.get("screenshot")],
            padding=True,
            return_tensors="pt",
            process_condition_type="test",
            min_pixels=256 * 28 * 28,
            max_pixels=MAX_PIXELS,
        ).to(device)

        with torch.inference_mode():
            # generation for predicting index
            output_ids = model.generate(**inputs, max_new_tokens=10)

        # decode generated output
        try:
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        except Exception:
            pred_text = "".join([str(x.item()) for x in output_ids.view(-1)])

        # parse predicted index
        pred_idx = None
        try:
            clean_pred = "".join(filter(str.isdigit, str(pred_text).strip()))
            pred_idx = int(clean_pred) if clean_pred else None
        except Exception:
            pred_idx = None

        # map to pred_element and pred_action heuristically
            pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None

            # try to infer action from the candidate string (common formats include "CLICK ...", "TYPE ...")
            def extract_action_from_text(s: str):
                if not s or not isinstance(s, str):
                    return None
                for a in ["CLICK", "TYPE", "SCROLL", "NOOP", "HOVER", "SELECT", "SUBMIT"]:
                    if a in s.upper().split():
                        return a
                # fallback: look for verbs at start
                toks = s.strip().split()
                if toks:
                    t0 = toks[0].upper().strip(':,')
                    if len(t0) <= 10 and t0.isalpha():
                        return t0
                return None

            pred_action = None
            if pred_element is not None:
                # if candidate carries action info, extract it
                cand_text = candidates[pred_idx]
                pred_action = extract_action_from_text(cand_text)
            # fallback strategies
            if pred_action is None:
                # if the prediction matches the target, use target_action if present
                if pred_idx is not None and target is not None and pred_idx == target:
                    pred_action = row.get("target_action") or row.get("gt_action") or "CLICK"
                else:
                    pred_action = "CLICK"

        # extract value_states by forwarding model with hidden states if requested
        value_states = None
        if extract_states:
            try:
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    last_hidden = outputs.hidden_states[-1]  # [batch, seq, dim]
                    pooled = last_hidden.mean(dim=1).cpu().numpy().tolist()[0]
                    value_states = pooled
            except Exception:
                value_states = None

        # ground-truth mapping
        gt_element = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
        # derive gt_action from dataset fields or from the gt candidate string
        gt_action = row.get("target_action") or row.get("gt_action") or None
        if gt_action is None and gt_element is not None and target is not None and 0 <= target < len(candidates):
            gt_action = extract_action_from_text(candidates[target])
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
            "value_states": value_states,
            "task_success": (pred_idx == target) if (pred_idx is not None and target is not None) else None,
        }
        results.append(rec)

        if pred_idx is not None and target is not None and pred_idx == target:
            correct += 1
        total += 1

        torch.cuda.empty_cache()

    # save preds JSONL
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
