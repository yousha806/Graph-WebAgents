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

# Setup Quantization Config for 4-bit (matching Qwen pattern)
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
    
    # ---- InternVL2 workarounds ----
    # 1) Patch torch.linspace to prevent meta tensor .item() crash
    _original_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return _original_linspace(*args, **kwargs)
    torch.linspace = _safe_linspace
    
    # 2) InternVL2 model lacks all_tied_weights_keys that newer transformers expects.
    #    Add it as a class property that returns an empty dict.
    from transformers.models.auto import auto_factory
    import torch.nn as nn
    
    # Patch nn.Module.__getattr__ globally to handle missing all_tied_weights_keys gracefully
    _orig_getattr = nn.Module.__getattr__
    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}  # Return empty dict, not set
        return _orig_getattr(self, name)
    nn.Module.__getattr__ = _patched_getattr
    
    try:
        # 3) Load with 4-bit quantization (reduces to ~4-6GB)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map=None,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
    finally:
        torch.linspace = _original_linspace
        nn.Module.__getattr__ = _orig_getattr
    # ---- end workarounds ----

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Load processor and tokenizer separately for InternVL2
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    
    # For chat template, need to load the tokenizer separately
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Warning: tokenizer loading failed ({e}), using minimal fallback")
        tokenizer = None
    
    # Set InternVL2's required img_context_token_id
    if not hasattr(model, 'img_context_token_id') or model.img_context_token_id is None:
        # Try to get from config first
        if hasattr(model.config, 'img_context_token_id'):
            model.img_context_token_id = model.config.img_context_token_id
        elif hasattr(model, 'config') and hasattr(model.config, 'image_token_id'):
            model.img_context_token_id = model.config.image_token_id
        else:
            # Default fallback: use token ID 0 or find from tokenizer
            if tokenizer and hasattr(tokenizer, 'img_context_token_id'):
                model.img_context_token_id = tokenizer.img_context_token_id
            else:
                # Last resort: use a common image token ID (often in range 100-1000 for special tokens)
                model.img_context_token_id = 151857  # Common for InternVL models

    # Load dataset from Hugging Face
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

        # Build messages for InternVL2
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {
                        "type": "text",
                        "text": f"Task: {task}\nHTML: {html}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number.",
                    },
                ],
            }
        ]

        # Apply chat template
        if tokenizer:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # fallback: just format text manually
            formatted_text = f"Task: {task}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number."
        
        # Process inputs separately: image processor and tokenizer
        try:
            from PIL import Image as PILImage
        except ImportError:
            PILImage = Image
            
        if isinstance(screenshot, str):
            # File path
            image = PILImage.open(screenshot).convert("RGB")
        elif isinstance(screenshot, Image.Image):
            image = screenshot.convert("RGB")
        else:
            # Assume it's already a PIL Image
            image = screenshot
        
        # Process image only with processor
        image_inputs = processor(images=image, return_tensors="pt")
        
        # Tokenize text only with tokenizer
        if tokenizer:
            text_inputs = tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
        else:
            text_inputs = {}
        
        # Combine: image_inputs has pixel_values, text_inputs has input_ids, attention_mask
        inputs = {**image_inputs, **text_inputs}
        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

        # Generate prediction
        pred_text = None
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=10)
            
            # Decode output
            try:
                if 'input_ids' in inputs:
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
                    ]
                else:
                    generated_ids = output_ids
                    
                if tokenizer:
                    pred_text = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                else:
                    pred_text = str(output_ids[0])
            except Exception as e:
                pred_text = str(output_ids[-1]) if len(output_ids) > 0 else "0"
                
        except (AssertionError, AttributeError, RuntimeError) as e:
            # Last resort: do forward pass + greedy decode manually
            print(f"Generate failed ({type(e).__name__}), using forward pass fallback")
            try:
                with torch.inference_mode():
                    # Get logits from forward pass
                    outputs = model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'pixel_values']})
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    
                    # Greedy: take argmax of last token
                    next_token_id = logits[0, -1, :].argmax().item()
                    if tokenizer:
                        pred_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                    else:
                        pred_text = str(next_token_id)
            except Exception as e2:
                print(f"Forward pass fallback also failed: {e2}")
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
        pred_action = extract_action_from_text(pred_element) if pred_element else None
        if pred_action is None:
            pred_action = "CLICK"  # default fallback

        # Extract value_states from hidden states if requested
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
            "value_states": value_states,
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

