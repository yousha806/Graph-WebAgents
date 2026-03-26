import argparse
import json
import torch
from PIL import Image
from io import BytesIO
from collections.abc import Mapping
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


def _resolve_pixel_values(model_inputs: dict):
    """Find image tensor in processor/model inputs across possible key names."""
    if "pixel_values" in model_inputs and torch.is_tensor(model_inputs["pixel_values"]):
        return model_inputs["pixel_values"]

    # Common alternates in multimodal processors / remote-code wrappers
    for k in ["images", "image", "vision_x", "pixel_values_videos", "pixel_values_images"]:
        if k in model_inputs and torch.is_tensor(model_inputs[k]):
            return model_inputs[k]

    # Last resort: any tensor key containing "pixel" or "image"
    for k, v in model_inputs.items():
        if torch.is_tensor(v) and ("pixel" in k.lower() or "image" in k.lower()):
            return v

    return None


def _to_pil_image(screenshot):
    """Convert dataset screenshot payloads to RGB PIL image."""
    if screenshot is None:
        return None

    if isinstance(screenshot, Image.Image):
        return screenshot.convert("RGB")

    if isinstance(screenshot, str):
        return Image.open(screenshot).convert("RGB")

    if isinstance(screenshot, dict):
        if screenshot.get("path"):
            return Image.open(screenshot["path"]).convert("RGB")
        if screenshot.get("bytes"):
            return Image.open(BytesIO(screenshot["bytes"])) .convert("RGB")

    # Numpy-like arrays from decoded image columns
    try:
        import numpy as np
        if isinstance(screenshot, np.ndarray):
            return Image.fromarray(screenshot).convert("RGB")
    except Exception:
        pass

    return None


def run(dataset_split: str = "test_website", preds_out: str = "out_preds.jsonl", extract_states: bool = True, wrong_out: str = "wrong_preds.jsonl", num_beams: int = 4, max_new_tokens: int = 10, do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50, early_stopping: bool = True, seed: int = None):
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
    wrong_results = []
    correct = 0
    total = 0

    for row in tqdm(dataset):
        task = row.get("confirmed_task", "")
        html = row.get("cleaned_html", "")[:15000]
        candidates = row.get("action_reprs") or []
        target = int(row["target_action_index"]) if row.get("target_action_index") is not None else None
        screenshot = row.get("screenshot")
        
        # Skip if no screenshot
        if screenshot is None:
            continue

        candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])

        # Build a text-only prompt for the tokenizer (avoid passing image objects into the chat template)
        text_content = f"Task: {task}\nHTML: {html}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number."
        messages = [{"role": "user", "content": text_content}]

        # Apply chat template if available; if it fails (e.g. template expects strings), fall back to plain text
        if tokenizer:
            try:
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                print(f"Warning: apply_chat_template failed ({type(e).__name__}), falling back to plain prompt")
                formatted_text = text_content
        else:
            formatted_text = text_content
        
        # Process inputs separately: image processor and tokenizer
        image = _to_pil_image(screenshot)
        if image is None:
            # Preserve row count and make failure explicit in logs/preds.
            pred_text = "0"
            pred_idx = None
            pred_element = None
            pred_action = "CLICK"
            gt_element = candidates[target] if (target is not None and 0 <= target < len(candidates)) else None
            gt_action = extract_action_from_text(gt_element) if gt_element else None
            gt_value = row.get("gt_value") if row.get("gt_value") is not None else None
            rec = {
                "id": row.get("annotation_id") or row.get("id"),
                "model": MODEL_NAME,
                "split": dataset_split,
                "input_task": task,
                "html_snippet": html[:200] if html else None,
                "screenshot": screenshot if isinstance(screenshot, str) else None,
                "candidates": candidates,
                "gt_index": target,
                "gt_element": gt_element,
                "gt_action": gt_action,
                "gt_value": gt_value,
                "pred_text": pred_text,
                "pred_idx": pred_idx,
                "pred_element": pred_element,
                "pred_action": pred_action,
                "pred_value": None,
                "value_states": None,
                "task_success": False,
            }
            results.append(rec)
            if target is not None:
                wrong_results.append(rec)
            total += 1
            continue
        
        # Build image inputs robustly across processor API variants.
        image_inputs = {}
        image_call_variants = [
            lambda: processor(images=image, return_tensors="pt"),
            lambda: processor(image, return_tensors="pt"),
            lambda: processor(images=[image], return_tensors="pt"),
            lambda: processor(images=image, text=None, return_tensors="pt"),
            lambda: processor(images=image, text="", return_tensors="pt"),
        ]
        for fn in image_call_variants:
            try:
                candidate_inputs = fn()
            except Exception:
                continue
            if isinstance(candidate_inputs, Mapping):
                image_inputs = dict(candidate_inputs)
                if _resolve_pixel_values(image_inputs) is not None:
                    break

        # Last resort: use processor.image_processor directly when available.
        if _resolve_pixel_values(image_inputs) is None and hasattr(processor, "image_processor"):
            try:
                direct_image_inputs = processor.image_processor(images=image, return_tensors="pt")
                if isinstance(direct_image_inputs, Mapping):
                    image_inputs = dict(direct_image_inputs)
            except Exception:
                pass
        
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
        
        # Combine: image_inputs should include image tensors, text_inputs has token fields.
        inputs = {**image_inputs, **text_inputs}

        # Ensure a canonical pixel_values key exists if we detected any image tensor key.
        resolved_pixels = _resolve_pixel_values(inputs)
        if resolved_pixels is None:
            resolved_pixels = _resolve_pixel_values(image_inputs)
        if resolved_pixels is not None and "pixel_values" not in inputs:
            inputs["pixel_values"] = resolved_pixels

        inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

        # Generate prediction
        pred_text = None
        try:
            # Optionally set RNG seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)

            with torch.inference_mode():
                pixel_values = _resolve_pixel_values(inputs)
                generate_inputs = {}
                if "input_ids" in inputs:
                    generate_inputs["input_ids"] = inputs["input_ids"]
                if "attention_mask" in inputs:
                    generate_inputs["attention_mask"] = inputs["attention_mask"]
                if pixel_values is not None:
                    generate_inputs["pixel_values"] = pixel_values

                if not generate_inputs:
                    generate_inputs = inputs

                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    early_stopping=early_stopping,
                )
                output_ids = model.generate(**generate_inputs, **gen_kwargs)

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
            except Exception:
                pred_text = str(output_ids[-1]) if len(output_ids) > 0 else "0"

        except (AssertionError, AttributeError, RuntimeError) as e:
            # Last resort: do forward pass + greedy decode manually
            print(f"Generate failed ({type(e).__name__}), using forward pass fallback")
            try:
                with torch.inference_mode():
                    # Get logits from forward pass. Some wrappers use non-standard image keys
                    # and/or require pixel_values as the first positional argument.
                    pixel_values = _resolve_pixel_values(inputs)
                    if pixel_values is None:
                        raise RuntimeError(f"No image tensor found in inputs. Keys: {list(inputs.keys())}")

                    fallback_inputs = {
                        k: v
                        for k, v in inputs.items()
                        if k in ["input_ids", "attention_mask", "position_ids", "image_flags"]
                    }
                    try:
                        outputs = model(pixel_values=pixel_values, **fallback_inputs)
                    except TypeError:
                        # Retry by passing pixel_values positionally
                        outputs = model(pixel_values, **fallback_inputs)

                    logits = outputs.logits if hasattr(outputs, 'logits') else (outputs[0] if isinstance(outputs, (list, tuple)) else None)

                    # Greedy: take argmax of last token
                    if logits is None:
                        raise RuntimeError('No logits available from forward pass')
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

        # Build evaluator-compatible record (standardized schema)
        html_snippet = html[:200] if html else None
        screenshot_ref = screenshot if isinstance(screenshot, str) else None
        rec = {
            "id": row.get("annotation_id") or row.get("id"),
            "model": MODEL_NAME,
            "split": dataset_split,
            "input_task": task,
            "html_snippet": html_snippet,
            "screenshot": screenshot_ref,
            "candidates": candidates,
            "gt_index": target,
            "gt_element": gt_element,
            "gt_action": gt_action,
            "gt_value": gt_value,
            "pred_text": pred_text,
            "pred_idx": pred_idx,
            "pred_element": pred_element,
            "pred_action": pred_action,
            "pred_value": None,
            "value_states": value_states,
            "task_success": bool(pred_idx is not None and target is not None and pred_idx == target),
        }
        results.append(rec)

        # Track incorrect examples separately
        if pred_idx is not None and target is not None and pred_idx != target:
            wrong_results.append(rec)

        if pred_idx is not None and target is not None and pred_idx == target:
            correct += 1
        total += 1

        torch.cuda.empty_cache()

    # Save predictions JSONL
    save_jsonl(preds_out, results)
    print(f"Wrote {len(results)} prediction records to {preds_out}")

    # Save incorrect predictions to separate file if requested
    if wrong_out:
        save_jsonl(wrong_out, wrong_results)
        print(f"Wrote {len(wrong_results)} incorrect prediction records to {wrong_out}")

    if total:
        print(f"Final Accuracy: {correct / total:.2%}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test_website", help="HF dataset split name")
    p.add_argument("--preds-out", required=True, help="Path to write predictions JSONL")
    p.add_argument("--wrong-out", default="wrong_preds.jsonl", help="Path to write incorrect predictions JSONL")
    p.add_argument("--no-states", dest="extract_states", action="store_false", help="Disable extraction of hidden-state value vectors")
    p.add_argument("--num-beams", type=int, default=4, help="Number of beams for generation")
    p.add_argument("--max-new-tokens", type=int, default=10, help="Max new tokens to generate")
    p.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy/beam search")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling")
    p.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    p.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", help="Disable early stopping for beam search")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = p.parse_args()
    run(dataset_split=args.split, preds_out=args.preds_out, extract_states=args.extract_states, wrong_out=args.wrong_out, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, early_stopping=args.early_stopping, seed=args.seed)

