import argparse
import json
import re
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
            return Image.open(BytesIO(screenshot["bytes"])).convert("RGB")

    # Numpy-like arrays from decoded image columns
    try:
        import numpy as np
        if isinstance(screenshot, np.ndarray):
            return Image.fromarray(screenshot).convert("RGB")
    except Exception:
        pass

    return None


def _build_manual_pixel_values(image, device):
    """Build a basic pixel tensor fallback for InternVL-style APIs.

    This is intentionally simple and robust: single 448x448 RGB crop.
    """
    if image is None:
        return None

    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        px = transform(image).unsqueeze(0)
    except Exception:
        # Fallback without torchvision
        import numpy as np
        arr = np.array(image.resize((448, 448))).astype("float32") / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = arr.transpose(2, 0, 1)
        px = torch.from_numpy(arr).unsqueeze(0)

    return px.to(device)


def _infer_vision_dtype(model):
    """Infer the expected dtype for image tensors from vision-related modules."""
    for attr in ["vision_model", "vision_tower", "vit"]:
        module = getattr(model, attr, None)
        if module is not None:
            try:
                return next(module.parameters()).dtype
            except StopIteration:
                pass
            except Exception:
                pass

    # Fallback to any model parameter dtype
    try:
        return next(model.parameters()).dtype
    except Exception:
        return torch.float32


def _build_default_image_flags(pixel_values):
    """Build minimal image_flags expected by InternVL forward paths."""
    if pixel_values is None or not torch.is_tensor(pixel_values):
        return None
    bsz = int(pixel_values.shape[0]) if pixel_values.ndim > 0 else 1
    return torch.ones((bsz, 1), dtype=torch.long, device=pixel_values.device)


def _infer_vocab_size(model):
    """Infer language embedding vocabulary size."""
    try:
        emb = model.language_model.get_input_embeddings()
        if hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)
    except Exception:
        pass
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    return None


def _resolve_safe_img_context_token_id(model, tokenizer):
    """Resolve a valid image-context token id that is guaranteed in embedding range."""
    vocab_size = _infer_vocab_size(model)

    candidates = []
    for src in [model, getattr(model, "config", None), tokenizer]:
        if src is None:
            continue
        for attr in ["img_context_token_id", "image_token_id"]:
            v = getattr(src, attr, None)
            if isinstance(v, int):
                candidates.append(v)

    # Try token-name lookups from tokenizer if available.
    if tokenizer is not None:
        for tok in ["<IMG_CONTEXT>", "<image>", "<img>"]:
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    candidates.append(tid)
            except Exception:
                pass
        for attr in ["unk_token_id", "eos_token_id", "bos_token_id", "pad_token_id"]:
            tid = getattr(tokenizer, attr, None)
            if isinstance(tid, int) and tid >= 0:
                candidates.append(tid)

    # Last-resort candidate.
    candidates.append(0)

    def _valid(tid):
        if not isinstance(tid, int) or tid < 0:
            return False
        if vocab_size is None:
            return True
        return tid < vocab_size

    for tid in candidates:
        if _valid(tid):
            return int(tid)

    # Absolute fallback: clamp into range when vocab is known.
    if vocab_size is not None and vocab_size > 0:
        return int(vocab_size - 1)
    return 0


def _sanitize_input_ids(inputs, vocab_size):
    """Ensure input_ids are long and within embedding range."""
    if vocab_size is None or "input_ids" not in inputs:
        return inputs
    ids = inputs["input_ids"]
    if not torch.is_tensor(ids):
        return inputs
    ids = ids.to(dtype=torch.long)
    ids = ids.clamp(min=0, max=vocab_size - 1)
    inputs["input_ids"] = ids
    return inputs


def _infer_num_image_tokens(model):
    """Infer how many image context tokens InternVL expects per image."""
    for attr in ["num_image_token", "image_token_len", "vision_token_len"]:
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ["num_image_token", "image_token_len", "vision_token_len"]:
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    # InternVL2-8B commonly uses 256 image tokens.
    return 256


def _ensure_image_context_tokens(inputs, model):
    """Ensure input_ids include image context placeholders when pixel_values are present."""
    if "pixel_values" not in inputs or "input_ids" not in inputs:
        return inputs

    img_context_id = getattr(model, "img_context_token_id", None)
    if img_context_id is None:
        return inputs

    input_ids = inputs["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        return inputs

    # If any context tokens already exist, keep as-is.
    if (input_ids == img_context_id).any().item():
        return inputs

    num_img_tokens = _infer_num_image_tokens(model)
    bsz = input_ids.shape[0]
    device = input_ids.device
    dtype = input_ids.dtype

    img_tokens = torch.full((bsz, num_img_tokens), int(img_context_id), dtype=dtype, device=device)
    inputs["input_ids"] = torch.cat([img_tokens, input_ids], dim=1)

    if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
        am = inputs["attention_mask"]
        if am.ndim == 2 and am.shape[0] == bsz:
            am_img = torch.ones((bsz, num_img_tokens), dtype=am.dtype, device=am.device)
            inputs["attention_mask"] = torch.cat([am_img, am], dim=1)

    if "position_ids" in inputs and torch.is_tensor(inputs["position_ids"]):
        pos = inputs["position_ids"]
        if pos.ndim == 2 and pos.shape[0] == bsz:
            inputs["position_ids"] = torch.arange(
                inputs["input_ids"].shape[1], device=pos.device, dtype=pos.dtype
            ).unsqueeze(0).repeat(bsz, 1)

    return inputs


def _predict_with_chat_api(model, tokenizer, image, prompt_text, gen_kwargs, device, manual_pixel_values=None):
    """Try InternVL-style chat APIs across common signatures."""
    if not hasattr(model, "chat"):
        return None

    # Keep generation kwargs conservative for chat APIs.
    chat_generation_config = {
        "max_new_tokens": gen_kwargs.get("max_new_tokens", 10),
        "do_sample": gen_kwargs.get("do_sample", False),
        "temperature": gen_kwargs.get("temperature", 1.0),
        "top_p": gen_kwargs.get("top_p", 1.0),
        "top_k": gen_kwargs.get("top_k", 50),
        "num_beams": gen_kwargs.get("num_beams", 1),
    }

    # Convert PIL image to tensor for chat calls that expect pixel_values directly.
    pixel_values = manual_pixel_values if manual_pixel_values is not None else _build_manual_pixel_values(image, device)

    attempts = [
        lambda: model.chat(tokenizer, pixel_values, prompt_text, generation_config=chat_generation_config),
        lambda: model.chat(tokenizer, pixel_values, prompt_text),
        lambda: model.chat(tokenizer, image, prompt_text, generation_config=chat_generation_config),
        lambda: model.chat(tokenizer, image, prompt_text),
        lambda: model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=prompt_text, generation_config=chat_generation_config),
        lambda: model.chat(tokenizer=tokenizer, pixel_values=pixel_values, question=prompt_text),
        lambda: model.chat(tokenizer=tokenizer, image=image, query=prompt_text, generation_config=chat_generation_config),
        lambda: model.chat(tokenizer=tokenizer, image=image, query=prompt_text),
        lambda: model.chat(tokenizer=tokenizer, query=prompt_text, image=image),
    ]

    for attempt in attempts:
        try:
            out = attempt()
            if isinstance(out, tuple):
                # Some APIs return (response, history)
                out = out[0]
            if out is not None:
                out = str(out).strip()
                if out:
                    return out
        except Exception:
            continue
    return None


def _extract_pred_idx(pred_text, num_candidates):
    """Parse a predicted candidate index from free-form model output."""
    if pred_text is None:
        return None
    s = str(pred_text).strip()
    if not s:
        return None

    # Prefer explicit standalone integers first.
    for m in re.finditer(r"\b\d+\b", s):
        try:
            v = int(m.group(0))
        except Exception:
            continue
        if 0 <= v < num_candidates:
            return v

    # Last resort: concatenate digits.
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        try:
            v = int(digits)
            if 0 <= v < num_candidates:
                return v
        except Exception:
            pass
    return None


def run(dataset_split: str = "test_website", preds_out: str = "out_preds.jsonl", extract_states: bool = True, wrong_out: str = "wrong_preds.jsonl", num_beams: int = 4, max_new_tokens: int = 10, do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 50, early_stopping: bool = True, seed: int = None, strict_vision: bool = True, inference_mode: str = "chat"):
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
    vision_dtype = _infer_vision_dtype(model)
    
    # Load processor and tokenizer separately for InternVL2
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    
    # For chat template, need to load the tokenizer separately
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Warning: tokenizer loading failed ({e}), using minimal fallback")
        tokenizer = None
    
    # Set a safe InternVL2 img_context_token_id guaranteed to be valid for embeddings.
    model.img_context_token_id = _resolve_safe_img_context_token_id(model, tokenizer)

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        manual_pixel_values = _build_manual_pixel_values(image, device) if image is not None else None
        if image is None:
            if strict_vision:
                raise RuntimeError(f"Unable to decode screenshot for id={row.get('annotation_id') or row.get('id')}; aborting in strict vision mode")
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
        
        # Primary path: build multimodal inputs in one processor call.
        # InternVL-style processors often require an <image> marker in the prompt.
        inputs = {}
        mm_text = f"<image>\n{formatted_text}"
        mm_call_variants = [
            lambda: processor(text=[mm_text], images=[image], return_tensors="pt", padding=True),
            lambda: processor(text=mm_text, images=image, return_tensors="pt"),
            lambda: processor(text=[mm_text], images=image, return_tensors="pt"),
            lambda: processor(text=mm_text, images=[image], return_tensors="pt"),
        ]
        for fn in mm_call_variants:
            try:
                candidate_inputs = fn()
            except Exception:
                continue
            if isinstance(candidate_inputs, Mapping):
                inputs = dict(candidate_inputs)
                if _resolve_pixel_values(inputs) is not None:
                    break

        # Fallback path: build image + text separately when unified call doesn't produce image tensors.
        image_inputs = {}
        if _resolve_pixel_values(inputs) is None:
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

            if _resolve_pixel_values(image_inputs) is None and hasattr(processor, "image_processor"):
                try:
                    direct_image_inputs = processor.image_processor(images=image, return_tensors="pt")
                    if isinstance(direct_image_inputs, Mapping):
                        image_inputs = dict(direct_image_inputs)
                except Exception:
                    pass

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

            inputs = {**image_inputs, **text_inputs}

        # Ensure a canonical pixel_values key exists if we detected any image tensor key.
        resolved_pixels = _resolve_pixel_values(inputs)
        if resolved_pixels is None:
            resolved_pixels = _resolve_pixel_values(image_inputs)
        if resolved_pixels is None and manual_pixel_values is not None:
            resolved_pixels = manual_pixel_values
        if resolved_pixels is not None:
            resolved_pixels = resolved_pixels.to(device=device, dtype=vision_dtype)
            inputs["pixel_values"] = resolved_pixels
            if "image_flags" not in inputs:
                inputs["image_flags"] = _build_default_image_flags(resolved_pixels)

        # InternVL requires image context tokens in input_ids to place vit_embeds.
        # If tokenizer/template omitted them, inject placeholders explicitly.
        inputs = _ensure_image_context_tokens(inputs, model)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = _sanitize_input_ids(inputs, _infer_vocab_size(model))

        # Generate prediction
        pred_text = None
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            early_stopping=early_stopping,
        )

        # Stable default path: chat API. Keep generate path optional for debugging.
        if inference_mode == "chat":
            chat_pred = _predict_with_chat_api(
                model=model,
                tokenizer=tokenizer,
                image=image,
                prompt_text=text_content,
                gen_kwargs=gen_kwargs,
                device=device,
                manual_pixel_values=manual_pixel_values,
            )
            if chat_pred is not None:
                pred_text = chat_pred
            elif strict_vision:
                raise RuntimeError(
                    f"Chat inference failed in strict vision mode for id={row.get('annotation_id') or row.get('id')}"
                )
            else:
                pred_text = "0"

        if inference_mode == "generate":
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
                        if "image_flags" in inputs:
                            generate_inputs["image_flags"] = inputs["image_flags"]
                        else:
                            generate_inputs["image_flags"] = _build_default_image_flags(pixel_values)

                    # If we have no image tensor, use chat API directly.
                    if pixel_values is None:
                        chat_pred = _predict_with_chat_api(
                            model=model,
                            tokenizer=tokenizer,
                            image=image,
                            prompt_text=text_content,
                            gen_kwargs=gen_kwargs,
                            device=device,
                            manual_pixel_values=manual_pixel_values,
                        )
                        if chat_pred is not None:
                            pred_text = chat_pred
                            output_ids = None
                        else:
                            if strict_vision:
                                raise RuntimeError(
                                    f"No image tensor available for generate and chat fallback failed for id={row.get('annotation_id') or row.get('id')}. "
                                    f"Input keys: {list(inputs.keys())}"
                                )
                            output_ids = model.generate(**generate_inputs, **gen_kwargs)
                    else:
                        if not generate_inputs:
                            generate_inputs = inputs
                        output_ids = model.generate(**generate_inputs, **gen_kwargs)

                # Decode output
                try:
                    if pred_text is not None:
                        pass
                    elif 'input_ids' in inputs and output_ids is not None:
                        generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
                        ]
                    else:
                        generated_ids = output_ids

                    if pred_text is not None:
                        pass
                    elif tokenizer:
                        pred_text = tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    else:
                        pred_text = str(output_ids[0])
                except Exception:
                    if pred_text is None:
                        pred_text = str(output_ids[-1]) if output_ids is not None and len(output_ids) > 0 else "0"

            except (AssertionError, AttributeError, RuntimeError):
                # Last resort: do forward pass + greedy decode manually
                print("Generate failed, using forward pass fallback")
                try:
                    with torch.inference_mode():
                        # Try chat API first in fallback path.
                        chat_pred = _predict_with_chat_api(
                            model=model,
                            tokenizer=tokenizer,
                            image=image,
                            prompt_text=text_content,
                            gen_kwargs=gen_kwargs,
                            device=device,
                            manual_pixel_values=manual_pixel_values,
                        )
                        if chat_pred is not None:
                            pred_text = chat_pred
                            raise StopIteration("chat-success")

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
                        if "image_flags" not in fallback_inputs:
                            fallback_inputs["image_flags"] = _build_default_image_flags(pixel_values)
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
                except StopIteration:
                    pass
                except Exception as e2:
                    if strict_vision:
                        raise RuntimeError(
                            f"Forward pass fallback failed in strict vision mode for id={row.get('annotation_id') or row.get('id')}: {e2}"
                        ) from e2
                    print(f"Forward pass fallback also failed: {e2}")
                    pred_text = "0"

        # Parse predicted index
        pred_idx = _extract_pred_idx(pred_text, len(candidates))

        # Map to pred_element and pred_action
        pred_element = candidates[pred_idx] if (pred_idx is not None and 0 <= pred_idx < len(candidates)) else None
        pred_action = extract_action_from_text(pred_element) if pred_element else None
        if pred_action is None:
            pred_action = "CLICK"  # default fallback

        # Extract value_states from hidden states if requested
        value_states = None
        if extract_states:
            try:
                if "pixel_values" not in inputs:
                    raise RuntimeError("Skipping value_states: no pixel_values in inputs")
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

        # Track correctness/incorrectness explicitly; parse failures are incorrect.
        is_correct = bool(pred_idx is not None and target is not None and pred_idx == target)
        is_incorrect = bool(target is not None and not is_correct)

        if is_incorrect:
            wrong_results.append(rec)

        if is_correct:
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
    p.add_argument("--allow-text-only-fallback", dest="strict_vision", action="store_false", help="Allow text-only fallback when image tensors are missing (not recommended)")
    p.add_argument("--inference-mode", choices=["chat", "generate"], default="chat", help="Inference path: chat (stable) or generate (experimental)")
    args = p.parse_args()
    run(dataset_split=args.split, preds_out=args.preds_out, extract_states=args.extract_states, wrong_out=args.wrong_out, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, early_stopping=args.early_stopping, seed=args.seed, strict_vision=args.strict_vision, inference_mode=args.inference_mode)

