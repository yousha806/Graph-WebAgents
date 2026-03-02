"""Run InternVL2 AXTree ablations for Mind2Web next-action prediction.

This script implements exactly two ablations on the Intern model:
1) image + all inputs + AXTree
2) image + all inputs + AXTree + chain-of-thought

It evaluates by default on all three test splits:
- test_task
- test_website
- test_domain

Expected input data:
- Precomputed AXTree splits under data/mind2web_axtree/<split>
  (produced by precompute_axtree.py)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# InternVL2 image preprocessing constants (ImageNet normalisation, 448px tiles)
_INTERN_INPUT_SIZE = 448
_INTERN_MEAN = (0.485, 0.456, 0.406)
_INTERN_STD  = (0.229, 0.224, 0.225)

DEFAULT_SPLITS = ["test_website"]

BASELINES = [
    {
        "name": "intern_image_allinputs_axtree",
        "cot": False,
    },
    {
        "name": "intern_image_allinputs_axtree_cot",
        "cot": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InternVL2 AXTree ablations for Mind2Web")
    parser.add_argument(
        "--model_name",
        default="OpenGVLab/InternVL2-8B",
        help="Hugging Face model id for InternVL2 checkpoint",
    )
    parser.add_argument(
        "--axtree_dir",
        default="data/mind2web_axtree",
        help="Directory containing AXTree-precomputed splits saved with datasets.save_to_disk",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=["test_task", "test_website", "test_domain"],
        help="Which test splits to run",
    )
    parser.add_argument(
        "--output_dir",
        default="inference_outputs/intern_axtree_ablations",
        help="Where to write JSONL predictions",
    )
    parser.add_argument("--max_axtree_chars", type=int, default=6000)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per split")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSONL outputs by skipping already written rows",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Optional bitsandbytes quantization mode",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def build_quantization_config(mode: str, dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
    if mode == "none":
        return None
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _patch_internvl2_meta_device_issue() -> None:
    """Patch InternVL2's cached model code to avoid .item() on meta tensors.

    InternVL2's InternVisionEncoder.__init__ calls:
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, ...)]

    In transformers >= 4.45, model __init__ runs inside accelerate's meta-device
    context even when low_cpu_mem_usage=False, causing:
        RuntimeError: Tensor.item() cannot be called on meta tensors

    This patch replaces that line with pure-Python arithmetic — equivalent in
    value, but creates no tensors, so it is safe in any device context.
    """
    BUGGY = (
        "dpr = [x.item() for x in torch.linspace("
        "0, config.drop_path_rate, config.num_hidden_layers)]"
    )
    FIXED = (
        "n = config.num_hidden_layers\n"
        "        dpr = ([0.0] + [config.drop_path_rate * i / (n - 1) for i in range(1, n)]"
        " if n > 1 else [0.0])"
    )
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    search_dir = hf_home / "modules" / "transformers_modules"
    if not search_dir.is_dir():
        return
    for fpath in search_dir.rglob("modeling_intern_vit.py"):
        content = fpath.read_text(encoding="utf-8")
        if BUGGY in content:
            fpath.write_text(content.replace(BUGGY, FIXED), encoding="utf-8")
            print(f"[patch] Applied meta-device fix to: {fpath}")


def load_intern_model(model_name: str, dtype: torch.dtype, quantization: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is required for InternVL2-8B inference but torch.cuda.is_available() returned False.\n"
            f"  torch version    : {torch.__version__}\n"
            f"  torch CUDA build : {torch.version.cuda}\n"
            f"  Likely cause     : driver version too old for this CUDA build, or kernel module\n"
            f"                     not loaded (reboot required after driver install).\n"
            f"  cu121 needs driver >= 520; cu124 needs >= 550; cu128 needs >= 570."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    quantization_config = build_quantization_config(quantization, dtype)

    # Flash Attention 2 gives ~2-4x speedup for long sequences by avoiding
    # materializing the O(N^2) attention matrix in HBM. InternVL2 can produce
    # up to 6144 image tokens (6 tiles × 1024), making FA2 highly beneficial.
    # Requires Ampere+ GPU (compute ≥ 8.0: A10G, A100, H100).
    # Automatically falls back to standard attention on older GPUs (T4=7.5, V100=7.0)
    # or if flash-attn is not installed.
    gpu_cc = torch.cuda.get_device_capability(0)
    try:
        import flash_attn  # noqa: F401
        use_flash_attn = gpu_cc >= (8, 0)
    except ImportError:
        use_flash_attn = False

    print(f"Loading Intern model: {model_name} (quantization={quantization}, dtype={dtype})")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  "
          f"compute={gpu_cc[0]}.{gpu_cc[1]}  |  "
          f"VRAM free: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB  |  "
          f"flash_attn={use_flash_attn}")

    # Patch InternVL2's cached model code before loading so that the
    # torch.linspace(...).item() call in InternVisionEncoder.__init__ does not
    # crash when transformers runs __init__ inside a meta-device context.
    _patch_internvl2_meta_device_issue()

    if quantization_config is not None:
        # Quantized path: bitsandbytes requires device_map.
        # Use {"": 0} (dict form, not "auto") — "auto" triggers accelerate's
        # meta-device dispatch which breaks InternVL2's __init__ (.item() on linspace).
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "device_map": {"": 0},
            "quantization_config": quantization_config,
            "use_flash_attn": use_flash_attn,
        }
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    else:
        # Non-quantized path: load on CPU then move to GPU.
        # torch_dtype is a recognised from_pretrained kwarg (handled before cls.__init__
        # is called), so it is never forwarded to InternVLChatModel.__init__().
        # low_cpu_mem_usage=False disables accelerate's meta-device init; the
        # _patch_internvl2_meta_device_issue() call above fixes the root cause
        # in the model source so this also works if meta init occurs anyway.
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            use_flash_attn=use_flash_attn,
        )
        model = model.to("cuda")

    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")

    if not hasattr(model, "chat"):
        raise RuntimeError(
            f"Model '{model_name}' does not expose chat(). "
            "Use an InternVL2 checkpoint with trust_remote_code chat API."
        )
    return model, tokenizer


def load_split_dataset(axtree_dir: str, split: str):
    split_path = Path(axtree_dir) / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"AXTree split not found: {split_path}. "
            "Run precompute_axtree.py for this split first."
        )
    return load_from_disk(str(split_path))


def to_pil_image(image_field: Any) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")

    if isinstance(image_field, str):
        return Image.open(image_field).convert("RGB")

    if isinstance(image_field, dict):
        path = image_field.get("path")
        if path:
            return Image.open(path).convert("RGB")
        if image_field.get("bytes") is not None:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")

    raise ValueError("Unsupported screenshot format from dataset row")


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text[:max_chars]


def candidate_lines(action_reprs: List[str]) -> str:
    return "\n".join(f"{idx}: {candidate}" for idx, candidate in enumerate(action_reprs))


def build_prompt(row: Dict[str, Any], use_cot: bool, max_axtree_chars: int) -> str:
    task = row.get("confirmed_task", "")
    axtree = truncate(row.get("axtree_text", ""), max_axtree_chars)
    actions = row.get("action_reprs", [])

    cot_instructions = (
        "Think through the page state and action options step by step. "
        "Then output your final answer in this exact format: FINAL_ANSWER: <index>."
        if use_cot
        else "Output only the selected action index as a single integer."
    )

    return (
        "You are predicting the next web action for a browser agent.\n"
        "Given task, AXTree, screenshot, and candidate actions, choose the correct action index.\n\n"
        f"Task:\n{task}\n\n"
        f"AXTree (truncated):\n{axtree}\n\n"
        f"Candidate actions:\n{candidate_lines(actions)}\n\n"
        f"{cot_instructions}"
    )


def parse_predicted_index(raw_output: str, num_candidates: int) -> Optional[int]:
    if not raw_output:
        return None

    tagged_match = re.search(r"FINAL_ANSWER\s*:\s*(-?\d+)", raw_output, flags=re.IGNORECASE)
    if tagged_match:
        value = int(tagged_match.group(1))
        return value if 0 <= value < num_candidates else None

    all_ints = re.findall(r"-?\d+", raw_output)
    if not all_ints:
        return None

    value = int(all_ints[-1])
    return value if 0 <= value < num_candidates else None


def extract_operation(action_repr: str) -> Optional[str]:
    if not action_repr:
        return None
    match = re.match(r"\s*([A-Za-z_]+)", action_repr)
    if not match:
        return None
    return match.group(1).upper()


def normalize_gt_operation(operation_field: Any) -> Optional[str]:
    if operation_field is None:
        return None

    if isinstance(operation_field, str):
        return operation_field.strip().upper() if operation_field.strip() else None

    if isinstance(operation_field, dict):
        for key in ("op", "operation", "action", "type", "name"):
            value = operation_field.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()

    return None


def pil_to_pixel_values(image: Image.Image, dtype: torch.dtype) -> torch.Tensor:
    """Preprocess a PIL image into the InternVL2 pixel_values tensor.

    InternVL2's chat() expects a CUDA tensor of shape (1, 3, 448, 448) with
    ImageNet normalisation, not a raw PIL image.
    """
    transform = transforms.Compose([
        transforms.Resize(
            (_INTERN_INPUT_SIZE, _INTERN_INPUT_SIZE),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=_INTERN_MEAN, std=_INTERN_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(dtype=dtype, device="cuda")


def run_intern_chat(
    model: Any,
    tokenizer: Any,
    pixel_values: torch.Tensor,
    prompt: str,
    generation_config: Dict[str, Any],
) -> str:
    """Call InternVL2 chat() with a preprocessed pixel_values tensor.

    InternVL2's chat signature:
        chat(tokenizer, pixel_values, question, generation_config) -> str

    If pixel_values is None the model treats it as text-only.
    The model prepends '<image>\\n' to the question automatically when
    pixel_values is provided and the token is absent from the prompt.
    """
    output = model.chat(tokenizer, pixel_values, prompt, generation_config)
    if isinstance(output, tuple):
        return str(output[0])
    return str(output)


def make_prediction_record(
    row: Dict[str, Any],
    split: str,
    baseline_name: str,
    pred_idx: Optional[int],
    raw_output: str,
) -> Dict[str, Any]:
    candidates = row.get("action_reprs", [])
    gt_idx = int(row.get("target_action_index", -1))
    pred_action_repr = candidates[pred_idx] if pred_idx is not None and 0 <= pred_idx < len(candidates) else None

    return {
        "split": split,
        "baseline": baseline_name,
        "annotation_id": row.get("annotation_id"),
        "action_uid": row.get("action_uid"),
        "gt_action_index": gt_idx,
        "pred_action_index": pred_idx,
        "gt_action": normalize_gt_operation(row.get("operation")),
        "pred_action": extract_operation(pred_action_repr or ""),
        "gt_action_repr": row.get("target_action_reprs"),
        "pred_action_repr": pred_action_repr,
        "raw_output": raw_output,
        "candidates": list(range(len(candidates))),
    }


def run_single_baseline(
    model: Any,
    tokenizer: Any,
    dataset,
    split: str,
    baseline: Dict[str, Any],
    args: argparse.Namespace,
    model_dtype: torch.dtype,
) -> int:
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
    }

    total_examples = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_dir = Path(args.output_dir)
    out_file = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if args.resume and out_file.exists():
        with out_file.open("r", encoding="utf8") as handle:
            start_idx = sum(1 for _ in handle)
        if start_idx >= total_examples:
            print(f"Skipping {baseline['name']} on {split}: already complete ({start_idx}/{total_examples})")
            return 0

    iterator = range(start_idx, total_examples)
    written = 0

    with out_file.open("a", encoding="utf8") as handle:
        for idx in tqdm(iterator, desc=f"{baseline['name']} | {split}"):
            row = dataset[idx]
            image = to_pil_image(row["screenshot"])
            pixel_values = pil_to_pixel_values(image, model_dtype)
            prompt = build_prompt(
                row=row,
                use_cot=baseline["cot"],
                max_axtree_chars=args.max_axtree_chars,
            )

            with torch.inference_mode():
                raw_output = run_intern_chat(model, tokenizer, pixel_values, prompt, generation_config)

            pred_idx = parse_predicted_index(raw_output, num_candidates=len(row.get("action_reprs", [])))
            record = make_prediction_record(row, split, baseline["name"], pred_idx, raw_output)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1

    return written


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    model, tokenizer = load_intern_model(args.model_name, dtype, args.quantization)

    out_dir = Path(args.output_dir)
    for split in args.splits:
        print(f"Loading split: {split}")
        dataset = load_split_dataset(args.axtree_dir, split)

        for baseline in BASELINES:
            print(f"Running baseline '{baseline['name']}' on {split}")
            out_file = out_dir / f"preds_{baseline['name']}_{split}.jsonl"
            num_written = run_single_baseline(model, tokenizer, dataset, split, baseline, args, dtype)
            print(f"Updated predictions: {out_file} (new rows written: {num_written})")


if __name__ == "__main__":
    main()
