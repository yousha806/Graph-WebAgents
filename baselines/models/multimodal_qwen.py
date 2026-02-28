import argparse
from typing import Dict, Optional

import torch
from datasets import load_from_disk
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2-VL baseline on Mind2Web")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--split", default="test_website")
    parser.add_argument("--data_dir", default="data/mind2web",
                        help="Directory produced by download_mind2web.py")
    parser.add_argument("--max_html_chars", type=int, default=15000)
    parser.add_argument(
        "--quantization",
        default="4bit",
        choices=["none", "4bit", "8bit"],
        help="Optional bitsandbytes quantization mode",
    )
    return parser.parse_args()


def build_quantization_kwargs(mode: str) -> Optional[Dict[str, object]]:
    if mode == "none":
        return None
    if mode == "4bit":
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    if mode == "8bit":
        return {"load_in_8bit": True}
    raise ValueError(f"Unsupported quantization mode: {mode}")


def load_qwen_model(model_name: str, quantization: str):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    quantization_kwargs = build_quantization_kwargs(quantization)
    quantization_config = BitsAndBytesConfig(**quantization_kwargs) if quantization_kwargs else None
    model_kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    print(f"Loading model {model_name} (quantization={quantization})...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def main() -> None:
    args = parse_args()
    model, processor = load_qwen_model(args.model_name, args.quantization)

    split_path = f"{args.data_dir}/{args.split}"
    print(f"Loading dataset split '{args.split}' from disk: {split_path}")
    dataset = load_from_disk(split_path)

    correct = 0
    total = 0
    results = []

    for row in tqdm(dataset):
        task = row["confirmed_task"]
        html = row["cleaned_html"][: args.max_html_chars]
        candidates = row["action_reprs"]
        target = int(row["target_action_index"])

        candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["screenshot"]},
                    {
                        "type": "text",
                        "text": (
                            f"Task: {task}\nHTML: {html}\nActions:\n{candidate_text}\n"
                            "Select the correct action index. Answer with ONLY the number."
                        ),
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[row["screenshot"]],
            padding=True,
            return_tensors="pt",
            process_condition_type="test",
            min_pixels=256 * 28 * 28,
            max_pixels=MAX_PIXELS,
        ).to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=10)

        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        pred_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        clean_pred = "".join(filter(str.isdigit, pred_text.strip()))
        pred_idx = int(clean_pred) if clean_pred else None

        if pred_idx is not None and pred_idx == target:
            correct += 1

        results.append(
            {
                "annotation_id": row["annotation_id"],
                "task": task,
                "html": html,
                "candidates": candidates,
                "target": target,
                "prediction": pred_idx,
            }
        )
        total += 1
        torch.cuda.empty_cache()

    print(f"\nFinal Accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()
