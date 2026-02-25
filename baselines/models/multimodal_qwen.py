import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
MAX_PIXELS = 1024 * 28 * 28
# 1. Setup Quantization Config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load dataset directly from Hugging Face
print("Loading dataset from Hugging Face...")
dataset = load_dataset("osunlp/Multimodal-Mind2Web", split="test_website")
correct = 0
total = 0
results = []
for row in tqdm(dataset):
    task = row["confirmed_task"]
    html = row["cleaned_html"][:15000]
    candidates = row["action_reprs"]
    target = int(row["target_action_index"])
    
    # Load image
    #image = Image.open(row["screenshot"]).convert("RGB")

    candidate_text = "".join([f"{i}: {c}\n" for i, c in enumerate(candidates)])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["screenshot"]},
                {"type": "text", "text": f"Task: {task}\nHTML: {html}\nActions:\n{candidate_text}\nSelect the correct action index. Answer with ONLY the number."},
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
        max_pixels=MAX_PIXELS
    ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=10)
        
    # Decode the output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    try:
        clean_pred = "".join(filter(str.isdigit, pred_text.strip()))
        if int(clean_pred) == target:
            correct += 1
    except:
        pass
    results.append({
        "annotation_id": row["annotation_id"],
        "task": task,
        "html": html,
        "candidates": candidates,
        "target": target,
        "prediction": int(clean_pred) })
    total += 1
    torch.cuda.empty_cache()

print(f"\nFinal Accuracy: {correct / total:.2%}")
