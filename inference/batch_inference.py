"""
Batch inference template for Mind2Web baselines.

For each baseline, this script:
- Loads the correct pretrained model
- Prepares the input according to the baseline’s modalities
- Runs inference and saves predictions in the required JSONL format

Fill in the TODOs with your actual model loading and preprocessing logic.
"""
import json
from typing import Dict, List

# Example baseline configs (add/modify as needed)
BASELINES = [
    {"name": "text_only", "modalities": ["text"], "model": "Qwen2-VL"},
    {"name": "image_only", "modalities": ["image"], "model": "InternVL2"},
    {"name": "multimodal_model_a", "modalities": ["text", "image"], "model": "Qwen2-VL"},
    {"name": "multimodal_model_b", "modalities": ["text", "image"], "model": "InternVL2"},
    {"name": "axtree_only", "modalities": ["axtree"], "model": "Qwen2-VL"},
    {"name": "multimodal_axtree_model_a", "modalities": ["text", "image", "axtree"], "model": "Qwen2-VL"},
    {"name": "multimodal_axtree_model_b", "modalities": ["text", "image", "axtree"], "model": "InternVL2"},
    {"name": "multimodal_cot", "modalities": ["text", "image"], "model": "Qwen2-VL", "cot": True},
    {"name": "multimodal_axtree_cot_model_a", "modalities": ["text", "image", "axtree"], "model": "Qwen2-VL", "cot": True},
    {"name": "multimodal_axtree_cot_model_b", "modalities": ["text", "image", "axtree"], "model": "InternVL2", "cot": True},
]

# TODO: Replace with your dataset loader
DATASET_PATH = "path/to/mind2web.jsonl"

def load_dataset(path: str) -> List[Dict]:
    # Dummy loader: replace with your actual dataset loading logic
    # Each sample should be a dict with keys: instruction, dom, image_path, axtree, gt_element, gt_action, etc.
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(line) for line in f]

# TODO: Replace with your actual model loading logic
def load_model(baseline: Dict):
    print(f"[stub] Loading model {baseline['model']} for {baseline['name']}")
    return None  # Replace with actual model object

# TODO: Replace with your actual inference logic
def run_model(model, sample: Dict, baseline: Dict) -> Dict:
    # Prepare input according to modalities
    # Run model and return prediction dict with required fields
    # Here we just return a dummy prediction
    return {
        "gt_element": sample.get("gt_element"),
        "gt_action": sample.get("gt_action"),
        "gt_value": sample.get("gt_value"),
        "pred_element": sample.get("gt_element"),  # Dummy: always correct
        "pred_action": sample.get("gt_action"),    # Dummy: always correct
        "pred_value": sample.get("gt_value"),
        "candidates": sample.get("candidates", []),
    }

def main():
    dataset = load_dataset(DATASET_PATH)
    for baseline in BASELINES:
        print(f"Running inference for baseline: {baseline['name']}")
        model = load_model(baseline)
        preds = []
        for sample in dataset:
            pred = run_model(model, sample, baseline)
            preds.append(pred)
        out_path = f"preds_{baseline['name']}.jsonl"
        with open(out_path, "w", encoding="utf8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
