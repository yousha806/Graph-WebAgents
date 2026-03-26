"""
Image-only baseline using Vision Transformer (ViT-B) + k-NN (zero-shot).
Extracts visual features and uses k-nearest neighbors for action prediction.
No training required.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import warnings

# Suppress decompression bomb warning
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

load_dotenv()

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def load_vitb_model():
    """Load pre-trained ViT-B model and image processor."""
    model_name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name)
    # Load ViT model (without classification head for feature extraction)
    model = ViTModel.from_pretrained(model_name)
    model = model.to(DEVICE)
    model.eval()
    return processor, model


def extract_image_features(image, processor, model, verbose=False):
    """Extract ViT-B features from a single image."""
    try:
        if isinstance(image, str):
            # Image path
            if not os.path.exists(image):
                if verbose:
                    print(f"Image path not found: {image}")
                return None
            image = Image.open(image).convert("RGB")
        elif image is None:
            return None
        elif not isinstance(image, Image.Image):
            if verbose:
                print(f"Invalid image type: {type(image)}")
            return None
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Extract features (CLS token from last hidden state)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token (first token of last hidden state)
            features = outputs.last_hidden_state[:, 0, :]  # [1, 768]
        
        return features.cpu().numpy()
    except Exception as e:
        if verbose:
            print(f"Error extracting features: {e}")
        return None


def build_knn_reference(processor, vitb_model, reference_split="test_website", cache_path="vitb_knn_reference.npz"):
    """Build k-NN reference database by extracting features from reference split."""
    if os.path.exists(cache_path):
        print(f"Loading cached reference features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        ref_features = data['features']
        ref_actions = data['actions']
        return ref_features, ref_actions
    
    print(f"Building k-NN reference database from {reference_split} split...")
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=reference_split)
    
    ref_features = []
    ref_actions = []
    
    for sample in tqdm(dataset):
        screenshot = sample.get("screenshot")
        candidates = sample.get("action_reprs") or []
        target_idx = sample.get("target_action_index")
        
        if screenshot is None or not candidates or target_idx is None:
            continue
        
        # Convert target_idx to int if needed
        if not isinstance(target_idx, int):
            try:
                target_idx = int(target_idx)
            except (ValueError, TypeError):
                continue
        
        features = extract_image_features(screenshot, processor, vitb_model)
        if features is None:
            continue
        
        ref_features.append(features[0])
        ref_actions.append(candidates[target_idx])
    
    ref_features = np.array(ref_features)
    ref_actions = np.array(ref_actions)
    
    # Cache the reference database
    np.savez(cache_path, features=ref_features, actions=ref_actions)
    print(f"Reference database: {len(ref_features)} samples, cached to {cache_path}")
    
    return ref_features, ref_actions


def predict_with_knn(features, ref_features, ref_actions, k=5):
    """Predict action using k-NN on reference features."""
    # Compute cosine similarity between query and reference
    similarities = cosine_similarity(features.reshape(1, -1), ref_features)[0]
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Majority vote on actions
    top_k_actions = ref_actions[top_k_indices]
    unique, counts = np.unique(top_k_actions, return_counts=True)
    pred_action = unique[np.argmax(counts)]
    
    return pred_action


def main():
    parser = argparse.ArgumentParser(description="ViT-B image-only baseline with k-NN (zero-shot)")
    parser.add_argument("--split", default="test_website", help="Dataset split to evaluate")
    parser.add_argument("--preds-out", default="vitb_preds.jsonl", help="Output predictions file")
    parser.add_argument("--wrong-out", default="wrong_vitb_preds.jsonl", help="Output incorrect predictions JSONL file")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors for k-NN")
    parser.add_argument("--reference-split", default="test_website", help="Reference split for k-NN database")
    parser.add_argument("--knn-cache", default="vitb_knn_reference.npz", help="Cache path for k-NN reference")
    
    args = parser.parse_args()
    
    # Load pre-trained ViT-B model
    print("Loading ViT-B model...")
    processor, vitb_model = load_vitb_model()
    
    # Build k-NN reference database
    ref_features, ref_actions = build_knn_reference(processor, vitb_model, args.reference_split, args.knn_cache)
    
    # Load dataset
    print(f"Loading dataset split: {args.split}")
    dataset = load_dataset("osunlp/Multimodal-Mind2Web", split=args.split)
    
    wrong_results = []

    # Generate predictions
    print(f"Generating predictions for {args.split} split using k-NN (k={args.k})...")
    predictions = []
    
    for sample in tqdm(dataset):
        sample_id = sample.get("annotation_id") or sample.get("id", "")
        input_task = sample.get("confirmed_task") or sample.get("task") or ""
        screenshot = sample.get("screenshot")
        html = sample.get("cleaned_html") or ""
        candidates = sample.get("action_reprs") or []  # Use action_reprs not candidates
        target_idx = sample.get("target_action_index")
        
        # Convert target_idx to int if it's a string
        if target_idx is not None and not isinstance(target_idx, int):
            try:
                target_idx = int(target_idx)
            except (ValueError, TypeError):
                target_idx = None
        
        # Skip if no screenshot
        if screenshot is None:
            continue
        if not candidates or target_idx is None:
            continue
        
        # Extract image features
        features = extract_image_features(screenshot, processor, vitb_model)
        if features is None:
            continue
        
        # Predict action using k-NN
        pred_action = predict_with_knn(features[0], ref_features, ref_actions, k=args.k)
        
        # Find index of predicted action
        pred_action_idx = 0
        for i, candidate in enumerate(candidates):
            if candidate == pred_action:
                pred_action_idx = i
                break
        
        # Extract ground truth
        gt_element = candidates[target_idx] if target_idx < len(candidates) else None
        
        # Helper to extract action type
        def extract_action(text):
            if not text:
                return None
            text_upper = str(text).upper()
            for action in ["CLICK", "TYPE", "SELECT", "SCROLL", "HOVER"]:
                if action in text_upper:
                    return action
            return None
        
        pred_action_text = extract_action(pred_action)
        gt_action = extract_action(gt_element) if gt_element else None
        
        # Determine success
        success = (pred_action_idx == target_idx) if target_idx is not None else False

        # Build prediction record (standardized schema)
        html_snippet = html[:200] if html else None
        screenshot_ref = screenshot if isinstance(screenshot, str) else None
        record = {
            "id": sample_id,
            "model": "vit-base-patch16-224",
            "split": args.split,
            "input_task": input_task,
            "html_snippet": html_snippet,
            "screenshot": screenshot_ref,
            "candidates": candidates,
            "gt_index": target_idx,
            "gt_element": gt_element,
            "gt_action": gt_action,
            "gt_value": sample.get("gt_value"),
            "pred_text": pred_action,
            "pred_idx": pred_action_idx,
            "pred_element": pred_action,
            "pred_action": pred_action_text,
            "pred_value": None,
            "value_states": None,
            "task_success": bool(success),
        }
        predictions.append(record)

        # Track incorrect examples separately
        if not success:
            wrong_results.append(record)
    
    # Save predictions
    save_jsonl(predictions, args.preds_out)
    print(f"Predictions saved to {args.preds_out}")
    print(f"Total predictions: {len(predictions)}")

    # Save incorrect predictions if requested
    if args.wrong_out:
        save_jsonl(wrong_results, args.wrong_out)
        print(f"Wrote {len(wrong_results)} incorrect prediction records to {args.wrong_out}")


def save_jsonl(data, filepath):
    """Save list of dicts as JSONL."""
    with open(filepath, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
