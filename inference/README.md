Inference scaffold for Multimodal VLMs

Run:

python -m inference.vlm_inference

## How to Run Baseline Inference and Evaluation

### 1. Run Batch Inference for All Baselines
Edit `batch_inference.py` to set your dataset path and fill in model/inference logic. Then run:

```bash
python batch_inference.py
```
This will generate prediction files like `preds_text_only.jsonl`, `preds_image_only.jsonl`, etc. for each baseline.

### 2. Evaluate Baseline Predictions
For each predictions file, run the evaluation script to compute metrics:

```bash
python vlm_inference_script.py --eval preds_text_only.jsonl --out metrics_text_only.json
python vlm_inference_script.py --eval preds_image_only.jsonl --out metrics_image_only.json
# ...repeat for each baseline
```
This will create metrics JSON files for each baseline.

### 3. (Optional) Generate a Checklist
To generate a checklist of recommended metrics and reporting items:

```bash
python vlm_inference_script.py --write-checklist CHECKLIST.md
```

### 4. Aggregate Results
Collect all metrics JSONs into a summary table for your report or paper.

---

## Evaluation Metrics Used and Rationale

The evaluation script (`vlm_inference_script.py`) computes the following metrics for each baseline:

- **Element Accuracy (Top-1):** Measures if the model correctly grounds the target DOM element. This is the most critical metric for web agents, as most failures are due to incorrect element selection.
- **Action Accuracy:** Measures if the predicted action (e.g., CLICK, TYPE) matches the ground truth. Important for verifying correct operation type.
- **Exact Match (Element + Action):** Both element and action must be correct. This is the main step-level metric reported in most web agent papers.
- **Parse Failure Rate:** Fraction of outputs that are invalid or unparseable (e.g., due to LLM/CoT errors). Important for robustness, especially with large models.
- **Top-K Element Accuracy (default Top-3):** Checks if the correct element is among the top K candidates. Useful for showing ranking improvements, especially with graph/AXTree modalities.
- **MRR (Mean Reciprocal Rank):** Evaluates the quality of the element ranking, rewarding higher placement of the correct element.
- **Per-action Precision/Recall/F1:** Gives a detailed breakdown of model performance for each action type, helping diagnose class imbalance or specific weaknesses.
- **Task Success Rate:** (if available) Fraction of records where the model completed the full multi-step task successfully (requires 'task_success' boolean in predictions). This is a strong, realistic metric for end-to-end agent performance.

These metrics are standard in web agent literature and allow you to:
- Compare grounding and action prediction quality across baselines and modalities
- Report robustness to parse failures
- Show improvements from multimodal and graph-based approaches
- Provide detailed breakdowns for ablation and error analysis

---

**Note:**
- You must implement the actual model loading and inference logic in `batch_inference.py` for your baselines.
- The evaluation script expects predictions in the required JSONL format (see script docstring for details).
