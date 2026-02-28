# Graph-WebAgents

Multimodal graph-based web agents built on the [Multimodal Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) dataset.

## Setup

```bash
pip install -r requirements.txt

# Playwright needs browser binaries (one-time)
playwright install chromium
```

## Dataset

The dataset has **four splits** (no validation split):

| Split | Actions | Tasks | Description |
|---|---|---|---|
| `train` | 7 775 | 1 009 | Training data |
| `test_task` | 1 339 | 177 | Same websites seen in training |
| `test_website` | 1 019 | 142 | Unseen websites |
| `test_domain` | 4 060 | 694 | Entire domains unseen |

### 1. Download the dataset

```bash
# Download all three test splits (default)
python download_mind2web.py

# Download all four splits
python download_mind2web.py --splits train test_task test_website test_domain

# Download only 1% of each split (saves disk space)
python download_mind2web.py --subset_fraction 0.01

# Use a different output directory or HF cache location
python download_mind2web.py --output_dir D:/data/mind2web --cache_dir D:/hf_cache
```

Splits are saved under `data/mind2web/<split>/` using Arrow format (`save_to_disk`).

### 2. Precompute AXTree representations (optional)

Renders each row's HTML in headless Chromium and extracts the accessibility tree:

```bash
python precompute_axtree.py --split test_website
python precompute_axtree.py --split train --html_field raw_html
```

Output is saved to `data/mind2web_axtree/<split>/` with two extra columns: `axtree_json` and `axtree_text`.

### 3. Use the PyTorch DataLoader

```python
from mind2web_dataloader import create_mind2web_dataloader, create_all_dataloaders

# Single split – loads from HF directly
loader = create_mind2web_dataloader("test_website", batch_size=4)

# Single split – loads from local download
loader = create_mind2web_dataloader("test_website", batch_size=4, local_dir="data/mind2web")

# All splits at once
loaders = create_all_dataloaders(local_dir="data/mind2web", batch_size=4)
for batch in loaders["test_website"]:
    print(batch["confirmed_task"])
```

Each batch is a `Dict[str, List]` with keys: `annotation_id`, `action_uid`, `website`, `domain`, `confirmed_task`, `html`, `screenshot`, `operation`, `action_reprs`, `target_action_index`, `target_action_reprs`, `pos_candidates`, `neg_candidates`.

## Project Structure

```
download_mind2web.py      # Download dataset from Hugging Face
precompute_axtree.py      # Precompute accessibility trees
mind2web_dataloader.py    # PyTorch Dataset + DataLoader
requirements.txt          # Python dependencies
baselines/                # Baseline model implementations
inference/                # Inference & evaluation scripts
```

## Baselines & Evaluation

See [inference/README.md](inference/README.md) for instructions on running batch inference and computing metrics.

## EC2 End-to-End Runbook (InternVL2 AXTree Ablations)

This section shows how to go from a fresh AWS instance to running inference and saving results in a persistent location.

### 0) What this runs

The script `inference/intern_axtree_ablations.py` runs two ablations for next-action prediction:

- `intern_image_allinputs_axtree`
- `intern_image_allinputs_axtree_cot`

Across all three test splits by default:

- `test_task`
- `test_website`
- `test_domain`

### 1) Create the EC2 instance

1. Launch an EC2 `g5.xlarge` (NVIDIA A10G 24GB) Ubuntu instance.
2. Attach an EBS volume large enough for model cache + dataset + outputs (recommend at least 150–250 GB total).
3. Security Group:
     - allow SSH (`22`) from your IP.
4. (Recommended) attach an IAM role with S3 write permissions if you want automatic backup to S3.

### 2) Connect and install system dependencies

```bash
ssh -i <your-key>.pem ubuntu@<EC2_PUBLIC_DNS>

sudo apt update
sudo apt install -y git python3-pip python3-venv tmux
nvidia-smi
```

### 3) Clone repo and set up Python environment

```bash
git clone <your-repo-url>
cd Graph-WebAgents

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Needed for AXTree precompute (one-time)
playwright install chromium
```

### 4) Prepare dataset and AXTree splits

```bash
# Download default 3 test splits
python download_mind2web.py

# Precompute AXTree for all 3 test splits
python precompute_axtree.py --split test_task
python precompute_axtree.py --split test_website
python precompute_axtree.py --split test_domain
```

### 5) Choose persistent output locations

Use directories on the EC2 disk (EBS-backed) so results remain even if your laptop disconnects:

```bash
mkdir -p /home/ubuntu/outputs/intern_axtree_ablations
mkdir -p /home/ubuntu/logs
```

### 6) Run inference detached (continues after disconnect)

Use `nohup` and `--resume`:

```bash
nohup python inference/intern_axtree_ablations.py \
    --model_name OpenGVLab/InternVL2-8B \
    --axtree_dir data/mind2web_axtree \
    --splits test_task test_website test_domain \
    --output_dir /home/ubuntu/outputs/intern_axtree_ablations \
    --resume \
    > /home/ubuntu/logs/intern_axtree_ablations.log 2>&1 &
```

The script writes JSONL rows incrementally and flushes continuously, so partial results are persisted.

### 7) Monitor progress

```bash
tail -f /home/ubuntu/logs/intern_axtree_ablations.log
ls -lh /home/ubuntu/outputs/intern_axtree_ablations
wc -l /home/ubuntu/outputs/intern_axtree_ablations/*.jsonl
```

Expected output files:

- `preds_intern_image_allinputs_axtree_test_task.jsonl`
- `preds_intern_image_allinputs_axtree_test_website.jsonl`
- `preds_intern_image_allinputs_axtree_test_domain.jsonl`
- `preds_intern_image_allinputs_axtree_cot_test_task.jsonl`
- `preds_intern_image_allinputs_axtree_cot_test_website.jsonl`
- `preds_intern_image_allinputs_axtree_cot_test_domain.jsonl`

### 8) Resume after interruption

Re-run the same command with `--resume`; completed rows are skipped based on current output line count.

### 9) Evaluate next-action outputs

Use the dedicated next-action evaluator (action index + action label metrics):

```bash
python inference/eval_next_action.py \
    --input /home/ubuntu/outputs/intern_axtree_ablations \
    --pattern "preds_*.jsonl" \
    --out /home/ubuntu/outputs/intern_axtree_ablations/metrics_summary.json
```

### 10) Optional: copy results to S3 for extra persistence

If your EC2 has IAM permission to write S3:

```bash
aws s3 sync /home/ubuntu/outputs/intern_axtree_ablations s3://<your-bucket>/intern_axtree_ablations/
aws s3 cp /home/ubuntu/logs/intern_axtree_ablations.log s3://<your-bucket>/intern_axtree_ablations.log
```

This protects results even if the instance is terminated.