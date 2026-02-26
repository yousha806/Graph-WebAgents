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