# AWS Setup Guide: Multimodal Mind2Web Baselines
## Qwen2-VL (7B) + InternVL2 (8B)

---

## 1. AWS Instance Recommendation

### Recommended: `g5.2xlarge` (per run) or `g5.12xlarge` (parallel runs)

| Instance | GPUs | GPU VRAM | vCPU | RAM | Est. Cost |
|---|---|---|---|---|---|
| `g5.2xlarge` | 1x A10G (24GB) | 24GB | 8 | 32GB | ~$1.21/hr |
| `g5.4xlarge` | 1x A10G (24GB) | 24GB | 16 | 64GB | ~$1.68/hr |
| `g5.12xlarge` | 4x A10G (24GB) | 96GB total | 48 | 192GB | ~$5.67/hr |

**Recommendation:** Start with **`g5.4xlarge`** — enough VRAM for 7-8B models in bfloat16, with headroom for batch sizes. Use `g5.12xlarge` if you want to run both models simultaneously or use larger batches.

Both Qwen2-VL-7B and InternVL2-8B fit comfortably in 24GB VRAM at bf16.

### Storage
- **EBS Volume:** 200GB gp3 (models ~15GB each, dataset + outputs)

---

## 2. Step-by-Step AWS Launch

### Step 1: Launch EC2 Instance

```bash
# Via AWS Console:
# 1. Go to EC2 → Launch Instance
# 2. Name: mind2web-baselines
# 3. AMI: Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)
#    - Search "Deep Learning AMI" in AMI catalog
#    - This comes with CUDA, PyTorch, conda pre-installed
# 4. Instance type: g5.4xlarge
# 5. Key pair: create or use existing .pem file
# 6. Storage: 200GB gp3
# 7. Security Group: allow SSH (port 22) from your IP
# 8. Launch!
```

### Step 2: Connect to Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem -L 8888:localhost:8888 ubuntu@<your-ec2-public-ip>
# The -L flag forwards Jupyter port to your local machine
```

### Step 3: Environment Setup (run on EC2)

```bash
# Activate the pre-installed PyTorch conda env
conda activate pytorch

# Install required packages
pip install transformers==4.45.0 \
            accelerate \
            einops \
            timm \
            sentencepiece \
            qwen-vl-utils \
            flash-attn --no-build-isolation \
            datasets \
            wandb \
            jupyter \
            matplotlib \
            scikit-learn \
            tqdm

# For InternVL2 specifically
pip install torchvision \
            Pillow \
            opencv-python-headless

# Optional: install from source for latest InternVL2
# pip install git+https://github.com/OpenGVLab/InternVL.git
```

### Step 4: Download Models (HuggingFace Hub)

```bash
# Set cache dir to your EBS volume
export HF_HOME=/home/ubuntu/hf_cache
mkdir -p $HF_HOME

# Download models (will cache to HF_HOME)
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
# Qwen2-VL
AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
print('Qwen2-VL downloaded')
"

python -c "
from transformers import AutoProcessor, AutoModel
# InternVL2
AutoProcessor.from_pretrained('OpenGVLab/InternVL2-8B')
AutoModel.from_pretrained('OpenGVLab/InternVL2-8B')
print('InternVL2 downloaded')
"
```

> ⚠️ You may need a HuggingFace token for gated models:
> ```bash
> huggingface-cli login
> # paste your token from huggingface.co/settings/tokens
> ```

### Step 5: Launch Jupyter

```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
# Copy the token from the output, open http://localhost:8888 in your browser
```

---

## 3. Cost Management Tips

- **Stop (not terminate) the instance** when not running experiments — you keep storage but stop compute billing
- Use **Spot Instances** for ~70% discount if you checkpoint frequently (g5.4xlarge spot ~$0.50/hr)
- Set a **CloudWatch billing alarm** at $50/$100 to avoid surprises
- Use `tmux` or `screen` so jobs survive SSH disconnects:
  ```bash
  tmux new -s baselines
  # run your job
  # Ctrl+B, D to detach
  tmux attach -t baselines  # reattach later
  ```

---

## 4. Experiment Tracking (Optional but Recommended)

```bash
# Set up W&B for free experiment tracking
wandb login  # get key from wandb.ai
```

---

## 5. Saving Results to S3

```bash
# Create S3 bucket for results
aws s3 mb s3://mind2web-results-yourname

# Sync results
aws s3 sync ./results/ s3://mind2web-results-yourname/results/
```

Make sure your EC2 instance has an **IAM role** with S3 access, or configure `aws configure` with your credentials.

---

## 6. Estimated Runtime per Baseline (g5.4xlarge)

| Baseline | Est. Time (Mind2Web test set ~2.4k samples) |
|---|---|
| Text DOM only | ~30-60 min |
| Image only | ~1-2 hrs |
| Multimodal (full) | ~2-3 hrs |
| + CoT variants | ~3-5 hrs (longer generation) |

Run overnight or use `tmux` + log to file:
```bash
python run_baselines.py --baseline text_dom 2>&1 | tee logs/text_dom.log
```
