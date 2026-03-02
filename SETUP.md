# EC2 Setup Guide — Multimodal Mind2Web

---

## 1. Create EC2 Instance

- **AMI**: Ubuntu 24.04
- **Storage**: 200 GB
- **Security**: SSH from my IP only
- **Key pair**: `mind2web-key.pem`

---

## 2. Connect

```bash
ssh -i mind2web-key.pem -L 8888:localhost:8888 ubuntu@<public-ip>
```

> ⚠️ Delete the instance after use — GPU instances are expensive.

---

## 3. System Setup

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt install -y nvidia-driver-535-server
sudo reboot   # wait ~10s, then reconnect
```

Verify GPU after reboot:
```bash
nvidia-smi
```

---

## 4. Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
eval "$(~/miniconda3/bin/conda shell.bash hook)"
```

---

## 5. Conda Environment

```bash
conda create -n mind2web python=3.10   # accept prompts with 'y'
conda activate mind2web
```

---

## 6. Clone Repo

```bash
git clone https://github.com/yousha806/Graph-WebAgents.git
cd Graph-WebAgents
git checkout mru1   # or whichever branch has the code
```

---

## 7. Install Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Install Playwright (headless browser for AXTree extraction):
```bash
playwright install chromium
sudo env PATH="$PATH" playwright install-deps chromium
```

Verify GPU is visible to PyTorch:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 8. Download Dataset

Default downloads `test_website` only (~5 min, no token needed):
```bash
python download_mind2web.py
```

See [README.md](README.md) for other splits. To speed up with a HF token:
```bash
huggingface-cli login
python download_mind2web.py
```

---

## 9. Precompute AXTree

```bash
python precompute_axtree.py \
  --split test_website \
  --data_dir data/mind2web \
  --out_dir data/mind2web_axtree
```

---

## 10. Run Inference

**Monitored** (output printed to terminal):
```bash
python inference/intern_axtree_ablations.py \
  --model_name OpenGVLab/InternVL2-8B \
  --axtree_dir data/mind2web_axtree \
  --splits test_website \
  --output_dir outputs/intern_axtree_ablations \
  --dtype bf16
```

**Detached** (runs in background, safe to disconnect):
```bash
mkdir -p logs
nohup python inference/intern_axtree_ablations.py \
  --model_name OpenGVLab/InternVL2-8B \
  --axtree_dir data/mind2web_axtree \
  --splits test_website \
  --output_dir outputs/intern_axtree_ablations \
  --dtype bf16 \
  --resume \
  > logs/inference.log 2>&1 &

echo "PID: $!  —  monitor with: tail -f logs/inference.log"
```

Monitor progress:
```bash
tail -f logs/inference.log
wc -l outputs/intern_axtree_ablations/*.jsonl
```

---

## 11. Compute Metrics

```bash
python inference/eval_next_action.py \
  --input outputs/intern_axtree_ablations \
  --pattern "preds_*.jsonl" \
  --out outputs/intern_axtree_ablations/metrics_summary.json
```
