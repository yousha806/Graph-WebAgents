#!/usr/bin/env bash
set -euo pipefail

# One-command EC2 launcher for InternVL2 AXTree ablations.
# - Downloads ONLY test splits (no train)
# - Precomputes AXTree for those test splits
# - Runs inference detached (nohup) with resume enabled
# - Saves outputs/logs to persistent EBS-backed paths

# ---------- Config (override with env vars) ----------
PROJECT_DIR="${PROJECT_DIR:-$HOME/Graph-WebAgents}"
MODEL_NAME="${MODEL_NAME:-OpenGVLab/InternVL2-8B}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mind2web}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-bf16}"
QUANTIZATION="${QUANTIZATION:-none}"
HF_TOKEN="${HF_TOKEN:-}"
ENABLE_S3_SYNC="${ENABLE_S3_SYNC:-1}"
S3_PREFIX="${S3_PREFIX:-intern_axtree_ablations}"
S3_SYNC_INTERVAL_SEC="${S3_SYNC_INTERVAL_SEC:-300}"
S3_ACCESS_POINT_URI=""

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/mind2web}"
AXTREE_DIR="${AXTREE_DIR:-$PROJECT_DIR/data/mind2web_axtree}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/intern_axtree_ablations}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
AWSCLI_INSTALL_DIR="${AWSCLI_INSTALL_DIR:-$HOME/awscli-v2}"

SPLITS=(test_task test_website test_domain)

usage() {
  cat <<'EOF'
Usage: scripts/run_ec2_inference.sh [--s3-access-point-uri <s3://...>] [--help]

Options:
  --s3-access-point-uri   S3 access point URI for syncing logs/outputs.
                          Example: s3://arn:aws:s3:us-east-1:123456789012:accesspoint/my-ap/mind2web-runs
  --help, -h              Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --s3-access-point-uri)
      if [[ $# -lt 2 ]]; then
        echo "Error: --s3-access-point-uri requires a value."
        usage
        exit 1
      fi
      S3_ACCESS_POINT_URI="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown argument '$1'"
      usage
      exit 1
      ;;
  esac
done

# ---------- Setup ----------
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$HF_CACHE_DIR"
cd "$PROJECT_DIR"

if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  if [[ -z "$S3_ACCESS_POINT_URI" ]]; then
    echo "S3 sync is enabled, but --s3-access-point-uri was not provided."
    read -r -p "Enter S3 access point URI: " S3_ACCESS_POINT_URI
    if [[ -z "$S3_ACCESS_POINT_URI" ]]; then
      echo "Error: S3 access point URI cannot be empty when ENABLE_S3_SYNC=1."
      exit 1
    fi
  fi
fi

# Conda-only flow: install Miniconda automatically if conda is not found anywhere.
if ! command -v conda >/dev/null 2>&1 && [[ ! -x "$MINICONDA_DIR/bin/conda" ]]; then
  echo "Conda not found. Installing Miniconda to $MINICONDA_DIR ..."

  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y || true
    sudo apt-get install -y wget bzip2 || true
  fi

  INSTALLER="/tmp/miniconda_installer.sh"
  wget -O "$INSTALLER" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
elif [[ ! -x "$MINICONDA_DIR/bin/conda" ]] && command -v conda >/dev/null 2>&1; then
  echo "Using existing conda from PATH."
else
  echo "Existing Miniconda found at $MINICONDA_DIR — skipping install."
fi

if [[ -x "$MINICONDA_DIR/bin/conda" ]]; then
  CONDA_BASE="$MINICONDA_DIR"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "Error: conda is unavailable and Miniconda install did not succeed."
  exit 1
fi

# Put conda's bin on PATH immediately so that subsequent subshells, nohup
# processes, and any direct 'conda' calls in this session all work without
# needing the user to source ~/.bashrc first.
export PATH="$CONDA_BASE/bin:$PATH"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ -n "$CONDA_ENV_PREFIX" ]]; then
  if [[ ! -d "$CONDA_ENV_PREFIX" ]]; then
    echo "Creating conda env at prefix: $CONDA_ENV_PREFIX (python=$PYTHON_VERSION)"
    conda create -y -p "$CONDA_ENV_PREFIX" "python=$PYTHON_VERSION" \
      --override-channels -c conda-forge
  fi
  conda activate "$CONDA_ENV_PREFIX"
elif [[ -n "$CONDA_ENV_NAME" ]]; then
  if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    echo "Creating conda env: $CONDA_ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" \
      --override-channels -c conda-forge
  fi
  conda activate "$CONDA_ENV_NAME"
else
  echo "Error: set CONDA_ENV_NAME or CONDA_ENV_PREFIX before running this script."
  echo "Example: CONDA_ENV_NAME=webagent ./scripts/run_ec2_inference.sh"
  exit 1
fi

# ---------- Verify conda env is active and will be used for ALL python calls ----------
ACTIVE_CONDA_ENV="$(conda info --envs | awk '/\*/ {print $1}')" 
PYTHON_BIN="$(command -v python)"
if [[ "$PYTHON_BIN" != "$CONDA_BASE"* && "$PYTHON_BIN" != *"/envs/"* ]]; then
  echo "Error: python resolved to '$PYTHON_BIN', which is outside the conda base '$CONDA_BASE'."
  echo "Conda env activation may have failed."
  exit 1
fi
echo "--------------------------------------------------------------"
echo "Conda env : $ACTIVE_CONDA_ENV"
echo "Python    : $PYTHON_BIN"
echo "Splits    : ${SPLITS[*]} (test-only, no train)"
echo "--------------------------------------------------------------"

python -m pip install --upgrade pip
# Install all deps; pass the PyTorch CUDA index so bitsandbytes gets the right CUDA-matched wheel
pip install \
  -r requirements.txt \
  -r baselines/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121

# ---------- NVIDIA driver check ----------
# Detect the EC2 instance family to know if a GPU is expected.
INSTANCE_TYPE="$(curl -sf --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-type || echo "unknown")"
echo "EC2 instance type: $INSTANCE_TYPE"

IS_GPU_INSTANCE=0
case "$INSTANCE_TYPE" in
  g4dn.*|g5.*|g5g.*|p2.*|p3.*|p4d.*|p5.*) IS_GPU_INSTANCE=1 ;;
esac

if [[ "$IS_GPU_INSTANCE" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU instance detected but nvidia-smi not found — installing NVIDIA drivers..."
    sudo apt-get update -y
    sudo apt-get install -y nvidia-utils-535 nvidia-driver-535
    echo "NVIDIA drivers installed. A reboot is required."
    echo "Please run: sudo reboot"
    echo "Then re-run this script after reconnecting."
    exit 1
  fi
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
elif [[ "$IS_GPU_INSTANCE" == "0" && "$INSTANCE_TYPE" != "unknown" ]]; then
  echo "ERROR: Instance type '$INSTANCE_TYPE' has no GPU."
  echo "InternVL2-8B requires a GPU instance. Stop this instance and switch to:"
  echo "  g4dn.xlarge  (T4,  16GB VRAM) — cheapest option"
  echo "  g5.xlarge    (A10G, 24GB VRAM) — supports Flash Attention 2"
  echo "  p3.2xlarge   (V100, 16GB VRAM)"
  exit 1
fi

# Flash Attention 2: must be installed AFTER torch, with --no-build-isolation so the
# build system finds the already-installed torch and CUDA headers.
# Only useful on Ampere+ GPUs (compute >= 8.0: A10G, A100, H100).
# Skipped automatically on older GPUs (T4=7.5) — the model falls back to standard attention.
GPU_CC=$(python - <<'PY'
import subprocess, sys
try:
    import torch
    cc = torch.cuda.get_device_capability(0)
    print(f"{cc[0]}.{cc[1]}")
except Exception:
    print("0.0")
PY
)
echo "GPU compute capability: $GPU_CC"
if python -c "import sys; cc=tuple(int(x) for x in '${GPU_CC}'.split('.')); sys.exit(0 if cc>=(8,0) else 1)" 2>/dev/null; then
  echo "Ampere+ GPU detected — installing flash-attn (this takes ~10-15 min on first run)..."
  pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn install failed; will use standard attention."
else
  echo "GPU compute < 8.0 — skipping flash-attn (not supported on this GPU)."
fi

if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI v2 not found; installing from official AWS installer..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -y || true
      sudo apt-get install -y unzip curl || true
    fi
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp/awscliv2
    sudo /tmp/awscliv2/aws/install --install-dir "$AWSCLI_INSTALL_DIR" --bin-dir /usr/local/bin
    rm -rf /tmp/awscliv2 /tmp/awscliv2.zip
  fi

  if ! command -v aws >/dev/null 2>&1; then
    echo "Error: aws CLI v2 installation failed."
    exit 1
  fi
  echo "Using: $(aws --version)"
fi

# Needed for precompute_axtree.py
# install-deps installs required OS-level shared libraries (libatk, libglib, etc.)
playwright install-deps chromium
playwright install chromium

# Export env vars so subprocesses (heredocs, parallel workers) can read them
export MODEL_NAME HF_CACHE_DIR HF_TOKEN
# Point HF dataset cache to our chosen dir for both download + precompute steps
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"

# ---------- Pre-download Intern model/checkpoint artifacts ----------
python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_name = os.environ.get("MODEL_NAME", "OpenGVLab/InternVL2-8B")
cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
token = os.environ.get("HF_TOKEN") or None

print(f"Pre-downloading model: {model_name} → cache: {cache_dir}")
local_path = snapshot_download(
    repo_id=model_name,
    cache_dir=cache_dir,
    token=token,
    resume_download=True,
)
print(f"Model cached at: {local_path}")
PY

sync_to_s3() {
  if [[ "$ENABLE_S3_SYNC" != "1" ]]; then
    return 0
  fi

  local s3_base="$S3_ACCESS_POINT_URI/$S3_PREFIX"
  echo "Syncing outputs/logs to $s3_base ..."

  aws s3 sync "$OUTPUT_DIR" "$s3_base/outputs" --only-show-errors
  aws s3 sync "$LOG_DIR" "$s3_base/logs" --only-show-errors
}

# ---------- Download ONLY test splits ----------
# HF_DATASETS_CACHE is already exported above; no --cache_dir needed.
python download_mind2web.py \
  --splits "${SPLITS[@]}" \
  --output_dir "$DATA_DIR"

# ---------- Precompute AXTree for test splits ----------
for split in "${SPLITS[@]}"; do
  python precompute_axtree.py \
    --split "$split" \
    --data_dir "$DATA_DIR" \
    --out_dir "$AXTREE_DIR"
done

# ---------- Launch detached inference ----------
RUN_LOG="$LOG_DIR/intern_axtree_ablations_$(date +%Y%m%d_%H%M%S).log"

nohup python inference/intern_axtree_ablations.py \
  --model_name "$MODEL_NAME" \
  --axtree_dir "$AXTREE_DIR" \
  --splits "${SPLITS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --dtype "$INFERENCE_DTYPE" \
  --quantization "$QUANTIZATION" \
  --resume \
  > "$RUN_LOG" 2>&1 &

PID=$!

if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  SYNC_LOG="$LOG_DIR/s3_sync_$(date +%Y%m%d_%H%M%S).log"
  (
    while kill -0 "$PID" >/dev/null 2>&1; do
      sync_to_s3 || echo "[sync] WARNING: sync_to_s3 returned non-zero" >&2
      sleep "$S3_SYNC_INTERVAL_SEC"
    done

    echo "[sync] Inference finished (PID $PID). Generating final metrics..."
    METRICS_OUT="$OUTPUT_DIR/metrics_summary.json"
    if ls "$OUTPUT_DIR"/preds_*.jsonl >/dev/null 2>&1; then
      python inference/eval_next_action.py \
        --input "$OUTPUT_DIR" \
        --pattern "preds_*.jsonl" \
        --out "$METRICS_OUT" \
        && echo "[sync] Metrics written: $METRICS_OUT" \
        || echo "[sync] WARNING: eval_next_action.py failed" >&2
    else
      echo "[sync] No prediction JSONL files found; skipping metrics."
    fi

    echo "[sync] Final S3 push..."
    sync_to_s3 || echo "[sync] WARNING: final sync_to_s3 returned non-zero" >&2
    echo "[sync] Done."
  ) >> "$SYNC_LOG" 2>&1 &
  SYNC_PID=$!
fi

echo "Started inference in background."
echo "Baselines: intern_image_allinputs_axtree, intern_image_allinputs_axtree_cot"
echo "PID: $PID"
if [[ -n "${SYNC_PID:-}" ]]; then
  echo "S3 sync watcher PID: $SYNC_PID"
  echo "Sync log: $SYNC_LOG"
fi
echo "Inference log: $RUN_LOG"
echo "Outputs: $OUTPUT_DIR"
echo ""
echo "Monitor with:"
echo "  tail -f $RUN_LOG"
if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  echo "  tail -f $SYNC_LOG"
fi
echo "  wc -l $OUTPUT_DIR/*.jsonl"
if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  echo "  aws s3 ls $S3_ACCESS_POINT_URI/$S3_PREFIX/outputs/"
  echo "  aws s3 ls $S3_ACCESS_POINT_URI/$S3_PREFIX/logs/"
fi
