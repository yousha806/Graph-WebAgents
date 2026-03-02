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

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/mind2web}"
AXTREE_DIR="${AXTREE_DIR:-$PROJECT_DIR/data/mind2web_axtree}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/intern_axtree_ablations}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

SPLITS=(test_website)

usage() {
  cat <<'EOF'
Usage: scripts/run_ec2_inference.sh [--help]

Results and logs are saved locally on the EC2 instance under OUTPUT_DIR and LOG_DIR.
Override any config variable via environment:
  PROJECT_DIR, MODEL_NAME, OUTPUT_DIR, LOG_DIR, HF_TOKEN, INFERENCE_DTYPE, QUANTIZATION
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
if [[ "$PYTHON_BIN" != "$CONDA_BASE"* && "$PYTHON_BIN" != *"/envs/"* && \
      ( -z "$CONDA_ENV_PREFIX" || "$PYTHON_BIN" != "$CONDA_ENV_PREFIX"* ) ]]; then
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
# Pin torch to cu121 BEFORE installing requirements so pip does not pull a newer
# cu12x build from PyPI. cu121 requires driver >= 520; nvidia-driver-535 satisfies this.
# --force-reinstall ensures a previously-installed cu128/cu124 build is replaced.
# --index-url (not --extra-index-url) forces resolution from the pytorch index only.
pip install --force-reinstall "torch>=2.1.0,<3.0" torchvision \
  --index-url https://download.pytorch.org/whl/cu121
# Install remaining deps; torch is already satisfied so it will not be reinstalled.
pip install \
  -r requirements.txt \
  -r baselines/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121

# Flash Attention 2: must be installed AFTER torch, with --no-build-isolation so the
# build system finds the already-installed torch and CUDA headers.
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

# flash-attn builds from source and requires nvcc + CUDA_HOME (the compiler/headers,
# not just the driver runtime that PyTorch ships). Detect from common EC2 install paths.
# Checks versioned paths first (cuda-12.1 matches the cu121 torch build), then the
# generic /usr/local/cuda symlink.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _cuda_dir in \
      /usr/local/cuda-12.1 /usr/local/cuda-12.2 /usr/local/cuda-12.3 /usr/local/cuda-12.4 \
      /usr/local/cuda-12 /usr/local/cuda \
      /usr/local/cuda-11.8; do
    if [[ -x "$_cuda_dir/bin/nvcc" ]]; then
      export CUDA_HOME="$_cuda_dir"
      export PATH="$CUDA_HOME/bin:$PATH"
      break
    fi
  done
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  # conda install -c nvidia cuda-nvcc triggers Anaconda ToS in non-interactive sessions.
  # Use pip's nvidia-cuda-nvcc-cu12 wheel instead — ToS-free and matches the cu121 torch build.
  echo "nvcc not found in standard paths — installing nvidia-cuda-nvcc-cu12 via pip..."
  pip install nvidia-cuda-nvcc-cu12 --quiet || true
  # nvidia.cuda_nvcc is a namespace package (__file__ == None); use site.getsitepackages()
  # to locate the nvcc binary directly.
  _NVCC_BIN="$(python - 2>/dev/null <<'PY'
import site, os
candidates = site.getsitepackages()
try:
    candidates.append(site.getusersitepackages())
except Exception:
    pass
for d in candidates:
    p = os.path.join(d, "nvidia", "cuda_nvcc", "bin", "nvcc")
    if os.path.isfile(p):
        print(os.path.dirname(p))
        break
PY
)"
  if [[ -n "$_NVCC_BIN" && -x "$_NVCC_BIN/nvcc" ]]; then
    export CUDA_HOME="$(dirname "$_NVCC_BIN")"
    export PATH="$_NVCC_BIN:$PATH"
  fi
fi
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "Installing flash-attn (this takes ~10-15 min on first run)..."
pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn install failed; will use standard attention."

# Needed for precompute_axtree.py
# install-deps calls apt-get internally and needs root; pass the conda PATH so
# sudo can find the playwright binary inside the conda env.
sudo env PATH="$PATH" playwright install-deps chromium
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

# ---------- CUDA pre-flight check ----------
# Run synchronously so failures surface immediately rather than being buried in the nohup log.
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    drv = "unknown"
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                       "--format=csv,noheader"], text=True).strip()
        drv = out.splitlines()[0]
    except Exception:
        pass
    print(
        f"ERROR: torch.cuda.is_available() = False\n"
        f"  torch version      : {torch.__version__}\n"
        f"  torch CUDA build   : {torch.version.cuda}\n"
        f"  nvidia-smi driver  : {drv}\n"
        f"\n"
        f"  Most likely causes:\n"
        f"    1. Driver version too old for this CUDA build.\n"
        f"       cu121 requires driver >= 520; cu124 requires >= 550.\n"
        f"    2. NVIDIA kernel module not loaded — reboot after driver install.\n"
        f"    3. /dev/nvidia* not accessible (check permissions).",
        file=sys.stderr,
    )
    sys.exit(1)

dev = torch.cuda.current_device()
print(
    f"CUDA OK: {torch.cuda.get_device_name(dev)} | "
    f"VRAM free: {torch.cuda.mem_get_info(dev)[0]/1e9:.1f} GB | "
    f"torch {torch.__version__} (CUDA {torch.version.cuda})"
)
PY

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

# Run metrics computation after inference completes (in background, waits for inference PID)
METRICS_LOG="$LOG_DIR/metrics_$(date +%Y%m%d_%H%M%S).log"
(
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  echo "[metrics] Inference finished. Computing metrics..."
  METRICS_OUT="$OUTPUT_DIR/metrics_summary.json"
  if ls "$OUTPUT_DIR"/preds_*.jsonl >/dev/null 2>&1; then
    python inference/eval_next_action.py \
      --input "$OUTPUT_DIR" \
      --pattern "preds_*.jsonl" \
      --out "$METRICS_OUT" \
      && echo "[metrics] Written: $METRICS_OUT" \
      || echo "[metrics] WARNING: eval_next_action.py failed" >&2
  else
    echo "[metrics] No prediction JSONL files found; skipping."
  fi
) >> "$METRICS_LOG" 2>&1 &

echo "Started inference in background."
echo "Baselines: intern_image_allinputs_axtree, intern_image_allinputs_axtree_cot"
echo "PID          : $PID"
echo "Inference log: $RUN_LOG"
echo "Metrics log  : $METRICS_LOG"
echo "Outputs      : $OUTPUT_DIR"
echo ""
echo "Monitor with:"
echo "  tail -f $RUN_LOG"
echo "  wc -l $OUTPUT_DIR/*.jsonl"
