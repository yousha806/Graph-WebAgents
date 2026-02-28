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
ENABLE_S3_SYNC="${ENABLE_S3_SYNC:-1}"
S3_PREFIX="${S3_PREFIX:-intern_axtree_ablations}"
S3_SYNC_INTERVAL_SEC="${S3_SYNC_INTERVAL_SEC:-300}"
S3_ACCESS_POINT_URI=""

DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/mind2web}"
AXTREE_DIR="${AXTREE_DIR:-$PROJECT_DIR/data/mind2web_axtree}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/intern_axtree_ablations}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

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

# Activate conda in non-interactive shell; fallback to venv if conda is unavailable.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"

  if [[ -n "$CONDA_ENV_PREFIX" ]]; then
    if [[ ! -d "$CONDA_ENV_PREFIX" ]]; then
      echo "Creating conda env at prefix: $CONDA_ENV_PREFIX (python=$PYTHON_VERSION)"
      conda create -y -p "$CONDA_ENV_PREFIX" "python=$PYTHON_VERSION"
    fi
    conda activate "$CONDA_ENV_PREFIX"
  elif [[ -n "$CONDA_ENV_NAME" ]]; then
    if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
      echo "Creating conda env: $CONDA_ENV_NAME (python=$PYTHON_VERSION)"
      conda create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION"
    fi
    conda activate "$CONDA_ENV_NAME"
  else
    echo "Error: set CONDA_ENV_NAME or CONDA_ENV_PREFIX before running this script."
    echo "Example: CONDA_ENV_NAME=webagent ./scripts/run_ec2_inference.sh"
    exit 1
  fi
else
  echo "Conda not found in PATH. Falling back to Python venv at $VENV_DIR"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found; cannot create venv fallback."
    exit 1
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip
pip install -r requirements.txt -r baselines/requirements.txt

if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI not found; attempting install via apt..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -y || true
      sudo apt-get install -y awscli || true
    fi
  fi

  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI still not found; attempting install via pip --user..."
    python -m pip install --user awscli
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if ! command -v aws >/dev/null 2>&1; then
    echo "Error: aws CLI not found and automatic installation failed."
    echo "Install manually: sudo apt-get install -y awscli"
    exit 1
  fi
fi

# Needed for precompute_axtree.py
playwright install chromium

# ---------- Pre-download Intern model/checkpoint artifacts ----------
python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_name = os.environ.get("MODEL_NAME", "OpenGVLab/InternVL2-8B")
cache_dir = os.environ.get("HF_CACHE_DIR")
token = os.environ.get("HF_TOKEN") or None

print(f"Pre-downloading model: {model_name}")
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
python download_mind2web.py \
  --splits "${SPLITS[@]}" \
  --output_dir "$DATA_DIR" \
  --cache_dir "$HF_CACHE_DIR"

# ---------- Precompute AXTree for test splits ----------
for split in "${SPLITS[@]}"; do
  python precompute_axtree.py \
    --split "$split" \
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
  (
    while kill -0 "$PID" >/dev/null 2>&1; do
      sync_to_s3 || true
      sleep "$S3_SYNC_INTERVAL_SEC"
    done

    METRICS_OUT="$OUTPUT_DIR/metrics_summary.json"
    if ls "$OUTPUT_DIR"/preds_*.jsonl >/dev/null 2>&1; then
      python inference/eval_next_action.py \
        --input "$OUTPUT_DIR" \
        --pattern "preds_*.jsonl" \
        --out "$METRICS_OUT" || true
    fi

    sync_to_s3 || true
  ) >/dev/null 2>&1 &
  SYNC_PID=$!
fi

echo "Started inference in background."
echo "Baselines: intern_image_allinputs_axtree, intern_image_allinputs_axtree_cot"
echo "PID: $PID"
if [[ -n "${SYNC_PID:-}" ]]; then
  echo "S3 sync watcher PID: $SYNC_PID"
fi
echo "Log: $RUN_LOG"
echo "Outputs: $OUTPUT_DIR"
echo ""
echo "Monitor with:"
echo "  tail -f $RUN_LOG"
echo "  wc -l $OUTPUT_DIR/*.jsonl"
if [[ "$ENABLE_S3_SYNC" == "1" ]]; then
  echo "  aws s3 ls $S3_ACCESS_POINT_URI/$S3_PREFIX/outputs"
fi
