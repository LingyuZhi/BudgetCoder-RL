#!/usr/bin/env bash
# Launch M4C GPU adapter persist/reload smoke inside tmux on n30158.
set -uo pipefail
source /etc/profile.d/modules.sh
module load Miniconda3/latest
module load cuda/12.8
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/.conda")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate verl
python -c "import pandas,verl; print('env_ok', pandas.__version__, verl.__version__)"
export http_proxy=http://10.36.204.1:3128
export https_proxy=http://10.36.204.1:3128
export ftp_proxy=http://10.36.204.1:3128
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
export BCRL_DATA_ROOT="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=true
cd "$(dirname "$0")/../.."
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES pwd=$(pwd) python=$(which python)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
LOG_DIR="outputs/experiments/E009"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1
python scripts/eval/run_m4c_gpu.py --experiment-id E009 --skip-gpu-pick
echo "m4c_exit=$?"
exec bash
