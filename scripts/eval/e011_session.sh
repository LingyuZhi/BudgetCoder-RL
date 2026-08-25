#!/usr/bin/env bash
# E011 SSH-independent session: start | status | attach | logs
# WANDB_API_KEY must be in the environment for `start`. Never echo it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${E011_SESSION_NAME:-E011}"
COMPUTE_HOST="${BCRL_COMPUTE_HOST:-n30158}"
LOG_DIR="${ROOT}/outputs/experiments/E011"
LOG_FILE="${LOG_DIR}/pipeline.log"
SESSION_INFO="${LOG_DIR}/session_info.json"
CMD="${1:-status}"

is_login_host() {
  local host
  host="$(hostname)"
  host="${host,,}"
  [[ "${host}" == sn* || "${host}" == *login* ]]
}

ssh_to_compute() {
  local remote_cmd="$1"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${COMPUTE_HOST}" \
    "cd $(printf '%q' "${ROOT}") && ${remote_cmd}"
}

maybe_forward() {
  if [[ "${E011_ON_COMPUTE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${CMD}" == "_inner" ]]; then
    return 0
  fi
  if is_login_host; then
    local quoted_key=""
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
      quoted_key="WANDB_API_KEY=$(printf '%q' "${WANDB_API_KEY}")"
    fi
    exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${COMPUTE_HOST}" \
      "cd $(printf '%q' "${ROOT}") && \
       export E011_ON_COMPUTE=1 BCRL_DATA_ROOT=$(printf '%q' "${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}") \
       ${quoted_key} && bash scripts/eval/e011_session.sh $(printf '%q' "${CMD}")"
  fi
}

write_session_info() {
  python3 - "$@" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
payload["hostname"] = socket.gethostname()
payload["timestamp"] = datetime.now(timezone.utc).isoformat()
payload["pid"] = os.getpid()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

inner_start() {
  source /etc/profile.d/modules.sh
  module load Miniconda3/latest
  module load cuda/12.8
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || echo "${HOME}/.conda")"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate verl
  export http_proxy="${http_proxy:-http://10.36.204.1:3128}"
  export https_proxy="${https_proxy:-http://10.36.204.1:3128}"
  export ftp_proxy="${ftp_proxy:-http://10.36.204.1:3128}"
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export BCRL_DATA_ROOT="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  export TOKENIZERS_PARALLELISM=true
  export BCRL_M5_OUTPUT_DIR="${LOG_DIR}"
  export WANDB_DIR="${LOG_DIR}/wandb"
  export WANDB_PROJECT=budget-coder-rl
  cd "${ROOT}"
  mkdir -p "${LOG_DIR}"
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "HARD FAIL: WANDB_API_KEY is not set" >&2
    exit 1
  fi
  echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} pwd=$(pwd) python=$(which python) slurm=${SLURM_JOB_ID:-<unset>}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  python scripts/eval/e011_gpu_sampler.py --output "${LOG_DIR}/gpu_sampler.jsonl" --interval 30 &
  local sampler_pid=$!
  echo "gpu_sampler_pid=${sampler_pid}"
  trap 'kill '"${sampler_pid}"' 2>/dev/null || true' EXIT
  python scripts/eval/run_m5b_preflight.py --experiment-id E011
  python scripts/eval/run_m5b_main.py --experiment-id E011
  local rc=$?
  echo "e011_exit=${rc}"
  kill "${sampler_pid}" 2>/dev/null || true
  exec bash
}

cmd_start() {
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "HARD FAIL: WANDB_API_KEY is not set" >&2
    exit 1
  fi
  mkdir -p "${LOG_DIR}"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "HARD FAIL: tmux session ${SESSION} already exists" >&2
    tmux ls
    exit 1
  fi
  local mechanism=tmux
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "NOTE: SLURM_JOB_ID=${SLURM_JOB_ID}; using tmux inside this allocation"
    mechanism=tmux_in_slurm
  fi
  write_session_info "${SESSION_INFO}" "{\"session\":\"${SESSION}\",\"mechanism\":\"${mechanism}\",\"log_file\":\"${LOG_FILE}\"}"
  # Inherit secrets via env(1) on the new pane process, not via tmux server globals / files.
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
    env \
      WANDB_API_KEY="${WANDB_API_KEY}" \
      BCRL_DATA_ROOT="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}" \
      http_proxy="${http_proxy:-http://10.36.204.1:3128}" \
      https_proxy="${https_proxy:-http://10.36.204.1:3128}" \
      ftp_proxy="${ftp_proxy:-http://10.36.204.1:3128}" \
      E011_ON_COMPUTE=1 \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
      bash -lc "cd $(printf '%q' "${ROOT}") && exec > >(tee -a $(printf '%q' "${LOG_FILE}")) 2>&1 && bash scripts/eval/e011_session.sh _inner"
  sleep 2
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "HARD FAIL: tmux session ${SESSION} vanished after start" >&2
    tail -n 50 "${LOG_FILE}" || true
    exit 1
  fi
  local pane_pid
  pane_pid="$(tmux list-panes -t "${SESSION}" -F '#{pane_pid}' | head -n 1)"
  local comm
  comm="$(ps -o comm= -p "${pane_pid}" 2>/dev/null || true)"
  write_session_info "${SESSION_INFO}" "{\"session\":\"${SESSION}\",\"mechanism\":\"${mechanism}\",\"log_file\":\"${LOG_FILE}\",\"tmux_pane_pid\":${pane_pid:-0},\"pane_comm\":\"${comm}\",\"ssh_independent\":true}"
  echo "E011 started: tmux session=${SESSION} pane_pid=${pane_pid} log=${LOG_FILE}"
  echo "Detach does not stop training. status/logs/attach via scripts/eval/e011_session.sh"
}

cmd_status() {
  mkdir -p "${LOG_DIR}"
  echo "host=$(hostname) slurm=${SLURM_JOB_ID:-<unset>}"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux_session=${SESSION} present"
    tmux list-panes -t "${SESSION}" -F 'pane_pid=#{pane_pid} cmd=#{pane_current_command}'
  else
    echo "tmux_session=${SESSION} missing"
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
  if [[ -f "${LOG_DIR}/run_status.json" ]]; then
    python3 -c "import json; p=json.load(open('${LOG_DIR}/run_status.json')); print('run_status', p.get('status'), p.get('stop_reason'), 'steps', p.get('n_steps_completed'))"
  else
    echo "run_status.json not yet written"
  fi
  if [[ -f "${LOG_DIR}/wandb_run.json" ]]; then
    python3 -c "import json; p=json.load(open('${LOG_DIR}/wandb_run.json')); print('wandb', p.get('url'))"
  fi
  if [[ -f "${LOG_FILE}" ]]; then
    echo "---- last 20 log lines ----"
    tail -n 20 "${LOG_FILE}"
  fi
}

cmd_attach() {
  tmux attach -t "${SESSION}"
}

cmd_logs() {
  mkdir -p "${LOG_DIR}"
  touch "${LOG_FILE}"
  tail -f "${LOG_FILE}"
}

if [[ "${CMD}" == "_inner" ]]; then
  inner_start
  exit 0
fi
maybe_forward
case "${CMD}" in
  start) cmd_start ;;
  status) cmd_status ;;
  attach) cmd_attach ;;
  logs) cmd_logs ;;
  *)
    echo "usage: $0 {start|status|attach|logs}" >&2
    exit 2
    ;;
esac
