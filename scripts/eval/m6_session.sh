#!/usr/bin/env bash
# E015 / M6 frozen held-out-task eval: start | status | attach | logs
# Orchestrates 2-task smoke then 244x9 matrix on 2xA100. No W&B required.
# Does not touch tmux E011 / E012 / E013 / E014.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${E015_SESSION_NAME:-E015}"
COMPUTE_HOST="${BCRL_COMPUTE_HOST:-n30158}"
LOG_DIR="${ROOT}/outputs/experiments/E015"
LOG_FILE="${LOG_DIR}/pipeline.log"
SESSION_INFO="${LOG_DIR}/session_info.json"
CMD="${1:-status}"
TRACE_DIR="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}/trajectories/m6/E015"
EPISODES="${TRACE_DIR}/episodes.jsonl"
SCORED="${TRACE_DIR}/episodes_scored.jsonl"

is_login_host() {
  local host
  host="$(hostname)"
  host="${host,,}"
  [[ "${host}" == sn* || "${host}" == *login* ]]
}

maybe_forward() {
  if [[ "${E015_ON_COMPUTE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${CMD}" == "_inner" ]]; then
    return 0
  fi
  if is_login_host; then
    exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${COMPUTE_HOST}" \
      "cd $(printf '%q' "${ROOT}") && \
       export E015_ON_COMPUTE=1 BCRL_DATA_ROOT=$(printf '%q' "${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}") && \
       bash scripts/eval/m6_session.sh $(printf '%q' "${CMD}")"
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
  export PYTHONUNBUFFERED=1
  cd "${ROOT}"
  mkdir -p "${LOG_DIR}" "${TRACE_DIR}"
  echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} pwd=$(pwd) python=$(which python) slurm=${SLURM_JOB_ID:-<unset>}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

  python - <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path("src").resolve()))
from budget_coder_rl.eval.m6 import inspect_tokenizer_warning, write_json
payload = inspect_tokenizer_warning(Path(".").resolve())
out = Path("outputs/experiments/E015/tokenizer_warning.json")
write_json(out, payload)
print(json.dumps({"tokenizer_warning": str(out), "n_log_hits": payload.get("n_log_hits"), "tito": payload.get("tito_correctness_bug")}))
PY

  python scripts/eval/e011_gpu_sampler.py --output "${LOG_DIR}/gpu_sampler.jsonl" --interval 30 &
  local sampler_pid=$!
  echo "gpu_sampler_pid=${sampler_pid}"
  trap 'kill '"${sampler_pid}"' 2>/dev/null || true' EXIT

  python scripts/eval/run_m6_eval.py --phase smoke --max-hours 2
  local smoke_rc=$?
  echo "e015_smoke_exit=${smoke_rc}"
  if [[ "${smoke_rc}" -ne 0 ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    exit "${smoke_rc}"
  fi
  python - <<PY
import json
from pathlib import Path
status = json.loads(Path("${LOG_DIR}/run_status_smoke.json").read_text())
if status.get("status") != "PASS":
    raise SystemExit(f"HARD FAIL: smoke status {status}")
n = int(status.get("n_completed") or 0)
# smoke writes 6 cells (2 tasks x 3 conditions x 4096), resume may already hold more
if n < 6:
    raise SystemExit(f"HARD FAIL: smoke completed {n} < 6")
print("smoke_ok", n)
PY

  python scripts/eval/run_m6_eval.py --phase base --max-hours 8
  local base_rc=$?
  echo "e015_base_exit=${base_rc}"
  if [[ "${base_rc}" -ne 0 ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    exit "${base_rc}"
  fi

  python scripts/eval/run_m6_eval.py --phase rl --max-hours 8
  local rl_rc=$?
  echo "e015_rl_exit=${rl_rc}"
  if [[ "${rl_rc}" -ne 0 ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    exit "${rl_rc}"
  fi

  python scripts/eval/score_episodes.py \
    --episodes "${EPISODES}" \
    --output "${SCORED}" \
    --summary "${LOG_DIR}/score_summary.json"
  python scripts/eval/summarize_m6.py --episodes "${SCORED}" --output-dir "${LOG_DIR}"
  python scripts/eval/select_m6_cases.py --episodes "${SCORED}" --output "${LOG_DIR}/m6_cases.json"

  kill "${sampler_pid}" 2>/dev/null || true
  echo "e015_pipeline_done"
  exit 0
}

cmd_start() {
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
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
    env \
      BCRL_DATA_ROOT="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}" \
      http_proxy="${http_proxy:-http://10.36.204.1:3128}" \
      https_proxy="${https_proxy:-http://10.36.204.1:3128}" \
      ftp_proxy="${ftp_proxy:-http://10.36.204.1:3128}" \
      E015_ON_COMPUTE=1 \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
      bash -lc "cd $(printf '%q' "${ROOT}") && exec > >(tee -a $(printf '%q' "${LOG_FILE}")) 2>&1 && bash scripts/eval/m6_session.sh _inner"
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
  echo "E015 started: tmux session=${SESSION} pane_pid=${pane_pid} log=${LOG_FILE}"
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
    python3 -c "import json; p=json.load(open('${LOG_DIR}/run_status.json')); print('run_status', p.get('status'), p.get('phase'), p.get('stop_reason'), 'written', p.get('n_written'), 'completed', p.get('n_completed'))"
  else
    echo "run_status.json not yet written"
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
