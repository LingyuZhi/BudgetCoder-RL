#!/usr/bin/env bash
# M7C matched Base replay: start | status | attach | logs
# Diagnostic only. Does not train, modify E017/E018, or enter intervention.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${M7C_SESSION_NAME:-M7C}"
COMPUTE_HOST="${BCRL_COMPUTE_HOST:-n30158}"
LOG_DIR="${ROOT}/outputs/experiments/M7C"
LOG_FILE="${LOG_DIR}/pipeline.log"
SESSION_INFO="${LOG_DIR}/session_info.json"
CMD="${1:-status}"
TRACE_DIR="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}/trajectories/m7c/M7C"
EPISODES="${TRACE_DIR}/episodes.jsonl"
SCORED="${TRACE_DIR}/episodes_scored.jsonl"

is_login_host() {
  local host
  host="$(hostname)"
  host="${host,,}"
  [[ "${host}" == sn* || "${host}" == *login* ]]
}

maybe_forward() {
  if [[ "${M7C_ON_COMPUTE:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${CMD}" == "_inner" || "${CMD}" == "_cpu" ]]; then
    return 0
  fi
  if is_login_host; then
    exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${COMPUTE_HOST}" \
      "cd $(printf '%q' "${ROOT}") && \
       export M7C_ON_COMPUTE=1 BCRL_DATA_ROOT=$(printf '%q' "${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}") && \
       bash scripts/eval/m7c_session.sh $(printf '%q' "${CMD}")"
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

activate_env() {
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
}

inner_start() {
  activate_env
  cd "${ROOT}"
  mkdir -p "${LOG_DIR}" "${TRACE_DIR}"
  echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} pwd=$(pwd) python=$(which python) slurm=${SLURM_JOB_ID:-<unset>}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

  python -m pytest tests/test_m7c.py tests/test_m7a.py tests/test_m7b.py -q
  echo "m7c_pytest_exit=$?"

  python scripts/eval/analyze_m7c_prompt_path.py --output-dir "${LOG_DIR}"
  local audit_rc=$?
  echo "m7c_audit_exit=${audit_rc}"
  python - <<PY
import json
from pathlib import Path
gate = json.loads(Path("${LOG_DIR}/audit_gate.json").read_text())
if not gate.get("allow_replay"):
    raise SystemExit(f"HARD FAIL: prompt-path confound {gate.get('confound_reasons')}")
print("allow_replay", gate.get("allow_replay"))
PY

  python scripts/eval/e011_gpu_sampler.py --output "${LOG_DIR}/gpu_sampler.jsonl" --interval 30 &
  local sampler_pid=$!
  echo "gpu_sampler_pid=${sampler_pid}"
  trap 'kill '"${sampler_pid}"' 2>/dev/null || true' EXIT

  python scripts/eval/run_m7c_replay.py --max-hours 6
  local replay_rc=$?
  echo "m7c_replay_exit=${replay_rc}"
  if [[ "${replay_rc}" -ne 0 ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    exit "${replay_rc}"
  fi

  python scripts/eval/score_episodes.py \
    --episodes "${EPISODES}" \
    --output "${SCORED}" \
    --summary "${LOG_DIR}/score_summary.json"

  python scripts/eval/analyze_m7c_replay.py \
    --episodes "${SCORED}" \
    --output-dir "${LOG_DIR}"

  ln -sfn "${EPISODES}" "${LOG_DIR}/episodes.jsonl"
  ln -sfn "${SCORED}" "${LOG_DIR}/episodes_scored.jsonl"

  kill "${sampler_pid}" 2>/dev/null || true
  echo "m7c_pipeline_done"
  echo "DO_NOT_ENTER_INTERVENTION"
  exit 0
}

cmd_cpu() {
  activate_env
  cd "${ROOT}"
  mkdir -p "${LOG_DIR}"
  python -m pytest tests/test_m7c.py -q
  python scripts/eval/analyze_m7c_prompt_path.py --output-dir "${LOG_DIR}"
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
      M7C_ON_COMPUTE=1 \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
      bash -lc "cd $(printf '%q' "${ROOT}") && exec > >(tee -a $(printf '%q' "${LOG_FILE}")) 2>&1 && bash scripts/eval/m7c_session.sh _inner"
  sleep 2
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "HARD FAIL: tmux session ${SESSION} vanished after start" >&2
    tail -n 50 "${LOG_FILE}" || true
    exit 1
  fi
  echo "M7C started: tmux session=${SESSION} log=${LOG_FILE}"
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
    python3 -c "import json; p=json.load(open('${LOG_DIR}/run_status.json')); print('run_status', p.get('status'), p.get('stop_reason'), 'written', p.get('n_written'))"
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
if [[ "${CMD}" == "_cpu" ]]; then
  cmd_cpu
  exit 0
fi
maybe_forward
case "${CMD}" in
  start) cmd_start ;;
  status) cmd_status ;;
  attach) cmd_attach ;;
  logs) cmd_logs ;;
  cpu) cmd_cpu ;;
  *)
    echo "usage: $0 {start|status|attach|logs|cpu}" >&2
    exit 2
    ;;
esac
