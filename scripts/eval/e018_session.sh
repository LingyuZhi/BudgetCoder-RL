#!/usr/bin/env bash
# E018 scaled-M6 frozen held-out eval: start | status | attach | logs
# Eval only. Does not train, resume E017, or enter M7.
# WANDB_API_KEY must be in the environment for `start`. Never echo it.
# Does not touch tmux E014 / E015 / E017.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${E018_SESSION_NAME:-E018}"
COMPUTE_HOST="${BCRL_COMPUTE_HOST:-n30158}"
LOG_DIR="${ROOT}/outputs/experiments/E018"
LOG_FILE="${LOG_DIR}/pipeline.log"
SESSION_INFO="${LOG_DIR}/session_info.json"
CMD="${1:-status}"
TRACE_DIR="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}/trajectories/m6/E018"
E015_SCORED="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}/trajectories/m6/E015/episodes_scored.jsonl"
EPISODES="${TRACE_DIR}/episodes.jsonl"
SCORED="${TRACE_DIR}/episodes_scored.jsonl"

is_login_host() {
  local host
  host="$(hostname)"
  host="${host,,}"
  [[ "${host}" == sn* || "${host}" == *login* ]]
}

maybe_forward() {
  if [[ "${E018_ON_COMPUTE:-0}" == "1" ]]; then
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
       export E018_ON_COMPUTE=1 BCRL_DATA_ROOT=$(printf '%q' "${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}") \
       ${quoted_key} && bash scripts/eval/e018_session.sh $(printf '%q' "${CMD}")"
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
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "HARD FAIL: WANDB_API_KEY is not set" >&2
    exit 1
  fi
  cd "${ROOT}"
  mkdir -p "${LOG_DIR}" "${TRACE_DIR}"
  echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} pwd=$(pwd) python=$(which python) slurm=${SLURM_JOB_ID:-<unset>}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

  python -m pytest tests/test_e018.py tests/test_m6.py -q
  echo "e018_pytest_exit=$?"

  python scripts/eval/e011_gpu_sampler.py --output "${LOG_DIR}/gpu_sampler.jsonl" --interval 30 &
  local sampler_pid=$!
  echo "gpu_sampler_pid=${sampler_pid}"
  trap 'kill '"${sampler_pid}"' 2>/dev/null || true' EXIT

  python scripts/eval/run_e018_eval.py --phase smoke --max-hours 2
  local smoke_rc=$?
  echo "e018_smoke_exit=${smoke_rc}"
  if [[ "${smoke_rc}" -ne 0 ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    echo "HARD FAIL: smoke/treatment-integrity failed; not starting 244-task matrix"
    exit "${smoke_rc}"
  fi
  python - <<PY
import json
from pathlib import Path
status = json.loads(Path("${LOG_DIR}/run_status_smoke.json").read_text())
if status.get("status") != "PASS":
    raise SystemExit(f"HARD FAIL: smoke status {status}")
integrity = Path("${LOG_DIR}/treatment_integrity.json")
if integrity.is_file():
    payload = json.loads(integrity.read_text())
    if payload.get("pass") is False:
        raise SystemExit(f"HARD FAIL: treatment integrity {payload.get('errors')}")
print("smoke_ok", status.get("n_completed"), "reuse_base", status.get("reuse_base"))
PY

  python - <<'PY'
import json
from pathlib import Path
audit = {}
path = Path("outputs/experiments/E018/reuse_audit.json")
if path.is_file():
    audit = json.loads(path.read_text())
print("reuse_decision", audit.get("decision"), "allow", audit.get("allow_reuse"))
Path("/tmp/e018_reuse_flag").write_text("1" if audit.get("allow_reuse") else "0")
PY
  local reuse_flag
  reuse_flag="$(cat /tmp/e018_reuse_flag)"

  if [[ "${reuse_flag}" != "1" ]]; then
    python scripts/eval/run_e018_eval.py --phase base --max-hours 8
    local base_rc=$?
    echo "e018_base_exit=${base_rc}"
    if [[ "${base_rc}" -ne 0 ]]; then
      kill "${sampler_pid}" 2>/dev/null || true
      exit "${base_rc}"
    fi
  else
    echo "e018_base_skipped reuse_e015_b0_b1"
  fi

  python scripts/eval/run_e018_eval.py --phase rl --max-hours 8
  local rl_rc=$?
  echo "e018_rl_exit=${rl_rc}"
  if [[ "${rl_rc}" -ne 0 ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    exit "${rl_rc}"
  fi

  python scripts/eval/score_episodes.py \
    --episodes "${EPISODES}" \
    --output "${SCORED}" \
    --summary "${LOG_DIR}/score_summary.json"

  local summarize_args=(--episodes "${SCORED}" --output-dir "${LOG_DIR}")
  if [[ -f "${E015_SCORED}" ]]; then
    summarize_args+=(--prototype-episodes "${E015_SCORED}")
    if [[ "${reuse_flag}" == "1" ]]; then
      summarize_args+=(--reuse-base-episodes "${E015_SCORED}")
    fi
  fi
  python scripts/eval/summarize_e018.py "${summarize_args[@]}"
  python scripts/eval/select_e018_cases.py --episodes "${SCORED}" --output "${LOG_DIR}/e018_cases.json"

  kill "${sampler_pid}" 2>/dev/null || true
  echo "e018_pipeline_done"
  echo "DO_NOT_ENTER_M7"
  exit 0
}

cmd_start() {
  mkdir -p "${LOG_DIR}"
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "HARD FAIL: WANDB_API_KEY is not set" >&2
    exit 1
  fi
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
      E018_ON_COMPUTE=1 \
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
      WANDB_API_KEY="${WANDB_API_KEY:-}" \
      bash -lc "cd $(printf '%q' "${ROOT}") && exec > >(tee -a $(printf '%q' "${LOG_FILE}")) 2>&1 && bash scripts/eval/e018_session.sh _inner"
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
  echo "E018 started: tmux session=${SESSION} pane_pid=${pane_pid} log=${LOG_FILE}"
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
