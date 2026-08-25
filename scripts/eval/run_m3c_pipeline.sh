#!/usr/bin/env bash
# M3C pipeline. CPU stages are safe anywhere; GPU stages require n30158 tmux.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAGE="${BCRL_M3C_STAGE:-all}"
DATA_ROOT="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}"
E006_OUT="$ROOT/outputs/experiments/E006"
E007_OUT="$ROOT/outputs/experiments/E007"
E007B_OUT="$ROOT/outputs/experiments/E007b"
E001_SCORED="$ROOT/outputs/experiments/E001/episodes_scored.jsonl"
E006_TRACE="$DATA_ROOT/trajectories/m3c/E006"
E007_TRACE="$DATA_ROOT/trajectories/m3c/E007"
E007B_TRACE="$DATA_ROOT/trajectories/m3c/E007b"

mkdir -p "$E006_OUT" "$E007_OUT"

cpu_stage() {
  python scripts/eval/calibrate_m3c_budgets.py \
    --episodes "$E001_SCORED" \
    --output "$E006_OUT/e001_budget_quantiles.json"
  python scripts/data/build_m3c_diagnostic_manifest.py
}

e006_stage() {
  python scripts/eval/run_m3c_gpu.py --experiment-id E006 --mode calibration --output-dir "$E006_OUT"
  python scripts/eval/score_episodes.py \
    --episodes "$E006_TRACE/episodes.jsonl" \
    --output "$E006_OUT/episodes_scored.jsonl" \
    --summary "$E006_OUT/score_summary.json"
  python scripts/eval/summarize_m3c.py \
    --mode calibration \
    --episodes "$E006_OUT/episodes_scored.jsonl" \
    --e001-visible "$E001_SCORED" \
    --output-dir "$E006_OUT" \
    --run-status "$E006_OUT/run_status.json"
}

read_primary_budget() {
  python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/experiments/E006/e006_regimes.json")
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get("primary_training_B_obs")
if value is None:
    raise SystemExit("HARD FAIL: primary_training_B_obs missing")
print(int(value))
PY
}

e007_stage() {
  local budget
  budget="$(read_primary_budget)"
  python scripts/eval/run_m3c_gpu.py \
    --experiment-id E007 \
    --mode grouped \
    --obs-tokens-limit "$budget" \
    --output-dir "$E007_OUT"
  python scripts/eval/score_episodes.py \
    --episodes "$E007_TRACE/episodes.jsonl" \
    --output "$E007_OUT/episodes_scored.jsonl" \
    --summary "$E007_OUT/score_summary.json"
  python scripts/eval/summarize_m3c.py \
    --mode grouped \
    --episodes "$E007_OUT/episodes_scored.jsonl" \
    --output-dir "$E007_OUT" \
    --run-status "$E007_OUT/run_status.json"
  python scripts/eval/analyze_m3c_behavior.py \
    --episodes "$E007_OUT/episodes_scored.jsonl" \
    --output-dir "$E007_OUT"
}

e007b_stage() {
  local budget ids
  budget="$(read_primary_budget)"
  ids="$(python - <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
from budget_coder_rl.eval.m3c import n8_probe_ids
payload = json.loads(Path("outputs/experiments/E007/m3c_groups.json").read_text(encoding="utf-8"))
if not (payload.get("group_summary") or {}).get("needs_n8_probe"):
    print("")
else:
    print(",".join(n8_probe_ids(payload.get("groups") or [], n_target=16)))
PY
)"
  if [[ -z "$ids" ]]; then
    echo "E007b skipped: n=8 probe not needed"
    return 0
  fi
  mkdir -p "$E007B_OUT"
  python scripts/eval/run_m3c_gpu.py \
    --experiment-id E007b \
    --mode grouped \
    --obs-tokens-limit "$budget" \
    --group-n 8 \
    --instance-ids "$ids" \
    --output-dir "$E007B_OUT"
  python scripts/eval/score_episodes.py \
    --episodes "$E007B_TRACE/episodes.jsonl" \
    --output "$E007B_OUT/episodes_scored.jsonl" \
    --summary "$E007B_OUT/score_summary.json"
  python scripts/eval/summarize_m3c.py \
    --mode grouped \
    --episodes "$E007B_OUT/episodes_scored.jsonl" \
    --output-dir "$E007B_OUT" \
    --group-n 8 \
    --run-status "$E007B_OUT/run_status.json"
}

offline_stage() {
  python scripts/data/build_m3c_train_candidates.py
  python scripts/eval/write_m3c_freeze.py
}

echo "=== M3C pipeline stage=$STAGE $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ==="
case "$STAGE" in
  cpu) cpu_stage ;;
  e006) e006_stage ;;
  e007) e007_stage ;;
  e007b) e007b_stage ;;
  offline) offline_stage ;;
  gpu)
    e006_stage
    e007_stage
    e007b_stage
    offline_stage
    ;;
  all)
    cpu_stage
    e006_stage
    e007_stage
    e007b_stage
    offline_stage
    ;;
  *)
    echo "HARD FAIL: unknown stage $STAGE" >&2
    exit 1
    ;;
esac
echo "=== M3C pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
