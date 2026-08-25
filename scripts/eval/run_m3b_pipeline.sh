#!/usr/bin/env bash
# Overnight M3B pipeline. Run inside tmux on n30158 after activating conda env verl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP="${BCRL_EXPERIMENT_ID:-E001}"
OUT="$ROOT/outputs/experiments/${EXP}"
DATA_ROOT="${BCRL_DATA_ROOT:-$HOME/my_data/budget-coder-rl}"
TRACE="$DATA_ROOT/trajectories/m3b/${EXP}"
LOG="$OUT/pipeline.log"
mkdir -p "$OUT" "$TRACE"

exec > >(tee -a "$LOG") 2>&1
echo "=== M3B pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ==="
echo "ROOT=$ROOT EXP=$EXP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python scripts/data/build_m3b_task_manifest.py
python scripts/eval/run_m3b_baseline.py --experiment-id "$EXP" --output-dir "$OUT"
python scripts/eval/score_episodes.py \
  --episodes "$TRACE/episodes.jsonl" \
  --output "$OUT/episodes_scored.jsonl" \
  --summary "$OUT/score_summary.json"
python scripts/eval/summarize_m3b.py \
  --episodes "$OUT/episodes_scored.jsonl" \
  --output-dir "$OUT" \
  --run-status "$OUT/run_status.json"
python scripts/eval/select_m3b_review_cases.py \
  --episodes "$OUT/episodes_scored.jsonl" \
  --output "$OUT/m3b_review_cases.json"

echo "=== M3B pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "SUMMARY=$OUT/SUMMARY.md"
echo "TRACE=$TRACE/episodes.jsonl"
