#!/usr/bin/env python
"""Post-rollout localization scoring for bcrl-episode-v1 JSONL.

Does not run AgentLoop, GRPO, or veRL RewardLoop.

Usage:

    python scripts/eval/score_episodes.py \\
        --episodes outputs/smoke/m3a_episodes.jsonl \\
        --oracle data/processed/swe_gym/evaluator_oracle.parquet \\
        --output outputs/smoke/m3a_episodes_scored.jsonl \\
        --summary outputs/smoke/m3a_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_materialize import oracle_parquet_path  # noqa: E402
from budget_coder_rl.eval.episode import summarize_episodes  # noqa: E402
from budget_coder_rl.eval.localization import evaluate_episode  # noqa: E402
from budget_coder_rl.eval.oracle import load_evaluator_oracle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    oracle_path = (
        args.oracle.resolve()
        if args.oracle is not None
        else oracle_parquet_path(args.repo_root.resolve())
    )
    if not args.episodes.is_file():
        print(f"HARD FAIL: missing episodes {args.episodes}", file=sys.stderr)
        return 1
    if not oracle_path.is_file():
        print(f"HARD FAIL: missing oracle {oracle_path}", file=sys.stderr)
        return 1
    index = load_evaluator_oracle(oracle_path)
    rows: list[dict[str, Any]] = []
    with args.episodes.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            identity = record.get("identity") or {}
            instance_id = identity.get("instance_id") or record.get("instance_id")
            oracle = index.get(str(instance_id))
            metrics = evaluate_episode(
                termination=record.get("termination"),
                submission=record.get("final_submission"),
                oracle=oracle,
            )
            record["localization"] = metrics.as_dict()
            rows.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = summarize_episodes(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"scored={args.output}")
    print(f"summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
