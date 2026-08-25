#!/usr/bin/env python
"""Offline M3C behavior vs within-group reward. Does not touch AgentLoop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m3c import (  # noqa: E402
    GROUP_N,
    behavior_reward_table,
    grouped_rows,
    load_jsonl,
    select_representative_groups,
    within_group_behavior_contrast,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-n", type=int, default=GROUP_N)
    parser.add_argument("--n-cases", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [
        row
        for row in load_jsonl(args.episodes)
        if row.get("termination") != "operational_error"
    ]
    if not rows:
        print(f"HARD FAIL: no episodes in {args.episodes}", file=sys.stderr)
        return 1
    groups = grouped_rows(rows, group_n=args.group_n)
    table = behavior_reward_table(rows)
    contrast = within_group_behavior_contrast(groups, rows)
    cases = select_representative_groups(groups, rows, n_target=args.n_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_episodes": len(rows),
        "n_groups": len(groups),
        "behavior_vs_reward": table,
        "within_group_contrast": contrast,
        "n_representative_groups": len(cases),
        "note": (
            "Offline analysis of existing JSONL. Do not change prompt/tools "
            "from these cases. AgentLoop hot path is unchanged."
        ),
    }
    (args.output_dir / "m3c_behavior.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "m3c_representative_groups.json").write_text(
        json.dumps({"n_cases": len(cases), "cases": cases}, indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_episodes": len(rows),
                "n_groups": len(groups),
                "n_representative_groups": len(cases),
                "within_group_contrast": contrast,
                "output": str(args.output_dir / "m3c_behavior.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
