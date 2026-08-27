#!/usr/bin/env python
"""Write E018 RL-win / Base-win / both-fail cases from scored episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.e018 import (  # noqa: E402
    attach_case_trajectories,
    default_e018_output_dir,
    load_jsonl,
    paired_cells,
    select_case_studies,
)
from budget_coder_rl.eval.m4b import write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--left-id", default="B1")
    parser.add_argument("--right-id", default="M_scaled")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [
        row
        for row in load_jsonl(args.episodes)
        if row.get("termination") != "operational_error"
    ]
    pairs = paired_cells(rows, left_id=args.left_id, right_id=args.right_id)
    selected = select_case_studies(pairs, left_id=args.left_id, right_id=args.right_id)
    payload = attach_case_trajectories(
        selected, pairs, left_id=args.left_id, right_id=args.right_id
    )
    payload["note"] = (
        "Programmatic ranking by paired localization delta on B1 vs M_scaled. "
        "Not reused E015 pretty examples unless the same rule selected them."
    )
    output = args.output
    if output is None:
        output = default_e018_output_dir(args.repo_root.resolve()) / "e018_cases.json"
    write_json(output, payload)
    print(
        json.dumps(
            {
                "n_pairs": len(pairs),
                "n_candidates": payload.get("n_candidates"),
                "selected": {
                    key: (value or {}).get("instance_id")
                    for key, value in (payload.get("selected") or {}).items()
                },
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
