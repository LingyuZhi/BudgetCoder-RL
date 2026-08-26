#!/usr/bin/env python
"""Write M6 RL-win / Base-win / both-fail case packet from scored episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m6 import (  # noqa: E402
    attach_case_trajectories,
    default_e015_output_dir,
    paired_cells,
    select_case_studies,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [
        row
        for row in load_jsonl(args.episodes)
        if row.get("termination") != "operational_error"
    ]
    pairs = paired_cells(rows, left_id="B1", right_id="M1")
    selected = select_case_studies(pairs)
    payload = attach_case_trajectories(selected, pairs)
    payload["note"] = (
        "Programmatic ranking by paired localization delta, then frozen "
        "cross-repo / interpretability rules. Not hand-picked pretty examples."
    )
    output = args.output
    if output is None:
        output = default_e015_output_dir(args.repo_root.resolve()) / "m6_cases.json"
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
