#!/usr/bin/env python
"""Deterministic stratified review-case packet for M3B trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m3b import REVIEW_SEED, select_review_cases  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=REVIEW_SEED)
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
    cases = select_review_cases(rows, n_target=args.n, seed=args.seed)
    n_expl = sum(
        1
        for case in cases
        if (case.get("taxonomy") or {}).get("failure_class") == "exploration_policy"
    )
    n_know = sum(
        1
        for case in cases
        if (case.get("taxonomy") or {}).get("failure_class") == "coding_knowledge"
    )
    payload = {
        "n_cases": len(cases),
        "seed": args.seed,
        "n_exploration_policy": n_expl,
        "n_coding_knowledge": n_know,
        "dominant_failure_class": (
            "exploration_policy"
            if n_expl >= n_know
            else "coding_knowledge"
            if n_know
            else "unknown"
        ),
        "cases": cases,
        "note": (
            "First-pass deterministic taxonomy. Human review should confirm "
            "labels. Do not change prompt/tools to chase these cases."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_cases": payload["n_cases"],
                "n_exploration_policy": n_expl,
                "n_coding_knowledge": n_know,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
