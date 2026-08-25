#!/usr/bin/env python
"""CPU budget-quantile report from frozen E001 scored episodes.

Does not start GPU, Ray, GRPO, or RewardLoop.

Usage:

    python scripts/eval/calibrate_m3c_budgets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m3c import (  # noqa: E402
    CALIBRATION_GPU_BUDGETS,
    CANDIDATE_BUDGETS,
    e001_budget_quantile_report,
    load_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments" / "E001" / "episodes_scored.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments" / "E006" / "e001_budget_quantiles.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.episodes.is_file():
        print(f"HARD FAIL: missing E001 scored episodes {args.episodes}", file=sys.stderr)
        return 1
    rows = load_jsonl(args.episodes)
    report = e001_budget_quantile_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    visible = report["visible"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_visible": visible.get("n"),
                "quantiles": visible.get("quantiles"),
                "frac_ge": visible.get("frac_ge"),
                "candidates": list(CANDIDATE_BUDGETS),
                "gpu_calibration_budgets": list(CALIBRATION_GPU_BUDGETS),
                "n_exhausted": len(report["exhausted"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
