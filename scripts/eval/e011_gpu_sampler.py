#!/usr/bin/env python
"""Append nvidia-smi samples for GPU0/GPU1 until killed."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m5b import OUTPUT_ENV, append_gpu_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    output = args.output
    if output is None:
        raw = os.environ.get(OUTPUT_ENV)
        if not raw:
            print(f"HARD FAIL: {OUTPUT_ENV} is not set", file=sys.stderr)
            return 1
        output = Path(raw) / "gpu_sampler.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            append_gpu_sample(output)
        except Exception as exc:
            sys.stderr.write(f"gpu_sampler: {exc}\n")
        time.sleep(max(5.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
