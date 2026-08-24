#!/usr/bin/env python
"""Dump the frozen Stage-1 runtime system-prompt audit (M3A).

Confirms search is a case-sensitive literal substring. Does not tune the
prompt from trajectories.

Usage:

    python scripts/data/audit_m3a_runtime_prompt.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.protocol.prompt import runtime_prompt_audit  # noqa: E402

STATS_RELPATH = "data/stats/swe_gym_m3a_runtime_prompt.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    payload = runtime_prompt_audit()
    output = args.output or (repo_root / STATS_RELPATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print(f"wrote {output}")
    if not payload["search_is_case_sensitive_literal_substring"]:
        print("HARD FAIL: search phrase missing from system prompt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
