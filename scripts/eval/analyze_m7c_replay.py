#!/usr/bin/env python
"""Score-aware M7C replay analysis. Writes aggregate/taxonomy/SUMMARY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.m4a import load_json  # noqa: E402
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m7c import (  # noqa: E402
    SCHEMA_VERSION,
    analyze_replay,
    default_m7c_output_dir,
    default_trace_dir,
    forbidden_output_dir_errors,
    iter_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_m7c_output_dir(repo_root)
    blocked = forbidden_output_dir_errors(output_dir, repo_root)
    if blocked:
        print(f"HARD FAIL: {blocked}", file=sys.stderr)
        return 1
    episodes = args.episodes
    if episodes is None:
        scored = default_trace_dir(bcrl_data_root()) / "episodes_scored.jsonl"
        raw = default_trace_dir(bcrl_data_root()) / "episodes.jsonl"
        episodes = scored if scored.is_file() else raw
    if not Path(episodes).is_file():
        print(f"HARD FAIL: missing episodes {episodes}", file=sys.stderr)
        return 1
    audit_path = args.audit or (output_dir / "prompt_path_audit.json")
    contract_path = args.contract or (output_dir / "execution_contract.json")
    if not Path(audit_path).is_file() or not Path(contract_path).is_file():
        print("HARD FAIL: run analyze_m7c_prompt_path.py first", file=sys.stderr)
        return 1
    audit = load_json(Path(audit_path))
    contract = load_json(Path(contract_path))
    rows = iter_jsonl(Path(episodes))
    payload = analyze_replay(rows, audit=audit, contract=contract)
    write_json(output_dir / "aggregate.json", {
        "schema_version": SCHEMA_VERSION,
        "train": payload["train"],
        "dev": payload["dev"],
        "comparison": payload["comparison"],
        "decision": payload["decision"],
        "q1": payload["q1"],
        "q2": payload["q2"],
        "n_rows": len(rows),
        "episodes_path": str(episodes),
    })
    write_json(output_dir / "taxonomy.json", payload["taxonomy"])
    (output_dir / "SUMMARY.md").write_text(payload["summary_markdown"], encoding="utf-8")
    print(payload["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
