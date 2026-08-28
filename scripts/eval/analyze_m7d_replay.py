#!/usr/bin/env python
"""Score-aware M7D replay analysis. Writes aggregates/taxonomy/SUMMARY."""

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
from budget_coder_rl.eval.m7d import (  # noqa: E402
    SCHEMA_VERSION,
    analyze_replay,
    default_m7d_output_dir,
    default_trace_dir,
    forbidden_output_dir_errors,
    iter_jsonl,
    render_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, default=None)
    parser.add_argument("--first-requests", type=Path, default=None)
    parser.add_argument("--first-outputs", type=Path, default=None)
    parser.add_argument("--cpu-audit", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_m7d_output_dir(repo_root)
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
    cpu_audit_path = args.cpu_audit or (output_dir / "first_request_cpu.json")
    cpu_audit = load_json(Path(cpu_audit_path)) if Path(cpu_audit_path).is_file() else {}
    request_cpu = output_dir / "first_request_audit.jsonl"
    request_gpu = output_dir / "first_request_audit_gpu.jsonl"
    first_requests = list(iter_jsonl(request_cpu))
    if args.first_requests:
        first_requests = iter_jsonl(Path(args.first_requests))
    if Path(request_gpu).is_file():
        first_requests.extend(iter_jsonl(request_gpu))
    first_outputs_path = args.first_outputs or (output_dir / "first_generation_outputs.jsonl")
    first_outputs = iter_jsonl(Path(first_outputs_path)) if Path(first_outputs_path).is_file() else []
    rows = iter_jsonl(Path(episodes))
    payload = analyze_replay(
        rows,
        first_requests=first_requests,
        first_outputs=first_outputs or None,
        aliasing=cpu_audit.get("aliasing"),
    )
    payload["trajectory_path"] = str(episodes)
    write_json(
        output_dir / "aggregates.json",
        {
            "schema_version": SCHEMA_VERSION,
            "cells": payload["cells"],
            "decision": payload["decision"],
            "n_episodes": payload["n_episodes"],
            "n_first_requests": payload["n_first_requests"],
            "episodes_path": str(episodes),
        },
    )
    write_json(output_dir / "taxonomy.json", payload["taxonomy"])
    if payload.get("first_divergence"):
        write_json(output_dir / "first_divergence.json", payload["first_divergence"])
    (output_dir / "SUMMARY.md").write_text(render_summary(payload), encoding="utf-8")
    print(payload["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
