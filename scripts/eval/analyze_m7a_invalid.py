#!/usr/bin/env python
"""Offline M7A invalid-action forensics. Does not touch AgentLoop or parser.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.e018 import default_trace_dir  # noqa: E402
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m5a import default_output_dir  # noqa: E402
from budget_coder_rl.eval.m7a import (  # noqa: E402
    E018_COMPARE_CONDITIONS,
    EXPERIMENT_ID,
    METRIC_SEMANTICS,
    PRIMARY_EVAL_BUDGET,
    SCHEMA_VERSION,
    analyze_corpus,
    e017_self_check,
    filter_e018_cell,
    iter_jsonl,
    mean_step_bcrl_rates,
    render_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e017-episodes", type=Path, default=None)
    parser.add_argument("--e017-step-bcrl", type=Path, default=None)
    parser.add_argument("--e018-episodes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        repo_root, EXPERIMENT_ID
    )
    e017_episodes = args.e017_episodes or (
        repo_root / "outputs" / "experiments" / "E017" / "episodes.jsonl"
    )
    e017_step = args.e017_step_bcrl or (
        repo_root / "outputs" / "experiments" / "E017" / "step_bcrl.jsonl"
    )
    e018_episodes = args.e018_episodes
    if e018_episodes is None:
        candidate = default_trace_dir(bcrl_data_root()) / "episodes_scored.jsonl"
        e018_episodes = candidate if candidate.is_file() else None

    if not e017_episodes.is_file():
        print(f"HARD FAIL: missing {e017_episodes}", file=sys.stderr)
        return 1

    e017 = analyze_corpus(iter_jsonl(e017_episodes), source="E017")
    step_rates = None
    if e017_step.is_file():
        step_rates = mean_step_bcrl_rates(e017_step)
    check = e017_self_check(e017, step_rates)

    e018_cells: dict[str, dict] = {}
    if e018_episodes is not None and Path(e018_episodes).is_file():
        scored = list(iter_jsonl(Path(e018_episodes)))
        for condition_id in E018_COMPARE_CONDITIONS:
            cell = analyze_corpus(
                filter_e018_cell(
                    scored,
                    condition_id=condition_id,
                    budget=PRIMARY_EVAL_BUDGET,
                ),
                source=f"E018:{condition_id}@{PRIMARY_EVAL_BUDGET}",
            )
            e018_cells[f"{condition_id}@{PRIMARY_EVAL_BUDGET}"] = cell

    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M7A",
        "note": (
            "Offline forensics of frozen E017/E018 research JSONL. "
            "Production parser/prompt/reward/training were not modified."
        ),
        "metric_semantics": METRIC_SEMANTICS,
        "e017": e017,
        "e017_self_check": check,
        "e017_step_bcrl": step_rates,
        "e018": {"cells": e018_cells, "n_source_rows": None},
        "paths": {
            "e017_episodes": str(e017_episodes),
            "e017_step_bcrl": str(e017_step) if e017_step.is_file() else None,
            "e018_episodes": str(e018_episodes) if e018_episodes else None,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = {
        "taxonomy": e017.get("examples") or {},
        "recoverability": e017.get("recoverability_examples") or {},
    }
    aggregates = dict(payload)
    aggregates["e017"] = {
        key: value
        for key, value in e017.items()
        if key not in {"examples", "recoverability_examples"}
    }
    for key, cell in e018_cells.items():
        e018_cells[key] = {
            k: v
            for k, v in cell.items()
            if k not in {"examples", "recoverability_examples"}
        }
    write_json(output_dir / "metric_semantics.json", METRIC_SEMANTICS)
    write_json(output_dir / "aggregates.json", aggregates)
    write_json(output_dir / "examples.json", examples)
    (output_dir / "SUMMARY.md").write_text(render_summary(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "e017_n_episodes": e017.get("n_episodes"),
                "episode_invalid_rate": e017.get("episode_invalid_rate"),
                "event_invalid_rate": e017.get("event_invalid_rate"),
                "episode_parse_ok_rate": e017.get("episode_parse_ok_rate"),
                "self_check_pass": check.get("pass"),
                "gate": (e017.get("gate") or {}).get("decision"),
                "e018_cells": sorted(e018_cells),
            },
            indent=2,
        )
    )
    return 0 if check.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
