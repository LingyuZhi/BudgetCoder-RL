#!/usr/bin/env python
"""Offline M7B train–eval invalid-action discrepancy audit.

Does not touch AgentLoop, parser, prompt, reward, or frozen E017/E018 artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.e018 import default_trace_dir  # noqa: E402
from budget_coder_rl.eval.m4a import load_json  # noqa: E402
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m5a import default_output_dir  # noqa: E402
from budget_coder_rl.eval.m7a import PRIMARY_EVAL_BUDGET, filter_e018_cell, iter_jsonl  # noqa: E402
from budget_coder_rl.eval.m7b import (  # noqa: E402
    BIN_CSV_FIELDS,
    BIN_SIZE,
    EARLY16,
    EXPERIMENT_ID,
    LATE16,
    MILESTONE,
    SCHEMA_VERSION,
    STEP_CSV_FIELDS,
    TAXONOMY_CSV_FIELDS,
    analyze_eval_cell,
    analyze_training_stream,
    audit_execution_contract,
    bin_step_rows,
    binned_csv_rows,
    cross_check_step_bcrl,
    drop_unserializable_sets,
    hypothesis_verdicts,
    iter_jsonl_indexed,
    load_optional_gold,
    load_step_bcrl,
    matched_comparison,
    render_summary,
    slice_steps,
    step_series_csv_rows,
    stratify_phases,
    taxonomy_over_time_rows,
    write_csv,
    write_curves_png,
    write_curves_svg,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e017-episodes", type=Path, default=None)
    parser.add_argument("--e017-step-bcrl", type=Path, default=None)
    parser.add_argument("--e017-provenance", type=Path, default=None)
    parser.add_argument("--e017-config", type=Path, default=None)
    parser.add_argument("--e018-episodes", type=Path, default=None)
    parser.add_argument("--e018-provenance", type=Path, default=None)
    parser.add_argument("--e018-integrity", type=Path, default=None)
    parser.add_argument("--e018-overlay", type=Path, default=None)
    parser.add_argument("--train-candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    return load_json(Path(path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        repo_root, EXPERIMENT_ID
    )
    e017_dir = repo_root / "outputs" / "experiments" / "E017"
    e018_dir = repo_root / "outputs" / "experiments" / "E018"
    e017_episodes = args.e017_episodes or (e017_dir / "episodes.jsonl")
    e017_step = args.e017_step_bcrl or (e017_dir / "step_bcrl.jsonl")
    e017_prov_path = args.e017_provenance or (e017_dir / "provenance.json")
    e017_config_path = args.e017_config or (e017_dir / "config_resolved.json")
    e018_prov_path = args.e018_provenance or (e018_dir / "provenance.json")
    e018_integrity_path = args.e018_integrity or (e018_dir / "treatment_integrity.json")
    e018_overlay_path = args.e018_overlay or (
        repo_root / "configs" / "experiments" / "stage1_m6_e018.json"
    )
    train_cand_path = args.train_candidates or (
        repo_root / "data" / "manifests" / "m5_scaled_train_candidates.json"
    )
    e018_episodes = args.e018_episodes
    if e018_episodes is None:
        candidate = default_trace_dir(bcrl_data_root()) / "episodes_scored.jsonl"
        e018_episodes = candidate if candidate.is_file() else None

    if not Path(e017_episodes).is_file():
        print(f"HARD FAIL: missing {e017_episodes}", file=sys.stderr)
        return 1

    gold_by_id = load_optional_gold(repo_root)
    train = analyze_training_stream(
        iter_jsonl_indexed(Path(e017_episodes)),
        gold_by_id=gold_by_id or None,
    )
    step_rows = [drop_unserializable_sets(row) for row in train["step_rows"]]
    bcrl_rows = load_step_bcrl(Path(e017_step)) if Path(e017_step).is_file() else []
    step_check = cross_check_step_bcrl(step_rows, bcrl_rows) if bcrl_rows else {
        "n_checked": 0,
        "n_rate_mismatch": 0,
        "n_count_mismatch": 0,
        "pass": False,
        "examples": [],
        "note": "step_bcrl.jsonl missing",
    }
    binned = bin_step_rows(step_rows, bin_size=BIN_SIZE)
    early16 = slice_steps(step_rows, *EARLY16)
    late16 = slice_steps(step_rows, *LATE16)
    last_bin = binned[-1] if binned else None
    strat = stratify_phases(step_rows, gold_available=bool(gold_by_id))

    e018_cells: dict[str, dict[str, Any]] = {}
    n_e018 = 0
    if e018_episodes is not None and Path(e018_episodes).is_file():
        scored = list(iter_jsonl(Path(e018_episodes)))
        n_e018 = len(scored)
        for condition_id in ("B1", "M_scaled"):
            cell = analyze_eval_cell(
                filter_e018_cell(
                    scored,
                    condition_id=condition_id,
                    budget=PRIMARY_EVAL_BUDGET,
                )
            )
            e018_cells[f"{condition_id}@{PRIMARY_EVAL_BUDGET}"] = cell

    e017_prov = _load_optional(e017_prov_path)
    e017_config = _load_optional(e017_config_path)
    e018_prov = _load_optional(e018_prov_path)
    e018_integrity = _load_optional(e018_integrity_path)
    e018_overlay = _load_optional(e018_overlay_path)
    _load_optional(train_cand_path)

    contract = audit_execution_contract(
        e017_provenance=e017_prov,
        e017_config=e017_config,
        e018_provenance=e018_prov,
        e018_overlay=e018_overlay,
        e018_integrity=e018_integrity,
        e017_empirical=train,
        e018_cells=e018_cells,
    )
    matched = matched_comparison(
        late16=late16,
        last_bin=last_bin,
        e018_cells=e018_cells,
        execution_matched=bool(contract.get("execution_matched")),
    )
    verdicts = hypothesis_verdicts(
        contract=contract,
        pooled=train["pooled"],
        early16=early16,
        late16=late16,
        stratification=strat,
        matched=matched,
        step_check=step_check,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "note": (
            "Offline train/eval discrepancy audit of frozen E017/E018 research JSONL. "
            "Production parser/prompt/reward/training were not modified."
        ),
        "paths": {
            "e017_episodes": str(e017_episodes),
            "e017_step_bcrl": str(e017_step) if Path(e017_step).is_file() else None,
            "e018_episodes": str(e018_episodes) if e018_episodes else None,
        },
        "e017_pooled": train["pooled"],
        "e017_n_jsonl_lines": train["n_jsonl_lines"],
        "e017_n_temp_zero": train["n_temp_zero"],
        "gold_available": bool(gold_by_id),
        "n_gold_ids": len(gold_by_id),
        "early16": _strip_repo_sets(early16),
        "late16": _strip_repo_sets(late16),
        "binned": [_strip_repo_sets(row) for row in binned],
        "e018": {"cells": e018_cells, "n_source_rows": n_e018},
        "e018_cells": e018_cells,
        "execution_contract": contract,
        "stratification": _strip_strat(strat),
        "matched_comparison": matched,
        "hypothesis_verdicts": verdicts,
        "step_bcrl_check": step_check,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "aggregates.json", payload)
    write_json(output_dir / "execution_contract.json", contract)
    write_json(output_dir / "stratification.json", _strip_strat(strat))
    write_json(output_dir / "matched_comparison.json", matched)
    write_json(output_dir / "hypothesis_verdicts.json", verdicts)
    write_json(
        output_dir / "curves.json",
        {
            "binned": [_strip_repo_sets(row) for row in binned],
            "early16": _strip_repo_sets(early16),
            "late16": _strip_repo_sets(late16),
        },
    )
    write_csv(output_dir / "step_series.csv", step_series_csv_rows(step_rows), STEP_CSV_FIELDS)
    write_csv(output_dir / "binned_series.csv", binned_csv_rows(binned), BIN_CSV_FIELDS)
    write_csv(
        output_dir / "taxonomy_over_time.csv",
        taxonomy_over_time_rows(binned),
        TAXONOMY_CSV_FIELDS,
    )
    write_curves_svg(binned, output_dir / "curves.svg")
    png_ok = write_curves_png(binned, output_dir / "curves.png")
    (output_dir / "SUMMARY.md").write_text(render_summary(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "e017_n_episodes": train["pooled"].get("n_episodes"),
                "e017_event_invalid_rate": train["pooled"].get("event_invalid_rate"),
                "e017_first_turn_protocol_rate": train["pooled"].get(
                    "first_turn_protocol_rate"
                ),
                "step_bcrl_check_pass": step_check.get("pass"),
                "png": png_ok,
                "primary_gap_contributor": verdicts.get("primary_gap_contributor"),
                "h2": next(
                    (
                        item.get("verdict")
                        for item in verdicts.get("items") or []
                        if item.get("id") == "H2"
                    ),
                    None,
                ),
                "e018_cells": sorted(e018_cells),
            },
            indent=2,
        )
    )
    return 0 if step_check.get("pass") else 2


def _strip_repo_sets(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"unique_ids", "repos"}}


def _strip_strat(strat: Mapping[str, Any]) -> dict[str, Any]:
    phases = {
        name: _strip_repo_sets(cell) for name, cell in (strat.get("phases") or {}).items()
    }
    out = dict(strat)
    out["phases"] = phases
    return out


if __name__ == "__main__":
    raise SystemExit(main())
