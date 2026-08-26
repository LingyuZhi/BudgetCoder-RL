#!/usr/bin/env python
"""Offline M6 tables, paired stats, plots. Does not run AgentLoop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m6 import (  # noqa: E402
    BUDGETS,
    EVAL_NAME,
    EXPERIMENT_ID,
    attach_case_trajectories,
    default_e015_output_dir,
    difficulty_breakdown,
    inspect_tokenizer_warning,
    load_correlation_groups,
    load_optional_features,
    lock_errors,
    main_table,
    paired_cells,
    paired_summary_stats,
    per_repo_breakdown,
    quality_budget_curve,
    select_case_studies,
    write_json,
)
from budget_coder_rl.eval.provenance import sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_curve_png(curve: Mapping[str, Any], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    series = curve.get("series") or {}
    for name in ("B0", "B1", "M1"):
        points = series.get(name) or []
        if not points:
            continue
        xs = [item["obs_tokens_limit"] for item in points]
        ys = [item["mean_localization_score"] for item in points]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("B_obs (tool observation tokens)")
    ax.set_ylabel("mean localization_score")
    ax.set_title("Localization vs observation budget")
    ax.set_xticks(list(BUDGETS))
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_obs_scatter_png(pairs: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for label, color, score_key, obs_key in (
        ("B1", "C0", "B1_score", "B1_repo_obs"),
        ("M1", "C1", "M1_score", "M1_repo_obs"),
    ):
        xs = [item.get(obs_key) for item in pairs if item.get(obs_key) is not None]
        ys = [item.get(score_key) for item in pairs if item.get(obs_key) is not None]
        if xs:
            ax.scatter(xs, ys, s=12, alpha=0.45, label=label, color=color)
    ax.set_xlabel("actual repo observation tokens")
    ax.set_ylabel("localization_score")
    ax.set_title("Actual C_obs vs localization")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_summary_md(
    path: Path,
    *,
    table: list[dict[str, Any]],
    paired: Mapping[str, Any],
    per_repo: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    n_rows: int,
    provenance: Mapping[str, Any] | None,
) -> None:
    lines = [
        "# M6 Frozen SWE-Gym held-out-task dev evaluation",
        "",
        f"- eval_name: `{EVAL_NAME}`",
        "- split_kind: **held-out-task** (not held-out-repository)",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- n_scored_episodes: {n_rows}",
        f"- RL checkpoint: E014 `global_step_32` only",
        "",
        "## Main metric table",
        "",
        "| cond | B_obs | n | loc | file F1 | symbol F1 | parse_ok | invalid | empty | exhaust | C_obs | U | turns | policy tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| {condition_id} | {obs_tokens_limit} | {n} | {loc} | {file_f1} | {sym} | {parse} | {inv} | {empty} | {exh} | {cobs} | {util} | {turns} | {ptok} |".format(
                condition_id=row["condition_id"],
                obs_tokens_limit=row["obs_tokens_limit"],
                n=row["n"],
                loc=fmt(row.get("mean_localization_score")),
                file_f1=fmt(row.get("mean_file_f1_parse_ok")),
                sym=fmt(row.get("mean_symbol_f1_scored")),
                parse=fmt(row.get("parse_ok_rate")),
                inv=fmt(row.get("invalid_tool_rate")),
                empty=fmt(row.get("empty_submission_rate")),
                exh=fmt(row.get("budget_exhaustion_rate")),
                cobs=fmt(row.get("mean_repo_observation_tokens"), 1),
                util=fmt(row.get("mean_budget_utilization")),
                turns=fmt(row.get("mean_n_events"), 2),
                ptok=fmt(row.get("mean_policy_token_count"), 1),
            )
        )
    lines.extend(["", "## Paired RL Visible vs Base Visible", ""])
    for budget in BUDGETS:
        stats = (paired.get("rl_vs_base") or {}).get(str(budget)) or {}
        boot = stats.get("bootstrap") or {}
        lines.append(
            f"- B_obs={budget}: n={stats.get('n_pairs')} meanΔ={fmt(stats.get('mean_delta'))} "
            f"medianΔ={fmt(stats.get('median_delta'))} "
            f"M1-win/tie/B1-win={stats.get('n_M1_win')}/{stats.get('n_tie')}/{stats.get('n_B1_win')} "
            f"CI95=[{fmt(boot.get('low'))}, {fmt(boot.get('high'))}] "
            f"clusters={boot.get('n_clusters')}"
        )
    lines.extend(["", "## Auxiliary Base Hidden vs Base Visible", ""])
    for budget in BUDGETS:
        stats = (paired.get("visibility") or {}).get(str(budget)) or {}
        lines.append(
            f"- B_obs={budget}: n={stats.get('n_pairs')} meanΔ={fmt(stats.get('mean_delta'))} "
            f"B1-win/tie/B0-win={stats.get('n_B1_win')}/{stats.get('n_tie')}/{stats.get('n_B0_win')}"
        )
    lines.extend(["", "## Per-repo (M1 vs B1, pooled budgets in per_repo.json)", ""])
    for row in per_repo.get("pooled") or []:
        flag = "" if row.get("significance_ok") else " (small n; no significance claim)"
        lines.append(
            f"- {row.get('repo')}: n={row.get('n')} meanΔ={fmt(row.get('mean_delta'))} "
            f"M1-win/tie/B1-win={row.get('n_M1_win')}/{row.get('n_tie')}/{row.get('n_B1_win')}{flag}"
        )
    lines.extend(
        [
            "",
            "## Tokenizer warning (E014, not an eval retune)",
            "",
            f"- call site: {tokenizer.get('call_site')}",
            f"- log hits: {tokenizer.get('n_log_hits')}",
            f"- tito_correctness_bug: {tokenizer.get('tito_correctness_bug')}",
            f"- eval_tokenization_changed: {tokenizer.get('eval_tokenization_changed')}",
            f"- note: {tokenizer.get('note')}",
            "",
            "## Provenance",
            "",
            f"- eval config sha256: {(provenance or {}).get('eval_config_sha256')}",
            f"- historical main table splice: false",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    errors = lock_errors(repo_root)
    if errors:
        print(f"HARD FAIL: eval lock {errors}", file=sys.stderr)
        return 1
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e015_output_dir(repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in load_jsonl(args.episodes)
        if row.get("termination") != "operational_error"
    ]
    if not rows:
        print(f"HARD FAIL: no scored episodes in {args.episodes}", file=sys.stderr)
        return 1
    groups = load_correlation_groups(repo_root)
    table = main_table(rows)
    paired_rl: dict[str, Any] = {}
    paired_vis: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        rl_pairs = paired_cells(rows, left_id="B1", right_id="M1", budget=budget)
        vis_pairs = paired_cells(rows, left_id="B0", right_id="B1", budget=budget)
        paired_rl[str(budget)] = paired_summary_stats(
            rl_pairs, left_id="B1", right_id="M1", group_ids=groups
        )
        paired_vis[str(budget)] = paired_summary_stats(
            vis_pairs, left_id="B0", right_id="B1", group_ids=groups
        )
        for item in rl_pairs:
            compact = {
                key: item[key]
                for key in item
                if key not in {"left", "right"}
            }
            compact["budget"] = budget
            paired_rows.append(compact)
    all_rl_pairs = paired_cells(rows, left_id="B1", right_id="M1")
    per_repo_pooled = per_repo_breakdown(all_rl_pairs)
    per_repo_by_budget = {
        str(budget): per_repo_breakdown(paired_cells(rows, left_id="B1", right_id="M1", budget=budget))
        for budget in BUDGETS
    }
    curve = quality_budget_curve(rows, condition_ids=("B0", "B1", "M1"))
    features = load_optional_features(repo_root)
    slices = difficulty_breakdown(all_rl_pairs, features)
    cases = select_case_studies(all_rl_pairs)
    case_payload = attach_case_trajectories(cases, all_rl_pairs)
    tokenizer = inspect_tokenizer_warning(repo_root)
    provenance = {
        "eval_name": EVAL_NAME,
        "eval_config_sha256": sha256_file(
            repo_root / "configs/experiments/stage1_m6_eval.json"
        ),
    }
    write_json(output_dir / "aggregates.json", {"table": table, "n_rows": len(rows)})
    write_json(
        output_dir / "paired_stats.json",
        {"rl_vs_base": paired_rl, "visibility": paired_vis},
    )
    write_json(
        output_dir / "per_repo.json",
        {"pooled": per_repo_pooled, "by_budget": per_repo_by_budget},
    )
    write_json(output_dir / "quality_budget_curve.json", curve)
    write_json(output_dir / "difficulty_slices.json", {"slices": slices, "note": "optional; not mined exhaustively"})
    write_json(output_dir / "case_selection.json", case_payload)
    write_json(output_dir / "tokenizer_warning.json", tokenizer)
    paired_path = output_dir / "paired_episodes.jsonl"
    with paired_path.open("w", encoding="utf-8") as handle:
        for item in paired_rows:
            handle.write(json.dumps(item, ensure_ascii=True, default=str) + "\n")
    png_ok = write_curve_png(curve, output_dir / "quality_budget_curve.png")
    scatter_ok = write_obs_scatter_png(all_rl_pairs, output_dir / "obs_tokens_vs_quality.png")
    write_summary_md(
        output_dir / "SUMMARY.md",
        table=table,
        paired={"rl_vs_base": paired_rl, "visibility": paired_vis},
        per_repo={"pooled": per_repo_pooled},
        tokenizer=tokenizer,
        n_rows=len(rows),
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "n_rows": len(rows),
                "n_table_cells": len(table),
                "curve_png": png_ok,
                "scatter_png": scatter_ok,
                "summary": str(output_dir / "SUMMARY.md"),
                "n_rl_pairs": len(all_rl_pairs),
                "case_n_candidates": case_payload.get("n_candidates"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
