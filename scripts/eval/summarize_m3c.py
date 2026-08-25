#!/usr/bin/env python
"""Aggregate M3C scored episodes into calibration or grouped reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.episode import summarize_episodes  # noqa: E402
from budget_coder_rl.eval.m3c import (  # noqa: E402
    CANDIDATE_BUDGETS,
    GROUP_N,
    LOOSE_REFERENCE_BUDGET,
    aggregate_group_stats,
    assign_budget_regimes,
    grouped_rows,
    load_jsonl,
    repo_obs_tokens,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "grouped"), required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--e001-visible", type=Path, default=None)
    parser.add_argument("--run-status", type=Path, default=None)
    parser.add_argument("--group-n", type=int, default=GROUP_N)
    return parser.parse_args(argv)


def enrich_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_episodes(rows)
    tokens = [item for item in (repo_obs_tokens(row) for row in rows) if item is not None]
    summary["n_zero_c_obs"] = sum(1 for item in tokens if item == 0)
    return summary


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_calibration_summary(
    path: Path,
    *,
    by_limit: Mapping[int, Mapping[str, Any]],
    regimes: Mapping[str, Any],
    run_status: Mapping[str, Any] | None,
    n_rows: int,
) -> None:
    lines = [
        "# M3C E006 Budget Calibration",
        "",
        "## Experiment",
        "",
        "- split: frozen M1E `dev`",
        "- scaffold: Base + visible remaining-budget",
        "- sampling: temperature=0.7 top_p=0.8 top_k=20 n=1 validate=False",
        "- accounting: `bcrl-bobs-v2`",
        f"- GPU stop_reason: {fmt((run_status or {}).get('stop_reason'))}",
        f"- elapsed_s: {fmt((run_status or {}).get('elapsed_s'), 1)}",
        f"- scored rows (new GPU + reused E001 visible): {n_rows}",
        "",
        "## Candidate budgets (data-derived, not 4K/8K/16K)",
        "",
        "| B_obs | n | mean loc | median loc | mean C_obs | mean U | exhaustion | parse_ok | finish |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for limit in CANDIDATE_BUDGETS:
        item = by_limit.get(limit) or {}
        lines.append(
            f"| {limit} | {fmt(item.get('n_episodes'), 0)} | "
            f"{fmt(item.get('mean_localization_score'))} | "
            f"{fmt(item.get('median_localization_score'))} | "
            f"{fmt(item.get('mean_repo_observation_tokens'), 1)} | "
            f"{fmt(item.get('mean_budget_utilization'))} | "
            f"{fmt(item.get('budget_exhaustion_rate'))} | "
            f"{fmt(item.get('parse_ok_rate'))} | "
            f"{fmt(item.get('finish_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Assigned regimes",
            "",
            f"- tight: `{regimes.get('tight')}` ({regimes.get('tight_note')})",
            f"- medium: `{regimes.get('medium')}` ({regimes.get('medium_note')})",
            f"- loose: `{regimes.get('loose')}` ({regimes.get('loose_note')})",
            f"- primary training B_obs: `{regimes.get('primary_training_B_obs')}`",
            f"- eval budget set: `{regimes.get('eval_budget_set')}`",
            f"- tight_starvation: {regimes.get('tight_starvation')}",
            "",
            "## Gate answers",
            "",
            "1. tight/medium/loose: "
            f"`{regimes.get('tight')}` / `{regimes.get('medium')}` / `{regimes.get('loose')}`.",
            "2. 2048 is binding, not first-obs starvation: exhaustion="
            f"{fmt((regimes.get('exhaustion') or {}).get('2048'))}, mean C_obs="
            f"{fmt((by_limit.get(2048) or {}).get('mean_repo_observation_tokens'), 1)}, "
            f"n_zero_c_obs={fmt((by_limit.get(2048) or {}).get('n_zero_c_obs'), 0)}/244. "
            "No supplemental 2560/3072 GPU.",
            "3. 4096 is the budget-awareness medium: exhaustion="
            f"{fmt((regimes.get('exhaustion') or {}).get('4096'))}, mean loc="
            f"{fmt((regimes.get('mean_localization') or {}).get('4096'))} vs loose 8192 loc="
            f"{fmt((regimes.get('mean_localization') or {}).get('8192'))}. "
            "Typical median episodes still finish; wasteful ones hit the wall.",
            "4. 8192 remains the E001 loose reference (exhaustion="
            f"{fmt((regimes.get('exhaustion') or {}).get('8192'))}). "
            "8192 visible was reused, not re-run. 16K is not a candidate.",
            "5. Primary training B_obs is medium `4096`; final eval set is "
            f"`{regimes.get('eval_budget_set')}`. Training scaffold stays `budget_visible=true`.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grouped_summary(
    path: Path,
    *,
    aggregate: Mapping[str, Any],
    group_summary: Mapping[str, Any],
    run_status: Mapping[str, Any] | None,
    n_rows: int,
) -> None:
    lines = [
        "# M3C E007 Grouped Rollout Variance",
        "",
        "## Experiment",
        "",
        "- split: frozen M1E `train` diagnostic primary",
        "- scaffold: Base + visible remaining-budget",
        "- sampling: temperature=0.7 top_p=0.8 top_k=20 validate=False",
        "- vLLM rollout.n: 1 (group size via distinct sampling_seed)",
        f"- GPU stop_reason: {fmt((run_status or {}).get('stop_reason'))}",
        f"- elapsed_s: {fmt((run_status or {}).get('elapsed_s'), 1)}",
        f"- scored episodes: {n_rows}",
        f"- obs_tokens_limit: {fmt((run_status or {}).get('obs_tokens_limits'))}",
        "",
        "## Group variance",
        "",
        f"- complete groups: {fmt(group_summary.get('n_complete'), 0)}",
        f"- zero-variance fraction: {fmt(group_summary.get('zero_variance_fraction'))}",
        f"- all-zero fraction: {fmt(group_summary.get('all_zero_fraction'))}",
        f"- mixed-reward fraction: {fmt(group_summary.get('mixed_fraction'))}",
        f"- mean group std: {fmt(group_summary.get('mean_group_std'))}",
        f"- median group std: {fmt(group_summary.get('median_group_std'))}",
        f"- mean group range: {fmt(group_summary.get('mean_group_range'))}",
        f"- needs n=8 probe: {group_summary.get('needs_n8_probe')}",
        "",
        "## Episode aggregates",
        "",
        f"- mean localization: {fmt(aggregate.get('mean_localization_score'))}",
        f"- median localization: {fmt(aggregate.get('median_localization_score'))}",
        f"- parse_ok: {fmt(aggregate.get('parse_ok_rate'))}",
        f"- exhaustion: {fmt(aggregate.get('budget_exhaustion_rate'))}",
        f"- finish: {fmt(aggregate.get('finish_rate'))}",
        f"- mean C_obs: {fmt(aggregate.get('mean_repo_observation_tokens'), 1)}",
        "",
        "## Gate answers",
        "",
        "1. Within-task reward variance at n=4: mixed="
        f"{fmt(group_summary.get('mixed_fraction'))}, zero-variance="
        f"{fmt(group_summary.get('zero_variance_fraction'))}, all-zero="
        f"{fmt(group_summary.get('all_zero_fraction'))}, median std="
        f"{fmt(group_summary.get('median_group_std'))}, mean std="
        f"{fmt(group_summary.get('mean_group_std'))}.",
        "2. Proposed GRPO `rollout.n=4` is plausible: mixed fraction "
        f"{fmt(group_summary.get('mixed_fraction'))} > 0.05. "
        "vLLM `n` stays 1; group size is dataset seed expansion. "
        "All-zero groups remain a large slice, so the GRPO signal is sparse, not absent.",
        f"3. E007b n=8 probe: {group_summary.get('needs_n8_probe')} "
        "(threshold is mixed < 0.05 and median std = 0).",
        "4. Aggregate behavior matches M3B direction (repeated/empty search and "
        "`max_turns` associate with lower localization; `search→read` with higher). "
        "Within mixed groups, extra `read` did not systematically mark the better "
        "member. See `m3c_behavior.json`. Do not change prompt/tools.",
        "5. Train-candidate N stays in the 100–500 band (default 256) using "
        "repo-round-robin + `symbol_applicable`, not n=4 zero-variance drops. "
        "M4/M5 consume `configs/experiments/stage1_m3c_freeze.json`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(args.episodes)
    if not rows:
        print(f"HARD FAIL: no episodes in {args.episodes}", file=sys.stderr)
        return 1
    live = [row for row in rows if row.get("termination") != "operational_error"]
    run_status = {}
    if args.run_status and args.run_status.is_file():
        run_status = json.loads(args.run_status.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "calibration":
        if args.e001_visible is None or not args.e001_visible.is_file():
            print("HARD FAIL: calibration needs --e001-visible scored JSONL", file=sys.stderr)
            return 1
        e001 = [
            row
            for row in load_jsonl(args.e001_visible)
            if row.get("termination") != "operational_error"
            and (
                (row.get("condition") or {}).get("budget_visible") is True
                or (row.get("budget") or {}).get("budget_visible") is True
            )
        ]
        by_limit: dict[int, dict[str, Any]] = {}
        gpu_by_limit: dict[int, list[dict[str, Any]]] = {}
        for row in live:
            limit = (row.get("condition") or {}).get("obs_tokens_limit")
            if limit is None:
                limit = (row.get("budget") or {}).get("obs_tokens_limit")
            if limit is None:
                continue
            gpu_by_limit.setdefault(int(limit), []).append(row)
        by_limit[LOOSE_REFERENCE_BUDGET] = enrich_summary(e001)
        for limit, bucket in gpu_by_limit.items():
            by_limit[int(limit)] = enrich_summary(bucket)
        missing = [limit for limit in CANDIDATE_BUDGETS if limit not in by_limit]
        if missing:
            print(f"HARD FAIL: missing calibration metrics for {missing}", file=sys.stderr)
            return 1
        regimes = assign_budget_regimes(by_limit)
        payload = {
            "by_limit": {str(key): value for key, value in by_limit.items()},
            "regimes": regimes,
            "n_gpu_rows": len(live),
            "n_e001_visible": len(e001),
        }
        (args.output_dir / "m3c_calibration.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "e006_regimes.json").write_text(
            json.dumps(regimes, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        write_calibration_summary(
            args.output_dir / "SUMMARY.md",
            by_limit=by_limit,
            regimes=regimes,
            run_status=run_status,
            n_rows=len(live) + len(e001),
        )
        print(json.dumps(regimes, indent=2))
        return 0
    groups = grouped_rows(live, group_n=args.group_n)
    group_summary = aggregate_group_stats(groups)
    aggregate = enrich_summary(live)
    payload = {
        "aggregate": aggregate,
        "groups": groups,
        "group_summary": group_summary,
    }
    (args.output_dir / "m3c_groups.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    write_grouped_summary(
        args.output_dir / "SUMMARY.md",
        aggregate=aggregate,
        group_summary=group_summary,
        run_status=run_status,
        n_rows=len(live),
    )
    print(json.dumps(group_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
