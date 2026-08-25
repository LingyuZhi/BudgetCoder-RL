#!/usr/bin/env python
"""Aggregate M3B scored episodes into paired reports and SUMMARY.md."""

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
from budget_coder_rl.eval.m3b import (  # noqa: E402
    PRIMARY_N,
    PROVISIONAL_OBS_TOKENS_LIMIT,
    default_manifest_path,
    load_manifest,
    pair_episodes,
    paired_rows,
    paired_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-status", type=Path, default=None)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subset_rows(rows: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        identity = row.get("identity") or {}
        if str(identity.get("instance_id") or "") in ids:
            out.append(row)
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_md(
    path: Path,
    *,
    aggregate: Mapping[str, Any],
    paired: Mapping[str, Any],
    primary_paired: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_status: Mapping[str, Any] | None,
    n_rows: int,
) -> None:
    hidden = paired.get("hidden") or {}
    visible = paired.get("visible") or {}
    util_h = hidden.get("mean_budget_utilization")
    util_v = visible.get("mean_budget_utilization")
    exh_h = hidden.get("budget_exhaustion_rate")
    exh_v = visible.get("budget_exhaustion_rate")
    binding = "unknown"
    rates = [rate for rate in (exh_h, exh_v, util_h, util_v) if rate is not None]
    if rates:
        max_exh = max(exh_h or 0.0, exh_v or 0.0)
        max_u = max(util_h or 0.0, util_v or 0.0)
        if max_exh >= 0.2 or max_u >= 0.85:
            binding = "often binding"
        elif max_u >= 0.4:
            binding = "moderate"
        else:
            binding = "loose for this base agent"
    lines = [
        "# M3B E001 Frozen Base-Policy Baseline",
        "",
        "## Experiment manifest",
        "",
        f"- split: frozen M1E `{manifest.get('split')}`",
        f"- universe: {manifest.get('n_universe')} (sha256 `{manifest.get('ordered_ids_sha256')}`)",
        f"- primary: {manifest.get('n_primary')} (sha256 `{manifest.get('primary_ids_sha256')}`)",
        f"- remainder: {manifest.get('n_remainder')}",
        f"- selection: `{manifest.get('selection_algorithm')}`",
        f"- oracle/gold used in selection: {manifest.get('oracle_used')}/{manifest.get('gold_used')}",
        f"- paired seed base: {manifest.get('paired_seed_base')}",
        "",
        "## Sampling / budget contract",
        "",
        "- model: `Qwen/Qwen3-4B-Instruct-2507`",
        "- sampling: temperature=0.7 top_p=0.8 top_k=20 n=1 validate=False",
        "- `do_sample` is Hydra-side only; not passed to vLLM SamplingParams",
        "- budget accounting: `bcrl-bobs-v2` (primary B_obs = inserted `# bcrl-obs-v1`)",
        f"- provisional B_obs: {PROVISIONAL_OBS_TOKENS_LIMIT} (not frozen Stage-1 training budget)",
        "- max_turns=6, envelope prompt/response=16384, max_model_len=32768",
        "",
        "## GPU / runtime",
        "",
        f"- stop_reason: {fmt((run_status or {}).get('stop_reason'))}",
        f"- elapsed_s: {fmt((run_status or {}).get('elapsed_s'), 1)}",
        f"- episodes scored: {n_rows}",
        f"- completed pairs (all): {paired.get('n_completed_pairs')}",
        f"- completed pairs (primary {PRIMARY_N}): {primary_paired.get('n_completed_pairs')}",
        "",
        "## Aggregate metrics (completed pairs)",
        "",
        "| metric | hidden B0 | visible B1 |",
        "|---|---:|---:|",
        f"| mean localization | {fmt(hidden.get('mean_localization_score'))} | {fmt(visible.get('mean_localization_score'))} |",
        f"| median localization | {fmt(hidden.get('median_localization_score'))} | {fmt(visible.get('median_localization_score'))} |",
        f"| mean file F1 | {fmt(hidden.get('mean_file_f1_parse_ok'))} | {fmt(visible.get('mean_file_f1_parse_ok'))} |",
        f"| mean symbol F1 | {fmt(hidden.get('mean_symbol_f1_scored'))} | {fmt(visible.get('mean_symbol_f1_scored'))} |",
        f"| symbol evaluable rate | {fmt(hidden.get('symbol_evaluable_rate'))} | {fmt(visible.get('symbol_evaluable_rate'))} |",
        f"| mean repo obs tokens | {fmt(hidden.get('mean_repo_observation_tokens'), 1)} | {fmt(visible.get('mean_repo_observation_tokens'), 1)} |",
        f"| mean budget metadata tokens | {fmt(hidden.get('mean_budget_metadata_tokens'), 1)} | {fmt(visible.get('mean_budget_metadata_tokens'), 1)} |",
        f"| mean total env tokens | {fmt(hidden.get('mean_total_env_tokens'), 1)} | {fmt(visible.get('mean_total_env_tokens'), 1)} |",
        f"| mean policy tokens | {fmt(hidden.get('mean_policy_token_count'), 1)} | {fmt(visible.get('mean_policy_token_count'), 1)} |",
        f"| mean utilization | {fmt(hidden.get('mean_budget_utilization'))} | {fmt(visible.get('mean_budget_utilization'))} |",
        f"| parse_ok | {fmt(hidden.get('parse_ok_rate'))} | {fmt(visible.get('parse_ok_rate'))} |",
        f"| budget exhaustion | {fmt(hidden.get('budget_exhaustion_rate'))} | {fmt(visible.get('budget_exhaustion_rate'))} |",
        f"| invalid tool | {fmt(hidden.get('invalid_tool_rate'))} | {fmt(visible.get('invalid_tool_rate'))} |",
        f"| empty submission | {fmt(hidden.get('empty_submission_rate'))} | {fmt(visible.get('empty_submission_rate'))} |",
        f"| finish rate | {fmt(hidden.get('finish_rate'))} | {fmt(visible.get('finish_rate'))} |",
        f"| max-turn rate | {fmt(hidden.get('max_turn_rate'))} | {fmt(visible.get('max_turn_rate'))} |",
        "",
        "## Hidden vs visible paired comparison",
        "",
        f"- visible-win / tie / hidden-win: "
        f"{paired.get('n_visible_win')} / {paired.get('n_tie')} / {paired.get('n_hidden_win')}",
        f"- mean Δ localization (visible-hidden): {fmt(paired.get('mean_delta_localization_score'))}",
        f"- median Δ localization: {fmt(paired.get('median_delta_localization_score'))}",
        f"- mean Δ repo obs tokens: {fmt(paired.get('mean_delta_repo_observation_tokens'), 1)}",
        f"- identical action sequences: {paired.get('n_action_sequence_equal')} / {paired.get('n_completed_pairs')}",
        "",
        "## Gate answers",
        "",
        "1. Frozen base localization / context-use: see aggregate table. "
        f"Mean localization hidden={fmt(hidden.get('mean_localization_score'))}, "
        f"visible={fmt(visible.get('mean_localization_score'))}; "
        f"mean C_obs hidden={fmt(hidden.get('mean_repo_observation_tokens'), 1)}, "
        f"visible={fmt(visible.get('mean_repo_observation_tokens'), 1)}.",
        f"2. Budget visibility on paired tasks: visible-win={paired.get('n_visible_win')}, "
        f"tie={paired.get('n_tie')}, hidden-win={paired.get('n_hidden_win')}; "
        f"mean Δ score={fmt(paired.get('mean_delta_localization_score'))}.",
        f"3. Provisional 8192 budget is **{binding}** "
        f"(exhaustion hidden={fmt(exh_h)}, visible={fmt(exh_v)}; "
        f"mean U hidden={fmt(util_h)}, visible={fmt(util_v)}).",
        "4. Failure class: see `m3b_review_cases.json` first-pass taxonomy "
        "(exploration_policy vs coding_knowledge). Do not retune prompt/tools from these cases.",
        "5. M3C multi-rollout is warranted if reward is non-degenerate (not all-zero) "
        "and exploration failures dominate. See unresolved items below.",
        "",
        "## Unresolved before M3C",
        "",
        "- 8192 remains provisional; do not freeze 4K/8K/16K from this run alone.",
        "- n=1 sampling cannot estimate per-task reward variance; that is M3C.",
        "- Do not freeze the M4/M5 training subset yet.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(args.episodes)
    if not rows:
        print(f"HARD FAIL: no episodes in {args.episodes}", file=sys.stderr)
        return 1
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else default_manifest_path(args.repo_root.resolve())
    )
    manifest = load_manifest(manifest_path) if manifest_path.is_file() else {}
    run_status = {}
    if args.run_status and args.run_status.is_file():
        run_status = json.loads(args.run_status.read_text(encoding="utf-8"))
    live = [row for row in rows if row.get("termination") != "operational_error"]
    grouped = pair_episodes(live)
    pairs = paired_rows(grouped)
    paired = paired_summary(live)
    primary_ids = set(manifest.get("primary_ids") or [])
    primary_rows = subset_rows(live, primary_ids) if primary_ids else live
    primary_paired = paired_summary(primary_rows) if primary_ids else paired
    hidden_rows = [item["hidden"] for item in pairs]
    visible_rows = [item["visible"] for item in pairs]
    aggregate = {
        "n_rows": len(rows),
        "n_live": len(live),
        "n_operational_error": len(rows) - len(live),
        "all": summarize_episodes(live),
        "hidden": summarize_episodes(hidden_rows),
        "visible": summarize_episodes(visible_rows),
        "paired": {
            "n_completed_pairs": paired["n_completed_pairs"],
            "n_visible_win": paired["n_visible_win"],
            "n_hidden_win": paired["n_hidden_win"],
            "n_tie": paired["n_tie"],
            "mean_delta_localization_score": paired["mean_delta_localization_score"],
            "median_delta_localization_score": paired["median_delta_localization_score"],
            "mean_delta_repo_observation_tokens": paired[
                "mean_delta_repo_observation_tokens"
            ],
            "n_action_sequence_equal": paired["n_action_sequence_equal"],
        },
        "primary_paired": {
            "n_completed_pairs": primary_paired["n_completed_pairs"],
            "n_visible_win": primary_paired["n_visible_win"],
            "n_hidden_win": primary_paired["n_hidden_win"],
            "n_tie": primary_paired["n_tie"],
            "mean_delta_localization_score": primary_paired[
                "mean_delta_localization_score"
            ],
        },
        "provisional_obs_tokens_limit": PROVISIONAL_OBS_TOKENS_LIMIT,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m3b_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "m3b_paired.json").write_text(
        json.dumps(paired, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    write_summary_md(
        args.output_dir / "SUMMARY.md",
        aggregate=aggregate,
        paired=paired,
        primary_paired=primary_paired,
        manifest=manifest,
        run_status=run_status,
        n_rows=len(rows),
    )
    print(json.dumps(aggregate["paired"], indent=2))
    print(f"wrote {args.output_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
