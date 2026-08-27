#!/usr/bin/env python
"""Offline E018 tables, paired stats, plots. Does not run AgentLoop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.e018 import (  # noqa: E402
    BUDGETS,
    CANONICAL_RL_STEP,
    EVAL_NAME,
    EXPERIMENT_ID,
    WANDB_EXPERIMENT_NAME,
    WANDB_PROJECT,
    cell_aggregate,
    default_e015_trace_dir,
    default_e018_output_dir,
    import_e015_condition,
    load_jsonl,
    main_table,
    overlay_lock_errors,
    paired_cells,
    per_repo_breakdown,
    quality_budget_curve,
    reliability_class,
    reuse_base_audit,
    scientific_conclusion,
    select_case_studies,
    attach_case_trajectories,
)
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m6 import load_correlation_groups, paired_summary_stats  # noqa: E402
from budget_coder_rl.eval.provenance import sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--prototype-episodes", type=Path, default=None)
    parser.add_argument("--reuse-base-episodes", type=Path, default=None)
    return parser.parse_args(argv)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_curve_png(curve: Mapping[str, Any], path: Path, names: tuple[str, ...]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    series = curve.get("series") or {}
    for name in names:
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


def _cell(table: list[dict[str, Any]], condition_id: str, budget: int) -> dict[str, Any]:
    for row in table:
        if row.get("condition_id") == condition_id and int(row.get("obs_tokens_limit") or 0) == int(budget):
            return row
    return {}


def write_summary_md(
    path: Path,
    *,
    table: list[dict[str, Any]],
    paired: Mapping[str, Any],
    per_repo: Mapping[str, Any],
    reuse: Mapping[str, Any],
    integrity: Mapping[str, Any],
    conclusion: str,
    rel_class: str,
    n_rows: int,
    n_error: int,
    provenance: Mapping[str, Any] | None,
    cases: Mapping[str, Any],
) -> None:
    lines = [
        "# E018 Scaled M6 Frozen SWE-Gym held-out-task evaluation",
        "",
        f"- eval_name: `{EVAL_NAME}`",
        "- split_kind: **held-out-task** (not held-out-repository)",
        f"- experiment_id: `{EXPERIMENT_ID}`",
        f"- n_scored_episodes (E018 native): {n_rows}",
        f"- operational_error: {n_error}",
        f"- RL checkpoint: E017 `global_step_{CANONICAL_RL_STEP}` only (256/32 forbidden)",
        f"- B0/B1 reuse: **{reuse.get('decision')}**",
        f"- scientific conclusion: **{conclusion}**",
        f"- reliability class: **{rel_class}**",
        f"- treatment_integrity.pass: {integrity.get('pass')}",
        "",
        "## Reuse / provenance",
        "",
        f"- reused B0/B1 cells: {reuse.get('decision') == 'reuse_e015_b0_b1'}",
        f"- reasons: {reuse.get('reasons')}",
        "- M_scaled 2048/4096/8192: always newly rolled out",
        "- Prototype RL: E015 M1 overlay only (not an independent randomized experiment)",
        "- E001/E006: not spliced",
        "",
        "## Treatment integrity (E017 global_step_275)",
        "",
        f"- checkpoint: `{integrity.get('checkpoint_actor_dir')}`",
        f"- load_ok: {integrity.get('load_ok')}",
        f"- update_weights_ok: {integrity.get('update_weights_ok')}",
        f"- listed_lora_ids: {integrity.get('listed_lora_ids')}",
        f"- lora_request_attached: {integrity.get('lora_request_attached')}",
        f"- lora_int_id: {integrity.get('lora_int_id')}",
        f"- load digest: {(integrity.get('load_fingerprint') or {}).get('digest')}",
        f"- sync digest: {(integrity.get('sync_payload') or {}).get('digest')}",
        f"- HTTP /v1/models saw adapter (supplementary only): {integrity.get('http_saw_adapter')}",
        "",
        "## Main metric table",
        "",
        "| cond | B_obs | n | loc | file P/R/F1 | symbol P/R/F1 | parse_ok | invalid | empty | exhaust | C_obs | U | turns | tools | policy tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        file_prf = "/".join(
            [
                fmt(row.get("mean_file_precision_parse_ok")),
                fmt(row.get("mean_file_recall_parse_ok")),
                fmt(row.get("mean_file_f1_parse_ok")),
            ]
        )
        symbol_prf = "/".join(
            [
                fmt(row.get("mean_symbol_precision_scored")),
                fmt(row.get("mean_symbol_recall_scored")),
                fmt(row.get("mean_symbol_f1_scored")),
            ]
        )
        lines.append(
            "| {condition_id} | {obs_tokens_limit} | {n} | {loc} | {file_prf} | {symbol_prf} | {parse} | {inv} | {empty} | {exh} | {cobs} | {util} | {turns} | {tools} | {ptok} |".format(
                condition_id=row["condition_id"],
                obs_tokens_limit=row["obs_tokens_limit"],
                n=row["n"],
                loc=fmt(row.get("mean_localization_score")),
                file_prf=file_prf,
                symbol_prf=symbol_prf,
                parse=fmt(row.get("parse_ok_rate")),
                inv=fmt(row.get("invalid_tool_rate")),
                empty=fmt(row.get("empty_submission_rate")),
                exh=fmt(row.get("budget_exhaustion_rate")),
                cobs=fmt(row.get("mean_repo_observation_tokens"), 1),
                util=fmt(row.get("mean_budget_utilization")),
                turns=fmt(row.get("mean_n_events"), 2),
                tools=fmt(row.get("mean_n_tool_ok"), 2),
                ptok=fmt(row.get("mean_policy_token_count"), 1),
            )
        )
    lines.extend(["", "## Primary: Scaled RL vs Base Visible", ""])
    for budget in BUDGETS:
        stats = (paired.get("scaled_vs_b1") or {}).get(str(budget)) or {}
        boot = stats.get("bootstrap") or {}
        lines.append(
            f"- B_obs={budget}: n={stats.get('n_pairs')} meanΔ={fmt(stats.get('mean_delta'))} "
            f"medianΔ={fmt(stats.get('median_delta'))} "
            f"M_scaled-win/tie/B1-win={stats.get('n_M_scaled_win')}/{stats.get('n_tie')}/{stats.get('n_B1_win')} "
            f"CI95=[{fmt(boot.get('low'))}, {fmt(boot.get('high'))}] "
            f"clusters={boot.get('n_clusters')}"
        )
    lines.extend(["", "## Auxiliary: Base Hidden vs Base Visible", ""])
    for budget in BUDGETS:
        stats = (paired.get("visibility") or {}).get(str(budget)) or {}
        lines.append(
            f"- B_obs={budget}: n={stats.get('n_pairs')} meanΔ={fmt(stats.get('mean_delta'))} "
            f"B1-win/tie/B0-win={stats.get('n_B1_win')}/{stats.get('n_tie')}/{stats.get('n_B0_win')}"
        )
    lines.extend(["", "## Scale-correction: Scaled RL vs Prototype RL (E015 M1)", ""])
    lines.append("Not an independent randomized experiment. Same freeze/seed; different training scale.")
    for budget in BUDGETS:
        stats = (paired.get("scaled_vs_proto") or {}).get(str(budget)) or {}
        boot = stats.get("bootstrap") or {}
        lines.append(
            f"- B_obs={budget}: n={stats.get('n_pairs')} meanΔ={fmt(stats.get('mean_delta'))} "
            f"M_scaled-win/tie/M1_proto-win={stats.get('n_M_scaled_win')}/{stats.get('n_tie')}/{stats.get('n_M1_proto_win')} "
            f"CI95=[{fmt(boot.get('low'))}, {fmt(boot.get('high'))}]"
        )
    lines.extend(["", "## Per-repo (M_scaled vs B1, pooled budgets)", ""])
    for row in per_repo.get("pooled") or []:
        flag = "" if row.get("significance_ok") else " (small n; no significance claim)"
        lines.append(
            f"- {row.get('repo')}: n={row.get('n')} meanΔ={fmt(row.get('mean_delta'))} "
            f"M_scaled-win/tie/B1-win={row.get('n_M_scaled_win')}/{row.get('n_tie')}/{row.get('n_B1_win')}{flag}"
        )
    selected = (cases.get("selected") or {}) if isinstance(cases, Mapping) else {}
    scaled_4096 = _cell(table, "M_scaled", 4096)
    b1_8192 = _cell(table, "B1", 8192)
    b1_4096 = _cell(table, "B1", 4096)
    scaled_2048 = _cell(table, "M_scaled", 2048)
    b1_2048 = _cell(table, "B1", 2048)
    scaled_8192 = _cell(table, "M_scaled", 8192)
    lines.extend(
        [
            "",
            "## Frontier / efficiency",
            "",
            f"- 2048: B1 loc={fmt(b1_2048.get('mean_localization_score'))} "
            f"M_scaled loc={fmt(scaled_2048.get('mean_localization_score'))}",
            f"- 4096 (primary): B1 loc={fmt(b1_4096.get('mean_localization_score'))} "
            f"M_scaled loc={fmt(scaled_4096.get('mean_localization_score'))}",
            f"- 8192: B1 loc={fmt(b1_8192.get('mean_localization_score'))} "
            f"M_scaled loc={fmt(scaled_8192.get('mean_localization_score'))}",
            f"- efficiency check Scaled@4096 vs Base@8192: "
            f"{fmt(scaled_4096.get('mean_localization_score'))} vs "
            f"{fmt(b1_8192.get('mean_localization_score'))}",
            "- Main plot: `quality_budget_curve.png` (B1 vs M_scaled). "
            "Prototype overlay: `quality_budget_curve_with_prototype.png`.",
            "",
            "## Programmatic trajectory cases (B1 vs M_scaled)",
            "",
            f"- RL-win: {(selected.get('rl_win') or {}).get('instance_id')}",
            f"- Base-win: {(selected.get('base_win') or {}).get('instance_id')}",
            f"- both-fail: {(selected.get('both_fail') or {}).get('instance_id')}",
            f"- n_candidates: {cases.get('n_candidates')}",
        ]
    )
    for item in cases.get("cases") or []:
        notes = item.get("behavior_notes") or {}
        lines.append(
            f"- {item.get('category')} `{item.get('instance_id')}` "
            f"B={item.get('obs_tokens_limit')} Δ={fmt(item.get('delta'))}: "
            f"first_query same={notes.get('same_first_query')} "
            f"B1_q={notes.get('B1_first_query')!r} "
            f"M_scaled_q={notes.get('M_scaled_first_query')!r} "
            f"first_read B1/M_scaled={notes.get('B1_first_read_turn')}/"
            f"{notes.get('M_scaled_first_read_turn')} "
            f"protocol_degradation={notes.get('protocol_degradation')}"
        )
    if conclusion == "NULL":
        lines.extend(
            [
                "",
                "If localization is NULL, 275-step scale may have changed behavior "
                "(queries / read timing / protocol) without changing outcome. "
                "See case `behavior_notes` and `e018_cases.json`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"**{conclusion}**. Reliability class **{rel_class}** "
            "(A loc+rel up; B loc up; C protocol only; D both null; E regression).",
            "",
            "Do not retune parser/reward/scaffold from this eval. Do not enter M7 from this script.",
            "",
            "## Provenance",
            "",
            f"- overlay sha256: {(provenance or {}).get('overlay_sha256')}",
            f"- parent E015 freeze sha256: {(provenance or {}).get('parent_eval_sha256')}",
            f"- wandb: {(provenance or {}).get('wandb_url')}",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_wandb(output_dir: Path, payload: Mapping[str, Any]) -> None:
    try:
        import wandb
    except Exception:
        return
    run_info = {}
    info_path = output_dir / "wandb_run.json"
    if info_path.is_file():
        run_info = json.loads(info_path.read_text(encoding="utf-8"))
    if wandb.run is None:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_EXPERIMENT_NAME,
            id=run_info.get("id"),
            resume="allow",
            dir=str(output_dir / "wandb"),
        )
    flat: dict[str, Any] = {
        "conclusion": payload.get("conclusion"),
        "reliability_class": payload.get("reliability_class"),
    }
    for budget, stats in (payload.get("paired", {}).get("scaled_vs_b1") or {}).items():
        flat[f"delta_{budget}"] = stats.get("mean_delta")
    wandb.log(flat)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    errors = overlay_lock_errors(repo_root)
    if errors:
        print(f"HARD FAIL: eval lock {errors}", file=sys.stderr)
        return 1
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e018_output_dir(repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    native = [
        row
        for row in load_jsonl(args.episodes)
        if row.get("termination") != "operational_error"
    ]
    n_error = sum(
        1 for row in load_jsonl(args.episodes) if row.get("termination") == "operational_error"
    )
    if not native:
        print(f"HARD FAIL: no scored episodes in {args.episodes}", file=sys.stderr)
        return 1
    proto_path = args.prototype_episodes
    if proto_path is None:
        proto_path = default_e015_trace_dir() / "episodes_scored.jsonl"
    proto = import_e015_condition(proto_path, condition_id="M1", remap_to="M1_proto")
    reuse_path = output_dir / "reuse_audit.json"
    reuse = json.loads(reuse_path.read_text(encoding="utf-8")) if reuse_path.is_file() else reuse_base_audit(repo_root)
    rows = list(native)
    if reuse.get("allow_reuse") and args.reuse_base_episodes:
        rows.extend(import_e015_condition(args.reuse_base_episodes, condition_id="B0"))
        rows.extend(import_e015_condition(args.reuse_base_episodes, condition_id="B1"))
    rows.extend(proto)
    groups = load_correlation_groups(repo_root)
    table = main_table(rows, condition_ids=("B0", "B1", "M1_proto", "M_scaled"))
    paired_scaled: dict[str, Any] = {}
    paired_vis: dict[str, Any] = {}
    paired_proto: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        scaled_pairs = paired_cells(rows, left_id="B1", right_id="M_scaled", budget=budget)
        vis_pairs = paired_cells(rows, left_id="B0", right_id="B1", budget=budget)
        proto_pairs = paired_cells(rows, left_id="M1_proto", right_id="M_scaled", budget=budget)
        paired_scaled[str(budget)] = paired_summary_stats(
            scaled_pairs, left_id="B1", right_id="M_scaled", group_ids=groups
        )
        paired_vis[str(budget)] = paired_summary_stats(
            vis_pairs, left_id="B0", right_id="B1", group_ids=groups
        )
        paired_proto[str(budget)] = paired_summary_stats(
            proto_pairs, left_id="M1_proto", right_id="M_scaled", group_ids=groups
        )
        for item in scaled_pairs:
            compact = {key: item[key] for key in item if key not in {"left", "right"}}
            compact["budget"] = budget
            paired_rows.append(compact)
    all_scaled_pairs = paired_cells(rows, left_id="B1", right_id="M_scaled")
    per_repo_pooled = per_repo_breakdown(all_scaled_pairs, left_id="B1", right_id="M_scaled")
    curve_main = quality_budget_curve(rows, condition_ids=("B1", "M_scaled"))
    curve_all = quality_budget_curve(rows, condition_ids=("B0", "B1", "M1_proto", "M_scaled"))
    cases = attach_case_trajectories(
        select_case_studies(all_scaled_pairs, left_id="B1", right_id="M_scaled"),
        all_scaled_pairs,
        left_id="B1",
        right_id="M_scaled",
    )
    conclusion = scientific_conclusion(paired_scaled)
    b1_cells = [_cell(table, "B1", budget) for budget in BUDGETS]
    scaled_cells = [_cell(table, "M_scaled", budget) for budget in BUDGETS]
    rel_class = reliability_class(
        conclusion=conclusion, b1_cells=b1_cells, scaled_cells=scaled_cells
    )
    integrity = {}
    integrity_path = output_dir / "treatment_integrity.json"
    if integrity_path.is_file():
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    provenance = {
        "overlay_sha256": sha256_file(repo_root / "configs/experiments/stage1_m6_e018.json"),
        "parent_eval_sha256": sha256_file(repo_root / "configs/experiments/stage1_m6_eval.json"),
    }
    wandb_info = output_dir / "wandb_run.json"
    if wandb_info.is_file():
        provenance["wandb_url"] = json.loads(wandb_info.read_text(encoding="utf-8")).get("url")
    write_json(output_dir / "aggregates.json", {"table": table, "n_rows": len(native), "n_merged": len(rows)})
    write_json(
        output_dir / "paired_stats.json",
        {
            "scaled_vs_b1": paired_scaled,
            "visibility": paired_vis,
            "scaled_vs_proto": paired_proto,
        },
    )
    write_json(output_dir / "per_repo.json", {"pooled": per_repo_pooled})
    write_json(output_dir / "quality_budget_curve.json", curve_all)
    write_json(output_dir / "reliability.json", {"class": rel_class, "conclusion": conclusion})
    write_json(output_dir / "e018_cases.json", cases)
    paired_path = output_dir / "paired_episodes.jsonl"
    with paired_path.open("w", encoding="utf-8") as handle:
        for item in paired_rows:
            handle.write(json.dumps(item, ensure_ascii=True, default=str) + "\n")
    png_ok = write_curve_png(curve_main, output_dir / "quality_budget_curve.png", ("B1", "M_scaled"))
    aux_ok = write_curve_png(
        curve_all, output_dir / "quality_budget_curve_with_prototype.png", ("B1", "M1_proto", "M_scaled")
    )
    write_summary_md(
        output_dir / "SUMMARY.md",
        table=table,
        paired={
            "scaled_vs_b1": paired_scaled,
            "visibility": paired_vis,
            "scaled_vs_proto": paired_proto,
        },
        per_repo={"pooled": per_repo_pooled},
        reuse=reuse,
        integrity=integrity,
        conclusion=conclusion,
        rel_class=rel_class,
        n_rows=len(native),
        n_error=n_error,
        provenance=provenance,
        cases=cases,
    )
    _log_wandb(
        output_dir,
        {
            "conclusion": conclusion,
            "reliability_class": rel_class,
            "paired": {"scaled_vs_b1": paired_scaled},
        },
    )
    print(
        json.dumps(
            {
                "n_rows": len(native),
                "n_merged": len(rows),
                "n_proto": len(proto),
                "conclusion": conclusion,
                "reliability_class": rel_class,
                "curve_png": png_ok,
                "aux_png": aux_ok,
                "summary": str(output_dir / "SUMMARY.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
