#!/usr/bin/env python
"""M5A CPU preflight: isolated veRL, seqlen stats, draft main/pilot freeze.

Does not run GPU or GRPO. Writes configs/experiments/stage1_m5_main.json
and the E010 provenance bundle.

Usage:

    python scripts/eval/run_m5a_preflight.py --experiment-id E010
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4a import (  # noqa: E402
    artifact_hashes,
    default_candidate_path,
    default_freeze_path,
    load_candidate_ordered_ids,
    load_json,
)
from budget_coder_rl.eval.m4b import PINNED_VERL_COMMIT, write_json  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    EXPERIMENT_ID,
    MAIN_CONFIG_RELPATH,
    MILESTONE,
    PILOT_CONFIG_RELPATH,
    SHARED_VERL_ROOT,
    build_main_contract,
    build_pilot_overlay,
    main_contract_immutable,
    characterize_training_seq_lengths,
    default_e007_episodes_path,
    default_isolated_verl_root,
    default_output_dir,
    ensure_isolated_verl_checkout,
    m5_freeze_consume_errors,
    select_prefix_instance_ids,
    write_verl_checkout_md,
)
from budget_coder_rl.eval.provenance import collect_run_provenance, git_info  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verl-source", type=Path, default=SHARED_VERL_ROOT)
    parser.add_argument("--isolated-verl", type=Path, default=None)
    parser.add_argument("--skip-worktree", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    data_root = Path(args.data_root) if args.data_root else bcrl_data_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(repo_root, args.experiment_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_path = default_freeze_path(repo_root)
    candidate_path = default_candidate_path(repo_root)
    freeze = load_json(freeze_path)
    freeze_errors = m5_freeze_consume_errors(
        freeze, freeze_path=freeze_path, candidate_path=candidate_path
    )
    if freeze_errors:
        print(f"HARD FAIL: freeze contract {freeze_errors}", file=sys.stderr)
        return 1

    isolated_root = (
        args.isolated_verl.resolve()
        if args.isolated_verl is not None
        else default_isolated_verl_root(data_root)
    )
    verl_info = ensure_isolated_verl_checkout(
        isolated_root=isolated_root,
        source_git=args.verl_source,
        pinned_commit=PINNED_VERL_COMMIT,
        create=not args.skip_worktree,
    )
    write_verl_checkout_md(output_dir / "verl_checkout.md", verl_info)

    ordered_ids = load_candidate_ordered_ids(candidate_path)
    prefix_ids = select_prefix_instance_ids(ordered_ids)
    episode_paths = [
        default_e007_episodes_path(data_root),
        repo_root / "outputs/experiments/E008/episodes.jsonl",
        repo_root / "outputs/experiments/E009/episodes.jsonl",
    ]
    seqlen = characterize_training_seq_lengths(
        episode_paths=[path for path in episode_paths if path.is_file()],
        mask_path=repo_root / "outputs/experiments/E003/loss_mask_evidence.json",
        candidate_ids=ordered_ids,
    )
    proposed = seqlen.get("proposed_ppo_max_token_len_per_gpu")
    if proposed is None:
        print("HARD FAIL: could not propose ppo_max_token_len_per_gpu", file=sys.stderr)
        return 1
    write_json(output_dir / "seqlen_stats.json", seqlen)

    project = git_info(repo_root)
    main_path = repo_root / MAIN_CONFIG_RELPATH
    existing_main = load_json(main_path) if main_path.is_file() else {}
    if main_contract_immutable(existing_main):
        main = existing_main
        print("main config is immutable; not rewriting", main_path)
    else:
        main = build_main_contract(
            freeze=freeze,
            freeze_path=freeze_path,
            candidate_path=candidate_path,
            seqlen=seqlen,
            verl_isolated=verl_info,
            project_commit=project.get("commit"),
            ppo_max_token_len_per_gpu=int(proposed),
        )
        write_json(main_path, main)
    overlay = build_pilot_overlay(output_dir=output_dir)
    overlay_path = repo_root / PILOT_CONFIG_RELPATH
    if not main_contract_immutable(existing_main):
        write_json(overlay_path, overlay)

    knob_audit = {
        "inherited_m3c": sorted((main.get("inherited_m3c") or {}).keys()),
        "inherited_m4_runtime": sorted((main.get("inherited_m4_runtime") or {}).keys()),
        "newly_frozen": sorted((main.get("newly_frozen") or {}).keys()),
        "not_swept": [
            "reward formula",
            "prompt/tools/parser",
            "B_obs",
            "LoRA rank/alpha",
            "actor lr",
            "group size",
        ],
    }
    write_json(output_dir / "knob_audit.json", knob_audit)

    provenance = collect_run_provenance(
        repo_root,
        verl_source=isolated_root,
        agent_loop_config=repo_root / "configs/agent_loop/repo_exploration_m3c.yaml",
    )
    provenance["experiment_id"] = args.experiment_id
    provenance["milestone"] = MILESTONE
    provenance["phase"] = "preflight"
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["isolated_verl"] = verl_info
    provenance["seqlen"] = {
        "measured_max": seqlen.get("measured_max"),
        "proposed_ppo_max_token_len_per_gpu": proposed,
        "training_seq_proxy": seqlen.get("training_seq_proxy"),
    }
    provenance["pilot_prefix_ids"] = prefix_ids
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "candidates": candidate_path,
            "main_config": main_path,
            "pilot_config": overlay_path,
            "oracle": oracle_parquet_path(repo_root),
            "train_parquet": train_parquet_path(repo_root),
        }
    )
    provenance["data_root"] = str(data_root)
    provenance["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "preflight_provenance.json", provenance)

    summary = {
        "status": "PREFLIGHT_OK",
        "isolated_verl": str(isolated_root),
        "verl_commit": verl_info.get("commit"),
        "verl_dirty": verl_info.get("dirty"),
        "ppo_max_token_len_per_gpu": proposed,
        "training_seq_proxy": seqlen.get("training_seq_proxy"),
        "main_config": str(main_path),
        "pilot_config": str(overlay_path),
        "n_prefix_ids": len(prefix_ids),
        "bcrl_commit": project.get("commit"),
        "bcrl_dirty": project.get("dirty"),
    }
    write_json(output_dir / "preflight_status.json", summary)
    print(json_dumps(summary))
    return 0


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
