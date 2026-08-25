#!/usr/bin/env python
"""M5B CPU + nvidia-smi preflight. Does not run GRPO or vLLM.

Usage:

    python scripts/eval/run_m5b_preflight.py --experiment-id E011
"""

from __future__ import annotations

import argparse
import json
import os
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
from budget_coder_rl.eval.m4a import (  # noqa: E402
    artifact_hashes,
    default_candidate_path,
    default_freeze_path,
    load_candidate_ordered_ids,
    load_json,
)
from budget_coder_rl.eval.m4b import PINNED_VERL_COMMIT, write_json  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    MAIN_STEPS,
    N_CANDIDATES,
    SHARED_VERL_ROOT,
    TRAIN_BATCH_SIZE,
    default_isolated_verl_root,
    default_main_config_path,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    m5_freeze_consume_errors,
    prepend_isolated_verl,
    write_verl_checkout_md,
)
from budget_coder_rl.eval.m5b import (  # noqa: E402
    DISK_MIN_GIB,
    EXPERIMENT_ID,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_M3C_SHA256,
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_TP,
    MILESTONE,
    checkpoint_dir_conflict_errors,
    consume_runtime_overlay,
    default_checkpoint_dir,
    default_e011_output_dir,
    default_runtime_config_path,
    disk_capacity_errors,
    expected_hybrid_placement,
    is_login_host,
    project_tree_dirty_errors,
    research_knob_errors,
    resource_lifecycle,
    sample_nvidia_gpus,
)
from budget_coder_rl.eval.provenance import git_info, sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    data_root = Path(args.data_root) if args.data_root else bcrl_data_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e011_output_dir(repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    if not args.allow_dirty:
        errors.extend(project_tree_dirty_errors(repo_root))

    freeze_path = default_freeze_path(repo_root)
    candidate_path = default_candidate_path(repo_root)
    main_path = default_main_config_path(repo_root)
    freeze = load_json(freeze_path)
    errors.extend(
        m5_freeze_consume_errors(
            freeze, freeze_path=freeze_path, candidate_path=candidate_path
        )
    )
    if sha256_file(freeze_path) != EXPECTED_M3C_SHA256:
        errors.append(
            f"M3C freeze sha256 {sha256_file(freeze_path)} != {EXPECTED_M3C_SHA256}"
        )
    if sha256_file(candidate_path) != EXPECTED_CANDIDATE_SHA256:
        errors.append(
            f"candidate sha256 {sha256_file(candidate_path)} != {EXPECTED_CANDIDATE_SHA256}"
        )
    if sha256_file(main_path) != EXPECTED_MAIN_SHA256:
        errors.append(
            f"main config sha256 {sha256_file(main_path)} != {EXPECTED_MAIN_SHA256}"
        )

    main_cfg = load_json(main_path)
    errors.extend(research_knob_errors(main_cfg))
    try:
        runtime = consume_runtime_overlay(repo_root=repo_root)
    except Exception as exc:
        errors.append(f"{exc}")
        runtime = {}

    isolated_root = default_isolated_verl_root(data_root)
    verl_info: dict[str, Any] = {}
    verl_runtime: dict[str, Any] = {}
    try:
        verl_info = ensure_isolated_verl_checkout(
            isolated_root=isolated_root,
            source_git=SHARED_VERL_ROOT,
            pinned_commit=PINNED_VERL_COMMIT,
            create=True,
        )
        prepend_isolated_verl(isolated_root, repo_root)
        verl_import_errors, verl_runtime = imported_verl_errors(
            isolated_root=isolated_root
        )
        errors.extend(verl_import_errors)
        write_verl_checkout_md(output_dir / "verl_checkout.md", verl_info)
    except Exception as exc:
        errors.append(f"isolated veRL: {exc}")

    ordered_ids = load_candidate_ordered_ids(candidate_path)
    if len(ordered_ids) != N_CANDIDATES:
        errors.append(f"ordered_ids n={len(ordered_ids)} != {N_CANDIDATES}")
    if N_CANDIDATES // TRAIN_BATCH_SIZE != MAIN_STEPS:
        errors.append("256 / 8 != 32")

    checkpoint_root = default_checkpoint_dir(data_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    errors.extend(
        checkpoint_dir_conflict_errors(
            checkpoint_root, allow_resume=bool(args.allow_resume)
        )
    )
    errors.extend(disk_capacity_errors(checkpoint_root, output_dir, min_gib=DISK_MIN_GIB))

    lifecycle = resource_lifecycle()
    gpus: list[dict[str, Any]] = []
    if not args.skip_gpu and not is_login_host():
        gpus = sample_nvidia_gpus()
        if len([row for row in gpus if "error" not in row]) < EXPECTED_N_GPUS:
            errors.append(f"need {EXPECTED_N_GPUS} GPUs, nvidia-smi={gpus}")

    placement = expected_hybrid_placement(
        n_gpus=int(runtime.get("n_gpus") or EXPECTED_N_GPUS),
        tensor_model_parallel_size=int(
            runtime.get("tensor_model_parallel_size") or EXPECTED_TP
        ),
    )
    model_path = data_root / "models" / "Qwen3-4B-Instruct-2507"
    if not model_path.exists():
        errors.append(f"missing model {model_path}")
    for required in (
        train_parquet_path(repo_root),
        oracle_parquet_path(repo_root),
        repo_root / "src/budget_coder_rl/reward/localization_score.py",
        repo_root / "configs/agent_loop/repo_exploration_m3c.yaml",
        default_runtime_config_path(repo_root),
    ):
        if not Path(required).is_file():
            errors.append(f"missing {required}")

    status = "PREFLIGHT_OK" if not errors else "PREFLIGHT_FAIL"
    summary = {
        "status": status,
        "errors": errors,
        "experiment_id": args.experiment_id,
        "milestone": MILESTONE,
        "hostname": lifecycle.get("hostname"),
        "lifecycle": lifecycle,
        "project": git_info(repo_root),
        "main_config_sha256": sha256_file(main_path),
        "m3c_freeze_sha256": sha256_file(freeze_path),
        "candidate_sha256": sha256_file(candidate_path),
        "overlay": runtime,
        "isolated_verl": verl_info,
        "verl_runtime": verl_runtime,
        "n_ordered_ids": len(ordered_ids),
        "checkpoint_root": str(checkpoint_root),
        "expected_placement": placement,
        "gpus": gpus,
        "artifacts": artifact_hashes(
            {
                "freeze": freeze_path,
                "candidates": candidate_path,
                "main_config": main_path,
                "overlay": default_runtime_config_path(repo_root),
            }
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "WANDB_API_KEY_set": bool(os.environ.get("WANDB_API_KEY")),
    }
    write_json(output_dir / "preflight_status.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
