#!/usr/bin/env python
"""Write scaled-M5 freeze JSON + lock + E016 preflight overlay.

Does not edit stage1_m3c_freeze.json, stage1_m5_main.json, E014, or E015.
Does not start GPU training.

Usage:

    python scripts/eval/write_m5_scaled_freeze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m3c import default_freeze_path  # noqa: E402
from budget_coder_rl.eval.m4a import load_json  # noqa: E402
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m5_scaled import (  # noqa: E402
    CANONICAL_ENVELOPE_RELPATH,
    PREFLIGHT_STEPS,
    build_preflight_overlay,
    build_scaled_contract,
    default_candidate_path,
    default_contract_lock_path,
    default_contract_path,
    default_preflight_checkpoint_dir,
    default_preflight_lock_path,
    default_preflight_output_dir,
    default_preflight_path,
    historical_untouched_errors,
    manifest_errors,
    preflight_overlay_errors,
    scaled_contract_errors,
)
from budget_coder_rl.eval.provenance import git_info, sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    untouched = historical_untouched_errors(repo_root)
    if untouched:
        print(f"HARD FAIL: historical artifacts mutated {untouched}", file=sys.stderr)
        return 1
    candidate_path = default_candidate_path(repo_root)
    freeze_path = default_freeze_path(repo_root)
    envelope_path = repo_root / CANONICAL_ENVELOPE_RELPATH
    if not candidate_path.is_file():
        print(f"HARD FAIL: missing scaled manifest {candidate_path}", file=sys.stderr)
        return 1
    candidate = load_json(candidate_path)
    errors = manifest_errors(candidate)
    if errors:
        print(f"HARD FAIL: scaled manifest {errors}", file=sys.stderr)
        return 1
    freeze = load_json(freeze_path)
    commit = git_info(repo_root).get("commit")
    contract = build_scaled_contract(
        freeze=freeze,
        freeze_path=freeze_path,
        candidate=candidate,
        candidate_path=candidate_path,
        envelope_path=envelope_path,
        project_commit=commit,
    )
    contract_errors = scaled_contract_errors(contract)
    if contract_errors:
        print(f"HARD FAIL: scaled contract {contract_errors}", file=sys.stderr)
        return 1
    contract_path = default_contract_path(repo_root)
    write_json(contract_path, contract)
    contract_sha = sha256_file(contract_path)
    lock = {
        "path": "configs/experiments/stage1_m5_scaled.json",
        "sha256": contract_sha,
        "manifest_path": "data/manifests/m5_scaled_train_candidates.json",
        "manifest_sha256": sha256_file(candidate_path),
        "unique_ids_sha256": candidate.get("unique_ids_sha256"),
        "padded_ids_sha256": candidate.get("padded_ids_sha256"),
        "n_unique": candidate.get("n_unique"),
        "n_rows": candidate.get("n_rows"),
        "optimizer_steps": candidate.get("optimizer_steps"),
        "group_n": candidate.get("group_n"),
        "ppo_max_token_len_per_gpu": 20480,
        "save_freq": 32,
        "note": (
            "Do not edit stage1_m3c_freeze.json, stage1_m5_main.json, E014, or E015. "
            "Do not start the 275-step run until READY_FOR_SCALED_M5_MAIN=true."
        ),
    }
    write_json(default_contract_lock_path(repo_root), lock)

    output_dir = default_preflight_output_dir(repo_root)
    overlay = build_preflight_overlay(
        output_dir=output_dir,
        checkpoint_dir=default_preflight_checkpoint_dir(),
        n_steps=PREFLIGHT_STEPS,
    )
    overlay_errors = preflight_overlay_errors(overlay, contract=contract)
    if overlay_errors:
        print(f"HARD FAIL: preflight overlay {overlay_errors}", file=sys.stderr)
        return 1
    overlay_path = default_preflight_path(repo_root)
    write_json(overlay_path, overlay)
    overlay_sha = sha256_file(overlay_path)
    write_json(
        default_preflight_lock_path(repo_root),
        {
            "path": "configs/experiments/stage1_m5_scaled_e016_preflight.json",
            "sha256": overlay_sha,
            "parent_path": "configs/experiments/stage1_m5_scaled.json",
            "parent_sha256": contract_sha,
            "experiment_id": "E016",
            "n_preflight_steps": PREFLIGHT_STEPS,
            "do_not_start_275": True,
            "note": "Disposable preflight overlay. Not a 275-step main run.",
        },
    )
    print(
        json.dumps(
            {
                "contract": str(contract_path),
                "contract_sha256": contract_sha,
                "manifest_sha256": lock["manifest_sha256"],
                "preflight": str(overlay_path),
                "preflight_sha256": overlay_sha,
                "optimizer_steps": 275,
                "preflight_steps": PREFLIGHT_STEPS,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
