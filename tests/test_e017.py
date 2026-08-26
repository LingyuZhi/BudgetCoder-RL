"""E017 namespace overlay; scaled freeze and E014/E015/E016 artifacts stay immutable."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.e017 import (
    CHECKPOINT_RELPATH,
    EXPECTED_OVERLAY_SHA256,
    EXPERIMENT_ID,
    LAUNCH_DISK_MIN_GIB,
    MAIN_STEPS,
    MIN_REMAINING_HOURS,
    WANDB_EXPERIMENT_NAME,
    checkpoint_path_errors,
    consume_e017_overlay,
    default_e017_output_dir,
    default_overlay_path,
    forbidden_output_dir_errors,
    overlay_errors,
    overlay_lock_errors,
    parse_slurm_duration,
)
from budget_coder_rl.eval.m5_scaled import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    EXPECTED_PADDED_IDS_SHA256,
    EXPECTED_UNIQUE_IDS_SHA256,
    N_ROWS,
    N_TRAJECTORIES,
    N_UNIQUE,
    consume_scaled_errors,
    default_preflight_checkpoint_dir,
    default_scaled_checkpoint_dir,
    save_event_steps,
)
from budget_coder_rl.eval.m5a import load_json
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scaled_contract_hashes_unchanged():
    assert consume_scaled_errors(REPO_ROOT) == []
    assert sha256_file(REPO_ROOT / "configs/experiments/stage1_m5_scaled.json") == (
        EXPECTED_CONTRACT_SHA256
    )
    assert sha256_file(REPO_ROOT / "data/manifests/m5_scaled_train_candidates.json") == (
        EXPECTED_MANIFEST_FILE_SHA256
    )


def test_e017_overlay_is_namespace_only():
    overlay_path = default_overlay_path(REPO_ROOT)
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_errors(overlay, repo_root=REPO_ROOT) == []
    from budget_coder_rl.eval.e017 import write_overlay_lock

    lock = write_overlay_lock(REPO_ROOT)
    digest = sha256_file(overlay_path)
    assert lock["sha256"] == digest
    if EXPECTED_OVERLAY_SHA256 != "0" * 64:
        assert digest == EXPECTED_OVERLAY_SHA256
    assert overlay_lock_errors(REPO_ROOT) == []
    consumed = consume_e017_overlay(repo_root=REPO_ROOT, overlay=overlay)
    assert consumed["n_steps"] == MAIN_STEPS == 275
    assert consumed["n_unique"] == N_UNIQUE == 2193
    assert consumed["n_rows"] == N_ROWS == 2200
    assert consumed["n_trajectories"] == N_TRAJECTORIES == 8800
    assert consumed["group_n"] == 4
    assert consumed["save_freq"] == 32
    assert consumed["ppo_max_token_len_per_gpu"] == 20480
    assert consumed["experiment_name"] == WANDB_EXPERIMENT_NAME
    assert consumed["resume_mode"] == "disable"
    assert "stage1_m5_scaled_e017" in consumed["default_local_dir"]
    assert "total_training_steps" not in (overlay.get("overrides") or {}).get("trainer", {})
    assert overlay["do_not_start_275"] is False
    assert overlay["not_preflight"] is True
    assert overlay["experiment_id"] == EXPERIMENT_ID
    unique = load_json(REPO_ROOT / "data/manifests/m5_scaled_train_candidates.json")
    assert unique["unique_ids_sha256"] == EXPECTED_UNIQUE_IDS_SHA256
    assert unique["padded_ids_sha256"] == EXPECTED_PADDED_IDS_SHA256


def test_e017_overlay_rejects_preflight_and_research_knobs():
    overlay = json.loads(default_overlay_path(REPO_ROOT).read_text(encoding="utf-8"))

    bad_steps = json.loads(json.dumps(overlay))
    bad_steps["overrides"]["trainer"]["total_training_steps"] = 2
    errors = overlay_errors(bad_steps, repo_root=None)
    assert errors

    bad_g = json.loads(json.dumps(overlay))
    bad_g["overrides"]["algorithm"] = {"rollout_n": 8}
    errors = overlay_errors(bad_g, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)

    bad_lr = json.loads(json.dumps(overlay))
    bad_lr["overrides"]["actor"] = {"optim_lr": 1e-4}
    errors = overlay_errors(bad_lr, repo_root=None)
    assert errors

    bad_save = json.loads(json.dumps(overlay))
    bad_save["overrides"]["trainer"]["save_freq"] = 1
    errors = overlay_errors(bad_save, repo_root=None)
    assert errors

    bad_pre = json.loads(json.dumps(overlay))
    bad_pre["do_not_start_275"] = True
    errors = overlay_errors(bad_pre, repo_root=None)
    assert any("do_not_start_275" in item for item in errors)


def test_e017_refuses_historical_output_and_checkpoint_dirs():
    repo = REPO_ROOT
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E014", repo)
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E015", repo)
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E016", repo)
    assert forbidden_output_dir_errors(default_e017_output_dir(repo), repo) == []
    data_root = Path("/tmp/bcrl-e017-test-data")
    e017_ckpt = data_root / CHECKPOINT_RELPATH
    e017_ckpt.mkdir(parents=True, exist_ok=True)
    assert checkpoint_path_errors(e017_ckpt, data_root) == []
    assert checkpoint_path_errors(default_scaled_checkpoint_dir(data_root), data_root)
    assert checkpoint_path_errors(default_preflight_checkpoint_dir(data_root), data_root)


def test_e017_save_cadence_and_slurm_parser():
    assert save_event_steps(275, 32) == [32, 64, 96, 128, 160, 192, 224, 256, 275]
    elapsed = parse_slurm_duration("11-18:45:33")
    limit = parse_slurm_duration("14-00:00:00")
    assert elapsed is not None and limit is not None
    remaining_h = (limit - elapsed) / 3600.0
    assert remaining_h > MIN_REMAINING_HOURS
    assert LAUNCH_DISK_MIN_GIB == 120.0
    assert parse_slurm_duration("20:00:00") == 20 * 3600
    assert parse_slurm_duration("UNLIMITED") == float("inf")
