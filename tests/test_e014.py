"""E014 canonical 32-step overlay; freeze and E011-E013 artifacts stay immutable."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.e012 import EXPECTED_E011_OVERLAY_SHA256
from budget_coder_rl.eval.e013 import EXPECTED_E012_OVERLAY_SHA256
from budget_coder_rl.eval.e014 import (
    CHOSEN_ENVELOPE,
    EXPECTED_CANONICAL_SHA256,
    EXPECTED_E013_OVERLAY_SHA256,
    consume_runtime_overlay,
    forbidden_output_dir_errors,
    historical_untouched_errors,
    overlay_errors,
    overlay_lock_errors,
)
from budget_coder_rl.eval.m5a import load_json
from budget_coder_rl.eval.m5b import EXPECTED_MAIN_SHA256, research_knob_errors
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_historical_freeze_and_overlays_untouched():
    assert historical_untouched_errors(REPO_ROOT) == []
    main = load_json(REPO_ROOT / "configs/experiments/stage1_m5_main.json")
    assert research_knob_errors(main) == []
    assert sha256_file(REPO_ROOT / "configs/experiments/stage1_m5_main.json") == EXPECTED_MAIN_SHA256
    assert sha256_file(REPO_ROOT / "configs/experiments/stage1_m5b_e011_runtime.json") == (
        EXPECTED_E011_OVERLAY_SHA256
    )
    assert sha256_file(REPO_ROOT / "configs/experiments/stage1_m5_e012_runtime.json") == (
        EXPECTED_E012_OVERLAY_SHA256
    )
    assert sha256_file(REPO_ROOT / "configs/experiments/stage1_m5_e013_runtime.json") == (
        EXPECTED_E013_OVERLAY_SHA256
    )
    assert sha256_file(
        REPO_ROOT / "configs/experiments/stage1_canonical_execution_envelope.json"
    ) == EXPECTED_CANONICAL_SHA256
    freeze_ppo = ((main.get("newly_frozen") or {}).get("actor") or {}).get(
        "ppo_max_token_len_per_gpu"
    )
    assert int(freeze_ppo) == 16384


def test_e014_overlay_is_20480_systems_only():
    overlay_path = REPO_ROOT / "configs/experiments/stage1_m5_e014_runtime.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_errors(overlay, repo_root=REPO_ROOT) == []
    assert overlay_lock_errors(REPO_ROOT) == []
    consumed = consume_runtime_overlay(repo_root=REPO_ROOT, overlay=overlay)
    assert consumed["n_gpus"] == 2
    assert consumed["tensor_model_parallel_size"] == 1
    assert consumed["parent_sha256"] == EXPECTED_MAIN_SHA256
    assert consumed["canonical_sha256"] == EXPECTED_CANONICAL_SHA256
    assert consumed["ppo_max_token_len_per_gpu"] == CHOSEN_ENVELOPE == 20480
    assert consumed["revision_reason"] == "canonical_32_step_rerun"
    assert consumed["n_steps"] == 32
    assert consumed["train_batch_size"] == 8
    assert consumed["n_candidates"] == 256


def test_e014_overlay_rejects_research_and_wrong_envelope():
    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5_e014_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    bad = json.loads(json.dumps(overlay))
    bad["overrides"]["actor"] = {"optim_lr": 1e-4}
    errors = overlay_errors(bad, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)

    bad_g = json.loads(json.dumps(overlay))
    bad_g["overrides"]["algorithm"] = {"rollout_n": 8}
    errors = overlay_errors(bad_g, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)

    bad_b = json.loads(json.dumps(overlay))
    bad_b["overrides"]["inherited_m3c"] = {"primary_training_B_obs": 8192}
    errors = overlay_errors(bad_b, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)

    bad_env = json.loads(json.dumps(overlay))
    bad_env["overrides"]["systems"]["ppo_max_token_len_per_gpu"] = 18432
    errors = overlay_errors(bad_env, repo_root=None)
    assert any("20480" in item for item in errors)

    bad_env["overrides"]["systems"]["ppo_max_token_len_per_gpu"] = 32768
    errors = overlay_errors(bad_env, repo_root=None)
    assert errors

    bad_tp = json.loads(json.dumps(overlay))
    bad_tp["overrides"]["gpu"]["tensor_model_parallel_size"] = 2
    errors = overlay_errors(bad_tp, repo_root=None)
    assert any("tensor_model_parallel_size" in item for item in errors)

    bad_canonical = json.loads(json.dumps(overlay))
    bad_canonical["canonical_envelope"]["sha256"] = "0" * 64
    errors = overlay_errors(bad_canonical, repo_root=None)
    assert any("canonical_envelope" in item for item in errors)


def test_e014_refuses_historical_output_dirs():
    repo = REPO_ROOT
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E011", repo)
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E012", repo)
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E013", repo)
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E014", repo) == []
