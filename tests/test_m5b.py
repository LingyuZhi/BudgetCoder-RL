"""M5B overlay allowlist, parent hash, TP=1, and checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m3c import FREEZE_RELPATH
from budget_coder_rl.eval.m4a import freeze_contract_errors
from budget_coder_rl.eval.m5a import MAIN_STEPS, TRAIN_BATCH_SIZE, load_json
from budget_coder_rl.eval.m5b import (
    CANONICAL_CHECKPOINT_STEP,
    CHECKPOINT_SELECTION_RULE,
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_TP,
    EXPERIMENT_ID,
    consume_runtime_overlay,
    expected_hybrid_placement,
    overlay_errors,
    overlay_lock_errors,
    placement_errors,
    research_knob_errors,
    selected_m6_candidate,
)
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_immutable_main_hash_unchanged():
    path = REPO_ROOT / "configs/experiments/stage1_m5_main.json"
    assert sha256_file(path) == EXPECTED_MAIN_SHA256
    lock = json.loads((REPO_ROOT / "configs/experiments/stage1_m5_main.lock.json").read_text())
    assert lock["sha256"] == EXPECTED_MAIN_SHA256
    assert lock["READY_FOR_M5B"] is True


def test_m3c_freeze_still_untouched():
    path = REPO_ROOT / FREEZE_RELPATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert freeze_contract_errors(payload) == []
    assert payload["not_trained"] is True
    assert payload["grpo_optimizer"] is False


def test_runtime_overlay_allowlist_and_parent_hash():
    overlay_path = REPO_ROOT / "configs/experiments/stage1_m5b_e011_runtime.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_errors(overlay, repo_root=REPO_ROOT) == []
    assert overlay_lock_errors(REPO_ROOT) == []
    consumed = consume_runtime_overlay(repo_root=REPO_ROOT, overlay=overlay)
    assert consumed["n_gpus"] == EXPECTED_N_GPUS
    assert consumed["tensor_model_parallel_size"] == EXPECTED_TP
    assert consumed["parent_sha256"] == EXPECTED_MAIN_SHA256
    assert consumed["canonical_global_step"] == CANONICAL_CHECKPOINT_STEP
    assert consumed["checkpoint_selection_rule"] == CHECKPOINT_SELECTION_RULE


def test_overlay_rejects_research_knob_changes():
    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5b_e011_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    bad_lr = json.loads(json.dumps(overlay))
    bad_lr["overrides"]["actor"] = {"optim_lr": 1e-4}
    errors = overlay_errors(bad_lr, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)

    bad_g = json.loads(json.dumps(overlay))
    bad_g["overrides"]["algorithm"] = {"rollout_n": 8}
    errors = overlay_errors(bad_g, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)

    bad_b = json.loads(json.dumps(overlay))
    bad_b["overrides"]["inherited_m3c"] = {"primary_training_B_obs": 8192}
    errors = overlay_errors(bad_b, repo_root=None)
    assert any("research" in item or "disallowed" in item for item in errors)


def test_overlay_rejects_tensor_parallel_size_2():
    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5b_e011_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    overlay["overrides"]["gpu"]["tensor_model_parallel_size"] = 2
    errors = overlay_errors(overlay, repo_root=None)
    assert any("tensor_model_parallel_size" in item for item in errors)


def test_expected_hybrid_placement_is_two_tp1_replicas():
    placement = expected_hybrid_placement(n_gpus=2, tensor_model_parallel_size=1)
    assert placement["fsdp_world_size"] == 2
    assert placement["n_vllm_replicas"] == 2
    assert placement["vllm_tensor_model_parallel_size"] == 1
    assert placement_errors(placement) == []
    tp2 = expected_hybrid_placement(n_gpus=2, tensor_model_parallel_size=2)
    assert tp2["n_vllm_replicas"] == 1
    assert placement_errors(tp2, tensor_model_parallel_size=2)


def test_checkpoint_selection_is_terminal_step_32_not_curve():
    chosen = selected_m6_candidate(
        Path("/tmp/stage1_m5_main"),
        metrics={"bcrl/reward/mean": 0.99, "best_step": 16},
    )
    assert chosen["rule"] == CHECKPOINT_SELECTION_RULE
    assert chosen["global_step"] == 32
    assert chosen["path"].endswith("global_step_32")
    assert chosen["post_hoc_curve_pick"] is False


def test_main_research_knobs_untouched_by_overlay_file():
    main = load_json(REPO_ROOT / "configs/experiments/stage1_m5_main.json")
    assert research_knob_errors(main) == []
    gpu = (main.get("newly_frozen") or {}).get("gpu") or {}
    assert int(gpu.get("n_gpus") or 0) == 1
    assert int((main.get("newly_frozen") or {}).get("trainer", {}).get("total_training_steps") or 0) == MAIN_STEPS
    assert int((main.get("newly_frozen") or {}).get("data", {}).get("train_batch_size") or 0) == TRAIN_BATCH_SIZE
    assert main["experiment_id"] == EXPERIMENT_ID
