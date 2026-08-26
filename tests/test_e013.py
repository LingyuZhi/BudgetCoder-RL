"""E013 20480 headroom overlay; E012 18432 overlay must stay frozen."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.e012 import EXPECTED_E011_OVERLAY_SHA256, REQUIRED_TASK
from budget_coder_rl.eval.e013 import (
    CHOSEN_ENVELOPE,
    EXPECTED_E012_ENVELOPE,
    EXPECTED_E012_OVERLAY_SHA256,
    STRESS_STEPS,
    STRESS_TRAIN_BATCH_SIZE,
    STRESS_UNIQUE_TASKS,
    consume_runtime_overlay,
    e012_untouched_errors,
    memory_healthy,
    overlay_errors,
    overlay_lock_errors,
    select_headroom_instance_ids,
)
from budget_coder_rl.eval.m5a import load_json
from budget_coder_rl.eval.m5b import EXPECTED_MAIN_SHA256, research_knob_errors
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_e012_overlay_and_main_freeze_untouched():
    main = REPO_ROOT / "configs/experiments/stage1_m5_main.json"
    e011 = REPO_ROOT / "configs/experiments/stage1_m5b_e011_runtime.json"
    e012 = REPO_ROOT / "configs/experiments/stage1_m5_e012_runtime.json"
    assert sha256_file(main) == EXPECTED_MAIN_SHA256
    assert sha256_file(e011) == EXPECTED_E011_OVERLAY_SHA256
    assert sha256_file(e012) == EXPECTED_E012_OVERLAY_SHA256
    assert e012_untouched_errors(REPO_ROOT) == []
    assert research_knob_errors(load_json(main)) == []
    freeze_ppo = ((load_json(main).get("newly_frozen") or {}).get("actor") or {}).get(
        "ppo_max_token_len_per_gpu"
    )
    assert int(freeze_ppo) == 16384
    e012_payload = json.loads(e012.read_text(encoding="utf-8"))
    assert int(e012_payload["overrides"]["systems"]["ppo_max_token_len_per_gpu"]) == (
        EXPECTED_E012_ENVELOPE
    )


def test_e013_overlay_is_20480_systems_only():
    overlay_path = REPO_ROOT / "configs/experiments/stage1_m5_e013_runtime.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_errors(overlay, repo_root=REPO_ROOT) == []
    assert overlay_lock_errors(REPO_ROOT) == []
    consumed = consume_runtime_overlay(repo_root=REPO_ROOT, overlay=overlay)
    assert consumed["n_gpus"] == 2
    assert consumed["tensor_model_parallel_size"] == 1
    assert consumed["parent_sha256"] == EXPECTED_MAIN_SHA256
    assert consumed["e012_sha256"] == EXPECTED_E012_OVERLAY_SHA256
    assert consumed["ppo_max_token_len_per_gpu"] == CHOSEN_ENVELOPE == 20480
    assert consumed["revision_reason"] == "sequence_capacity_headroom"
    assert consumed["stress_train_batch_size"] == STRESS_TRAIN_BATCH_SIZE == 2
    assert consumed["freeze_train_batch_size"] == 8


def test_e013_overlay_rejects_research_and_e012_mutation():
    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5_e013_runtime.json").read_text(
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

    bad_e012 = json.loads(json.dumps(overlay))
    bad_e012["e012_runtime"]["sha256"] = "0" * 64
    errors = overlay_errors(bad_e012, repo_root=None)
    assert any("e012_runtime" in item for item in errors)


def test_headroom_selection_is_two_long_prompts_including_dask():
    ranked = [
        {"instance_id": REQUIRED_TASK, "prompt_token_count": 12811},
        {"instance_id": "bokeh__bokeh-13535", "prompt_token_count": 5675},
        {"instance_id": "iterative__dvc-1650", "prompt_token_count": 5106},
    ]
    selected = select_headroom_instance_ids(ranked)
    assert selected == [REQUIRED_TASK, "bokeh__bokeh-13535"]
    assert len(selected) == STRESS_UNIQUE_TASKS
    assert STRESS_STEPS == 1


def test_memory_healthy_threshold():
    ok, reasons = memory_healthy(oom=False, peak_mib=33771)
    assert ok and reasons == []
    ok, reasons = memory_healthy(oom=True, peak_mib=20000)
    assert not ok and "OOM" in reasons
    ok, reasons = memory_healthy(oom=False, peak_mib=39000)
    assert not ok
    ok, reasons = memory_healthy(oom=False, peak_mib=None)
    assert not ok


EXPECTED_CANONICAL_SHA256 = (
    "0b5928dbf28fd3f5949b3f62dcac47b23970b900a42b595c6fee6514c2986f65"
)


def test_canonical_execution_envelope_frozen_at_20480():
    path = REPO_ROOT / "configs/experiments/stage1_canonical_execution_envelope.json"
    lock_path = REPO_ROOT / "configs/experiments/stage1_canonical_execution_envelope.lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert sha256_file(path) == EXPECTED_CANONICAL_SHA256
    assert lock["sha256"] == EXPECTED_CANONICAL_SHA256
    assert payload["status"] == "frozen"
    assert int(payload["ppo_max_token_len_per_gpu"]) == 20480
    assert payload["source_experiment_id"] == "E013"
    assert payload["canonical_32_step_not_started"] is True
    assert payload["e012_kept"]["sha256"] == EXPECTED_E012_OVERLAY_SHA256
    assert int(payload["e012_kept"]["ppo_max_token_len_per_gpu"]) == 18432
    assert payload["parent_freeze"]["sha256"] == EXPECTED_MAIN_SHA256
    assert int(payload["parent_freeze"]["ppo_max_token_len_per_gpu_in_freeze_json"]) == 16384
