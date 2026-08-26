"""E012 systems overlay, envelope choice, and long-prompt selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from budget_coder_rl.eval.e012 import (
    ENVELOPE_CANDIDATES,
    EXPECTED_E011_OVERLAY_SHA256,
    REQUIRED_TASK,
    STRESS_STEPS,
    STRESS_UNIQUE_TASKS,
    choose_envelope,
    consume_runtime_overlay,
    overlay_errors,
    overlay_lock_errors,
    repeat_ids_for_steps,
    select_long_prompt_instance_ids,
)
from budget_coder_rl.eval.m5a import load_json
from budget_coder_rl.eval.m5b import EXPECTED_MAIN_SHA256, research_knob_errors
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_immutable_main_and_e011_overlay_untouched():
    main = REPO_ROOT / "configs/experiments/stage1_m5_main.json"
    e011 = REPO_ROOT / "configs/experiments/stage1_m5b_e011_runtime.json"
    assert sha256_file(main) == EXPECTED_MAIN_SHA256
    assert sha256_file(e011) == EXPECTED_E011_OVERLAY_SHA256
    assert research_knob_errors(load_json(main)) == []
    freeze_ppo = ((load_json(main).get("newly_frozen") or {}).get("actor") or {}).get(
        "ppo_max_token_len_per_gpu"
    )
    assert int(freeze_ppo) == 16384


def test_e012_overlay_allowlist_parent_and_e011_hashes():
    overlay_path = REPO_ROOT / "configs/experiments/stage1_m5_e012_runtime.json"
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_errors(overlay, repo_root=REPO_ROOT) == []
    assert overlay_lock_errors(REPO_ROOT) == []
    consumed = consume_runtime_overlay(repo_root=REPO_ROOT, overlay=overlay)
    assert consumed["n_gpus"] == 2
    assert consumed["tensor_model_parallel_size"] == 1
    assert consumed["parent_sha256"] == EXPECTED_MAIN_SHA256
    assert consumed["e011_sha256"] == EXPECTED_E011_OVERLAY_SHA256
    assert consumed["ppo_max_token_len_per_gpu"] in ENVELOPE_CANDIDATES
    assert consumed["ppo_max_token_len_per_gpu"] == 18432
    assert consumed["revision_reason"] == "sequence_capacity_exceeded"
    assert sha256_file(overlay_path) == (
        "a71f4557e6ca752715f81b21a18d3a70d810950af47d99c9bed7c9bfdf30fcf1"
    )


def test_overlay_rejects_research_knob_changes():
    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5_e012_runtime.json").read_text(
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

    bad_sys = json.loads(json.dumps(overlay))
    bad_sys["overrides"]["systems"]["optim_lr"] = 1e-4
    errors = overlay_errors(bad_sys, repo_root=None)
    assert any("systems overrides disallowed" in item for item in errors)


def test_overlay_rejects_tp2_and_envelope_32768():
    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5_e012_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    overlay["overrides"]["gpu"]["tensor_model_parallel_size"] = 2
    errors = overlay_errors(overlay, repo_root=None)
    assert any("tensor_model_parallel_size" in item for item in errors)

    overlay = json.loads(
        (REPO_ROOT / "configs/experiments/stage1_m5_e012_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    overlay["overrides"]["systems"]["ppo_max_token_len_per_gpu"] = 32768
    errors = overlay_errors(overlay, repo_root=None)
    assert errors


def test_choose_envelope_smallest_defensible():
    assert choose_envelope(16751 + 256) == 18432
    assert choose_envelope(18432) == 18432
    assert choose_envelope(18433) == 20480
    with pytest.raises(ValueError, match="32768"):
        choose_envelope(20481)


def test_long_prompt_selection_includes_dask_and_repeats_for_two_steps():
    ranked = [
        {"instance_id": "aa", "prompt_token_count": 9000},
        {"instance_id": REQUIRED_TASK, "prompt_token_count": 8000},
        {"instance_id": "bb", "prompt_token_count": 7000},
        {"instance_id": "cc", "prompt_token_count": 6000},
        {"instance_id": "dd", "prompt_token_count": 5000},
        {"instance_id": "ee", "prompt_token_count": 4000},
        {"instance_id": "ff", "prompt_token_count": 3000},
        {"instance_id": "gg", "prompt_token_count": 2000},
        {"instance_id": "hh", "prompt_token_count": 1000},
    ]
    selected = select_long_prompt_instance_ids(ranked)
    assert len(selected) == STRESS_UNIQUE_TASKS
    assert REQUIRED_TASK in selected
    assert selected[0] == "aa"
    assert "hh" not in selected
    repeated = repeat_ids_for_steps(selected, n_steps=STRESS_STEPS)
    assert len(repeated) == STRESS_UNIQUE_TASKS * STRESS_STEPS
    assert repeated[:8] == repeated[8:]

    ranked_short = [
        {"instance_id": f"t{i}", "prompt_token_count": 1000 - i} for i in range(10)
    ]
    ranked_short.append({"instance_id": REQUIRED_TASK, "prompt_token_count": 10})
    forced = select_long_prompt_instance_ids(ranked_short)
    assert REQUIRED_TASK in forced
    assert len(forced) == 8


def test_inspect_hard_stop_envelope_is_parameterized(tmp_path):
    from budget_coder_rl.eval.m5b import inspect_step_metrics_for_hard_stop

    inspect_step_metrics_for_hard_stop(
        {"bcrl/seq/training_proxy_max": 16751, "bcrl/any_nonzero_advantage": True},
        step=1,
        output_dir=tmp_path,
        ppo_max_token_len=18432,
    )
