"""M5A preflight helpers: freeze consume, seqlen cap, prefix selection, gate."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m3c import FREEZE_RELPATH
from budget_coder_rl.eval.m4a import freeze_contract_errors, load_candidate_ordered_ids
from budget_coder_rl.eval.m5a import (
    EXPERIMENT_ID,
    GROUP_N,
    MAIN_STEPS,
    MILESTONE,
    N_CANDIDATES,
    OBS_TOKENS_LIMIT,
    PILOT_STEPS,
    TRAIN_BATCH_SIZE,
    build_main_contract,
    build_pilot_overlay,
    compute_bcrl_step_metrics,
    length_stats,
    m5_freeze_consume_errors,
    m5a_gate,
    merge_main_and_pilot,
    pilot_override_errors,
    propose_ppo_max_token_len_per_gpu,
    ready_for_m5b_errors,
    select_prefix_instance_ids,
    stamp_main_contract_gate,
    coerce_sequence,
    main_contract_immutable,
)
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m3c_freeze_not_edited_by_m5a():
    path = REPO_ROOT / FREEZE_RELPATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert freeze_contract_errors(payload) == []
    assert payload["not_trained"] is True
    assert payload["grpo_optimizer"] is False
    assert payload["lora_update"] is False
    assert payload["primary_training_B_obs"] == OBS_TOKENS_LIMIT
    assert m5_freeze_consume_errors(
        payload,
        freeze_path=path,
        candidate_path=REPO_ROOT / "data/manifests/m3c_train_candidates.json",
    ) == []


def test_m5a_constants():
    assert EXPERIMENT_ID == "E010"
    assert MILESTONE == "M5A"
    assert TRAIN_BATCH_SIZE == 8
    assert PILOT_STEPS == 4
    assert MAIN_STEPS == 32
    assert N_CANDIDATES == 256
    assert N_CANDIDATES % TRAIN_BATCH_SIZE == 0
    assert N_CANDIDATES // TRAIN_BATCH_SIZE == MAIN_STEPS
    assert GROUP_N == 4


def test_coerce_sequence_does_not_truth_test_arrays():
    class Ambiguous:
        def __bool__(self):
            raise ValueError("ambiguous")

        def __iter__(self):
            return iter(["u1", "u2"])

        def __len__(self):
            return 2

    assert coerce_sequence(None) == []
    assert coerce_sequence(Ambiguous()) == ["u1", "u2"]
    ordered = [f"id_{i:03d}" for i in range(40)]
    selected = select_prefix_instance_ids(ordered, n_tasks=8, n_steps=4)
    assert selected == ordered[:32]
    assert selected[0] == "id_000"
    assert "mixed" not in "".join(selected)


def test_propose_token_cap_rejects_8192_when_max_exceeds_it():
    assert propose_ppo_max_token_len_per_gpu(7000) == 12288
    assert propose_ppo_max_token_len_per_gpu(15251) == 16384
    assert propose_ppo_max_token_len_per_gpu(8193) == 12288
    cap = propose_ppo_max_token_len_per_gpu(15251)
    assert cap >= 15251 + 256
    assert cap != 32768
    assert cap != 8192


def test_length_stats_percentiles():
    stats = length_stats(list(range(1, 101)))
    assert stats["n"] == 100
    assert stats["min"] == 1
    assert stats["max"] == 100
    assert stats["median"] == 50.5
    assert stats["p95"] is not None
    assert stats["p99"] is not None


def test_bcrl_step_metrics_group_variance():
    metrics = compute_bcrl_step_metrics(
        uids=["a", "a", "a", "a", "b", "b", "b", "b"],
        rewards=[0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        extra_rows=[
            {
                "file_f1": 0.0,
                "symbol_f1": 0.0,
                "parse_ok": False,
                "termination": "budget_exhausted",
                "budget_exhausted": True,
                "obs_tokens_used": 4096,
                "prompt_token_count": 800,
                "policy_token_count": 100,
                "observation_token_count": 3000,
                "events": [{"error_kind": "protocol", "parse_error_code": "invalid_arguments"}],
            },
            {
                "file_f1": 1.0,
                "symbol_f1": 0.0,
                "parse_ok": True,
                "termination": "finish",
                "budget_exhausted": False,
                "obs_tokens_used": 2000,
                "prompt_token_count": 800,
                "policy_token_count": 120,
                "observation_token_count": 2100,
                "events": [{"action_name": "search"}, {"action_name": "finish"}],
            },
        ]
        + [
            {
                "file_f1": 0.5,
                "parse_ok": True,
                "termination": "finish",
                "obs_tokens_used": 1500,
                "prompt_token_count": 500,
                "policy_token_count": 80,
                "observation_token_count": 1600,
                "events": [],
            }
        ]
        * 6,
    )
    assert metrics["bcrl/group/n"] == 2
    assert metrics["bcrl/group/mixed_fraction"] == 0.5
    assert metrics["bcrl/group/zero_var_fraction"] == 0.5
    assert metrics["bcrl/any_nonzero_advantage"] is True
    assert metrics["bcrl/any_mixed_group"] is True
    assert metrics["bcrl/invalid_action_rate"] == 0.125
    assert metrics["bcrl/seq/training_proxy_max"] == 3900


def test_pilot_overlay_cannot_change_research_knobs():
    freeze = json.loads((REPO_ROOT / FREEZE_RELPATH).read_text(encoding="utf-8"))
    candidate = REPO_ROOT / "data/manifests/m3c_train_candidates.json"
    main = build_main_contract(
        freeze=freeze,
        freeze_path=REPO_ROOT / FREEZE_RELPATH,
        candidate_path=candidate,
        seqlen={"training_seq_proxy": {"max": 15251}, "proposed_ppo_max_token_len_per_gpu": 16384},
        verl_isolated={"commit": "8481f9f9880d0f46a75b3db0329d3de8abad3d81", "dirty": False},
        project_commit="deadbeef",
        ppo_max_token_len_per_gpu=16384,
    )
    overlay = build_pilot_overlay(output_dir=Path("/tmp/e010"), n_steps=4, n_tasks=8)
    assert pilot_override_errors(overlay, main=main) == []
    merged = merge_main_and_pilot(main, overlay)
    assert merged["newly_frozen"]["trainer"]["total_training_steps"] == 4
    assert merged["newly_frozen"]["data"]["train_batch_size"] == 8
    assert merged["newly_frozen"]["actor"]["ppo_max_token_len_per_gpu"] == 16384
    assert merged["inherited_m3c"]["primary_training_B_obs"] == 4096
    assert merged["newly_frozen"]["actor"]["optim_lr"] == 1e-6
    assert main["gate"]["READY_FOR_M5B"] is False
    stamped = stamp_main_contract_gate(
        main,
        evidence={
            "n_steps_completed": 4,
            "n_steps_nonzero_advantage": 4,
            "ppo_max_token_len_per_gpu": 16384,
            "wandb_run": {"id": "test"},
        },
        stamped_at="2026-08-26T00:00:00+00:00",
    )
    assert main_contract_immutable(stamped) is True
    assert stamped["newly_frozen"]["trainer"]["total_training_steps"] == 32
    errors = pilot_override_errors(
        {
            "schema_version": overlay["schema_version"],
            "overrides": {"actor": {"optim_lr": 1e-4}},
        },
        main=main,
    )
    assert any("disallowed" in item for item in errors)


def test_ready_for_m5b_requires_full_evidence():
    incomplete = m5a_gate({})
    assert incomplete["READY_FOR_M5B"] is False
    assert "veRL checkout" in " ".join(incomplete["reasons"])
    complete = {
        "verl_isolated_clean": True,
        "m3c_freeze_ok": True,
        "seqlen_characterized": True,
        "ppo_max_token_len_frozen": True,
        "assert_risk_if_8192": True,
        "ppo_max_token_len_per_gpu": 16384,
        "pilot_completed": True,
        "pilot_oom": False,
        "n_steps_completed": 4,
        "n_steps_nonzero_advantage": 2,
        "metrics_jsonl_present": True,
        "wandb_logged": True,
        "main_config_written": True,
        "checkpoint_policy_frozen": True,
    }
    gate = m5a_gate(complete)
    assert gate["READY_FOR_M5B"] is True
    assert ready_for_m5b_errors(complete) == []
    still_8192 = dict(complete)
    still_8192["ppo_max_token_len_per_gpu"] = 8192
    assert any("8192" in item for item in ready_for_m5b_errors(still_8192))


def test_candidate_hash_matches_freeze():
    freeze = json.loads((REPO_ROOT / FREEZE_RELPATH).read_text(encoding="utf-8"))
    path = REPO_ROOT / "data/manifests/m3c_train_candidates.json"
    ids = load_candidate_ordered_ids(path)
    assert len(ids) == 256
    assert sha256_file(path) == freeze["train_candidate_manifest"]["file_sha256"]
