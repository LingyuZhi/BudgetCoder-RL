"""M4B one-step GRPO helpers: mask contract, LoRA fingerprints, freeze consume."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from budget_coder_rl.eval.m4a import freeze_contract_errors, select_smoke_instance_ids
from budget_coder_rl.eval.m4b import (
    EXPERIMENT_ID,
    GROUP_N,
    MILESTONE,
    N_TASKS,
    PINNED_VERL_COMMIT,
    assemble_groups_from_members,
    assemble_loss_mask_evidence,
    compare_param_snapshots,
    count_mask_tokens,
    fingerprint_numeric,
    is_lora_param_name,
    m4b_gate,
    mask_correctness_errors,
    metric_finite_nonzero,
    step_learning_signal,
    unwrap_metric_value,
    write_smoke_parquet,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_freeze_json_not_edited_by_m4b():
    path = REPO_ROOT / "configs" / "experiments" / "stage1_m3c_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert freeze_contract_errors(payload) == []
    assert payload["grpo_optimizer"] is False
    assert payload["lora_update"] is False
    assert payload["not_trained"] is True


def test_m4b_constants():
    assert EXPERIMENT_ID == "E003"
    assert MILESTONE == "M4B"
    assert N_TASKS == 2
    assert GROUP_N == 4
    assert PINNED_VERL_COMMIT == "8481f9f9880d0f46a75b3db0329d3de8abad3d81"


def test_select_two_tasks_keeps_candidate_order():
    ordered = ["keep_later", "mixed_first", "mixed_second", "zero_var"]
    groups = [
        {"instance_id": "mixed_second", "stats": {"mixed": True}},
        {"instance_id": "mixed_first", "stats": {"mixed": True}},
        {"instance_id": "zero_var", "stats": {"mixed": False}},
        {"instance_id": "keep_later", "stats": {"mixed": True}},
    ]
    selected = select_smoke_instance_ids(ordered, groups, n=N_TASKS)
    assert selected == ["keep_later", "mixed_first"]


def test_observation_mask_must_differ_from_attention_fallback():
    response_mask = [[1, 1, 1, 1, 0]]
    attention = [[1, 1, 1, 1, 0]]
    advantages = [[0.5, 0.5, 0.5, 0.5, 0.0]]
    errors = mask_correctness_errors(
        response_mask=response_mask,
        attention_response=attention,
        advantages=advantages,
        n_observation_tokens=[2],
    )
    assert any("equals attention" in item for item in errors)


def test_padding_and_obs_advantages_must_be_zero():
    response_mask = [[1, 0, 0]]
    attention = [[1, 1, 0]]
    advantages = [[1.0, 0.2, 0.0]]
    errors = mask_correctness_errors(
        response_mask=response_mask,
        attention_response=attention,
        advantages=advantages,
        n_observation_tokens=[1],
    )
    assert any("advantage" in item for item in errors)


def test_policy_tokens_counted_and_pad_separated():
    counts = count_mask_tokens([1, 1, 0, 0, 0, 0], n_obs=2)
    assert counts["n_policy"] == 2
    assert counts["n_observation"] == 2
    assert counts["n_pad"] == 2


def test_lora_name_classifier():
    assert is_lora_param_name("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight")
    assert is_lora_param_name("...lora_B.weight")
    assert not is_lora_param_name("base_model.model.layers.0.self_attn.q_proj.weight")


def test_compare_param_snapshots_detects_lora_change_and_frozen_base():
    before = {
        "n_trainable": 1,
        "n_frozen": 1,
        "unexpected_trainable": [],
        "lora": {
            "lora_A": fingerprint_numeric(sha256="aaa", numel=4, mean=0.0, max_abs=0.1),
        },
        "frozen": {
            "base.weight": fingerprint_numeric(sha256="bbb", numel=8, mean=1.0, max_abs=2.0, full_hash=False),
        },
    }
    after_ok = {
        "n_trainable": 1,
        "n_frozen": 1,
        "unexpected_trainable": [],
        "lora": {
            "lora_A": fingerprint_numeric(sha256="ccc", numel=4, mean=0.01, max_abs=0.12),
        },
        "frozen": {
            "base.weight": fingerprint_numeric(sha256="bbb", numel=8, mean=1.0, max_abs=2.0, full_hash=False),
        },
    }
    ok = compare_param_snapshots(before, after_ok)
    assert ok["lora_changed"] is True
    assert ok["base_frozen"] is True
    after_bad = {
        **after_ok,
        "frozen": {
            "base.weight": fingerprint_numeric(sha256="zzz", numel=8, mean=1.1, max_abs=2.1, full_hash=False),
        },
    }
    bad = compare_param_snapshots(before, after_bad)
    assert bad["base_frozen"] is False


def test_m4b_gate_requires_all_evidence():
    groups = assemble_groups_from_members(
        [
            {
                "instance_id": "t",
                "uid": "u",
                "rm_score": 0.0,
                "localization_score": 0.0,
                "advantage_scalar": -0.5,
                "termination": "max_turns",
                "rollout_n": i,
            }
            for i in range(3)
        ]
        + [
            {
                "instance_id": "t",
                "uid": "u",
                "rm_score": 0.5,
                "localization_score": 0.5,
                "advantage_scalar": 0.5,
                "termination": "finish",
                "rollout_n": 3,
            }
        ]
    )
    learning = step_learning_signal(groups)
    loss_mask = assemble_loss_mask_evidence(
        [
            {
                "response_mask_present": True,
                "n_policy": 10,
                "n_observation": 4,
                "n_pad": 2,
                "advantages_zero_on_mask0": True,
            }
        ],
        tito_errors=[],
        mask_errors=[],
    )
    lora = {
        "lora_changed": True,
        "base_frozen": True,
        "trainable_are_lora": True,
        "n_lora_changed": 1,
        "n_lora_tensors": 1,
    }
    gate = m4b_gate(
        learning=learning,
        loss_mask=loss_mask,
        grad=metric_finite_nonzero(0.4),
        pg_loss=metric_finite_nonzero(-0.1),
        lora=lora,
    )
    assert gate["pass"] is True
    zero = m4b_gate(
        learning={"ok": False},
        loss_mask=loss_mask,
        grad=metric_finite_nonzero(0.4),
        pg_loss=metric_finite_nonzero(-0.1),
        lora=lora,
    )
    assert zero["pass"] is False


def test_metric_unwrap_handles_list_and_mapping():
    assert unwrap_metric_value([{"value": 0.4}]) == 0.4
    assert metric_finite_nonzero([0.4])["nonzero"] is True
    assert metric_finite_nonzero(None)["finite"] is False


def test_write_smoke_parquet_preserves_order(tmp_path: Path):
    source = tmp_path / "src.parquet"
    dest = tmp_path / "out.parquet"
    frame = pd.DataFrame(
        [
            {
                "data_source": "x",
                "prompt": [{"role": "user", "content": "b"}],
                "reward_model": {"style": "rule", "ground_truth": "b"},
                "extra_info": {"instance_id": "b", "repo": "r", "split": "train"},
            },
            {
                "data_source": "x",
                "prompt": [{"role": "user", "content": "a"}],
                "reward_model": {"style": "rule", "ground_truth": "a"},
                "extra_info": {"instance_id": "a", "repo": "r", "split": "train"},
            },
        ]
    )
    frame.to_parquet(source, index=False)
    info = write_smoke_parquet(source, dest, ["a", "b"])
    assert info["n_rows"] == 2
    loaded = pd.read_parquet(dest)
    ids = [row["instance_id"] for row in loaded["extra_info"].tolist()]
    assert ids == ["a", "b"]
    assert loaded["extra_info"].iloc[0]["obs_tokens_limit"] == 4096
    assert loaded["extra_info"].iloc[0]["budget_visible"] is True


def test_m4b_collate_injects_multi_modal_inputs():
    import torch

    from budget_coder_rl.train.m4b_trainer import m4b_collate_fn

    batch = m4b_collate_fn(
        [
            {
                "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                "extra_info": {"instance_id": "a"},
            }
        ]
    )
    assert "multi_modal_inputs" in batch
    assert batch["multi_modal_inputs"][0] == {}


def test_agent_loop_still_does_not_import_reward():
    agent = (
        REPO_ROOT / "src" / "budget_coder_rl" / "agent_loop" / "repo_exploration.py"
    ).read_text(encoding="utf-8")
    assert "budget_coder_rl.reward" not in agent
    assert "budget_coder_rl.train" not in agent
