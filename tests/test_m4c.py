"""M4C persist/reload helpers: fingerprints, FSDP inventory, vLLM adapter gate."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m4a import freeze_contract_errors
from budget_coder_rl.eval.m4b import fingerprint_numeric
from budget_coder_rl.eval.m4c import (
    EXPERIMENT_ID,
    MILESTONE,
    N_TASKS,
    PINNED_VERL_COMMIT,
    VLLM_LORA_INT_ID,
    checkpoint_integrity_errors,
    compare_lora_fingerprints,
    fingerprint_digest,
    lora_sha256_map,
    m4c_gate,
    persist_lora_fingerprint,
    vllm_sync_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fp(sha256: str, *, mean: float = 0.0, max_abs: float = 0.1) -> dict:
    return fingerprint_numeric(sha256=sha256, numel=4, mean=mean, max_abs=max_abs)


def _snapshot(lora_a: str, lora_b: str) -> dict:
    return persist_lora_fingerprint(
        {
            "rank": 0,
            "n_trainable": 2,
            "n_frozen": 1,
            "unexpected_trainable": [],
            "lora": {
                "layer.lora_A": _fp(lora_a, mean=0.1),
                "layer.lora_B": _fp(lora_b, mean=1e-6, max_abs=1e-6),
            },
            "frozen": {"base.weight": _fp("frozen", mean=1.0, max_abs=2.0)},
        }
    )


def _payload(*, digest: str, max_abs: float = 1e-6) -> dict:
    return {
        "peft_config_present": True,
        "n_adapter_tensors": 2,
        "digest": digest,
        "lora_b_max_abs": max_abs,
        "adapter_nonzero": max_abs > 0,
        "tensors": {
            "lora_A": _fp("aaa"),
            "lora_B": _fp("bbb", max_abs=max_abs),
        },
    }


def _generate_ok() -> dict:
    return {
        "phase": "reload",
        "lora_as_adapter": True,
        "listed_lora_ids": [VLLM_LORA_INT_ID],
        "lora_int_id": VLLM_LORA_INT_ID,
        "lora_loaded": True,
        "lora_request_attached": True,
    }


def test_freeze_json_not_edited_by_m4c():
    path = REPO_ROOT / "configs" / "historical" / "stage1_m3c_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert freeze_contract_errors(payload) == []
    assert payload["grpo_optimizer"] is False
    assert payload["lora_update"] is False
    assert payload["not_trained"] is True


def test_m4c_constants():
    assert EXPERIMENT_ID == "E009"
    assert MILESTONE == "M4C"
    assert N_TASKS == 2
    assert PINNED_VERL_COMMIT == "8481f9f9880d0f46a75b3db0329d3de8abad3d81"
    assert VLLM_LORA_INT_ID == 123


def test_verl_path_text_formats_commit_without_brace_errors():
    from budget_coder_rl.eval.m4c import VERL_PATH_TEXT

    text = VERL_PATH_TEXT.format(commit="deadbeef")
    assert "deadbeef" in text
    assert "global_step_{N}" in text


def test_fingerprint_digest_stable_and_order_independent():
    left = {"b": "2", "a": "1"}
    right = {"a": "1", "b": "2"}
    assert fingerprint_digest(left) == fingerprint_digest(right)
    assert fingerprint_digest(left) != fingerprint_digest({"a": "1", "b": "3"})


def test_compare_lora_fingerprints_requires_full_sha256_equality():
    theta0 = _snapshot("aaa", "bbb")
    theta1 = _snapshot("aaa", "ccc")
    reloaded = _snapshot("aaa", "ccc")
    vs_step = compare_lora_fingerprints(theta0, theta1)
    vs_reload = compare_lora_fingerprints(theta1, reloaded)
    assert theta1["n_lora_tensors"] == 2
    assert set(lora_sha256_map(theta1)) == {"layer.lora_A", "layer.lora_B"}
    assert theta1["digest"] != theta0["digest"]
    assert vs_step["equal"] is False
    assert vs_step["n_mismatched"] == 1
    assert vs_reload["equal"] is True
    assert lora_sha256_map(theta1)["layer.lora_B"] == "ccc"


def test_checkpoint_integrity_errors_on_missing_dir(tmp_path: Path):
    errors = checkpoint_integrity_errors(tmp_path / "missing", expected_step=1)
    assert any("missing checkpoint root" in item for item in errors)


def test_checkpoint_integrity_accepts_official_fsdp_layout(tmp_path: Path):
    root = tmp_path / "checkpoints"
    actor = root / "global_step_1" / "actor"
    hf = actor / "huggingface"
    hf.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    for name in (
        "model_world_size_1_rank_0.pt",
        "optim_world_size_1_rank_0.pt",
        "extra_state_world_size_1_rank_0.pt",
    ):
        (actor / name).write_bytes(b"ckpt")
    (actor / "fsdp_config.json").write_text(
        json.dumps({"FSDP_version": 1, "world_size": 1}),
        encoding="utf-8",
    )
    (hf / "config.json").write_text("{}", encoding="utf-8")
    assert checkpoint_integrity_errors(root, expected_step=1) == []


def test_checkpoint_integrity_fails_empty_shard_and_wrong_step(tmp_path: Path):
    root = tmp_path / "checkpoints"
    actor = root / "global_step_1" / "actor" / "huggingface"
    actor.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("2\n", encoding="utf-8")
    (root / "global_step_1" / "actor" / "model_world_size_1_rank_0.pt").write_bytes(b"")
    errors = checkpoint_integrity_errors(root, expected_step=1)
    assert any("latest_checkpointed_iteration" in item for item in errors)
    assert any("empty actor/model_world_size_1_rank_0.pt" in item for item in errors)


def test_vllm_sync_errors_when_generate_omits_lora_request():
    errors = vllm_sync_errors(
        payload=_payload(digest="abc"),
        generate_rows=[
            {
                "phase": "reload",
                "lora_as_adapter": True,
                "listed_lora_ids": [VLLM_LORA_INT_ID],
                "lora_int_id": None,
                "lora_loaded": True,
                "lora_request_attached": False,
            }
        ],
        saved_payload=_payload(digest="abc"),
    )
    assert any("LoRARequest" in item for item in errors)


def test_vllm_sync_errors_when_list_loras_missing_123():
    errors = vllm_sync_errors(
        payload=_payload(digest="abc"),
        generate_rows=[
            {
                "phase": "reload",
                "lora_as_adapter": True,
                "listed_lora_ids": [],
                "lora_int_id": None,
                "lora_loaded": False,
                "lora_request_attached": False,
            }
        ],
    )
    assert any("list_loras missing adapter id 123" in item for item in errors)


def test_vllm_sync_ok_when_payload_and_request_match():
    errors = vllm_sync_errors(
        payload=_payload(digest="same"),
        generate_rows=[_generate_ok()],
        saved_payload=_payload(digest="same"),
    )
    assert errors == []


def test_m4c_gate_pass_and_fail_reload_mismatch():
    theta0 = _snapshot("aaa", "zero")
    theta1 = _snapshot("aaa", "updated")
    reloaded = _snapshot("aaa", "updated")
    gate = m4c_gate(
        optimizer_gate={"pass": True, "reasons": []},
        theta0=theta0,
        theta1=theta1,
        reloaded=reloaded,
        checkpoint_errors=[],
        vllm_errors=[],
        n_reload_episodes=1,
    )
    assert gate["pass"] is True
    assert gate["theta1_ne_theta0"] is True
    assert gate["reloaded_eq_theta1"] is True
    bad = m4c_gate(
        optimizer_gate={"pass": True, "reasons": []},
        theta0=theta0,
        theta1=theta1,
        reloaded=theta0,
        checkpoint_errors=[],
        vllm_errors=["generate did not attach LoRARequest"],
        n_reload_episodes=0,
    )
    assert bad["pass"] is False
    assert any("reloaded LoRA fingerprint != saved θ1" in item for item in bad["reasons"])
    assert any("LoRARequest" in item for item in bad["reasons"])
    assert any("no post-reload AgentLoop episode" in item for item in bad["reasons"])


def test_agent_loop_still_does_not_import_reward_or_train():
    agent = (
        REPO_ROOT / "src" / "budget_coder_rl" / "agent_loop" / "repo_exploration.py"
    ).read_text(encoding="utf-8")
    assert "budget_coder_rl.reward" not in agent
    assert "budget_coder_rl.train" not in agent
