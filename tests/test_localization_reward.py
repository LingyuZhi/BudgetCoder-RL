"""M4A RewardLoop compute_score: frozen evaluator + sidecar, no AgentLoop gold."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from budget_coder_rl.eval.localization import evaluate_episode
from budget_coder_rl.eval.m4a import (
    assemble_group_evidence,
    freeze_contract_errors,
    leakage_errors,
    select_smoke_instance_ids,
)
from budget_coder_rl.eval.oracle import OracleRow, load_evaluator_oracle
from budget_coder_rl.reward.localization_score import (
    LocalizationScoreError,
    compute_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_sidecar(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "instance_id": "owner__1",
                "split": "train",
                "base_changed_files": ["src/foo.py"],
                "oracle_symbols": [{"path": "src/foo.py", "qualname": "Foo.bar"}],
                "symbol_applicable": True,
            }
        ]
    )
    frame.to_parquet(path)


def _oracle_row() -> OracleRow:
    return OracleRow(
        instance_id="owner__1",
        split="train",
        base_changed_files=("src/foo.py",),
        oracle_symbols=(("src/foo.py", "Foo.bar"),),
        symbol_applicable=True,
    )


def test_compute_score_matches_evaluate_episode(tmp_path: Path):
    sidecar = tmp_path / "oracle.parquet"
    _write_sidecar(sidecar)
    extra = {
        "instance_id": "owner__1",
        "termination": "finish",
        "final_submission": {
            "locations": [{"path": "src/foo.py", "symbol": "Foo.bar"}]
        },
    }
    extra.update(
        {
            "events": [{"observation": "not used"}],
            "segments": [{"kind": "assistant", "token_ids": [1]}],
        }.items()
    )
    result = compute_score(
        data_source="budget_coder_swe_gym_localization",
        solution_str="<tool_call>should not be parsed</tool_call><final>ignored</final>",
        ground_truth="owner__1",
        extra_info=extra,
        oracle_parquet=sidecar,
    )
    expected = evaluate_episode(
        termination="finish",
        submission=extra["final_submission"],
        oracle=_oracle_row(),
    )
    assert result["score"] == expected.localization_score == 1.0
    assert result["file_f1"] == 1.0
    assert result["symbol_f1"] == 1.0
    assert result["parse_ok"] is True
    assert "base_changed_files" not in result
    assert "oracle_symbols" not in result
    assert result["instance_id"] == "owner__1"


def test_compute_score_ignores_leaked_gold_in_extra_info(tmp_path: Path):
    sidecar = tmp_path / "oracle.parquet"
    _write_sidecar(sidecar)
    extra = {
        "instance_id": "owner__1",
        "termination": "finish",
        "final_submission": {
            "locations": [{"path": "src/foo.py", "symbol": "Foo.bar"}]
        },
        "oracle_symbols": [("wrong.py", "Nope")],
        "base_changed_files": ["wrong.py"],
    }
    result = compute_score(
        "budget_coder_swe_gym_localization",
        "",
        "owner__1",
        extra_info=extra,
        oracle_parquet=sidecar,
    )
    assert result["score"] == 1.0
    assert "wrong.py" not in str(result)


def test_missing_termination_hard_fails(tmp_path: Path):
    sidecar = tmp_path / "oracle.parquet"
    _write_sidecar(sidecar)
    with pytest.raises(LocalizationScoreError, match="termination"):
        compute_score(
            "budget_coder_swe_gym_localization",
            "<final>{\"locations\":[]}</final>",
            "owner__1",
            extra_info={"instance_id": "owner__1", "final_submission": {"locations": []}},
            oracle_parquet=sidecar,
        )


def test_budget_exhausted_without_final_is_zero(tmp_path: Path):
    sidecar = tmp_path / "oracle.parquet"
    _write_sidecar(sidecar)
    result = compute_score(
        "budget_coder_swe_gym_localization",
        "partial",
        "owner__1",
        extra_info={
            "instance_id": "owner__1",
            "termination": "budget_exhausted",
            "final_submission": None,
        },
        oracle_parquet=sidecar,
    )
    assert result["score"] == 0.0
    assert result["parse_ok"] is False


def test_instance_id_mismatch_hard_fails(tmp_path: Path):
    sidecar = tmp_path / "oracle.parquet"
    _write_sidecar(sidecar)
    with pytest.raises(LocalizationScoreError, match="mismatch"):
        compute_score(
            "x",
            "",
            "owner__1",
            extra_info={
                "instance_id": "other__2",
                "termination": "finish",
                "final_submission": {"locations": []},
            },
            oracle_parquet=sidecar,
        )


def test_sidecar_roundtrip_matches_index(tmp_path: Path):
    sidecar = tmp_path / "oracle.parquet"
    _write_sidecar(sidecar)
    row = load_evaluator_oracle(sidecar).get("owner__1")
    assert row.base_changed_files == ("src/foo.py",)


def test_agent_loop_and_m3c_do_not_import_reward_or_oracle():
    agent = (
        REPO_ROOT / "src" / "budget_coder_rl" / "agent_loop" / "repo_exploration.py"
    ).read_text(encoding="utf-8")
    m3c = (REPO_ROOT / "src" / "budget_coder_rl" / "eval" / "m3c.py").read_text(
        encoding="utf-8"
    )
    assert "budget_coder_rl.reward" not in agent
    assert "budget_coder_rl.reward" not in m3c
    assert "localization_score" not in agent
    assert "budget_coder_rl.eval" not in agent
    assert "load_evaluator_oracle" not in m3c
    assert "oracle_parquet" not in m3c


def test_select_smoke_ids_uses_candidate_order_not_e007_order():
    ordered = ["keep_later", "mixed_first", "mixed_second", "zero_var"]
    groups = [
        {"instance_id": "mixed_second", "stats": {"mixed": True}},
        {"instance_id": "mixed_first", "stats": {"mixed": True}},
        {"instance_id": "zero_var", "stats": {"mixed": False}},
        {"instance_id": "keep_later", "stats": {"mixed": True}},
    ]
    selected = select_smoke_instance_ids(ordered, groups, n=2)
    assert selected == ["keep_later", "mixed_first"]


def test_assemble_group_evidence_gate():
    members = [
        {
            "instance_id": "t",
            "uid": "u1",
            "rm_score": 0.0,
            "localization_score": 0.0,
            "advantage_scalar": -0.87,
            "termination": "max_turns",
            "rollout_n": i,
        }
        for i in range(3)
    ]
    members.append(
        {
            "instance_id": "t",
            "uid": "u1",
            "rm_score": 0.5,
            "localization_score": 0.5,
            "advantage_scalar": 0.87,
            "termination": "finish",
            "rollout_n": 3,
        }
    )
    evidence = assemble_group_evidence(members)
    assert evidence["same_task"] is True
    assert evidence["same_uid"] is True
    assert evidence["mixed"] is True
    assert evidence["nonzero_advantage"] is True
    assert evidence["gate_pass"] is True


def test_leakage_errors_flag_field_names_not_repo_paths():
    errors = leakage_errors(
        decoded_prompt="locate the cache bug",
        decoded_observations=["# bcrl-obs-v1\npath: src/foo.py"],
        extra_field_keys=("instance_id", "final_submission", "termination"),
    )
    assert errors == []
    leaked = leakage_errors(
        decoded_prompt="oracle_symbols dumped here",
        decoded_observations=[],
        extra_field_keys=("oracle_symbols",),
    )
    assert any("oracle_symbols" in item for item in leaked)


def test_freeze_contract_errors_empty_on_m3c_freeze():
    freeze_path = REPO_ROOT / "configs" / "historical" / "stage1_m3c_freeze.json"
    payload = __import__("json").loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze_contract_errors(payload) == []
