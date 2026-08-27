"""M7B train–eval invalid-action discrepancy audit. CPU-only; no AgentLoop."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m7a import episode_parse_ok, filter_e018_cell, naive_bool
from budget_coder_rl.eval.m7b import (
    analyze_eval_cell,
    analyze_training_stream,
    audit_execution_contract,
    bin_step_rows,
    coarsen_taxonomy,
    compact_episode_metrics,
    cross_check_step_bcrl,
    first_turn_event,
    global_step_from_index,
    hypothesis_verdicts,
    is_padding_index,
    iter_jsonl_indexed,
    matched_comparison,
    slice_steps,
    temperature_is_greedy,
)


def _tool(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _event(
    raw: str,
    *,
    error_kind: str | None,
    parse_error_code: str | None = None,
    turn: int = 1,
    action_name: str | None = None,
) -> dict:
    return {
        "turn": turn,
        "raw_action": raw,
        "error_kind": error_kind,
        "parse_error_code": parse_error_code,
        "action_name": action_name,
    }


def _episode(
    *,
    instance_id: str,
    events: list[dict],
    parse_ok: object = False,
    loc: float = 0.0,
    repo: str = "acme/repo",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    obs: int = 4096,
    budget_visible: bool = True,
    max_turns: int = 6,
    seed: int | None = None,
    prompt_tokens: int = 100,
    condition_id: str | None = None,
) -> dict:
    sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
    condition = {
        "obs_tokens_limit": obs,
        "budget_visible": budget_visible,
        "max_turns": max_turns,
        "max_new_tokens_per_turn": 2048,
        "sampling": sampling,
        "sampling_seed": seed,
    }
    if condition_id is not None:
        condition["condition_id"] = condition_id
        condition["policy"] = "rl" if condition_id == "M_scaled" else "base"
    return {
        "identity": {"instance_id": instance_id, "repo": repo, "split": "train"},
        "condition": condition,
        "budget": {"obs_tokens_limit": obs, "budget_visible": budget_visible},
        "tokens": {"prompt_token_count": prompt_tokens},
        "localization": {"parse_ok": parse_ok, "localization_score": loc},
        "events": events,
        "termination": "max_turns",
    }


def test_global_step_from_index_32_per_step():
    assert global_step_from_index(0) == 1
    assert global_step_from_index(31) == 1
    assert global_step_from_index(32) == 2
    assert global_step_from_index(8799) == 275
    assert is_padding_index(8771, n_unique=2193, group_n=4) is False
    assert is_padding_index(8772, n_unique=2193, group_n=4) is True


def test_first_turn_protocol_only_vs_later_tool_error():
    proto_then_tool = _episode(
        instance_id="a",
        events=[
            _event("just prose", error_kind="protocol", parse_error_code="malformed_action", turn=1),
            _event(
                _tool("read", {"path": "nope.py", "start_line": 1, "end_line": 2}),
                error_kind="tool",
                action_name="read",
                turn=2,
            ),
        ],
        parse_ok="False",
    )
    ok_then_proto = _episode(
        instance_id="b",
        events=[
            _event(_tool("tree", {}), error_kind=None, action_name="tree", turn=1),
            _event(
                _tool("tree", {}) + "\n" + _tool("search", {"query": "x"}),
                error_kind="protocol",
                parse_error_code="multiple_actions",
                turn=2,
            ),
        ],
        parse_ok=True,
    )
    first = compact_episode_metrics(proto_then_tool)
    second = compact_episode_metrics(ok_then_proto)
    assert first["first_turn_invalid"] is True
    assert first["first_turn_protocol"] is True
    assert first["n_tool_error_events"] == 1
    assert second["first_turn_invalid"] is False
    assert second["first_turn_protocol"] is False
    assert second["first_turn_taxonomy"] is None
    assert first_turn_event(ok_then_proto)["turn"] == 1
    assert coarsen_taxonomy("wrong_schema") == "other_protocol"
    assert coarsen_taxonomy("multiple_actions") == "multiple_actions"
    assert coarsen_taxonomy("framing_unbalanced_tags") == "framing_unbalanced_tags"
    assert coarsen_taxonomy("tool_semantic_misuse") == "tool_semantic_misuse"


def test_truthy_string_false_still_held():
    row = _episode(
        instance_id="c",
        events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
        parse_ok="False",
    )
    metrics = compact_episode_metrics(row)
    assert metrics["parse_ok"] is False
    assert episode_parse_ok(row) is False
    assert naive_bool("False") is True


def test_binning_does_not_rewrite_source_rows():
    rows = []
    source = []
    for index in range(6):
        step = global_step_from_index(index, traj_per_step=2)
        ep = _episode(
            instance_id=f"id-{index}",
            events=[
                _event("no tags", error_kind="protocol", parse_error_code="malformed_action", turn=1)
            ],
            parse_ok="False",
            loc=0.1 * (index + 1),
        )
        source.append(dict(ep))
        rows.append((index, ep))
    train = analyze_training_stream(rows, traj_per_step=2, n_unique=6, group_n=1)
    original = json.dumps(source)
    binned = bin_step_rows(train["step_rows"], bin_size=2)
    assert json.dumps(source) == original
    assert len(train["step_rows"]) == 3
    assert len(binned) == 2
    assert binned[0]["bin_start"] == 1
    assert binned[0]["bin_end"] == 2
    assert binned[0]["n_episodes"] == 4
    sliced = slice_steps(train["step_rows"], 1, 2)
    assert sliced["n_episodes"] == 4
    assert source[0]["events"][0]["raw_action"] == "no tags"


def test_temperature_zero_detected_in_execution_audit():
    greedy = _episode(
        instance_id="g",
        events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
        temperature=0.0,
    )
    sampled = _episode(
        instance_id="s",
        events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
        temperature=0.7,
    )
    assert temperature_is_greedy(0) is True
    assert temperature_is_greedy(0.0) is True
    assert temperature_is_greedy(0.7) is False
    greedy_train = analyze_training_stream([(0, greedy)], traj_per_step=1, n_unique=1, group_n=1)
    ok_train = analyze_training_stream([(0, sampled)], traj_per_step=1, n_unique=1, group_n=1)
    assert greedy_train["n_temp_zero"] == 1
    assert ok_train["n_temp_zero"] == 0
    contract = audit_execution_contract(
        e017_provenance={"sampling_rollout": {"temperature": 0.7, "n": 4, "do_sample": True}},
        e017_config={
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 0.7,
                    "n": 4,
                    "val_kwargs": {"temperature": 0.0, "do_sample": False},
                }
            },
            "trainer": {"val_before_train": False, "test_freq": -1},
        },
        e018_provenance={"validate": False, "vllm_rollout_n": 1, "sampling_intended": {"temperature": 0.7}},
        e018_overlay={"frozen_from_parent": {"sampling": {"temperature": 0.7, "n": 1}, "validate": False}},
        e018_integrity={"pass": True, "listed_lora_ids": [123]},
        e017_empirical=greedy_train,
        e018_cells={},
    )
    assert contract["verl_validate_override"]["greedy_override_detected"] is True
    ok_contract = audit_execution_contract(
        e017_provenance={"sampling_rollout": {"temperature": 0.7, "n": 4}},
        e017_config={
            "actor_rollout_ref": {"rollout": {"temperature": 0.7, "n": 4, "val_kwargs": {"temperature": 0.0}}},
            "trainer": {"val_before_train": False, "test_freq": -1},
        },
        e018_provenance={"validate": False, "vllm_rollout_n": 1, "sampling_intended": {"temperature": 0.7}},
        e018_overlay={"frozen_from_parent": {"sampling": {"temperature": 0.7, "n": 1}, "validate": False}},
        e018_integrity={"pass": True},
        e017_empirical=ok_train,
        e018_cells={},
    )
    assert ok_contract["verl_validate_override"]["greedy_override_detected"] is False
    assert ok_contract["matched_sampling_temperature"] is True
    assert ok_contract["execution_matched"] is False


def test_e018_cell_filter_b1_and_m_scaled_4096():
    rows = [
        _episode(
            instance_id="dev-1",
            events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
            condition_id="B1",
            obs=4096,
            seed=20260827,
        ),
        _episode(
            instance_id="dev-1",
            events=[_event("prose", error_kind="protocol", parse_error_code="malformed_action")],
            condition_id="M_scaled",
            obs=4096,
            seed=20260827,
        ),
        _episode(
            instance_id="dev-1",
            events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
            condition_id="B1",
            obs=2048,
            seed=20260827,
        ),
        _episode(
            instance_id="dev-2",
            events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
            condition_id="B0",
            obs=4096,
            budget_visible=False,
        ),
    ]
    b1 = list(filter_e018_cell(rows, condition_id="B1", budget=4096))
    scaled = list(filter_e018_cell(rows, condition_id="M_scaled", budget=4096))
    assert len(b1) == 1
    assert len(scaled) == 1
    b1_cell = analyze_eval_cell(b1)
    scaled_cell = analyze_eval_cell(scaled)
    assert b1_cell["event_invalid_rate"] == 0.0
    assert scaled_cell["first_turn_protocol_rate"] == 1.0


def test_error_rows_still_advance_step_index(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    good = _episode(
        instance_id="ok",
        events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"error": "boom", "instance_id": "bad"}) + "\n")
        handle.write(json.dumps(good) + "\n")
    indexed = list(iter_jsonl_indexed(path))
    assert indexed[0][0] == 0
    assert indexed[1][0] == 1
    assert indexed[0][1]["error"] == "boom"
    train = analyze_training_stream(indexed, traj_per_step=1, n_unique=2, group_n=1)
    assert train["n_jsonl_lines"] == 2
    assert train["pooled"]["n_episodes"] == 1
    assert train["pooled"]["n_error_rows"] == 1


def test_step_bcrl_cross_check_and_h2_rejected_when_flat():
    two = _tool("tree", {}) + "\n" + _tool("search", {"query": "x"})
    rows = []
    for index in range(4):
        rows.append(
            (
                index,
                _episode(
                    instance_id=f"t-{index}",
                    events=[
                        _event(two, error_kind="protocol", parse_error_code="multiple_actions", turn=1)
                    ],
                    parse_ok="False",
                    loc=0.0,
                    prompt_tokens=800,
                ),
            )
        )
    train = analyze_training_stream(rows, traj_per_step=2, n_unique=4, group_n=1)
    bcrl = [
        {
            "global_steps": 1,
            "n_trajectories": 2,
            "invalid_action_rate": train["step_rows"][0]["episode_invalid_rate"],
            "protocol_error_count": train["step_rows"][0]["n_protocol_events"],
        },
        {
            "global_steps": 2,
            "n_trajectories": 2,
            "invalid_action_rate": train["step_rows"][1]["episode_invalid_rate"],
            "protocol_error_count": train["step_rows"][1]["n_protocol_events"],
        },
    ]
    check = cross_check_step_bcrl(train["step_rows"], bcrl)
    assert check["pass"] is True
    late = slice_steps(train["step_rows"], 1, 2)
    e018 = {
        "B1@4096": {
            "event_invalid_rate": 0.11,
            "first_turn_protocol_rate": 0.05,
            "n_episodes": 244,
            "parse_ok_rate": 0.5,
            "mean_localization_score": 0.2,
        },
        "M_scaled@4096": {
            "event_invalid_rate": 0.12,
            "first_turn_protocol_rate": 0.05,
            "n_episodes": 244,
            "parse_ok_rate": 0.5,
            "mean_localization_score": 0.2,
        },
    }
    contract = audit_execution_contract(
        e017_provenance={"sampling_rollout": {"temperature": 0.7, "n": 4}},
        e017_config={
            "actor_rollout_ref": {"rollout": {"temperature": 0.7, "n": 4, "val_kwargs": {"temperature": 0.0}}},
            "trainer": {"val_before_train": False, "test_freq": -1},
        },
        e018_provenance={"validate": False, "vllm_rollout_n": 1, "sampling_intended": {"temperature": 0.7}},
        e018_overlay={"frozen_from_parent": {"sampling": {"temperature": 0.7}, "validate": False}},
        e018_integrity={"pass": True},
        e017_empirical=train,
        e018_cells=e018,
    )
    matched = matched_comparison(
        late16=late,
        last_bin=None,
        e018_cells=e018,
        execution_matched=False,
    )
    assert matched["forbid_rl_attribution"] is True
    verdicts = hypothesis_verdicts(
        contract=contract,
        pooled=train["pooled"],
        early16=late,
        late16=late,
        stratification={
            "prompt_mix_stable": True,
            "prompt_length_rel_change_early_to_late": 0.01,
            "phases": {"early": late, "late": late},
        },
        matched=matched,
        step_check=check,
    )
    by_id = {item["id"]: item for item in verdicts["items"]}
    assert by_id["H2"]["verdict"] == "rejected"
    assert by_id["H1"]["greedy_validate_override"] == "rejected"
    assert by_id["H4"]["verdict"] == "rejected"
    assert verdicts["gate"]["do_not_start_intervention"] is True
    assert verdicts["gate"]["protocol_compliance_learning_credible"] is False
