"""M7D first-request / sibling-expansion contracts. CPU-only."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m3b import QWEN3_SAMPLING
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID
from budget_coder_rl.eval.m7a import classify_event
from budget_coder_rl.eval.m7c import (
    assemble_first_turn,
    dataset_to_agent_kwargs,
    synthetic_policy_row,
)
from budget_coder_rl.eval.m7d import (
    CELL_SPECS,
    E017_CHECKPOINT_MARKER,
    E017_FINGERPRINT_BUCKETS,
    EXPERIMENT_ID,
    FORBIDDEN_OUTPUT_IDS,
    GROUP_N,
    HIGH_FIRST_TURN,
    LOW_FIRST_TURN,
    N_SUBSET,
    SEED_POLICY,
    VALIDATE,
    VERDICTS,
    VLLM_ROLLOUT_N,
    analyze_cell_rows,
    apply_get_gen_batch_semantics,
    assign_logical_uids,
    build_execution_cells,
    build_first_request_record,
    build_unseeded_extra_info,
    canonicalize_sampling_params,
    cell_is_high,
    cell_is_low,
    compare_prompt_identity,
    compare_sampling_identity,
    decide_verdict,
    expand_trainer_siblings,
    extract_seed_report,
    first_generation_prompt_ids_unseeded,
    forbidden_output_dir_errors,
    lora_runtime_metadata,
    map_first_generation_bucket,
    probe_repeat_aliasing,
    sampling_contract,
    sibling_group_errors,
    subset_tasks,
    trajectory_info_from_index,
)


class FakeTokenizer:
    """Deterministic chat-template stand-in. Does not call HuggingFace."""

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        **kwargs,
    ):
        if tools is not None:
            raise AssertionError("M7D must not pass HF tools=")
        blob = json.dumps(list(messages), ensure_ascii=True, sort_keys=True)
        suffix = "<gen>" if add_generation_prompt else ""
        text = f"<bos>{blob}{suffix}"
        if not tokenize:
            return text
        return [ord(ch) % 97 + 11 for ch in text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr((int(item) - 11) + 32) for item in token_ids)


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
    split: str = "train",
    parse_ok: object = False,
    loc: float = 0.0,
) -> dict:
    return {
        "identity": {"instance_id": instance_id, "repo": "acme/repo", "split": split},
        "condition": {
            "obs_tokens_limit": 4096,
            "budget_visible": True,
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        },
        "localization": {"parse_ok": parse_ok, "localization_score": loc},
        "events": events,
        "termination": "max_turns",
    }


def _rows_for_cell(bucket: str, *, n: int, cell: str) -> list[dict]:
    if bucket == "valid":
        events = [_event(_tool("tree", {}), error_kind=None, action_name="tree")]
    elif bucket == "multiple":
        raw = (
            "<tool_call>\n{\"name\":\"tree\",\"arguments\":{}}\n</tool_call>\n"
            "<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"x\"}}\n</tool_call>"
        )
        events = [_event(raw, error_kind="protocol", parse_error_code="malformed_action")]
    else:
        events = [_event("just prose", error_kind="protocol", parse_error_code="malformed_action")]
    rows = []
    for index in range(n):
        row = _episode(instance_id=f"id-{index}", events=events, parse_ok=bucket == "valid")
        row["m7d"] = {"cell": cell, "sibling_index": 0, "uid": f"uid-{index}"}
        rows.append(row)
    return rows


def test_g4_sibling_expansion_count_and_uid():
    extras = [
        {"instance_id": "a", "index": 10},
        {"instance_id": "b", "index": 11},
    ]
    prompts = [[{"role": "user", "content": "issue a"}], [{"role": "user", "content": "issue b"}]]
    expansion = expand_trainer_siblings(
        extras=extras,
        raw_prompts=prompts,
        indices=[10, 11],
        group_n=GROUP_N,
    )
    assert expansion["n_rows"] == 8
    assert expansion["n_logical"] == 2
    assert sibling_group_errors(expansion) == []
    uids = [str(item) for item in expansion["expanded"]["uid"]]
    assert uids[0] == uids[1] == uids[2] == uids[3]
    assert uids[4] == uids[5] == uids[6] == uids[7]
    assert uids[0] != uids[4]
    rollouts = [item["rollout_n"] for item in expansion["trajectory_info"]]
    assert rollouts == [0, 1, 2, 3, 0, 1, 2, 3]


def test_sibling_message_objects_alias_under_np_repeat():
    extras = [{"instance_id": "a", "tags": ["keep"]}]
    prompts = [[{"role": "user", "content": "hello"}]]
    aliasing = probe_repeat_aliasing(extras=extras, raw_prompts=prompts, group_n=4)
    assert aliasing["any_extra_info_aliased"] is True
    assert aliasing["any_raw_prompt_aliased"] is True
    expansion = expand_trainer_siblings(extras=extras, raw_prompts=prompts, group_n=4)
    sibling_extras = list(expansion["expanded"]["extra_info"])
    assert sibling_extras[0] is sibling_extras[1]
    sibling_extras[0]["tags"].append("mutated")
    assert "mutated" in sibling_extras[3]["tags"]


def test_same_task_across_cells_identical_first_prompt_ids():
    tokenizer = FakeTokenizer()
    row = synthetic_policy_row(
        problem_statement="same issue body",
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="train",
        index=0,
    )
    ids = first_generation_prompt_ids_unseeded(row, tokenizer)
    kwargs = dataset_to_agent_kwargs(row)
    kwargs["extra_info"] = build_unseeded_extra_info(kwargs["extra_info"])
    records = []
    for cell, spec in CELL_SPECS.items():
        for sibling in range(spec["group_n"]):
            records.append(
                build_first_request_record(
                    cell=cell,
                    logical_task_index=0,
                    sibling_index=sibling,
                    instance_id="acme__widget-1",
                    uid="uid-1",
                    dataset_index=0,
                    extra_info=kwargs["extra_info"],
                    kwargs=kwargs,
                    tokenizer=tokenizer,
                )
            )
    cmp = compare_prompt_identity(records)
    assert cmp["identical"] is True
    assert cmp["divergences"] == []
    assert all(item["prompt_ids_sha256"] == records[0]["prompt_ids_sha256"] for item in records)
    assert ids
    assert "sampling_seed" not in kwargs["extra_info"]
    ctx = assemble_first_turn(kwargs, tokenizer)
    assert ctx["prompt_ids"] == ids


def test_sampling_parameter_canonicalization():
    contract = sampling_contract(group_n=4)
    canon = canonicalize_sampling_params(contract)
    assert canon["temperature"] == QWEN3_SAMPLING["temperature"] == 0.7
    assert canon["top_p"] == 0.8
    assert canon["top_k"] == 20
    assert canon["n"] == VLLM_ROLLOUT_N == 1
    assert canon["validate"] is VALIDATE is False
    assert contract["seed_policy"] == SEED_POLICY
    missing_n = canonicalize_sampling_params({"temperature": 0.7, "top_p": 0.8, "top_k": 20})
    assert missing_n["n"] == 1


def test_seed_extraction_reporting_unseeded():
    record = {
        "sampling_seed": None,
        "sibling_index": 2,
        "rollout_n": 2,
        "dataset_index": 7,
        "engine_seed": 20260826,
        "extra_info": {"sampling_seed": None},
        "canonical_sampling": {"seed": None, "temperature": 0.7},
    }
    report = extract_seed_report(record)
    assert report["extra_info_sampling_seed"] is None
    assert report["sampling_params_seed"] is None
    assert report["sibling_index"] == 2
    assert report["rollout_n"] == 2
    assert report["engine_seed"] == 20260826
    assert report["seed_policy"] == SEED_POLICY
    seeded = {
        "cell": "A",
        "instance_id": "x",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "validate": False,
        "effective_n": 1,
        "extra_info": {"sampling_seed": 9},
        "sampling_params": {"seed": 9},
    }
    cmp = compare_sampling_identity([seeded])
    assert cmp["identical"] is False
    assert any(item["field"] == "extra_info.sampling_seed" for item in cmp["divergences"])


def test_base_vs_lora_runtime_metadata():
    base = lora_runtime_metadata(
        cell="A",
        attached=False,
        lora_int_id=None,
        listed_ids=[],
        checkpoint_path=None,
    )
    assert base["ok"] is True
    residual = lora_runtime_metadata(
        cell="B",
        attached=False,
        lora_int_id=None,
        listed_ids=[VLLM_LORA_INT_ID],
    )
    assert residual["ok"] is False
    fresh = lora_runtime_metadata(
        cell="D",
        attached=True,
        lora_int_id=VLLM_LORA_INT_ID,
        listed_ids=[VLLM_LORA_INT_ID],
        checkpoint_path=None,
        lora_b_max_abs=0.0,
    )
    assert fresh["ok"] is True
    e017 = lora_runtime_metadata(
        cell="D",
        attached=True,
        lora_int_id=VLLM_LORA_INT_ID,
        listed_ids=[VLLM_LORA_INT_ID],
        checkpoint_path=f"/data/{E017_CHECKPOINT_MARKER}/global_step_275/actor",
    )
    assert e017["ok"] is False
    assert any("E017 checkpoint" in item for item in e017["errors"])


def test_taxonomy_denominator_first_turn():
    valid = _rows_for_cell("valid", n=3, cell="A")
    proto = _rows_for_cell("multiple", n=1, cell="A")
    stats = analyze_cell_rows(valid + proto, cell="A")
    assert stats["n_episodes"] == 4
    assert stats["denominators"]["first_turn_protocol_rate"] == 4
    assert stats["first_turn_protocol_rate"] == 0.25
    assert stats["first_generation_taxonomy"]["multiple_actions"] == 1
    assert stats["first_generation_taxonomy"]["valid_single_action"] == 3
    event = {
        "error_kind": "protocol",
        "raw_action": (
            "<tool_call>\n{\"name\":\"tree\",\"arguments\":{}}\n</tool_call>\n"
            "<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"x\"}}\n</tool_call>"
        ),
    }
    assert classify_event(event) == "multiple_actions"
    assert map_first_generation_bucket(event) == "multiple_actions"
    assert map_first_generation_bucket(
        {"error_kind": "protocol", "raw_action": "hello"}
    ) == "no_action"
    unbalanced = {
        "error_kind": "protocol",
        "raw_action": "<tool_call>\n{\"name\":\"tree\",\"arguments\":{}}\n",
    }
    assert map_first_generation_bucket(unbalanced) == "framing_unbalanced_tags"
    prose = {
        "error_kind": "protocol",
        "raw_action": "Sure.\n<tool_call>\n{\"name\":\"tree\",\"arguments\":{}}\n</tool_call>\nThanks.",
    }
    assert map_first_generation_bucket(prose) == "extra_prose"
    assert map_first_generation_bucket(None) == "valid_single_action"


def test_historical_artifacts_readonly():
    repo = Path("/tmp/bcrl-fake-root")
    for experiment_id in ("E017", "E018", "M7C"):
        errors = forbidden_output_dir_errors(
            repo / "outputs" / "experiments" / experiment_id,
            repo,
        )
        assert errors
    ok = forbidden_output_dir_errors(repo / "outputs" / "experiments" / EXPERIMENT_ID, repo)
    assert ok == []
    assert "M7C" in FORBIDDEN_OUTPUT_IDS
    assert "E017" in FORBIDDEN_OUTPUT_IDS


def test_verdict_tree_and_get_gen_batch_preserves_keys():
    low = analyze_cell_rows(_rows_for_cell("valid", n=8, cell="A"), cell="A")
    high = analyze_cell_rows(_rows_for_cell("multiple", n=8, cell="C"), cell="C")
    assert low["low"] is True
    assert high["high"] is True
    g4 = decide_verdict(
        request_divergences=[],
        cell_stats={"A": low, "B": low, "C": high, "D": high},
    )
    assert g4["verdict"] == "G4_sibling_path_implicated"
    trainer = decide_verdict(
        request_divergences=[],
        cell_stats={"A": low, "B": high, "C": high, "D": high},
    )
    assert trainer["verdict"] == "trainer_rollout_path_implicated"
    lora = decide_verdict(
        request_divergences=[],
        cell_stats={"A": low, "B": low, "C": low, "D": high},
    )
    assert lora["verdict"] == "fresh_lora_runtime_implicated"
    none = decide_verdict(
        request_divergences=[],
        cell_stats={"A": low, "B": low, "C": low, "D": low},
    )
    assert none["verdict"] == "E017_historical_pathology_not_reproduced"
    request = decide_verdict(
        request_divergences=[{"field": "prompt_ids_sha256", "why": "differ"}],
        cell_stats={"A": low, "B": low, "C": low, "D": low},
    )
    assert request["verdict"] == "first_request_divergence_found"
    non_tensor = {
        "raw_prompt": ["p"],
        "extra_info": [{"instance_id": "a"}],
        "uid": ["u"],
        "index": [0],
        "agent_name": ["repo_exploration"],
        "data_source": ["swe"],
        "reward_model": [{"ground_truth": "a"}],
    }
    survived = apply_get_gen_batch_semantics(non_tensor)
    assert set(survived) == set(non_tensor)
    assert "extra_info" in survived and "raw_prompt" in survived and "uid" in survived
    uids = assign_logical_uids(3)
    assert len(set(str(item) for item in uids)) == 3
    info = trajectory_info_from_index([1, 1, 1, 1], validate=False)
    assert [item["rollout_n"] for item in info] == [0, 1, 2, 3]
    cells = build_execution_cells()
    assert cells["no_optimizer_step"] is True
    assert cells["sampling"]["n"] == 1
    assert N_SUBSET == 16
    assert set(VERDICTS) >= {
        "G4_sibling_path_implicated",
        "E017_historical_pathology_not_reproduced",
    }
    assert cell_is_low(0.02, 0) is True
    assert cell_is_high(0.66, 10) is True
    assert LOW_FIRST_TURN == 0.10
    assert HIGH_FIRST_TURN == 0.20
    assert E017_FINGERPRINT_BUCKETS == ("multiple_actions", "framing_unbalanced_tags")


def test_subset_is_prefix(tmp_path: Path):
    manifest = tmp_path / "data" / "manifests"
    manifest.mkdir(parents=True)
    payload = {"ordered_ids": [f"id-{i}" for i in range(20)]}
    (manifest / "m5_scaled_train_candidates.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    subset = subset_tasks(repo_root=tmp_path, n=16)
    assert subset["train_ids"] == payload["ordered_ids"][:16]
    assert subset["n"] == 16
    assert subset["seed_policy"] == SEED_POLICY
    extra = build_unseeded_extra_info(
        {
            "instance_id": "a",
            "repo": "acme/widget",
            "split": "train",
            "sampling_seed": 123,
            "condition_id": "B1",
        }
    )
    assert "sampling_seed" not in extra
    assert extra["obs_tokens_limit"] == 4096
    assert extra["budget_visible"] is True
