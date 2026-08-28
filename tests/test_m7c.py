"""M7C prompt-path and replay-analysis contracts. CPU-only."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.agent_loop.tokenization import encode_chat_messages
from budget_coder_rl.data.swe_gym_fields import POLICY_FORBIDDEN_DERIVED_FIELDS
from budget_coder_rl.data.swe_gym_materialize import EXTRA_INFO_KEYS, POLICY_COLUMNS
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING
from budget_coder_rl.eval.m7a import episode_parse_ok, naive_bool
from budget_coder_rl.eval.m7c import (
    APPLY_CHAT_TEMPLATE_KWARGS,
    N_SUBSET,
    OBS_TOKENS_LIMIT,
    SEED_BASE,
    VALIDATE,
    VERDICT_NOT_REPRODUCED_ABS,
    VERDICT_STRENGTHENED_DELTA,
    VLLM_ROLLOUT_N,
    analyze_split_rows,
    assemble_first_turn,
    build_matched_extra_info,
    coarsen_taxonomy,
    compact_episode_metrics,
    compare_splits,
    dataset_to_agent_kwargs,
    decide_verdict,
    first_generation_prompt_ids,
    leakage_scan,
    policy_row_schema_errors,
    replay_seed,
    sampling_contract,
    select_subset,
    synthetic_policy_row,
    truthy,
)
from budget_coder_rl.protocol.prompt import build_stage1_messages, policy_safe_repo


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
            raise AssertionError("M7C must not pass HF tools=")
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


def test_train_dev_task_view_schema_identical():
    train = synthetic_policy_row(
        problem_statement="issue A",
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="train",
        index=0,
    )
    dev = synthetic_policy_row(
        problem_statement="issue A",
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="dev",
        index=1,
    )
    assert list(train.keys()) == list(dev.keys()) == list(POLICY_COLUMNS)
    assert set(train["extra_info"]) == set(dev["extra_info"]) == set(EXTRA_INFO_KEYS)
    assert not policy_row_schema_errors(train)
    assert not policy_row_schema_errors(dev)
    for field in POLICY_FORBIDDEN_DERIVED_FIELDS:
        assert field not in train["extra_info"]
        assert field not in dev["extra_info"]


def test_evaluator_only_fields_do_not_enter_prompt():
    sentinels = {
        field: f"LEAK_{field.upper()}_SENTINEL"
        for field in ("oracle_symbols", "patch", "difficulty", "search_space")
    }
    extra = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "split": "train",
        **sentinels,
    }
    messages = build_stage1_messages(
        "benign issue text",
        repo=policy_safe_repo(extra),
    )
    blob = "\n".join(str(item.get("content") or "") for item in messages)
    for value in sentinels.values():
        assert value not in blob
    assert extra["split"] not in messages[0]["content"]
    assert extra["instance_id"] not in blob
    errors = leakage_scan(
        rendered=blob,
        decoded=blob,
        system_prompt=messages[0]["content"],
        user_prompt=messages[1]["content"],
        problem_statement="benign issue text",
        extra_info={"instance_id": "owner__repo-1", "repo": "owner/repo", "split": "train"},
    )
    assert errors == []


def test_same_synthetic_issue_same_runtime_messages():
    issue = "identical synthetic problem_statement"
    train = synthetic_policy_row(
        problem_statement=issue,
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="train",
        index=0,
    )
    dev = synthetic_policy_row(
        problem_statement=issue,
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="dev",
        index=99,
    )
    tokenizer = FakeTokenizer()
    train_kw = dataset_to_agent_kwargs(train)
    train_kw["extra_info"] = build_matched_extra_info(
        train_kw["extra_info"], sampling_seed=SEED_BASE
    )
    dev_kw = dataset_to_agent_kwargs(dev)
    dev_kw["extra_info"] = build_matched_extra_info(
        dev_kw["extra_info"], sampling_seed=SEED_BASE
    )
    train_ctx = assemble_first_turn(train_kw, tokenizer)
    dev_ctx = assemble_first_turn(dev_kw, tokenizer)
    assert train_ctx["messages"] == dev_ctx["messages"]
    assert train_kw["extra_info"]["split"] == "train"
    assert dev_kw["extra_info"]["split"] == "dev"


def test_same_messages_identical_prompt_token_ids():
    tokenizer = FakeTokenizer()
    messages = build_stage1_messages("issue body", repo="acme/widget")
    left = encode_chat_messages(tokenizer, messages)
    right = encode_chat_messages(tokenizer, messages)
    assert left == right
    assert left


def test_synthetic_train_dev_loader_prompt_ids_equal():
    tokenizer = FakeTokenizer()
    issue = "same fake task routed through train vs dev extra_info.split"
    train = synthetic_policy_row(
        problem_statement=issue,
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="train",
        index=0,
    )
    dev = synthetic_policy_row(
        problem_statement=issue,
        repo="acme/widget",
        instance_id="acme__widget-1",
        split="dev",
        index=7,
    )
    train_ids = first_generation_prompt_ids(train, tokenizer, sampling_seed=SEED_BASE)
    dev_ids = first_generation_prompt_ids(dev, tokenizer, sampling_seed=SEED_BASE)
    assert train_ids == dev_ids
    assert train["extra_info"]["split"] != dev["extra_info"]["split"]


def test_sampling_contract_matches_qwen3():
    contract = sampling_contract()
    assert contract["temperature"] == QWEN3_SAMPLING["temperature"] == 0.7
    assert contract["top_p"] == 0.8
    assert contract["top_k"] == 20
    assert contract["n"] == VLLM_ROLLOUT_N == 1
    assert contract["validate"] is VALIDATE is False
    assert contract["obs_tokens_limit"] == OBS_TOKENS_LIMIT == 4096
    assert contract["apply_chat_template_kwargs"] == dict(APPLY_CHAT_TEMPLATE_KWARGS)
    assert replay_seed(0) == SEED_BASE
    assert replay_seed(3) == SEED_BASE + 3


def test_truthy_string_false_does_not_inflate_parse_ok():
    row = _episode(
        instance_id="c",
        events=[_event(_tool("tree", {}), error_kind=None, action_name="tree")],
        parse_ok="False",
    )
    metrics = compact_episode_metrics(row)
    assert metrics["parse_ok"] is False
    assert episode_parse_ok(row) is False
    assert truthy("False") is False
    assert naive_bool("False") is True


def test_replay_denominators_first_turn_and_events():
    proto = _episode(
        instance_id="a",
        split="train",
        events=[
            _event("just prose", error_kind="protocol", parse_error_code="malformed_action", turn=1),
            _event(_tool("tree", {}), error_kind=None, action_name="tree", turn=2),
        ],
        parse_ok="False",
    )
    tool_later = _episode(
        instance_id="b",
        split="train",
        events=[
            _event(_tool("tree", {}), error_kind=None, action_name="tree", turn=1),
            _event(
                _tool("read", {"path": "nope.py", "start_line": 1, "end_line": 2}),
                error_kind="tool",
                action_name="read",
                turn=2,
            ),
        ],
        parse_ok=True,
        loc=0.2,
    )
    stats = analyze_split_rows([proto, tool_later], split="train")
    assert stats["n_episodes"] == 2
    assert stats["n_events"] == 4
    assert stats["n_invalid_events"] == 2
    assert stats["event_invalid_rate"] == 0.5
    assert stats["first_turn_protocol_rate"] == 0.5
    assert stats["denominators"]["first_turn_protocol_rate"] == 2
    assert stats["denominators"]["event_invalid_rate"] == 4
    assert stats["episode_invalid_rate"] == 1.0
    assert coarsen_taxonomy("multiple_actions") == "multiple_actions"
    assert coarsen_taxonomy("wrong_schema") == "other_protocol"
    assert coarsen_taxonomy("runtime_infra", error_kind="tool") == "other_tool_error"


def test_pre_registered_verdicts():
    strengthened = decide_verdict(
        allow_replay=True,
        confound_reasons=[],
        comparison={"first_turn_protocol_delta_train_minus_dev": VERDICT_STRENGTHENED_DELTA},
    )
    reproduced = decide_verdict(
        allow_replay=True,
        confound_reasons=[],
        comparison={"first_turn_protocol_delta_train_minus_dev": 0.05},
    )
    confound = decide_verdict(
        allow_replay=False,
        confound_reasons=["synthetic ids differ"],
        comparison={"first_turn_protocol_delta_train_minus_dev": 0.9},
    )
    ambiguous = decide_verdict(
        allow_replay=True,
        confound_reasons=[],
        comparison={"first_turn_protocol_delta_train_minus_dev": 0.15},
    )
    assert strengthened["verdict"] == "H3_causally_strengthened"
    assert reproduced["verdict"] == "H3_not_reproduced"
    assert reproduced["primary_delta"] < VERDICT_NOT_REPRODUCED_ABS
    assert confound["verdict"] == "execution_path_confound_found"
    assert ambiguous["ambiguous_band"] is True
    assert compare_splits(
        {"first_turn_protocol_rate": 0.5},
        {"first_turn_protocol_rate": 0.1},
    )["first_turn_protocol_delta_train_minus_dev"] == 0.4


def test_select_subset_is_prefix():
    ids = [f"id-{i}" for i in range(10)]
    assert select_subset(ids, 4) == ids[:4]
    assert N_SUBSET == 64


def test_matched_extra_info_strips_condition_and_keeps_split():
    extra = build_matched_extra_info(
        {
            "instance_id": "a",
            "repo": "acme/widget",
            "split": "dev",
            "condition_id": "B1",
            "policy": "base",
            "patch": "should be dropped if present as key",
        },
        sampling_seed=7,
    )
    assert extra["split"] == "dev"
    assert extra["sampling_seed"] == 7
    assert extra["obs_tokens_limit"] == 4096
    assert extra["budget_visible"] is True
    assert "condition_id" not in extra
    assert "policy" not in extra
    assert "patch" not in extra
