"""M7A invalid-action forensics. CPU-only; does not touch AgentLoop."""

from __future__ import annotations

import json

from budget_coder_rl.eval.m5a import compute_bcrl_step_metrics
from budget_coder_rl.eval.m7a import (
    RECOVERABLE,
    TOOL_FAILURE,
    UNRECOVERABLE,
    analyze_corpus,
    analyze_episode,
    classify_event,
    episode_is_invalid,
    episode_parse_ok,
    event_is_invalid,
    naive_bool,
    recoverability_label,
    try_recover_action,
    truthy,
)
from budget_coder_rl.protocol.parser import parse_action


def _tool(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


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


def test_truthy_rejects_string_false():
    assert truthy("False") is False
    assert truthy("false") is False
    assert truthy("True") is True
    assert naive_bool("False") is True
    assert episode_parse_ok({"localization": {"parse_ok": "False"}}) is False
    assert episode_parse_ok({"localization": {"parse_ok": "True"}}) is True
    assert episode_parse_ok({"localization": {"parse_ok": False}}) is False


def test_invalid_action_is_episode_any_protocol_or_tool_error():
    clean = {
        "events": [
            _event(_tool("search", {"query": "x"}), error_kind=None, action_name="search"),
        ],
        "localization": {"parse_ok": False},
        "termination": "budget_exhausted",
    }
    mixed = {
        "events": [
            _event(_tool("read", {"path": "missing.py", "start_line": 1, "end_line": 2}), error_kind="tool", action_name="read"),
            _event(_tool("search", {"query": "x"}), error_kind=None, action_name="search"),
        ],
        "localization": {"parse_ok": True},
        "termination": "finish",
    }
    proto = {
        "events": [
            _event("just prose", error_kind="protocol", parse_error_code="malformed_action"),
        ],
        "localization": {"parse_ok": False},
        "termination": "max_turns",
    }
    assert episode_is_invalid(clean) is False
    assert episode_parse_ok(clean) is False
    assert episode_is_invalid(mixed) is True
    assert episode_parse_ok(mixed) is True
    assert episode_is_invalid(proto) is True
    assert event_is_invalid(mixed["events"][0]) is True
    assert event_is_invalid(mixed["events"][1]) is False


def test_invalid_rate_matches_training_step_metric():
    extras = [
        {
            "parse_ok": "False",
            "events": [_event("no tags", error_kind="protocol", parse_error_code="malformed_action")],
        },
        {
            "parse_ok": True,
            "events": [_event(_tool("tree", {}), error_kind=None, action_name="tree")],
        },
        {
            "parse_ok": False,
            "events": [
                _event(
                    _tool("read", {"path": "nope.py", "start_line": 1, "end_line": 2}),
                    error_kind="tool",
                    action_name="read",
                )
            ],
        },
        {"parse_ok": True, "events": []},
    ]
    metrics = compute_bcrl_step_metrics(
        uids=["a", "a", "b", "b"],
        rewards=[0.0, 0.5, 0.0, 0.0],
        extra_rows=extras,
    )
    corpus = analyze_corpus(
        [
            {"events": row["events"], "localization": {"parse_ok": row["parse_ok"]}}
            for row in extras
        ],
        source="fixture",
        keep_examples=False,
    )
    assert metrics["bcrl/invalid_action_rate"] == 0.5
    assert metrics["bcrl/parse_ok_rate"] == 0.5
    assert corpus["episode_invalid_rate"] == 0.5
    assert corpus["episode_parse_ok_rate"] == 0.5
    assert corpus["event_invalid_rate"] == 2 / 3


def test_taxonomy_from_real_e017_shapes():
    assert (
        classify_event(
            _event(
                "<tool_call>\n{\"name\":\"tree\",\"arguments\":{\"path\":\".\"}\n",
                error_kind="protocol",
                parse_error_code="malformed_action",
            )
        )
        == "framing_unbalanced_tags"
    )
    assert (
        classify_event(
            _event("search \"instantiate\" .", error_kind="protocol", parse_error_code="malformed_action")
        )
        == "no_recognizable_action"
    )
    prose = (
        "please look around\n"
        + _tool("tree", {"path": "."})
    )
    assert (
        classify_event(
            _event(prose, error_kind="protocol", parse_error_code="multiple_actions")
        )
        == "surrounding_prose"
    )
    two = _tool("tree", {}) + "\n" + _tool("search", {"query": "x"})
    assert (
        classify_event(
            _event(two, error_kind="protocol", parse_error_code="multiple_actions")
        )
        == "multiple_actions"
    )
    untagged = '{"name":"search","arguments":{"query":"instantiate","path":"src"}}'
    assert (
        classify_event(
            _event(untagged, error_kind="protocol", parse_error_code="malformed_action")
        )
        == "framing_wrong_envelope"
    )
    wrong_env = '<tool_call>\n{"locations":[{"path":"src/bokeh/bokeh.py"}]}\n</tool_call>'
    assert (
        classify_event(
            _event(wrong_env, error_kind="protocol", parse_error_code="invalid_arguments")
        )
        == "framing_wrong_envelope"
    )
    max_depth = '<tool_call>\n{"name":"tree","arguments":{"path":".","max_depth":8}}\n</tool_call>'
    assert (
        classify_event(
            _event(max_depth, error_kind="protocol", parse_error_code="invalid_arguments")
        )
        == "wrong_schema"
    )
    bad_type = '<tool_call>\n{"name":"tree","arguments":{"depth":2.0}}\n</tool_call>'
    assert (
        classify_event(
            _event(bad_type, error_kind="protocol", parse_error_code="invalid_arguments")
        )
        == "bad_args"
    )
    unknown = _tool("bash", {"cmd": "ls"})
    assert (
        classify_event(
            _event(unknown, error_kind="protocol", parse_error_code="unknown_tool")
        )
        == "unknown_tool"
    )
    malformed = "<tool_call>{not json}</tool_call>"
    assert (
        classify_event(
            _event(malformed, error_kind="protocol", parse_error_code="malformed_json")
        )
        == "malformed_json"
    )
    tool_err = _event(
        _tool("read", {"path": "src/README.md", "start_line": 1, "end_line": 100}),
        error_kind="tool",
        action_name="read",
    )
    assert classify_event(tool_err) == "tool_semantic_misuse"
    assert classify_event(_event(_tool("tree", {}), error_kind=None, action_name="tree")) is None


def test_recover_surrounding_prose_and_untagged_json_and_unclosed_tag():
    block = _tool("search", {"query": "example", "path": "."})
    prose = "I will search now.\n" + block + "\nthanks"
    recovered = try_recover_action(prose)
    assert recovered is not None
    parsed = parse_action(recovered)
    assert parsed.name == "search"
    assert recoverability_label(
        _event(prose, error_kind="protocol", parse_error_code="multiple_actions")
    ) == RECOVERABLE

    untagged = '{"name":"search","arguments":{"query":"instantiate","path":"src"}}'
    recovered_json = try_recover_action(untagged)
    assert recovered_json is not None
    assert parse_action(recovered_json).name == "search"

    final_json = '{"locations":[{"path":"src/foo.py","symbol":"Foo.bar"}]}'
    recovered_final = try_recover_action(final_json)
    assert recovered_final is not None
    assert len(parse_action(recovered_final).locations) == 1

    unclosed = '<tool_call>\n{"name":"tree","arguments":{"path":"."}}\n'
    recovered_close = try_recover_action(unclosed)
    assert recovered_close is not None
    assert parse_action(recovered_close).name == "tree"

    final_prose = "Final localization submission:\n" + _final(
        {"locations": [{"path": "src/hydra/utils.py", "symbol": "instantiate"}]}
    )
    assert try_recover_action(final_prose) is not None


def test_recover_rejects_ambiguous_or_schema_guesses():
    two = _tool("tree", {}) + "\n" + _tool("search", {"query": "x"})
    assert try_recover_action(two) is None
    with_missing_close_and_broken_json = (
        '<tool_call>\n{"name":"tree","arguments":{"path":"."}\n'
    )
    assert try_recover_action(with_missing_close_and_broken_json) is None
    max_depth = '<tool_call>\n{"name":"tree","arguments":{"path":".","max_depth":8}}\n</tool_call>'
    assert try_recover_action(max_depth) is None
    location_object = '{"path":"src/conanfile.py","symbol":"Foo.bar"}'
    assert try_recover_action(location_object) is None
    flattened = '<tool_call>\n{"name": "tree", "path": "src", "depth": 2}\n</tool_call>'
    assert try_recover_action(flattened) is None
    assert try_recover_action("just prose") is None
    valid = _tool("tree", {})
    assert try_recover_action(valid) is None
    parse_action(valid)


def test_tool_failure_is_not_protocol_recovery():
    event = _event(
        _tool("read", {"path": "src/README.md", "start_line": 1, "end_line": 100}),
        error_kind="tool",
        action_name="read",
    )
    assert recoverability_label(event) == TOOL_FAILURE
    assert try_recover_action(event["raw_action"]) is None


def test_analyze_episode_first_invalid_and_counts():
    row = {
        "identity": {"instance_id": "ex-1"},
        "termination": "max_turns",
        "localization": {"parse_ok": "False", "localization_score": 0.0},
        "events": [
            _event(_tool("tree", {}), error_kind=None, action_name="tree", turn=1),
            _event(
                "please\n" + _tool("search", {"query": "x"}),
                error_kind="protocol",
                parse_error_code="multiple_actions",
                turn=2,
            ),
            _event(
                _tool("read", {"path": "nope.py", "start_line": 1, "end_line": 2}),
                error_kind="tool",
                action_name="read",
                turn=3,
            ),
        ],
    }
    analyzed = analyze_episode(row)
    assert analyzed["invalid"] is True
    assert analyzed["parse_ok"] is False
    assert analyzed["first_invalid_turn"] == 2
    assert analyzed["n_invalid_events"] == 2
    assert analyzed["n_recoverable"] == 1
    assert analyzed["n_tool_fail"] == 1
    assert analyzed["episode_recoverability"] == "any_invalid_recoverable"
    corpus = analyze_corpus([row], source="fixture")
    assert corpus["taxonomy_event_counts"]["surrounding_prose"] == 1
    assert corpus["taxonomy_event_counts"]["tool_semantic_misuse"] == 1
    assert corpus["recoverability_event"][RECOVERABLE] == 1
    assert corpus["recoverability_event"][UNRECOVERABLE] == 0
    assert corpus["recoverability_event"][TOOL_FAILURE] == 1
    assert corpus["gate"]["start_m7b"] is False
