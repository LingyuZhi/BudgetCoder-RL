"""Strict action parser tests. No filesystem, workspace, or oracle."""

from __future__ import annotations

import json

import pytest

from budget_coder_rl.protocol import (
    OBS_VERSION,
    FinalAction,
    ProtocolError,
    ToolCall,
    format_error,
    parse_action,
)


def _tool(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


def test_valid_tool_call_tree_defaults():
    action = parse_action(_tool("tree", {}))
    assert isinstance(action, ToolCall)
    assert action.name == "tree"
    assert action.arguments == {"path": ".", "depth": 2}


def test_valid_search_and_read():
    search = parse_action(
        _tool("search", {"query": "cache invalidation", "path": "src", "max_results": 10})
    )
    assert isinstance(search, ToolCall)
    assert search.arguments["query"] == "cache invalidation"
    assert search.arguments["path"] == "src"
    assert search.arguments["max_results"] == 10
    read = parse_action(_tool("read", {"path": "src/a.py", "start_line": 1, "end_line": 20}))
    assert isinstance(read, ToolCall)
    assert read.arguments == {"path": "src/a.py", "start_line": 1, "end_line": 20}


def test_valid_final_with_optional_symbol():
    action = parse_action(
        _final({"locations": [{"path": "src/foo.py", "symbol": "Foo.bar"}]})
    )
    assert isinstance(action, FinalAction)
    assert len(action.locations) == 1
    assert action.locations[0].path == "src/foo.py"
    assert action.locations[0].symbol == "Foo.bar"
    empty = parse_action(_final({"locations": []}))
    assert empty.locations == ()
    path_only = parse_action(_final({"locations": [{"path": "src/foo.py"}]}))
    assert path_only.locations[0].symbol is None


def test_malformed_json():
    with pytest.raises(ProtocolError) as exc:
        parse_action("<tool_call>{not json}</tool_call>")
    assert exc.value.code == "malformed_json"


def test_duplicate_json_keys_rejected():
    raw = '<tool_call>{"name":"tree","name":"search","arguments":{}}</tool_call>'
    with pytest.raises(ProtocolError) as exc:
        parse_action(raw)
    assert exc.value.code == "duplicate_keys"


def test_multiple_actions_and_prose():
    two = _tool("tree", {}) + "\n" + _tool("search", {"query": "x"})
    with pytest.raises(ProtocolError) as exc:
        parse_action(two)
    assert exc.value.code == "multiple_actions"
    mixed = _tool("tree", {}) + "\n" + _final({"locations": []})
    with pytest.raises(ProtocolError) as exc:
        parse_action(mixed)
    assert exc.value.code == "multiple_actions"
    prose = "please run this\n" + _tool("tree", {})
    with pytest.raises(ProtocolError) as exc:
        parse_action(prose)
    assert exc.value.code == "multiple_actions"


def test_empty_and_untagged():
    with pytest.raises(ProtocolError) as exc:
        parse_action("   \n")
    assert exc.value.code == "empty_action"
    with pytest.raises(ProtocolError) as exc:
        parse_action("just prose")
    assert exc.value.code == "malformed_action"


def test_unknown_tool_and_finish_as_tool():
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("bash", {"cmd": "ls"}))
    assert exc.value.code == "unknown_tool"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("finish", {"locations": []}))
    assert exc.value.code == "unknown_tool"


def test_invalid_argument_types_not_coerced():
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("tree", {"depth": 2.0}))
    assert exc.value.code == "invalid_arguments"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("tree", {"depth": True}))
    assert exc.value.code == "invalid_arguments"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("tree", {"extra": 1}))
    assert exc.value.code == "invalid_arguments"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("search", {"query": 1}))
    assert exc.value.code == "invalid_arguments"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_tool("read", {"path": "a.py", "start_line": "1", "end_line": 2}))
    assert exc.value.code == "invalid_arguments"


def test_malformed_final_cases():
    with pytest.raises(ProtocolError) as exc:
        parse_action(_final({"files": ["a.py"]}))
    assert exc.value.code == "malformed_final"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_final({"locations": [{"path": "../etc/passwd"}]}))
    assert exc.value.code == "malformed_final"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_final({"locations": [{"path": "/abs/foo.py"}]}))
    assert exc.value.code == "malformed_final"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_final({"locations": [{"path": "src/a.py", "symbol": ""}]}))
    assert exc.value.code == "malformed_final"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_final({"locations": [{"path": "src/a.py", "line": 3}]}))
    assert exc.value.code == "malformed_final"
    with pytest.raises(ProtocolError) as exc:
        parse_action(_final({"locations": [{"path": ""}]}))
    assert exc.value.code == "malformed_final"


def test_no_path_canonicalization_or_dedup():
    action = parse_action(
        _final(
            {
                "locations": [
                    {"path": "src/foo.py", "symbol": "A"},
                    {"path": "src/foo.py", "symbol": "A"},
                    {"path": "src/./foo.py"},
                ]
            }
        )
    )
    assert [item.path for item in action.locations] == [
        "src/foo.py",
        "src/foo.py",
        "src/./foo.py",
    ]


def test_error_observation_is_stable():
    first = format_error(tool="protocol", code="malformed_json", message="malformed JSON")
    second = format_error(tool="protocol", code="malformed_json", message="malformed JSON")
    assert first == second
    assert first.startswith(f"# {OBS_VERSION}\n")
    assert "status: error\n" in first
    assert first.endswith("\n")
    assert "---" not in first


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_numbers_are_rejected(token: str):
    raw = f'<tool_call>{{"name":"tree","arguments":{{"depth":{token}}}}}</tool_call>'
    with pytest.raises(ProtocolError) as exc:
        parse_action(raw)
    assert exc.value.code == "malformed_json"
    assert "non-finite" in exc.value.message
    first = format_error(tool="protocol", code="malformed_json", message="malformed JSON")
    second = format_error(tool="protocol", code="malformed_json", message="malformed JSON")
    assert first == second
    assert first.startswith(f"# {OBS_VERSION}\n")
    assert "status: error\n" in first
    assert first.endswith("\n")
    assert "---" not in first
