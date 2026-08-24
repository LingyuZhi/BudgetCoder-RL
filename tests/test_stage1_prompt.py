"""Runtime Stage-1 prompt builder: M2B constants, no privileged leakage."""

from __future__ import annotations

from budget_coder_rl.data.swe_gym_fields import POLICY_FORBIDDEN_DERIVED_FIELDS
from budget_coder_rl.env.tools import (
    QUERY_MAX_CHARS,
    READ_MAX_CHARS,
    READ_MAX_LINES,
    SEARCH_MAX_RESULTS,
    TREE_MAX_DEPTH,
    TREE_MAX_ENTRIES,
)
from budget_coder_rl.protocol.parser import (
    FINAL_CLOSE,
    FINAL_OPEN,
    SEARCH_DEFAULT_MAX_RESULTS,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_NAMES,
    TREE_DEFAULT_DEPTH,
)
from budget_coder_rl.protocol.prompt import (
    build_stage1_messages,
    build_system_prompt,
    extract_issue_text,
    policy_safe_repo,
    rendered_prompt_text,
)


def test_system_prompt_uses_m2b_constants():
    text = build_system_prompt()
    for name in TOOL_NAMES:
        assert name in text
    assert TOOL_CALL_OPEN in text
    assert TOOL_CALL_CLOSE in text
    assert FINAL_OPEN in text
    assert FINAL_CLOSE in text
    assert str(TREE_DEFAULT_DEPTH) in text
    assert str(TREE_MAX_DEPTH) in text
    assert str(TREE_MAX_ENTRIES) in text
    assert str(SEARCH_DEFAULT_MAX_RESULTS) in text
    assert str(SEARCH_MAX_RESULTS) in text
    assert str(READ_MAX_LINES) in text
    assert str(READ_MAX_CHARS) in text
    assert str(QUERY_MAX_CHARS) in text
    assert "Do not edit" in text
    assert "tests" in text.lower()


def test_stage1_messages_wrap_issue_and_repo():
    messages = build_stage1_messages("the issue body", repo="owner/repo")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "the issue body" in messages[1]["content"]
    assert "owner/repo" in messages[1]["content"]
    assert extract_issue_text([{"role": "user", "content": "the issue body"}]) == (
        "the issue body"
    )


def test_prompt_does_not_inject_privileged_extra_info_values():
    sentinels = {
        field: f"LEAK_{field.upper()}_SENTINEL"
        for field in POLICY_FORBIDDEN_DERIVED_FIELDS
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
    blob = rendered_prompt_text(messages)
    assert "owner/repo" in blob
    assert "benign issue text" in blob
    for field, value in sentinels.items():
        assert value not in blob, field
    assert extra["base_commit"] not in blob
    assert "owner__repo-1" not in blob
    assert extra["split"] not in messages[1]["content"]
