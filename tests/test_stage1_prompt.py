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
from budget_coder_rl.budget.state import BudgetState, BUDGET_OBS_VERSION
from budget_coder_rl.protocol.prompt import (
    SEARCH_LITERAL_PHRASE,
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
    assert SEARCH_LITERAL_PHRASE in text
    assert "case-sensitive literal substring" in text


def test_hidden_messages_byte_identical_without_budget_kwargs():
    a = build_stage1_messages("the issue body", repo="owner/repo")
    b = build_stage1_messages(
        "the issue body",
        repo="owner/repo",
        budget_visible=False,
        budget_state=None,
    )
    assert a == b
    assert BUDGET_OBS_VERSION not in a[0]["content"]


def test_visible_budget_appends_to_system_not_issue():
    state = BudgetState(
        obs_tokens_used=0,
        obs_tokens_limit=8192,
        turns_used=0,
        turns_limit=6,
    )
    hidden = build_stage1_messages("the issue body", repo="owner/repo")
    visible = build_stage1_messages(
        "the issue body",
        repo="owner/repo",
        budget_state=state,
        budget_visible=True,
    )
    assert visible[1]["content"] == hidden[1]["content"]
    assert visible[0]["content"].startswith(hidden[0]["content"])
    assert BUDGET_OBS_VERSION in visible[0]["content"]
    assert "obs_tokens_remaining: 8192" in visible[0]["content"]
    assert SEARCH_LITERAL_PHRASE in visible[0]["content"]


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


def test_runtime_prompt_audit_flags_literal_search():
    from budget_coder_rl.protocol.prompt import runtime_prompt_audit

    audit = runtime_prompt_audit()
    assert audit["search_is_case_sensitive_literal_substring"] is True
    assert audit["search_literal_phrase"] == SEARCH_LITERAL_PHRASE
    assert len(audit["system_prompt_sha256"]) == 64
