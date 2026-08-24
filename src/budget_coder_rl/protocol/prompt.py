"""Stage-1 runtime prompt. Dataset ``prompt`` stays raw issue text.

Built at AgentLoop time from problem_statement plus optional policy-safe
repo identity. Interpolates frozen M2B parser/tool constants so the
prompt cannot drift from the parser/tool contract by hand-copied numbers.

Must not receive or serialize privileged extra_info (patch, oracle, etc.).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from budget_coder_rl.budget.state import BudgetState, format_budget_state
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
    SEARCH_DEFAULT_PATH,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_NAMES,
    TREE_DEFAULT_DEPTH,
    TREE_DEFAULT_PATH,
)

_TOOL_NAME_LIST = ", ".join(sorted(TOOL_NAMES))


SEARCH_LITERAL_PHRASE = "case-sensitive literal substring"


def build_system_prompt(
    *,
    budget_state: BudgetState | None = None,
    budget_visible: bool = False,
) -> str:
    """Localization role, tool contract, and strict one-action protocol.

    Hidden / no-budget default text is frozen. Visible remaining-budget state
    is appended and must not change tools or the action protocol.
    """
    text = (
        "You are a repository-localization agent. Given a software issue and a "
        "read-only snapshot of the repository at a fixed base commit, explore "
        "the tree and submit the code locations that must be changed to address "
        "the issue.\n"
        "\n"
        "Stage 1 allows only structured exploration. Do not edit files, run "
        "tests, use a shell, or invent tools outside this contract.\n"
        "\n"
        f"Available tools: {_TOOL_NAME_LIST}.\n"
        "\n"
        f"- tree: list a directory. Arguments: path (default "
        f"{TREE_DEFAULT_PATH!r}), depth (default {TREE_DEFAULT_DEPTH}, max "
        f"{TREE_MAX_DEPTH}). At most {TREE_MAX_ENTRIES} entries are returned. "
        "Does not follow directory symlinks.\n"
        f"- search: {SEARCH_LITERAL_PHRASE} search. Arguments: query "
        f"(required, max {QUERY_MAX_CHARS} characters, single line), path "
        f"(default {SEARCH_DEFAULT_PATH!r}), max_results (default "
        f"{SEARCH_DEFAULT_MAX_RESULTS}, cap {SEARCH_MAX_RESULTS}). Hidden "
        "files are included. .gitignore is not honored. Symlink files are "
        "not followed. Binary / non-UTF-8 files are skipped.\n"
        f"- read: read a line range. Arguments: path, start_line, end_line "
        f"(1-based inclusive integers). At most {READ_MAX_LINES} lines or "
        f"{READ_MAX_CHARS} characters are returned.\n"
        "\n"
        "Each turn you must output exactly one action and nothing else: either "
        "one tool call or one final submission.\n"
        "\n"
        f"Tool call:\n{TOOL_CALL_OPEN}\n"
        '{"name":"search","arguments":{"query":"example","path":"."}}\n'
        f"{TOOL_CALL_CLOSE}\n"
        "\n"
        f"Final localization submission:\n{FINAL_OPEN}\n"
        '{"locations":[{"path":"src/foo.py","symbol":"Foo.bar"}]}\n'
        f"{FINAL_CLOSE}\n"
        "\n"
        "The locations array may be empty. Each location requires path "
        "(repo-relative, no traversal). symbol is optional. Do not wrap the "
        "action in markdown. Extra prose, multiple actions, unknown tools, "
        "or finish-as-a-tool-call are rejected.\n"
        "\n"
        "Submit when you have identified the relevant files/symbols. Do not "
        "attempt to patch the issue."
    )
    if not budget_visible:
        return text
    if budget_state is None or budget_state.obs_tokens_limit is None:
        raise ValueError("visible budget prompt requires a numeric obs_tokens_limit")
    return text + "\n\n" + format_budget_state(budget_state).rstrip("\n")


def build_user_prompt(problem_statement: str, *, repo: str | None = None) -> str:
    """Issue text plus optional repository identity. No oracle fields."""
    issue = problem_statement if isinstance(problem_statement, str) else str(
        problem_statement
    )
    repo_name = (repo or "").strip()
    if repo_name:
        return f"Repository: {repo_name}\n\nIssue:\n{issue}"
    return f"Issue:\n{issue}"


def build_stage1_messages(
    problem_statement: str,
    *,
    repo: str | None = None,
    budget_state: BudgetState | None = None,
    budget_visible: bool = False,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                budget_state=budget_state,
                budget_visible=budget_visible,
            ),
        },
        {"role": "user", "content": build_user_prompt(problem_statement, repo=repo)},
    ]


def extract_issue_text(raw_prompt: Any) -> str:
    """Take the last user-message content from dataset raw_prompt."""
    messages = _coerce_messages(raw_prompt)
    for message in reversed(messages):
        if str(message.get("role") or "") == "user":
            return str(message.get("content") or "")
    if messages:
        return str(messages[-1].get("content") or "")
    return ""


def policy_safe_repo(extra_info: Mapping[str, Any] | None) -> str | None:
    """Repo identity only. Does not copy extra_info into the prompt."""
    if extra_info is None:
        return None
    repo = extra_info.get("repo") if isinstance(extra_info, Mapping) else None
    if repo is None:
        return None
    text = str(repo).strip()
    return text or None


def _coerce_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    if raw_prompt is None:
        return []
    if hasattr(raw_prompt, "tolist"):
        try:
            raw_prompt = raw_prompt.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(raw_prompt, dict):
        return [dict(raw_prompt)]
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    if not isinstance(raw_prompt, list):
        return [{"role": "user", "content": str(raw_prompt)}]
    messages: list[dict[str, Any]] = []
    for item in raw_prompt:
        if hasattr(item, "items") and not isinstance(item, dict):
            item = {str(key): item[key] for key in item}
        if isinstance(item, dict):
            messages.append(
                {
                    "role": str(item.get("role") or ""),
                    "content": str(item.get("content") or ""),
                }
            )
        else:
            messages.append({"role": "user", "content": str(item)})
    return messages


def rendered_prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages)


def runtime_prompt_audit() -> dict[str, Any]:
    """Frozen hidden system-prompt facts. Not a trajectory-tuning surface."""
    text = build_system_prompt()
    return {
        "search_is_case_sensitive_literal_substring": SEARCH_LITERAL_PHRASE in text,
        "search_literal_phrase": SEARCH_LITERAL_PHRASE,
        "system_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "system_prompt_chars": len(text),
        "obs_version": "bcrl-obs-v1",
        "budget_envelope_version": "bcrl-budget-v1",
        "note": (
            "Confirmed: search is a case-sensitive literal substring. "
            "Do not trajectory-tune this prompt."
        ),
    }
