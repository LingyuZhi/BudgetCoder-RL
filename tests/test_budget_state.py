"""BudgetState formatting, remaining-after v2 header, extra_info knobs."""

from __future__ import annotations

import pytest

from budget_coder_rl.budget.state import (
    BUDGET_ACCOUNTING_VERSION,
    BUDGET_OBS_VERSION,
    BudgetState,
    BudgetVisibleRequiresLimitError,
    converge_visible_observation,
    format_budget_state,
    resolve_episode_budget,
    wrap_observation_with_budget,
)


def test_remaining_and_can_insert():
    state = BudgetState(obs_tokens_used=10, obs_tokens_limit=12, turns_used=1, turns_limit=6)
    assert state.obs_tokens_remaining == 2
    assert state.turns_remaining == 5
    assert state.can_insert(2)
    assert not state.can_insert(3)
    state.consume(2)
    assert state.obs_tokens_used == 12
    assert state.obs_tokens_remaining == 0


def test_unlimited_always_inserts():
    state = BudgetState(obs_tokens_used=99, obs_tokens_limit=None)
    assert state.obs_tokens_remaining is None
    assert state.can_insert(10_000)
    state.consume(5)
    assert state.obs_tokens_used == 104


def test_format_requires_limit():
    with pytest.raises(BudgetVisibleRequiresLimitError):
        format_budget_state(BudgetState(obs_tokens_limit=None))


def test_format_and_wrap_keep_v1_body():
    state = BudgetState(
        obs_tokens_used=40,
        obs_tokens_limit=100,
        turns_used=2,
        turns_limit=6,
    )
    header = format_budget_state(state)
    assert header.startswith(f"# {BUDGET_OBS_VERSION}\n")
    assert "obs_tokens_remaining: 60" in header
    assert "turns_remaining: 4" in header
    v1 = "# bcrl-obs-v1\ntool: tree\nstatus: ok\n---\nf pkg.py\n"
    wrapped = wrap_observation_with_budget(v1, state)
    assert wrapped.startswith(f"# {BUDGET_OBS_VERSION}\n")
    assert wrapped.endswith(v1)
    assert "# bcrl-obs-v1" in wrapped


def test_visible_header_uses_v1_primary_cost_not_wrapped_length():
    v1 = "# bcrl-obs-v1\nbody\n"

    def encode(content: str) -> list[int]:
        return [1] * len(content)

    ids, content, v1_n = converge_visible_observation(
        v1,
        used_before=0,
        limit=10_000,
        turns_used=1,
        turns_limit=6,
        encode=encode,
    )
    assert v1_n == len(v1)
    assert len(ids) == len(content)
    assert len(ids) > v1_n
    assert f"obs_tokens_used: {v1_n}" in content
    assert f"obs_tokens_remaining: {10_000 - v1_n}" in content
    assert f"obs_tokens_used: {len(ids)}" not in content
    assert content.endswith(v1)


def test_as_dict_includes_accounting_version():
    state = BudgetState(obs_tokens_used=1, obs_tokens_limit=8)
    payload = state.as_dict()
    assert payload["budget_accounting_version"] == BUDGET_ACCOUNTING_VERSION


def test_resolve_episode_budget_override_and_visible_requires_limit():
    limit, visible = resolve_episode_budget(
        {},
        default_limit=None,
        default_visible=False,
    )
    assert limit is None
    assert visible is False
    limit, visible = resolve_episode_budget(
        {"obs_tokens_limit": "512", "budget_visible": "true"},
        default_limit=None,
        default_visible=False,
    )
    assert limit == 512
    assert visible is True
    with pytest.raises(BudgetVisibleRequiresLimitError):
        resolve_episode_budget(
            {"budget_visible": True},
            default_limit=None,
            default_visible=False,
        )
    limit, visible = resolve_episode_budget(
        {"obs_tokens_limit": None},
        default_limit=8192,
        default_visible=False,
    )
    assert limit is None
