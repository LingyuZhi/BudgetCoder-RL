"""Observation-token hard budget. Tokenizer accounting lives in AgentLoop.

``# bcrl-obs-v1`` tool bodies are unchanged. Visible remaining-budget text is a
separate ``# bcrl-budget-v1`` envelope assembled at encode time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BUDGET_OBS_VERSION = "bcrl-budget-v1"
BUDGET_HEADER_MAX_ITERS = 4


class BudgetHeaderFixpointError(RuntimeError):
    """Visible budget header token length did not stabilize. Do not guess."""


class BudgetVisibleRequiresLimitError(ValueError):
    """``budget_visible=True`` is meaningless without a numeric obs-token limit."""


@dataclass
class BudgetState:
    """Cumulative inserted observation-token usage and turn safety cap."""

    obs_tokens_used: int = 0
    obs_tokens_limit: int | None = None
    turns_used: int = 0
    turns_limit: int = 6

    @property
    def obs_tokens_remaining(self) -> int | None:
        if self.obs_tokens_limit is None:
            return None
        return max(0, self.obs_tokens_limit - self.obs_tokens_used)

    @property
    def turns_remaining(self) -> int:
        return max(0, self.turns_limit - self.turns_used)

    def can_insert(self, cost: int) -> bool:
        if cost < 0:
            raise ValueError(f"observation token cost must be >= 0, got {cost}")
        if self.obs_tokens_limit is None:
            return True
        return self.obs_tokens_used + cost <= self.obs_tokens_limit

    def consume(self, cost: int) -> None:
        if not self.can_insert(cost):
            raise RuntimeError(
                "refusing to consume over the observation-token budget; "
                "caller must not insert truncated observations"
            )
        self.obs_tokens_used += cost

    def as_dict(self) -> dict[str, int | None]:
        return {
            "obs_tokens_used": self.obs_tokens_used,
            "obs_tokens_limit": self.obs_tokens_limit,
            "obs_tokens_remaining": self.obs_tokens_remaining,
            "turns_used": self.turns_used,
            "turns_limit": self.turns_limit,
            "turns_remaining": self.turns_remaining,
        }


def format_budget_state(state: BudgetState) -> str:
    """Frozen ``bcrl-budget-v1`` text. Requires a numeric obs-token limit."""
    if state.obs_tokens_limit is None:
        raise BudgetVisibleRequiresLimitError(
            "cannot format budget state without obs_tokens_limit"
        )
    remaining = max(0, state.obs_tokens_limit - state.obs_tokens_used)
    return (
        f"# {BUDGET_OBS_VERSION}\n"
        f"obs_tokens_used: {state.obs_tokens_used}\n"
        f"obs_tokens_limit: {state.obs_tokens_limit}\n"
        f"obs_tokens_remaining: {remaining}\n"
        f"turns_used: {state.turns_used}\n"
        f"turns_limit: {state.turns_limit}\n"
        f"turns_remaining: {state.turns_remaining}\n"
    )


def wrap_observation_with_budget(v1_text: str, state: BudgetState) -> str:
    """Envelope around frozen ``# bcrl-obs-v1`` text. Does not rewrite the body."""
    header = format_budget_state(state).rstrip("\n")
    return header + "\n\n" + v1_text


def converge_visible_observation(
    v1_text: str,
    *,
    used_before: int,
    limit: int,
    turns_used: int,
    turns_limit: int,
    encode: Callable[[str], Sequence[int]],
    max_iters: int = BUDGET_HEADER_MAX_ITERS,
) -> tuple[list[int], str]:
    """Encode visible observation with remaining-after numbers.

    Displayed ``obs_tokens_used`` is the post-insert total so the next generate
    sees an up-to-date remaining value. Digit-length effects are resolved by
    iterating the displayed used count; failure hard-fails.
    """
    if limit < 0:
        raise ValueError(f"obs_tokens_limit must be >= 0, got {limit}")
    displayed_used = used_before
    last_ids: list[int] | None = None
    for _ in range(max_iters):
        state = BudgetState(
            obs_tokens_used=displayed_used,
            obs_tokens_limit=limit,
            turns_used=turns_used,
            turns_limit=turns_limit,
        )
        content = wrap_observation_with_budget(v1_text, state)
        ids = list(encode(content))
        last_ids = ids
        used_after = used_before + len(ids)
        if displayed_used == used_after:
            return ids, content
        displayed_used = used_after
    raise BudgetHeaderFixpointError(
        "bcrl-budget-v1 header fixpoint did not converge after "
        f"{max_iters} iterations (used_before={used_before}, "
        f"last_encoded={0 if last_ids is None else len(last_ids)})"
    )


def resolve_episode_budget(
    extra_info: Mapping[str, Any] | None,
    *,
    default_limit: int | None,
    default_visible: bool,
) -> tuple[int | None, bool]:
    """Per-episode knobs from runtime extra_info; constructor values are defaults.

    Does not read oracle fields. ``budget_visible=True`` requires a numeric limit.
    """
    info = extra_info if isinstance(extra_info, Mapping) else {}
    if "obs_tokens_limit" in info:
        limit = _coerce_optional_int(info.get("obs_tokens_limit"))
    else:
        limit = default_limit
    if "budget_visible" in info:
        visible = _coerce_bool(info.get("budget_visible"))
    else:
        visible = bool(default_visible)
    if visible and limit is None:
        raise BudgetVisibleRequiresLimitError(
            "budget_visible=True requires a numeric obs_tokens_limit"
        )
    if limit is not None and limit < 0:
        raise ValueError(f"obs_tokens_limit must be >= 0, got {limit}")
    return limit, visible


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("obs_tokens_limit must be an int or None, not bool")
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            raise ValueError("obs_tokens_limit must be an int or None, not bool")
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "none", "null"}:
            return None
        return int(stripped)
    return int(value)


def _coerce_bool(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "1", "yes"}:
            return True
        if stripped in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"cannot coerce budget_visible from {value!r}")
    return bool(value)
