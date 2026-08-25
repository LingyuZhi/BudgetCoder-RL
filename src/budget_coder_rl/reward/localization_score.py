"""veRL RewardLoop ``compute_score`` for Stage-1 localization.

Looks up the frozen M1E evaluator sidecar by opaque ``instance_id``.
Does not parse gold from the policy prompt or from ``solution_str``.
AgentLoop must not import this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping as MappingABC
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from budget_coder_rl.data.swe_gym_materialize import oracle_parquet_path
from budget_coder_rl.eval.localization import evaluate_episode
from budget_coder_rl.eval.oracle import EvaluatorOracleIndex, load_evaluator_oracle

GOLD_EXTRA_KEYS = (
    "oracle_symbols",
    "base_changed_files",
    "gold_edit_files",
    "patch",
    "test_patch",
)


class LocalizationScoreError(ValueError):
    """RewardLoop wiring or instance_id contract failure."""


def resolve_oracle_parquet(explicit: str | Path | None = None) -> Path:
    """Sidecar path: explicit arg, ``BCRL_ORACLE_PARQUET``, then repo default."""
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("BCRL_ORACLE_PARQUET")
    if env:
        return Path(env).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return oracle_parquet_path(repo_root).resolve()


def compute_score(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    oracle_parquet: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """NaiveRewardManager entrypoint. Return a dict with ``score``.

    ``ground_truth`` is the opaque M1E ``instance_id``. Submission and
    termination come from AgentLoop ``extra_fields`` merged into
    ``extra_info`` by pinned veRL NaiveRewardManager. ``solution_str`` is
    the decoded full response (including observations) and is not parsed.
    """
    del data_source, kwargs
    extra = _copy_mapping(extra_info)
    for key in GOLD_EXTRA_KEYS:
        extra.pop(key, None)

    instance_id = str(ground_truth or "").strip()
    extra_id = str(extra.get("instance_id") or "").strip()
    if not instance_id:
        raise LocalizationScoreError("reward_model.ground_truth (instance_id) is empty")
    if extra_id and extra_id != instance_id:
        raise LocalizationScoreError(
            f"instance_id mismatch: ground_truth={instance_id!r} extra_info={extra_id!r}"
        )

    if "termination" not in extra:
        raise LocalizationScoreError(
            f"{instance_id}: extra_info missing termination "
            "(RewardLoop extra_fields merge failed; do not parse solution_str)"
        )
    if "final_submission" not in extra:
        raise LocalizationScoreError(
            f"{instance_id}: extra_info missing final_submission "
            "(RewardLoop extra_fields merge failed; do not parse solution_str)"
        )

    oracle = _oracle_index(str(resolve_oracle_parquet(oracle_parquet))).get(instance_id)
    metrics = evaluate_episode(
        termination=extra.get("termination"),
        submission=_as_optional_mapping(extra.get("final_submission")),
        oracle=oracle,
    )
    payload = metrics.as_dict()
    payload["score"] = float(metrics.localization_score)
    payload["instance_id"] = instance_id
    payload["solution_str_chars"] = len(str(solution_str or ""))
    return payload


@lru_cache(maxsize=4)
def _oracle_index(path: str) -> EvaluatorOracleIndex:
    return load_evaluator_oracle(path)


def _copy_mapping(value: Any) -> dict[str, Any]:
    value = _maybe_item(value)
    if value is None:
        return {}
    if isinstance(value, MappingABC):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise LocalizationScoreError(f"extra_info is not a mapping: {type(value)!r}")


def _as_optional_mapping(value: Any) -> Mapping[str, Any] | None:
    value = _maybe_item(value)
    if value is None:
        return None
    if isinstance(value, MappingABC):
        return value
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    return None


def _maybe_item(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, dict)):
        return value
    if hasattr(value, "item") and not isinstance(value, MappingABC):
        try:
            return value.item()
        except Exception:
            return value
    return value
