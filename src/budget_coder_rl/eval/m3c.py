"""M3C measurement: budget quantiles, grouped-rollout stats, offline behavior.

Does not run GRPO, RewardLoop, or LoRA. Diagnostic manifest construction
reads only policy-visible identities (instance_id / repo / split). Oracle
sidecar counts are accepted as explicit arguments for the train-candidate
rule and must not enter policy parquet or AgentLoop prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_materialize import (
    EXPECTED_TRAIN_ROWS,
    train_parquet_path,
)
from budget_coder_rl.eval.m3b import (
    extra_mapping,
    repo_round_robin_ids,
    sha256_ids,
)

DIAGNOSTIC_SCHEMA = "bcrl-m3c-diagnostic-v1"
CANDIDATE_SCHEMA = "bcrl-m3c-train-candidates-v1"
DIAGNOSTIC_RELPATH = "data/manifests/m3c_diagnostic_tasks.json"
CANDIDATE_RELPATH = "data/manifests/m3c_train_candidates.json"
FREEZE_RELPATH = "configs/experiments/stage1_m3c_freeze.json"
SELECTION_ALGORITHM = "repo_name_sort + instance_id_lexicographic + repo_round_robin"
PRIMARY_N = 80
GROUP_N = 4
GROUP_SEED_STRIDE = 8
GROUP_SEED_BASE = 20260826
HIGH_SCORE = 0.66
CANDIDATE_BUDGETS = (2048, 4096, 8192)
CALIBRATION_GPU_BUDGETS = (2048, 4096)
LOOSE_REFERENCE_BUDGET = 8192
OVERLONG_INSTANCE_IDS = frozenset({"Project-MONAI__MONAI-6344"})
TRAIN_CANDIDATE_TARGET_N = 256
MIXED_FRACTION_N8_THRESHOLD = 0.05
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def load_split_identities(
    parquet_path: Path,
    *,
    expected_split: str,
) -> list[dict[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path, columns=["extra_info"])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in frame.to_dict(orient="records"):
        extra = extra_mapping(record.get("extra_info"))
        instance_id = str(extra.get("instance_id") or "").strip()
        repo = str(extra.get("repo") or "").strip()
        split = str(extra.get("split") or "").strip()
        if not instance_id or not repo:
            raise ValueError(f"{expected_split} parquet row missing instance_id or repo")
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id in {expected_split} parquet: {instance_id}")
        if split and split != expected_split:
            raise ValueError(
                f"{instance_id}: extra_info.split={split!r} expected {expected_split}"
            )
        seen.add(instance_id)
        rows.append(
            {"instance_id": instance_id, "repo": repo, "split": expected_split}
        )
    return rows


def group_seed(
    task_index: int,
    group_index: int,
    *,
    base: int = GROUP_SEED_BASE,
    stride: int = GROUP_SEED_STRIDE,
) -> int:
    if group_index < 0 or group_index >= int(stride):
        raise ValueError(f"group_index {group_index} out of range for stride={stride}")
    return int(base) + int(task_index) * int(stride) + int(group_index)


def build_diagnostic_manifest(
    rows: Sequence[Mapping[str, str]],
    *,
    primary_n: int = PRIMARY_N,
    expected_n: int = EXPECTED_TRAIN_ROWS,
    group_n: int = GROUP_N,
    seed_stride: int = GROUP_SEED_STRIDE,
    seed_base: int = GROUP_SEED_BASE,
    overlong_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(rows) != expected_n:
        raise ValueError(f"train identities {len(rows)} != expected {expected_n}")
    ordered = repo_round_robin_ids(rows)
    if len(ordered) != expected_n:
        raise ValueError(f"ordered ids {len(ordered)} != expected {expected_n}")
    if primary_n > len(ordered):
        raise ValueError(f"primary_n {primary_n} > universe {len(ordered)}")
    blocked = frozenset(overlong_ids or OVERLONG_INSTANCE_IDS)
    by_id = {row["instance_id"]: row for row in rows}
    repo_counts: dict[str, int] = defaultdict(int)
    primary_repo_counts: dict[str, int] = defaultdict(int)
    skipped_overlong: list[str] = []
    for instance_id in ordered:
        repo_counts[by_id[instance_id]["repo"]] += 1
    for instance_id in ordered[:primary_n]:
        primary_repo_counts[by_id[instance_id]["repo"]] += 1
        if instance_id in blocked:
            skipped_overlong.append(instance_id)
    tasks = []
    for index, instance_id in enumerate(ordered):
        skipped = instance_id in blocked
        item: dict[str, Any] = {
            "task_index": index,
            "instance_id": instance_id,
            "repo": by_id[instance_id]["repo"],
            "set": "primary" if index < primary_n else "remainder",
            "skipped_overlong": skipped,
            "group_seeds": [
                group_seed(index, group_index, base=seed_base, stride=seed_stride)
                for group_index in range(seed_stride)
            ],
        }
        tasks.append(item)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "split": "train",
        "selection_algorithm": SELECTION_ALGORITHM,
        "oracle_used": False,
        "gold_used": False,
        "group_n": group_n,
        "group_seed_stride": seed_stride,
        "group_seed_base": seed_base,
        "group_seed_formula": "GROUP_SEED_BASE + task_index * GROUP_SEED_STRIDE + group_index",
        "obs_tokens_limit_note": (
            "diagnostic rollouts use the E006-assigned primary B_obs; "
            "this manifest does not freeze a budget"
        ),
        "n_universe": len(ordered),
        "n_primary": primary_n,
        "n_remainder": len(ordered) - primary_n,
        "n_primary_runnable": primary_n - len(skipped_overlong),
        "skipped_overlong": skipped_overlong,
        "ordered_ids": ordered,
        "primary_ids": ordered[:primary_n],
        "remainder_ids": ordered[primary_n:],
        "ordered_ids_sha256": sha256_ids(ordered),
        "primary_ids_sha256": sha256_ids(ordered[:primary_n]),
        "repo_counts_universe": dict(sorted(repo_counts.items())),
        "repo_counts_primary": dict(sorted(primary_repo_counts.items())),
        "n_repos_universe": len(repo_counts),
        "n_repos_primary": len(primary_repo_counts),
        "tasks": tasks,
    }


def load_diagnostic_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"unexpected diagnostic schema {payload.get('schema_version')!r}"
        )
    if payload.get("oracle_used") or payload.get("gold_used"):
        raise ValueError("M3C diagnostic manifest must not be built from oracle/gold")
    return payload


def default_diagnostic_path(repo_root: Path) -> Path:
    return Path(repo_root) / DIAGNOSTIC_RELPATH


def build_diagnostic_manifest_from_train_parquet(
    repo_root: Path,
    *,
    parquet_path: Path | None = None,
    primary_n: int = PRIMARY_N,
    group_n: int = GROUP_N,
    seed_stride: int = GROUP_SEED_STRIDE,
) -> dict[str, Any]:
    path = parquet_path or train_parquet_path(repo_root)
    rows = load_split_identities(path, expected_split="train")
    return build_diagnostic_manifest(
        rows, primary_n=primary_n, group_n=group_n, seed_stride=seed_stride
    )


def quantile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    if p < 0 or p > 1:
        raise ValueError(f"quantile p must be in [0, 1], got {p}")
    ordered = sorted(float(item) for item in values)
    index = (len(ordered) - 1) * float(p)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def repo_obs_tokens(row: Mapping[str, Any]) -> float | None:
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    value = budget.get("repo_observation_tokens")
    if value is None:
        value = budget.get("obs_tokens_used")
    if value is None:
        return None
    return float(value)


def localization_score(row: Mapping[str, Any]) -> float | None:
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    value = loc.get("localization_score")
    if value is None:
        return None
    return float(value)


def budget_visible_flag(row: Mapping[str, Any]) -> bool | None:
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    if condition.get("budget_visible") is not None:
        return bool(condition.get("budget_visible"))
    if budget.get("budget_visible") is not None:
        return bool(budget.get("budget_visible"))
    return None


def _token_stats(values: Sequence[float], *, candidates: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    n = len(values)
    return {
        "n": n,
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / n),
        "quantiles": {f"p{int(level * 100)}": quantile(values, level) for level in QUANTILE_LEVELS},
        "n_zero": sum(1 for item in values if item == 0),
        "frac_ge": {
            str(limit): sum(1 for item in values if item >= float(limit)) / n
            for limit in candidates
        },
        "frac_u_ge_0.70_at_8192": sum(1 for item in values if item / 8192.0 >= 0.70) / n,
        "frac_u_ge_0.85_at_8192": sum(1 for item in values if item / 8192.0 >= 0.85) / n,
    }


def _exhausted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
        if row.get("termination") != "budget_exhausted" and not budget.get("budget_exhausted"):
            continue
        identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
        out.append(
            {
                "instance_id": identity.get("instance_id"),
                "budget_visible": budget_visible_flag(row),
                "repo_observation_tokens": repo_obs_tokens(row),
                "termination": row.get("termination"),
                "obs_tokens_limit": budget.get("obs_tokens_limit"),
            }
        )
    return out


def e001_budget_quantile_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[int] = CANDIDATE_BUDGETS,
) -> dict[str, Any]:
    live = [row for row in rows if row.get("termination") != "operational_error"]
    visible = [row for row in live if budget_visible_flag(row) is True]
    hidden = [row for row in live if budget_visible_flag(row) is False]
    visible_tokens = [item for item in (repo_obs_tokens(row) for row in visible) if item is not None]
    hidden_tokens = [item for item in (repo_obs_tokens(row) for row in hidden) if item is not None]
    all_tokens = [item for item in (repo_obs_tokens(row) for row in live) if item is not None]
    return {
        "schema_version": "bcrl-m3c-e001-quantiles-v1",
        "n_rows": len(rows),
        "n_live": len(live),
        "budget_accounting_version": "bcrl-bobs-v2",
        "primary_metric": "repo_observation_tokens",
        "candidates": list(candidates),
        "candidate_rationale": {
            "2048": "visible p25-p40 band; tests binding vs first-obs starvation",
            "4096": "slightly above visible p75; typical median episode still has headroom",
            "8192": "E001 loose reference; do not re-run; 16K is not a candidate",
        },
        "note": (
            "Unconstrained C_obs at 8192 is not the same as behavior under a "
            "hard cap. Exhaustion can occur with C_obs << B_obs when the next "
            "observation cannot be inserted (no silent truncate)."
        ),
        "all": _token_stats(all_tokens, candidates=candidates),
        "visible": _token_stats(visible_tokens, candidates=candidates),
        "hidden": _token_stats(hidden_tokens, candidates=candidates),
        "exhausted": _exhausted_rows(live),
        "gpu_calibration_budgets": list(CALIBRATION_GPU_BUDGETS),
        "loose_reference_budget": LOOSE_REFERENCE_BUDGET,
    }


def is_starvation(summary: Mapping[str, Any]) -> bool:
    exhaustion = float(summary.get("budget_exhaustion_rate") or 0.0)
    mean_obs = summary.get("mean_repo_observation_tokens")
    n = int(summary.get("n_episodes") or 0)
    n_zero = int(summary.get("n_zero_c_obs") or 0)
    frac_zero = (n_zero / n) if n else 0.0
    mean_value = float(mean_obs) if mean_obs is not None else 0.0
    return exhaustion >= 0.40 and (mean_value < 400.0 or frac_zero >= 0.25)


def assign_budget_regimes(
    metrics_by_limit: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assign tight/medium/loose after measured calibration metrics exist."""
    required = set(CANDIDATE_BUDGETS)
    present = set(int(key) for key in metrics_by_limit)
    if not required.issubset(present):
        raise ValueError(f"regime assignment needs metrics for {sorted(required)}, got {sorted(present)}")
    tight_metrics = metrics_by_limit[2048]
    medium_metrics = metrics_by_limit[4096]
    loose_metrics = metrics_by_limit[8192]
    tight_starved = is_starvation(tight_metrics)
    tight_limit = None if tight_starved else 2048
    medium_limit = 4096
    loose_limit = 8192
    primary = medium_limit
    return {
        "tight": tight_limit,
        "medium": medium_limit,
        "loose": loose_limit,
        "primary_training_B_obs": primary,
        "eval_budget_set": [item for item in (tight_limit, medium_limit, loose_limit) if item is not None],
        "tight_starvation": tight_starved,
        "tight_note": (
            "2048 looks like first-obs starvation; consider one supplemental "
            "GPU at 2560 or 3072. Do not change reward/tools/prompt."
            if tight_starved
            else "2048 is binding without universal first-obs starvation."
        ),
        "medium_note": (
            "4096 is the default primary training budget (budget-awareness axis)."
        ),
        "loose_note": "8192 remains the E001 loose reference.",
        "exhaustion": {
            str(limit): (metrics_by_limit[limit] or {}).get("budget_exhaustion_rate")
            for limit in CANDIDATE_BUDGETS
        },
        "mean_utilization": {
            str(limit): (metrics_by_limit[limit] or {}).get("mean_budget_utilization")
            for limit in CANDIDATE_BUDGETS
        },
        "mean_localization": {
            str(limit): (metrics_by_limit[limit] or {}).get("mean_localization_score")
            for limit in CANDIDATE_BUDGETS
        },
    }


def _distinct_rewards(rewards: Sequence[float]) -> list[float]:
    seen: list[float] = []
    for value in rewards:
        if not any(abs(value - existing) < 1e-12 for existing in seen):
            seen.append(float(value))
    return seen


def group_reward_stats(
    rewards: Sequence[float],
    *,
    high_score: float = HIGH_SCORE,
) -> dict[str, Any]:
    values = [float(item) for item in rewards]
    n = len(values)
    if n == 0:
        raise ValueError("group_reward_stats requires at least one reward")
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    else:
        variance = sum((item - mean) ** 2 for item in values) / (n - 1)
        std = math.sqrt(variance)
    lo = min(values)
    hi = max(values)
    distinct = _distinct_rewards(values)
    zero_variance = abs(hi - lo) < 1e-12
    all_zero = all(abs(item) < 1e-12 for item in values)
    all_equal_positive = zero_variance and mean > 0
    all_high = all(item >= high_score for item in values)
    mixed = len(distinct) >= 2
    return {
        "n": n,
        "rewards": values,
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "range": hi - lo,
        "distinct_count": len(distinct),
        "distinct_rewards": distinct,
        "zero_variance": zero_variance,
        "all_zero": all_zero,
        "all_equal_positive": all_equal_positive,
        "all_high": all_high,
        "mixed": mixed,
    }


def group_index_of_row(row: Mapping[str, Any]) -> int | None:
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    extra = row.get("group") if isinstance(row.get("group"), MappingABC) else {}
    value = extra.get("group_index")
    if value is None:
        value = condition.get("group_index")
    if value is None:
        return None
    return int(value)


def grouped_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_n: int = GROUP_N,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[int, Mapping[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        if row.get("termination") == "operational_error":
            continue
        identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
        instance_id = str(identity.get("instance_id") or "")
        if not instance_id:
            continue
        index = group_index_of_row(row)
        if index is None:
            continue
        if instance_id not in buckets:
            buckets[instance_id] = {}
            order.append(instance_id)
        buckets[instance_id][int(index)] = row
    out: list[dict[str, Any]] = []
    for instance_id in order:
        members = buckets[instance_id]
        complete = all(index in members for index in range(group_n))
        rewards: list[float | None] = []
        member_rows: list[Mapping[str, Any] | None] = []
        for index in range(group_n):
            member = members.get(index)
            member_rows.append(member)
            rewards.append(None if member is None else localization_score(member))
        scored = [float(item) for item in rewards if item is not None]
        stats = group_reward_stats(scored) if len(scored) == group_n else None
        repo = None
        first = next((item for item in member_rows if item is not None), None)
        if first is not None:
            identity = first.get("identity") if isinstance(first.get("identity"), MappingABC) else {}
            repo = identity.get("repo")
        out.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "complete": complete,
                "n_members": len(scored),
                "group_n": group_n,
                "rewards": rewards,
                "stats": stats,
            }
        )
    return out


def aggregate_group_stats(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [item for item in groups if item.get("complete") and item.get("stats")]
    n = len(complete)
    if n == 0:
        return {"n_groups": 0, "n_complete": 0}
    stds = [float(item["stats"]["std"]) for item in complete]
    ranges = [float(item["stats"]["range"]) for item in complete]
    n_zero_var = sum(1 for item in complete if item["stats"]["zero_variance"])
    n_all_zero = sum(1 for item in complete if item["stats"]["all_zero"])
    n_mixed = sum(1 for item in complete if item["stats"]["mixed"])
    n_all_pos = sum(1 for item in complete if item["stats"]["all_equal_positive"])
    n_all_high = sum(1 for item in complete if item["stats"]["all_high"])
    all_rewards = [float(value) for item in complete for value in item["stats"]["rewards"]]
    return {
        "n_groups": len(groups),
        "n_complete": n,
        "zero_variance_fraction": n_zero_var / n,
        "all_zero_fraction": n_all_zero / n,
        "mixed_fraction": n_mixed / n,
        "all_equal_positive_fraction": n_all_pos / n,
        "all_high_fraction": n_all_high / n,
        "mean_group_std": sum(stds) / n,
        "median_group_std": quantile(stds, 0.5),
        "mean_group_range": sum(ranges) / n,
        "median_group_range": quantile(ranges, 0.5),
        "reward_range_quantiles": {
            f"p{int(level * 100)}": quantile(ranges, level) for level in (0.50, 0.75, 0.90)
        },
        "task_reward_quantiles": {
            f"p{int(level * 100)}": quantile(all_rewards, level)
            for level in QUANTILE_LEVELS
        },
        "n_zero_variance": n_zero_var,
        "n_all_zero": n_all_zero,
        "n_mixed": n_mixed,
        "needs_n8_probe": (
            (n_mixed / n) < MIXED_FRACTION_N8_THRESHOLD and (quantile(stds, 0.5) or 0.0) == 0.0
        ),
    }


def _headers_from_event(event: Mapping[str, Any]) -> dict[str, str]:
    text = event.get("observation")
    if not isinstance(text, str):
        text = event.get("observation_preview")
    if not isinstance(text, str):
        return {}
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() == "---":
            break
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(": ")
        if not sep or key in headers:
            continue
        headers[key] = value
    return headers


def _search_hit_paths(event: Mapping[str, Any]) -> list[str]:
    text = event.get("observation")
    if not isinstance(text, str):
        text = event.get("observation_preview")
    if not isinstance(text, str) or "---" not in text:
        return []
    body = text.split("---", 1)[1]
    paths: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit() and parts[0]:
            paths.append(parts[0])
    return paths


def episode_behavior_flags(row: Mapping[str, Any]) -> dict[str, Any]:
    events = list(row.get("events") or [])
    behavior = row.get("behavior") if isinstance(row.get("behavior"), MappingABC) else {}
    n_read = int(behavior.get("n_read") or 0)
    n_search = int(behavior.get("n_search") or 0)
    n_empty = int(behavior.get("n_empty_search_hits") or 0)
    n_repeat = int(behavior.get("n_repeated_search_queries") or 0)
    termination = row.get("termination")
    n_nonempty_search = 0
    n_conversions = 0
    for index, event in enumerate(events):
        if not isinstance(event, MappingABC) or event.get("action_name") != "search":
            continue
        headers = _headers_from_event(event)
        raw_count = headers.get("match_count")
        try:
            match_count = int(raw_count) if raw_count is not None else 0
        except (TypeError, ValueError):
            match_count = 0
        if match_count <= 0:
            continue
        n_nonempty_search += 1
        hits = set(_search_hit_paths(event))
        for later in events[index + 1 :]:
            if not isinstance(later, MappingABC) or later.get("action_name") != "read":
                continue
            args = later.get("action_arguments") if isinstance(later.get("action_arguments"), MappingABC) else {}
            path = str(args.get("path") or "")
            if path and path in hits:
                n_conversions += 1
                break
    return {
        "read_count": n_read,
        "n_search": n_search,
        "finish_with_zero_read": termination == "finish" and n_read == 0,
        "repeated_search": n_repeat > 0,
        "zero_hit_search": n_empty > 0,
        "n_empty_search_hits": n_empty,
        "n_repeated_search_queries": n_repeat,
        "termination_finish": termination == "finish",
        "termination_max_turns": termination == "max_turns",
        "n_nonempty_search": n_nonempty_search,
        "n_search_to_read_conversions": n_conversions,
        "search_to_read_after_nonempty": n_nonempty_search > 0 and n_conversions > 0,
    }


def attach_behavior_flags(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["m3c_behavior"] = episode_behavior_flags(row)
    return payload


def behavior_reward_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    live = [row for row in rows if row.get("termination") != "operational_error"]

    def mean_score(pred) -> dict[str, Any]:
        scores = [
            localization_score(row)
            for row in live
            if pred(row) and localization_score(row) is not None
        ]
        values = [float(item) for item in scores if item is not None]
        return {
            "n": len(values),
            "mean_localization": (sum(values) / len(values)) if values else None,
        }

    def flags(row: Mapping[str, Any]) -> dict[str, Any]:
        extra = row.get("m3c_behavior")
        if isinstance(extra, MappingABC):
            return dict(extra)
        return episode_behavior_flags(row)

    return {
        "n": len(live),
        "read_gt_0": mean_score(lambda row: flags(row)["read_count"] > 0),
        "read_eq_0": mean_score(lambda row: flags(row)["read_count"] == 0),
        "finish_with_zero_read": mean_score(lambda row: flags(row)["finish_with_zero_read"]),
        "repeated_search": mean_score(lambda row: flags(row)["repeated_search"]),
        "no_repeated_search": mean_score(lambda row: not flags(row)["repeated_search"]),
        "zero_hit_search": mean_score(lambda row: flags(row)["zero_hit_search"]),
        "no_zero_hit_search": mean_score(lambda row: not flags(row)["zero_hit_search"]),
        "search_to_read": mean_score(lambda row: flags(row)["search_to_read_after_nonempty"]),
        "no_search_to_read": mean_score(
            lambda row: flags(row)["n_nonempty_search"] > 0
            and not flags(row)["search_to_read_after_nonempty"]
        ),
        "finish": mean_score(lambda row: flags(row)["termination_finish"]),
        "max_turns": mean_score(lambda row: flags(row)["termination_max_turns"]),
    }


def within_group_behavior_contrast(
    groups: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
        instance_id = str(identity.get("instance_id") or "")
        index = group_index_of_row(row)
        if instance_id and index is not None:
            by_id[instance_id][int(index)] = row
    n_mixed_with_read_gap = 0
    n_mixed_with_conversion_gap = 0
    n_mixed_with_repeat_gap = 0
    n_mixed = 0
    for group in groups:
        if not group.get("complete") or not (group.get("stats") or {}).get("mixed"):
            continue
        n_mixed += 1
        instance_id = str(group["instance_id"])
        members = [
            by_id[instance_id][index]
            for index in range(int(group["group_n"]))
            if index in by_id[instance_id]
        ]
        scored = []
        for member in members:
            flags = episode_behavior_flags(member)
            scored.append((float(localization_score(member) or 0.0), flags, member))
        scored.sort(key=lambda item: item[0])
        low_flags = scored[0][1]
        high_flags = scored[-1][1]
        if high_flags["read_count"] > low_flags["read_count"]:
            n_mixed_with_read_gap += 1
        if high_flags["search_to_read_after_nonempty"] and not low_flags["search_to_read_after_nonempty"]:
            n_mixed_with_conversion_gap += 1
        if low_flags["repeated_search"] and not high_flags["repeated_search"]:
            n_mixed_with_repeat_gap += 1
    return {
        "n_mixed_complete": n_mixed,
        "n_mixed_higher_read_on_better_member": n_mixed_with_read_gap,
        "n_mixed_conversion_on_better_member": n_mixed_with_conversion_gap,
        "n_mixed_repeated_search_on_worse_member": n_mixed_with_repeat_gap,
    }


def compact_group_member(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    flags = episode_behavior_flags(row)
    events = row.get("events") or []
    actions = [
        str(event.get("action_name") or event.get("action_type"))
        for event in events
        if isinstance(event, MappingABC) and (event.get("action_name") or event.get("action_type"))
    ]
    queries = [
        (event.get("action_arguments") or {}).get("query")
        for event in events
        if isinstance(event, MappingABC) and event.get("action_name") == "search"
    ]
    return {
        "group_index": group_index_of_row(row),
        "sampling_seed": (row.get("condition") or {}).get("sampling_seed"),
        "termination": row.get("termination"),
        "localization_score": loc.get("localization_score"),
        "file_f1": loc.get("file_f1"),
        "symbol_f1": loc.get("symbol_f1"),
        "parse_ok": loc.get("parse_ok"),
        "actions": actions,
        "search_queries": queries,
        "read_paths": (row.get("behavior") or {}).get("read_paths"),
        "behavior": flags,
        "budget": {
            "repo_observation_tokens": (row.get("budget") or {}).get("repo_observation_tokens"),
            "obs_tokens_limit": (row.get("budget") or {}).get("obs_tokens_limit"),
            "budget_exhausted": (row.get("budget") or {}).get("budget_exhausted"),
        },
        "instance_id": identity.get("instance_id"),
        "repo": identity.get("repo"),
    }


def select_representative_groups(
    groups: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    n_target: int = 8,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
        instance_id = str(identity.get("instance_id") or "")
        index = group_index_of_row(row)
        if instance_id and index is not None:
            by_id[instance_id][int(index)] = row
    mixed = [
        item
        for item in groups
        if item.get("complete") and (item.get("stats") or {}).get("mixed")
    ]
    mixed.sort(
        key=lambda item: (
            -float(item["stats"]["range"]),
            -float(item["stats"]["std"]),
            str(item["instance_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    for group in mixed:
        instance_id = str(group["instance_id"])
        members = by_id.get(instance_id) or {}
        if len(members) < int(group["group_n"]):
            continue
        compact_members = [
            compact_group_member(members[index])
            for index in range(int(group["group_n"]))
            if index in members
        ]
        flags = [item["behavior"] for item in compact_members]
        scores = [float(item["localization_score"] or 0.0) for item in compact_members]
        low = compact_members[scores.index(min(scores))]
        high = compact_members[scores.index(max(scores))]
        selected.append(
            {
                "instance_id": instance_id,
                "repo": group.get("repo"),
                "stats": group["stats"],
                "contrast": {
                    "low_read_count": low["behavior"]["read_count"],
                    "high_read_count": high["behavior"]["read_count"],
                    "low_repeated_search": low["behavior"]["repeated_search"],
                    "high_search_to_read": high["behavior"]["search_to_read_after_nonempty"],
                    "hypothesis_aligned": (
                        high["behavior"]["read_count"] > low["behavior"]["read_count"]
                        or (
                            high["behavior"]["search_to_read_after_nonempty"]
                            and not low["behavior"]["search_to_read_after_nonempty"]
                        )
                    ),
                },
                "members": compact_members,
                "n_finish_zero_read": sum(1 for item in flags if item["finish_with_zero_read"]),
            }
        )
        if len(selected) >= n_target:
            break
    return selected


def build_train_candidate_manifest(
    rows: Sequence[Mapping[str, str]],
    *,
    eligible_ids: Sequence[str],
    skipped: Mapping[str, Sequence[str]],
    rule_text: str,
    target_n: int = TRAIN_CANDIDATE_TARGET_N,
    expected_n: int = EXPECTED_TRAIN_ROWS,
) -> dict[str, Any]:
    if len(rows) != expected_n:
        raise ValueError(f"train identities {len(rows)} != expected {expected_n}")
    ordered_all = repo_round_robin_ids(rows)
    eligible = set(eligible_ids)
    ordered = [instance_id for instance_id in ordered_all if instance_id in eligible]
    if target_n > len(ordered):
        raise ValueError(f"target_n {target_n} > eligible {len(ordered)}")
    selected = ordered[:target_n]
    by_id = {row["instance_id"]: row for row in rows}
    repo_counts: dict[str, int] = defaultdict(int)
    for instance_id in selected:
        repo_counts[by_id[instance_id]["repo"]] += 1
    rule_hash = hashlib.sha256(rule_text.encode("utf-8")).hexdigest()
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "split": "train",
        "selection_algorithm": SELECTION_ALGORITHM,
        "oracle_used_for_class_filter": True,
        "gold_used_for_cherry_pick": False,
        "zero_variance_used_as_drop": False,
        "rule_text": rule_text,
        "rule_text_sha256": rule_hash,
        "n_universe": len(ordered_all),
        "n_eligible": len(ordered),
        "n_selected": len(selected),
        "target_n": target_n,
        "skipped": {key: list(value) for key, value in skipped.items()},
        "ordered_ids": selected,
        "ordered_ids_sha256": sha256_ids(selected),
        "eligible_ids_sha256": sha256_ids(ordered),
        "repo_counts": dict(sorted(repo_counts.items())),
        "n_repos": len(repo_counts),
        "tasks": [
            {
                "task_index": index,
                "instance_id": instance_id,
                "repo": by_id[instance_id]["repo"],
            }
            for index, instance_id in enumerate(selected)
        ],
    }


def default_candidate_path(repo_root: Path) -> Path:
    return Path(repo_root) / CANDIDATE_RELPATH


def default_freeze_path(repo_root: Path) -> Path:
    return Path(repo_root) / FREEZE_RELPATH


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def episode_key(row: Mapping[str, Any]) -> tuple[str, int, bool, int] | None:
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    instance_id = str(identity.get("instance_id") or "")
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    limit = condition.get("obs_tokens_limit")
    if limit is None:
        limit = budget.get("obs_tokens_limit")
    visible = budget_visible_flag(row)
    group_index = group_index_of_row(row)
    if not instance_id or limit is None or visible is None or group_index is None:
        return None
    return (instance_id, int(limit), bool(visible), int(group_index))


def with_group_fields(
    extra: Mapping[str, Any],
    *,
    visible: bool,
    limit: int,
    seed: int,
    group_index: int,
    group_n: int,
) -> dict[str, Any]:
    out = dict(extra)
    out["budget_visible"] = bool(visible)
    out["obs_tokens_limit"] = int(limit)
    out["sampling_seed"] = int(seed)
    out["group_index"] = int(group_index)
    out["group_n"] = int(group_n)
    return out


def n8_probe_ids(
    groups: Sequence[Mapping[str, Any]],
    *,
    n_target: int = 16,
) -> list[str]:
    ranked = [
        item
        for item in groups
        if item.get("complete") and item.get("stats")
    ]
    ranked.sort(
        key=lambda item: (
            0 if item["stats"]["mixed"] else 1,
            0 if item["stats"]["zero_variance"] else 1,
            str(item["instance_id"]),
        )
    )
    return [str(item["instance_id"]) for item in ranked[:n_target]]
