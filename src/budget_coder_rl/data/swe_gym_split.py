"""SWE-Gym M1D-B eligibility policy and grouped/repo-stratified split.

Stage-1 keeps all 2438 instances. SWE-Gym is split into train/dev only.
Correlation groups are atomic. Secondary features are audited, not optimized.
This module does not materialize veRL parquet, implement reward, or start M1E.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_N_REPOS,
    EXPECTED_N_ROWS,
    EXPECTED_SHA256,
    HF_REPO_ID,
    HF_REVISION,
    length_stats,
    sha256_file,
)
from budget_coder_rl.data.swe_gym_features import (
    FEATURE_JSONL_RELPATH,
    FEATURE_SUMMARY_RELPATH,
    float_stats,
)

POLICY_VERSION = "swe-gym-stage1-eligible-all-v1"
SPLIT_VERSION = "swe-gym-group-repo-v1"
SPLIT_SEED = 42
DEV_FRACTION_NUM = 1
DEV_FRACTION_DEN = 10
GROUPING_KEY = "correlation_group_id"
STRATIFICATION_KEY = "repo"
BINARY_RATE_SHIFT_THRESHOLD = 0.10

POLICY_RELPATH = "data/manifests/swe_gym_m1d_policy.json"
SPLIT_RELPATH = "data/manifests/swe_gym_m1d_split.json"
SPLIT_SUMMARY_RELPATH = "data/manifests/swe_gym_m1d_split_summary.json"

SPLIT_SPOTLIGHT_INSTANCE_IDS: tuple[str, ...] = (
    "pandas-dev__pandas-53805",
    "pandas-dev__pandas-53809",
    "pandas-dev__pandas-53830",
)

ASSIGNMENT_FIELDS: tuple[str, ...] = (
    "instance_id",
    "repo",
    "correlation_group_id",
    "split",
)
ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "dev"})

FORBIDDEN_SPLIT_KEYS: frozenset[str] = frozenset(
    {
        "patch",
        "test_patch",
        "hints_text",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "problem_statement",
        "gold_edit_files",
        "oracle_symbols",
        "unmapped_sites",
        "file_results",
        "keep",
        "drop",
    }
)

NOT_DROP_CRITERIA: tuple[str, ...] = (
    "zero_symbol_oracle",
    "non_python_target",
    "non_code_only_taxonomy",
    "large_patch",
    "many_gold_or_base_changed_files",
    "added_heavy_patch",
    "issue_filename_or_symbol_hints",
    "m1b_test_metadata_audit_flags",
    "structural_difficulty",
)


class SplitInputError(ValueError):
    """Invalid split input (size mismatch, missing ids, etc.)."""


class CrossRepoCorrelationError(SplitInputError):
    """A frozen correlation group spans more than one repo."""

    def __init__(self, groups: Sequence[Mapping[str, Any]]):
        self.groups = [dict(item) for item in groups]
        details = []
        for item in self.groups:
            details.append(
                f"{item['correlation_group_id']} repos={item['repos']} "
                f"instances={item['instance_ids']}"
            )
        super().__init__(
            "cross-repo correlation group(s) are not allowed; "
            "do not split the group and do not ignore the relation. "
            f"count={len(self.groups)}: " + "; ".join(details)
        )


class GroupUnit(NamedTuple):
    correlation_group_id: str
    size: int
    instance_ids: tuple[str, ...]
    repo: str


class SubsetSumResult(NamedTuple):
    selected_ids: tuple[str, ...]
    target: int
    actual: int
    delta: int
    exact: bool


class SplitResult(NamedTuple):
    assignments: list[dict[str, str]]
    repo_allocations: list[dict[str, Any]]
    target_dev_rows: int
    target_train_rows: int
    actual_dev_rows: int
    actual_train_rows: int
    n_rows: int
    n_train_groups: int
    n_dev_groups: int
    cross_repo_correlation_groups: int


def policy_path(repo_root: Path) -> Path:
    return Path(repo_root) / POLICY_RELPATH


def split_path(repo_root: Path) -> Path:
    return Path(repo_root) / SPLIT_RELPATH


def split_summary_path(repo_root: Path) -> Path:
    return Path(repo_root) / SPLIT_SUMMARY_RELPATH


def target_dev_rows(n_rows: int, *, numerator: int = DEV_FRACTION_NUM, denominator: int = DEV_FRACTION_DEN) -> int:
    if n_rows < 0:
        raise SplitInputError(f"n_rows must be >= 0, got {n_rows}")
    if denominator <= 0:
        raise SplitInputError(f"denominator must be > 0, got {denominator}")
    return (int(n_rows) * int(numerator) + int(denominator) // 2) // int(denominator)


def group_priority(split_version: str, seed: int, correlation_group_id: str) -> str:
    payload = f"{split_version}|{seed}|{correlation_group_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def largest_remainder_quotas(
    repo_counts: Mapping[str, int],
    *,
    n_rows: int | None = None,
    numerator: int = DEV_FRACTION_NUM,
    denominator: int = DEV_FRACTION_DEN,
) -> dict[str, Any]:
    """Hamilton / largest-remainder allocation. Integer remainder ranking."""
    counts = {str(repo): int(count) for repo, count in repo_counts.items()}
    if any(count < 0 for count in counts.values()):
        raise SplitInputError("repo counts must be non-negative")
    observed = sum(counts.values())
    if n_rows is None:
        n_rows = observed
    elif int(n_rows) != observed:
        raise SplitInputError(
            f"n_rows={n_rows} != sum(repo_counts)={observed}"
        )
    n_rows = int(n_rows)
    target_dev = target_dev_rows(n_rows, numerator=numerator, denominator=denominator)
    bases: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for repo, count in counts.items():
        bases[repo] = (count * numerator) // denominator
        remainders[repo] = (count * numerator) % denominator
    leftover = target_dev - sum(bases.values())
    if leftover < 0:
        raise SplitInputError(f"Hamilton leftover is negative: {leftover}")
    ranked = sorted(counts, key=lambda repo: (-remainders[repo], repo))
    if leftover > len(ranked):
        raise SplitInputError(
            f"Hamilton leftover {leftover} exceeds n_repos {len(ranked)}"
        )
    extras = {repo: 0 for repo in counts}
    for repo in ranked[:leftover]:
        extras[repo] = 1
    quotas: dict[str, dict[str, int]] = {}
    for repo in sorted(counts):
        quotas[repo] = {
            "n_rows": counts[repo],
            "base_quota": bases[repo],
            "remainder": remainders[repo],
            "extra": extras[repo],
            "target_dev_rows": bases[repo] + extras[repo],
        }
    if sum(item["target_dev_rows"] for item in quotas.values()) != target_dev:
        raise SplitInputError("Hamilton quotas do not sum to target_dev_rows")
    return {
        "n_rows": n_rows,
        "target_dev_rows": target_dev,
        "target_train_rows": n_rows - target_dev,
        "quotas": quotas,
    }


def collect_correlation_groups(
    records: Sequence[Mapping[str, Any]],
) -> list[GroupUnit]:
    seen_instances: dict[str, str] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        instance_id = str(item.get("instance_id") or "").strip()
        repo = str(item.get("repo") or "").strip()
        group_id = str(item.get("correlation_group_id") or "").strip()
        if not instance_id:
            raise SplitInputError("empty instance_id")
        if not repo:
            raise SplitInputError(f"empty repo for {instance_id}")
        if not group_id:
            raise SplitInputError(f"empty correlation_group_id for {instance_id}")
        if instance_id in seen_instances:
            raise SplitInputError(f"duplicate instance_id {instance_id}")
        seen_instances[instance_id] = group_id
        bucket = by_id.get(group_id)
        if bucket is None:
            bucket = {"repos": set(), "instance_ids": [], "declared_sizes": set()}
            by_id[group_id] = bucket
        bucket["repos"].add(repo)
        bucket["instance_ids"].append(instance_id)
        if item.get("correlation_group_size") is not None:
            bucket["declared_sizes"].add(int(item["correlation_group_size"]))

    if len(seen_instances) != len(records):
        raise SplitInputError("instance_id coverage does not match n_rows")

    cross: list[dict[str, Any]] = []
    size_errors: list[str] = []
    groups: list[GroupUnit] = []
    for group_id in sorted(by_id):
        bucket = by_id[group_id]
        members = sorted(str(value) for value in bucket["instance_ids"])
        if len(members) != len(set(members)):
            raise SplitInputError(
                f"duplicate instance_id inside correlation group {group_id}"
            )
        repos = sorted(str(value) for value in bucket["repos"])
        size = len(members)
        payload = {
            "correlation_group_id": group_id,
            "repos": repos,
            "instance_ids": members,
            "size": size,
        }
        if len(repos) != 1:
            cross.append(payload)
            continue
        declared = bucket["declared_sizes"]
        if declared and declared != {size}:
            size_errors.append(
                f"{group_id}: declared {sorted(declared)} != member count {size}"
            )
        groups.append(
            GroupUnit(
                correlation_group_id=group_id,
                size=size,
                instance_ids=tuple(members),
                repo=repos[0],
            )
        )
    if cross:
        raise CrossRepoCorrelationError(cross)
    if size_errors:
        raise SplitInputError(
            "correlation_group_size disagrees with member count: "
            + "; ".join(size_errors)
        )
    return groups


def select_groups_subset_sum(
    groups: Sequence[GroupUnit],
    target: int,
    *,
    split_version: str = SPLIT_VERSION,
    seed: int = SPLIT_SEED,
) -> SubsetSumResult:
    """Deterministic 0/1 subset-sum. Exact target wins; else nearest, then smaller w."""
    if int(target) < 0:
        raise SplitInputError(f"subset-sum target must be >= 0, got {target}")
    ordered = sorted(
        groups,
        key=lambda group: (
            group_priority(split_version, seed, group.correlation_group_id),
            group.correlation_group_id,
        ),
    )
    total = sum(group.size for group in ordered)
    reachable = [False] * (total + 1)
    reachable[0] = True
    parent: list[tuple[int, int] | None] = [None] * (total + 1)
    for index, group in enumerate(ordered):
        size = int(group.size)
        if size < 1:
            raise SplitInputError(
                f"correlation group {group.correlation_group_id} has size {size}"
            )
        for weight in range(total, size - 1, -1):
            if reachable[weight - size] and not reachable[weight]:
                reachable[weight] = True
                parent[weight] = (weight - size, index)

    target = int(target)
    if target <= total and reachable[target]:
        chosen = target
        exact = True
    else:
        attainable = [weight for weight, ok in enumerate(reachable) if ok]
        chosen = min(attainable, key=lambda weight: (abs(weight - target), weight))
        exact = chosen == target

    selected_index: set[int] = set()
    weight = chosen
    while weight > 0:
        step = parent[weight]
        if step is None:
            raise RuntimeError(f"subset-sum parent missing at weight={weight}")
        prev_weight, index = step
        selected_index.add(index)
        weight = prev_weight

    selected_ids = tuple(
        sorted(ordered[index].correlation_group_id for index in selected_index)
    )
    return SubsetSumResult(
        selected_ids=selected_ids,
        target=target,
        actual=chosen,
        delta=chosen - target,
        exact=exact,
    )


def _assignment_record(
    instance_id: str, repo: str, correlation_group_id: str, split: str
) -> dict[str, str]:
    if split not in ALLOWED_SPLITS:
        raise SplitInputError(f"invalid split {split!r}")
    record = {
        "instance_id": instance_id,
        "repo": repo,
        "correlation_group_id": correlation_group_id,
        "split": split,
    }
    leaked = FORBIDDEN_SPLIT_KEYS.intersection(record)
    if leaked:
        raise RuntimeError(f"forbidden keys leaked into assignment: {sorted(leaked)}")
    return record


def assign_split(
    records: Sequence[Mapping[str, Any]],
    *,
    split_version: str = SPLIT_VERSION,
    seed: int = SPLIT_SEED,
    numerator: int = DEV_FRACTION_NUM,
    denominator: int = DEV_FRACTION_DEN,
) -> SplitResult:
    if not records:
        raise SplitInputError("no records to split")
    groups = collect_correlation_groups(records)
    groups_by_repo: dict[str, list[GroupUnit]] = defaultdict(list)
    repo_counts: dict[str, int] = defaultdict(int)
    for group in groups:
        groups_by_repo[group.repo].append(group)
        repo_counts[group.repo] += group.size
    n_rows = len(records)
    if sum(repo_counts.values()) != n_rows:
        raise SplitInputError("correlation groups do not cover every instance")

    hamilton = largest_remainder_quotas(
        repo_counts,
        n_rows=n_rows,
        numerator=numerator,
        denominator=denominator,
    )
    assignments_by_id: dict[str, dict[str, str]] = {}
    allocations: list[dict[str, Any]] = []
    for repo in sorted(repo_counts):
        quota = hamilton["quotas"][repo]
        selection = select_groups_subset_sum(
            groups_by_repo[repo],
            int(quota["target_dev_rows"]),
            split_version=split_version,
            seed=seed,
        )
        dev_group_ids = set(selection.selected_ids)
        train_group_ids: list[str] = []
        train_rows = 0
        for group in groups_by_repo[repo]:
            split_name = "dev" if group.correlation_group_id in dev_group_ids else "train"
            if split_name == "train":
                train_group_ids.append(group.correlation_group_id)
                train_rows += group.size
            for instance_id in group.instance_ids:
                assignments_by_id[instance_id] = _assignment_record(
                    instance_id,
                    repo,
                    group.correlation_group_id,
                    split_name,
                )
        allocations.append(
            {
                "repo": repo,
                "n_rows": int(quota["n_rows"]),
                "n_groups": len(groups_by_repo[repo]),
                "base_quota": int(quota["base_quota"]),
                "remainder": int(quota["remainder"]),
                "extra": int(quota["extra"]),
                "target_dev_rows": int(quota["target_dev_rows"]),
                "actual_dev_rows": int(selection.actual),
                "delta": int(selection.delta),
                "exact": bool(selection.exact),
                "train_rows": int(train_rows),
                "dev_rows": int(selection.actual),
                "train_groups": len(train_group_ids),
                "dev_groups": len(selection.selected_ids),
                "dev_group_ids": list(selection.selected_ids),
                "train_group_ids": sorted(train_group_ids),
            }
        )

    assignments = [
        assignments_by_id[instance_id] for instance_id in sorted(assignments_by_id)
    ]
    actual_dev = sum(1 for item in assignments if item["split"] == "dev")
    actual_train = len(assignments) - actual_dev
    n_dev_groups = sum(item["dev_groups"] for item in allocations)
    n_train_groups = sum(item["train_groups"] for item in allocations)
    return SplitResult(
        assignments=assignments,
        repo_allocations=allocations,
        target_dev_rows=int(hamilton["target_dev_rows"]),
        target_train_rows=int(hamilton["target_train_rows"]),
        actual_dev_rows=actual_dev,
        actual_train_rows=actual_train,
        n_rows=n_rows,
        n_train_groups=n_train_groups,
        n_dev_groups=n_dev_groups,
        cross_repo_correlation_groups=0,
    )


def validate_split_invariants(
    records: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    *,
    expected_n_rows: int | None = None,
    expected_n_repos: int | None = None,
    require_all_repos_in_dev: bool = False,
) -> list[str]:
    errors: list[str] = []
    record_ids = [str(item.get("instance_id") or "").strip() for item in records]
    assign_ids = [str(item.get("instance_id") or "").strip() for item in assignments]
    if expected_n_rows is not None and len(records) != int(expected_n_rows):
        errors.append(f"record count {len(records)} != expected {expected_n_rows}")
    if len(assignments) != len(records):
        errors.append(
            f"assignment count {len(assignments)} != record count {len(records)}"
        )
    if len(set(record_ids)) != len(record_ids):
        errors.append("input instance_id is not unique")
    if len(set(assign_ids)) != len(assign_ids):
        errors.append("assignment instance_id is not unique")
    if set(record_ids) != set(assign_ids):
        errors.append(
            "assignment instance_id set != input instance_id set: "
            f"only_input={len(set(record_ids) - set(assign_ids))} "
            f"only_assign={len(set(assign_ids) - set(record_ids))}"
        )

    train_ids = {item["instance_id"] for item in assignments if item.get("split") == "train"}
    dev_ids = {item["instance_id"] for item in assignments if item.get("split") == "dev"}
    other = [
        str(item.get("split"))
        for item in assignments
        if item.get("split") not in ALLOWED_SPLITS
    ]
    if other:
        errors.append(f"assignments contain non train/dev split values: {sorted(set(other))}")
    if train_ids.intersection(dev_ids):
        errors.append(
            f"train ∩ dev is not empty ({len(train_ids.intersection(dev_ids))} ids)"
        )
    union = train_ids.union(dev_ids)
    if union != set(record_ids):
        errors.append(
            f"train ∪ dev size {len(union)} != n_rows {len(record_ids)}"
        )

    for item in assignments:
        leaked = FORBIDDEN_SPLIT_KEYS.intersection(item)
        extra = set(item).difference(ASSIGNMENT_FIELDS)
        if leaked:
            errors.append(
                f"assignment {item.get('instance_id')} has forbidden keys {sorted(leaked)}"
            )
        if extra:
            errors.append(
                f"assignment {item.get('instance_id')} has extra keys {sorted(extra)}"
            )

    group_splits: dict[str, set[str]] = defaultdict(set)
    group_repos: dict[str, set[str]] = defaultdict(set)
    for item in assignments:
        group_id = str(item.get("correlation_group_id") or "")
        group_splits[group_id].add(str(item.get("split") or ""))
        group_repos[group_id].add(str(item.get("repo") or ""))
    leakage = sorted(
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    )
    if leakage:
        errors.append(
            f"correlation leakage: {len(leakage)} group(s) in both splits "
            f"(preview={leakage[:12]})"
        )
    cross = sorted(
        group_id for group_id, repos in group_repos.items() if len(repos) > 1
    )
    if cross:
        errors.append(
            f"cross-repo correlation group(s) in assignments: {cross[:12]}"
        )

    record_repos = {str(item.get("repo") or "").strip() for item in records}
    train_repos = {item["repo"] for item in assignments if item.get("split") == "train"}
    dev_repos = {item["repo"] for item in assignments if item.get("split") == "dev"}
    if expected_n_repos is not None and len(record_repos) != int(expected_n_repos):
        errors.append(
            f"unique repo count {len(record_repos)} != expected {expected_n_repos}"
        )
    missing_train = sorted(record_repos - train_repos)
    if missing_train:
        errors.append(f"repos missing from train: {missing_train}")
    if require_all_repos_in_dev:
        missing_dev = sorted(record_repos - dev_repos)
        if missing_dev:
            errors.append(f"repos missing from dev: {missing_dev}")
    return errors


def correlation_leakage_count(assignments: Sequence[Mapping[str, Any]]) -> int:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for item in assignments:
        group_splits[str(item.get("correlation_group_id") or "")].add(
            str(item.get("split") or "")
        )
    return sum(1 for splits in group_splits.values() if len(splits) > 1)


def build_eligibility_policy(
    *,
    n_rows: int,
    revision: str = HF_REVISION,
    sha256: str = EXPECTED_SHA256,
) -> dict[str, Any]:
    return {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": revision,
        "sha256": sha256,
        "policy_version": POLICY_VERSION,
        "master_pool_size": int(n_rows),
        "eligible_count": int(n_rows),
        "excluded_count": 0,
        "excluded_instance_ids": [],
        "primary_task": "budget-constrained repository localization",
        "primary_supervision": "file-level localization",
        "primary_oracle": "base_changed_files",
        "symbol_oracle": {
            "role": "auxiliary / where available",
            "not_an_eligibility_criterion": True,
        },
        "difficulty_features": (
            "analysis/sampling metadata, not eligibility criteria"
        ),
        "curriculum": "not enabled at dataset stage",
        "external_final_test": "not from SWE-Gym",
        "not_drop_criteria": list(NOT_DROP_CRITERIA),
        "reward": "not implemented in M1D-B",
        "notes": {
            "keep_all": True,
            "no_filter": True,
            "no_internal_test": True,
            "no_verl_parquet": True,
            "not_m1e": True,
            "no_curriculum": True,
            "no_weighted_sampler": True,
        },
    }


def _rate(count: int, denominator: int) -> dict[str, Any]:
    return {
        "n": int(count),
        "denominator": int(denominator),
        "rate": (round(count / denominator, 6) if denominator else None),
    }


def _int_stats(values: Iterable[int], keys: Sequence[str]) -> dict[str, Any]:
    stats = length_stats(values)
    payload = {"n": stats["n"]}
    for key in keys:
        payload[key] = stats[key]
    return payload


def _float_stats_subset(values: Iterable[float | None], keys: Sequence[str]) -> dict[str, Any]:
    stats = float_stats(values)
    payload = {"n": stats["n"]}
    for key in keys:
        payload[key] = stats[key]
    return payload


def _split_feature_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_rows = len(records)
    modality_keys = (
        "has_python_ast_target",
        "has_code_like_target",
        "symbol_oracle_available",
    )
    hint_keys = (
        "gold_full_path_mentioned",
        "gold_basename_mentioned",
        "gold_qualified_symbol_mentioned",
        "gold_symbol_name_mentioned",
    )

    def _flag_count(section: str, key: str) -> int:
        n_true = 0
        for item in records:
            block = item.get(section) or {}
            if isinstance(block, Mapping) and bool(block.get(key)):
                n_true += 1
        return n_true

    return {
        "n_rows": n_rows,
        "denominator": n_rows,
        "symbol_capability": {
            "symbol_oracle_available": _rate(
                _flag_count("modality", "symbol_oracle_available"), n_rows
            )
        },
        "target_scope": {
            "base_changed_file_count": _int_stats(
                (
                    int((item.get("target_scope") or {}).get("base_changed_file_count") or 0)
                    for item in records
                ),
                ("mean", "p50", "p90", "p95", "max"),
            ),
            "oracle_symbol_count": _int_stats(
                (
                    int((item.get("target_scope") or {}).get("oracle_symbol_count") or 0)
                    for item in records
                ),
                ("mean", "p50", "p90"),
            ),
        },
        "search_space": {
            "repo_tracked_files": _int_stats(
                (
                    int(item["search_space"]["repo_tracked_files"])
                    for item in records
                    if isinstance(item.get("search_space"), Mapping)
                    and item["search_space"].get("repo_tracked_files") is not None
                ),
                ("p50", "p90"),
            ),
            "file_target_density": _float_stats_subset(
                (
                    item["search_space"].get("file_target_density")
                    if isinstance(item.get("search_space"), Mapping)
                    else None
                    for item in records
                ),
                ("p50", "p90"),
            ),
        },
        "hint_strength": {
            key: _rate(_flag_count("hint_strength", key), n_rows) for key in hint_keys
        },
        "modality": {
            key: _rate(_flag_count("modality", key), n_rows) for key in modality_keys
        },
        "notes": {
            "gold_symbol_name_mentioned_is_not_a_balancing_criterion": True,
            "secondary_features_not_optimized": True,
        },
    }


def audit_secondary_distributions(
    records: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(item.get("instance_id") or ""): item for item in records}
    buckets: dict[str, list[Mapping[str, Any]]] = {"train": [], "dev": []}
    for item in assignments:
        split_name = str(item.get("split") or "")
        if split_name not in buckets:
            continue
        record = by_id.get(str(item.get("instance_id") or ""))
        if record is not None:
            buckets[split_name].append(record)
    train_audit = _split_feature_audit(buckets["train"])
    dev_audit = _split_feature_audit(buckets["dev"])
    flags: list[dict[str, Any]] = []

    def _collect_rate_paths(block: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
        found: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        for key, value in block.items():
            if key == "notes":
                continue
            path = prefix + (str(key),)
            if isinstance(value, Mapping) and "rate" in value and "denominator" in value:
                found.append((path, dict(value)))
            elif isinstance(value, Mapping):
                found.extend(_collect_rate_paths(value, path))
        return found

    train_rates = {path: payload for path, payload in _collect_rate_paths(train_audit)}
    for path, train_payload in train_rates.items():
        if path[-1] == "gold_symbol_name_mentioned":
            continue
        cursor: Any = dev_audit
        for key in path:
            cursor = cursor[key]
        dev_payload = cursor
        train_rate = train_payload.get("rate")
        dev_rate = dev_payload.get("rate")
        if train_rate is None or dev_rate is None:
            continue
        abs_delta = round(abs(float(dev_rate) - float(train_rate)), 6)
        if abs_delta > BINARY_RATE_SHIFT_THRESHOLD:
            flags.append(
                {
                    "metric": ".".join(path),
                    "train_rate": train_rate,
                    "dev_rate": dev_rate,
                    "abs_delta": abs_delta,
                    "threshold": BINARY_RATE_SHIFT_THRESHOLD,
                    "denominator_train": train_payload["denominator"],
                    "denominator_dev": dev_payload["denominator"],
                }
            )
    flags.sort(key=lambda item: item["metric"])
    return {
        "train": train_audit,
        "dev": dev_audit,
        "pathological_flags": flags,
        "binary_rate_shift_threshold": BINARY_RATE_SHIFT_THRESHOLD,
        "notes": {
            "secondary_features_not_optimized": True,
            "seed_was_not_changed_to_balance_distributions": True,
            "gold_symbol_name_mentioned_is_reported_but_not_a_balancing_criterion": True,
        },
    }


def spotlight_assignment(
    assignments: Sequence[Mapping[str, Any]],
    *,
    instance_ids: Sequence[str] = SPLIT_SPOTLIGHT_INSTANCE_IDS,
) -> dict[str, Any]:
    by_id = {item["instance_id"]: item for item in assignments}
    present = [instance_id for instance_id in instance_ids if instance_id in by_id]
    missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
    payload: dict[str, Any] = {
        "instance_ids": list(instance_ids),
        "present": present,
        "missing": missing,
        "present_in_input": not missing,
    }
    if missing:
        payload["same_split"] = False
        payload["same_group"] = False
        return payload
    splits = {by_id[instance_id]["split"] for instance_id in instance_ids}
    groups = {by_id[instance_id]["correlation_group_id"] for instance_id in instance_ids}
    payload["same_split"] = len(splits) == 1
    payload["same_group"] = len(groups) == 1
    payload["split"] = next(iter(splits))
    payload["correlation_group_id"] = next(iter(groups))
    payload["correlation_group_size"] = len(instance_ids)
    payload["members"] = [
        {
            "instance_id": instance_id,
            "repo": by_id[instance_id]["repo"],
            "correlation_group_id": by_id[instance_id]["correlation_group_id"],
            "split": by_id[instance_id]["split"],
        }
        for instance_id in instance_ids
    ]
    return payload


def build_split_manifest(
    result: SplitResult,
    *,
    revision: str = HF_REVISION,
    sha256: str = EXPECTED_SHA256,
    split_version: str = SPLIT_VERSION,
    seed: int = SPLIT_SEED,
    feature_jsonl_sha256: str | None = None,
    feature_jsonl: str = FEATURE_JSONL_RELPATH,
    feature_summary: str = FEATURE_SUMMARY_RELPATH,
) -> dict[str, Any]:
    repo_quotas = {}
    for item in result.repo_allocations:
        repo_quotas[item["repo"]] = {
            "n_rows": item["n_rows"],
            "n_groups": item["n_groups"],
            "base_quota": item["base_quota"],
            "remainder": item["remainder"],
            "extra": item["extra"],
            "target_dev_rows": item["target_dev_rows"],
            "actual_dev_rows": item["actual_dev_rows"],
            "delta": item["delta"],
            "exact": item["exact"],
            "train_rows": item["train_rows"],
            "dev_rows": item["dev_rows"],
            "train_groups": item["train_groups"],
            "dev_groups": item["dev_groups"],
        }
    return {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": revision,
        "sha256": sha256,
        "split_version": split_version,
        "seed": int(seed),
        "dev_fraction": {
            "numerator": DEV_FRACTION_NUM,
            "denominator": DEV_FRACTION_DEN,
        },
        "target_dev_rows": result.target_dev_rows,
        "target_train_rows": result.target_train_rows,
        "actual_dev_rows": result.actual_dev_rows,
        "actual_train_rows": result.actual_train_rows,
        "n_rows": result.n_rows,
        "n_train_groups": result.n_train_groups,
        "n_dev_groups": result.n_dev_groups,
        "allocation_algorithm": {
            "repo_quota": "largest_remainder_hamilton",
            "group_selection": "group_subset_sum_dp",
            "tie_break": "sha256(split_version|seed|correlation_group_id)",
        },
        "grouping_key": GROUPING_KEY,
        "stratification_key": STRATIFICATION_KEY,
        "policy_version": POLICY_VERSION,
        "eligible_count": result.n_rows,
        "excluded_count": 0,
        "m1d_a": {
            "feature_summary": feature_summary,
            "feature_jsonl": feature_jsonl,
            "feature_jsonl_sha256": feature_jsonl_sha256,
        },
        "repo_quotas": repo_quotas,
        "assignments": list(result.assignments),
        "notes": {
            "no_internal_test": True,
            "no_verl_parquet": True,
            "not_m1e": True,
            "seed_search": False,
            "secondary_features_not_optimized": True,
        },
    }


def build_split_summary(
    records: Sequence[Mapping[str, Any]],
    result: SplitResult,
    *,
    revision: str = HF_REVISION,
    sha256: str = EXPECTED_SHA256,
    split_version: str = SPLIT_VERSION,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    secondary = audit_secondary_distributions(records, result.assignments)
    exact_misses = [
        {
            "repo": item["repo"],
            "target_dev_rows": item["target_dev_rows"],
            "actual_dev_rows": item["actual_dev_rows"],
            "delta": item["delta"],
        }
        for item in result.repo_allocations
        if not item["exact"]
    ]
    train_repos = sorted(
        {item["repo"] for item in result.assignments if item["split"] == "train"}
    )
    dev_repos = sorted(
        {item["repo"] for item in result.assignments if item["split"] == "dev"}
    )
    leakage = correlation_leakage_count(result.assignments)
    return {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": revision,
        "sha256": sha256,
        "split_version": split_version,
        "seed": int(seed),
        "n_rows": result.n_rows,
        "denominator": result.n_rows,
        "global": {
            "train_rows": result.actual_train_rows,
            "dev_rows": result.actual_dev_rows,
            "train_groups": result.n_train_groups,
            "dev_groups": result.n_dev_groups,
            "target_dev_rows": result.target_dev_rows,
            "actual_dev_rows": result.actual_dev_rows,
            "delta": result.actual_dev_rows - result.target_dev_rows,
            "correlation_leakage_count": leakage,
            "denominator": result.n_rows,
        },
        "cross_repo_correlation_groups": result.cross_repo_correlation_groups,
        "repo_distributions": [
            {
                "repo": item["repo"],
                "n_rows": item["n_rows"],
                "target_dev_rows": item["target_dev_rows"],
                "actual_dev_rows": item["actual_dev_rows"],
                "quota_delta": item["delta"],
                "exact": item["exact"],
                "train_rows": item["train_rows"],
                "dev_rows": item["dev_rows"],
                "train_groups": item["train_groups"],
                "dev_groups": item["dev_groups"],
                "denominator": item["n_rows"],
            }
            for item in result.repo_allocations
        ],
        "repos_in_train": train_repos,
        "repos_in_dev": dev_repos,
        "exact_quota_misses": exact_misses,
        "spotlight": spotlight_assignment(result.assignments),
        "secondary": secondary,
        "invariants": {
            "assigned_once": len(result.assignments) == result.n_rows,
            "train_dev_disjoint": True,
            "union_complete": (
                result.actual_train_rows + result.actual_dev_rows == result.n_rows
            ),
            "correlation_leakage": leakage,
            "no_internal_test": True,
        },
        "notes": {
            "no_internal_test": True,
            "no_verl_parquet": True,
            "not_m1e": True,
            "seed_search": False,
            "secondary_features_not_optimized": True,
        },
    }


def manifest_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def feature_jsonl_identity(path: Path) -> str:
    return sha256_file(path)


def format_split_report(
    policy: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    global_stats = summary["global"]
    den = summary["denominator"]
    lines = [
        "SWE-Gym M1D-B eligibility policy + grouped train/dev split",
        f"policy_version: {policy['policy_version']}",
        f"split_version: {split_manifest['split_version']}",
        f"seed: {split_manifest['seed']}",
        f"revision: {split_manifest['revision']}",
        f"sha256: {split_manifest['sha256']}",
        (
            f"eligibility: {policy['eligible_count']}/{policy['master_pool_size']} "
            f"(excluded {policy['excluded_count']})"
        ),
        f"cross-repo correlation groups: {summary['cross_repo_correlation_groups']}",
        (
            f"rows: train={global_stats['train_rows']} dev={global_stats['dev_rows']} "
            f"target_dev={global_stats['target_dev_rows']} "
            f"delta={global_stats['delta']} denominator={den}"
        ),
        (
            f"groups: train={global_stats['train_groups']} "
            f"dev={global_stats['dev_groups']} "
            f"leakage={global_stats['correlation_leakage_count']}"
        ),
        "",
        "repo quotas (target / actual / delta):",
    ]
    for item in summary["repo_distributions"]:
        lines.append(
            f"  {item['repo']}: n={item['n_rows']} "
            f"target={item['target_dev_rows']} actual={item['actual_dev_rows']} "
            f"delta={item['quota_delta']} exact={item['exact']}"
        )
    misses = summary["exact_quota_misses"]
    if misses:
        lines.append("")
        lines.append("exact quota misses:")
        for item in misses:
            lines.append(
                f"  {item['repo']}: target={item['target_dev_rows']} "
                f"actual={item['actual_dev_rows']} delta={item['delta']}"
            )
    else:
        lines.append("")
        lines.append("exact quota misses: none")
    spotlight = summary["spotlight"]
    lines.extend(
        [
            "",
            "spotlight:",
            f"  ids={spotlight.get('instance_ids')}",
            f"  group={spotlight.get('correlation_group_id')} "
            f"size={spotlight.get('correlation_group_size')} "
            f"split={spotlight.get('split')} "
            f"same_split={spotlight.get('same_split')} "
            f"same_group={spotlight.get('same_group')}",
        ]
    )
    flags = summary["secondary"]["pathological_flags"]
    lines.append("")
    if flags:
        lines.append("secondary pathological flags:")
        for item in flags:
            lines.append(
                f"  {item['metric']}: train={item['train_rate']} "
                f"dev={item['dev_rate']} abs_delta={item['abs_delta']} "
                f"(den train={item['denominator_train']} "
                f"dev={item['denominator_dev']})"
            )
    else:
        lines.append("secondary pathological flags: none")
    lines.extend(
        [
            "",
            "M1E: not started (no veRL parquet, no RL).",
        ]
    )
    return "\n".join(lines) + "\n"
