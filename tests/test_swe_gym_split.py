"""Unit tests for M1D-B eligibility policy and grouped train/dev split.

Uses synthetic records only. Does not read the official parquet, M1D-A JSONL,
or SWE-Gym Git mirrors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from budget_coder_rl.data.swe_gym import write_json
from budget_coder_rl.data.swe_gym_features import correlation_assignments
from budget_coder_rl.data.swe_gym_split import (
    ASSIGNMENT_FIELDS,
    DEV_FRACTION_DEN,
    DEV_FRACTION_NUM,
    FORBIDDEN_SPLIT_KEYS,
    POLICY_VERSION,
    SPLIT_SEED,
    SPLIT_SPOTLIGHT_INSTANCE_IDS,
    SPLIT_VERSION,
    CrossRepoCorrelationError,
    GroupUnit,
    SplitInputError,
    assign_split,
    build_eligibility_policy,
    build_split_manifest,
    build_split_summary,
    collect_correlation_groups,
    correlation_leakage_count,
    group_priority,
    largest_remainder_quotas,
    manifest_json_bytes,
    select_groups_subset_sum,
    spotlight_assignment,
    target_dev_rows,
    validate_split_invariants,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _group(group_id: str, size: int, repo: str = "owner/name") -> GroupUnit:
    instance_ids = tuple(f"{group_id}__{index}" for index in range(size))
    return GroupUnit(group_id, size, instance_ids, repo)


def _record(
    instance_id: str,
    repo: str,
    group_id: str,
    *,
    size: int | None = None,
    symbol: bool = True,
    files: int = 1,
    symbols: int = 1,
    tracked: int = 100,
    density: float = 0.01,
    full_path: bool = False,
    basename: bool = False,
    qualified: bool = False,
    symbol_name: bool = False,
) -> dict:
    payload = {
        "instance_id": instance_id,
        "repo": repo,
        "correlation_group_id": group_id,
        "modality": {
            "has_python_ast_target": True,
            "has_code_like_target": True,
            "symbol_oracle_available": symbol,
        },
        "target_scope": {
            "base_changed_file_count": files,
            "oracle_symbol_count": symbols,
        },
        "search_space": {
            "repo_tracked_files": tracked,
            "file_target_density": density,
        },
        "hint_strength": {
            "gold_full_path_mentioned": full_path,
            "gold_basename_mentioned": basename,
            "gold_qualified_symbol_mentioned": qualified,
            "gold_symbol_name_mentioned": symbol_name,
        },
    }
    if size is not None:
        payload["correlation_group_size"] = size
    return payload


def _records_from_groups(groups: list[GroupUnit]) -> list[dict]:
    records = []
    for group in groups:
        for instance_id in group.instance_ids:
            records.append(
                _record(
                    instance_id,
                    group.repo,
                    group.correlation_group_id,
                    size=group.size,
                )
            )
    return records


def test_hamilton_quotas_sum_exactly_and_match_remainder_ranking():
    repo_counts = {
        "Project-MONAI/MONAI": 374,
        "bokeh/bokeh": 26,
        "conan-io/conan": 75,
        "dask/dask": 145,
        "facebookresearch/hydra": 66,
        "getmoto/moto": 343,
        "iterative/dvc": 225,
        "modin-project/modin": 107,
        "pandas-dev/pandas": 737,
        "pydantic/pydantic": 83,
        "python/mypy": 257,
    }
    n_rows = sum(repo_counts.values())
    assert n_rows == 2438
    assert target_dev_rows(n_rows) == 244
    allocated = largest_remainder_quotas(repo_counts, n_rows=n_rows)
    quotas = allocated["quotas"]
    assert allocated["target_dev_rows"] == 244
    assert allocated["target_train_rows"] == 2194
    assert sum(item["target_dev_rows"] for item in quotas.values()) == 244
    expected = {
        "pandas-dev/pandas": 74,
        "Project-MONAI/MONAI": 37,
        "getmoto/moto": 34,
        "python/mypy": 26,
        "iterative/dvc": 22,
        "dask/dask": 14,
        "modin-project/modin": 11,
        "pydantic/pydantic": 8,
        "conan-io/conan": 8,
        "facebookresearch/hydra": 7,
        "bokeh/bokeh": 3,
    }
    for repo, target in expected.items():
        assert quotas[repo]["target_dev_rows"] == target
        assert quotas[repo]["base_quota"] == repo_counts[repo] // 10
        assert quotas[repo]["remainder"] == repo_counts[repo] % 10
    leftover = 244 - sum(repo_counts[repo] // 10 for repo in repo_counts)
    ranked = sorted(
        repo_counts,
        key=lambda repo: (-(repo_counts[repo] % 10), repo),
    )
    extras = set(ranked[:leftover])
    for repo, item in quotas.items():
        assert item["extra"] == (1 if repo in extras else 0)
        assert item["target_dev_rows"] == item["base_quota"] + item["extra"]


def test_group_atomicity():
    groups = [
        _group("cg:a", 3, "owner/one"),
        _group("cg:b", 1, "owner/one"),
        _group("cg:c", 1, "owner/two"),
        _group("cg:d", 1, "owner/two"),
    ]
    records = _records_from_groups(groups)
    result = assign_split(records)
    by_group: dict[str, set[str]] = {}
    for item in result.assignments:
        by_group.setdefault(item["correlation_group_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    assert correlation_leakage_count(result.assignments) == 0


def test_exact_subset_sum_is_preferred():
    groups = [_group("cg:a", 1), _group("cg:b", 2), _group("cg:c", 3)]
    selected = select_groups_subset_sum(groups, 3)
    assert selected.exact is True
    assert selected.actual == 3
    assert selected.delta == 0
    assert sum(group.size for group in groups if group.correlation_group_id in selected.selected_ids) == 3


def test_nearest_attainable_when_no_exact_sum():
    groups = [_group("cg:a", 2), _group("cg:b", 2), _group("cg:c", 2)]
    selected = select_groups_subset_sum(groups, 3)
    assert selected.exact is False
    assert selected.actual == 2
    assert selected.delta == -1
    assert selected.target == 3
    assert len(selected.selected_ids) == 1
    even_groups = [_group(f"cg:{index}", 2, "owner/name") for index in range(15)]
    records = _records_from_groups(even_groups)
    result = assign_split(records)
    repo_one = result.repo_allocations[0]
    assert repo_one["target_dev_rows"] == 3
    assert repo_one["exact"] is False
    assert repo_one["actual_dev_rows"] == 2
    assert repo_one["delta"] == -1
    by_group: dict[str, set[str]] = {}
    for item in result.assignments:
        by_group.setdefault(item["correlation_group_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in by_group.values())


def test_sha256_priority_is_stable_and_not_builtin_hash():
    digest = group_priority(SPLIT_VERSION, SPLIT_SEED, "cg:foo")
    expected = hashlib.sha256(
        f"{SPLIT_VERSION}|{SPLIT_SEED}|cg:foo".encode("utf-8")
    ).hexdigest()
    assert digest == expected
    assert digest == group_priority(SPLIT_VERSION, SPLIT_SEED, "cg:foo")
    assert digest != group_priority(SPLIT_VERSION, SPLIT_SEED, "cg:bar")
    # Built-in hash is not part of the contract and is not a hex digest of this payload.
    assert digest != format(abs(hash("cg:foo")), "x")


def test_same_seed_and_input_yield_identical_split():
    groups = [
        _group("cg:a", 2, "owner/one"),
        _group("cg:b", 1, "owner/one"),
        _group("cg:c", 1, "owner/one"),
        _group("cg:d", 3, "owner/two"),
        _group("cg:e", 1, "owner/two"),
    ]
    records = _records_from_groups(groups)
    first = assign_split(records)
    second = assign_split(records)
    assert first.assignments == second.assignments
    manifest_one = build_split_manifest(first, feature_jsonl_sha256="abc")
    manifest_two = build_split_manifest(second, feature_jsonl_sha256="abc")
    assert manifest_json_bytes(manifest_one) == manifest_json_bytes(manifest_two)


def test_distinct_groups_can_be_assigned_independently():
    groups = [_group(f"cg:{index}", 1, "owner/name") for index in range(10)]
    records = _records_from_groups(groups)
    result = assign_split(records)
    assert result.target_dev_rows == 1
    assert result.actual_dev_rows == 1
    splits = {item["correlation_group_id"]: item["split"] for item in result.assignments}
    assert set(splits.values()) == {"train", "dev"}
    assert sum(1 for split in splits.values() if split == "dev") == 1
    assert sum(1 for split in splits.values() if split == "train") == 9


def test_transitive_correlation_group_is_never_split():
    identities = [
        {
            "instance_id": "a",
            "repo": "owner/name",
            "base_commit": "c1",
            "problem_statement": "same issue",
        },
        {
            "instance_id": "b",
            "repo": "owner/name",
            "base_commit": "c1",
            "problem_statement": "other issue",
        },
        {
            "instance_id": "c",
            "repo": "owner/name",
            "base_commit": "c2",
            "problem_statement": "same issue",
        },
        {
            "instance_id": "d",
            "repo": "owner/name",
            "base_commit": "c3",
            "problem_statement": "unrelated",
        },
    ]
    assigned = correlation_assignments(identities)
    group_ids = {item[0] for item in assigned.values()}
    assert len({assigned[key][0] for key in ("a", "b", "c")}) == 1
    assert assigned["d"][0] not in {assigned["a"][0]}
    records = []
    for item in identities:
        group_id, size = assigned[item["instance_id"]]
        records.append(_record(item["instance_id"], item["repo"], group_id, size=size))
    result = assign_split(records)
    transitive_group = assigned["a"][0]
    splits = {
        item["split"]
        for item in result.assignments
        if item["correlation_group_id"] == transitive_group
    }
    members = [
        item["instance_id"]
        for item in result.assignments
        if item["correlation_group_id"] == transitive_group
    ]
    assert set(members) == {"a", "b", "c"}
    assert splits == {next(iter(splits))}
    assert correlation_leakage_count(result.assignments) == 0
    assert len(group_ids) == 2


def test_cross_repo_correlation_group_fails_explicitly():
    records = [
        _record("a", "owner/one", "cg:shared", size=2),
        _record("b", "owner/two", "cg:shared", size=2),
        _record("c", "owner/one", "cg:solo", size=1),
    ]
    with pytest.raises(CrossRepoCorrelationError) as caught:
        assign_split(records)
    assert caught.value.groups
    assert caught.value.groups[0]["correlation_group_id"] == "cg:shared"
    assert set(caught.value.groups[0]["repos"]) == {"owner/one", "owner/two"}
    assert "do not split the group" in str(caught.value)
    with pytest.raises(CrossRepoCorrelationError):
        collect_correlation_groups(records)


def test_train_dev_union_intersection_invariants():
    groups = [
        _group("cg:a", 2, "owner/one"),
        _group("cg:b", 1, "owner/one"),
        _group("cg:c", 4, "owner/two"),
        _group("cg:d", 1, "owner/two"),
        _group("cg:e", 1, "owner/two"),
    ]
    records = _records_from_groups(groups)
    result = assign_split(records)
    errors = validate_split_invariants(records, result.assignments)
    assert errors == []
    train = {item["instance_id"] for item in result.assignments if item["split"] == "train"}
    dev = {item["instance_id"] for item in result.assignments if item["split"] == "dev"}
    all_ids = {item["instance_id"] for item in records}
    assert train.isdisjoint(dev)
    assert train.union(dev) == all_ids
    assert len(result.assignments) == len(records)
    assert "test" not in {item["split"] for item in result.assignments}


def test_spotlight_style_size_three_group_is_assigned_wholly():
    groups = [
        GroupUnit(
            "cg:pandas-dev__pandas-53805",
            3,
            SPLIT_SPOTLIGHT_INSTANCE_IDS,
            "pandas-dev/pandas",
        ),
        _group("cg:solo-a", 1, "pandas-dev/pandas"),
        _group("cg:solo-b", 1, "pandas-dev/pandas"),
        _group("cg:other", 1, "owner/other"),
    ]
    records = []
    for group in groups:
        for instance_id in group.instance_ids:
            records.append(
                _record(
                    instance_id,
                    group.repo,
                    group.correlation_group_id,
                    size=group.size,
                )
            )
    result = assign_split(records)
    spotlight = spotlight_assignment(result.assignments)
    assert spotlight["same_group"] is True
    assert spotlight["same_split"] is True
    assert spotlight["correlation_group_id"] == "cg:pandas-dev__pandas-53805"
    assert spotlight["correlation_group_size"] == 3
    members = [
        item
        for item in result.assignments
        if item["instance_id"] in SPLIT_SPOTLIGHT_INSTANCE_IDS
    ]
    assert len(members) == 3
    assert len({item["split"] for item in members}) == 1


def test_split_manifest_deterministic_serialization(tmp_path: Path):
    groups = [
        _group("cg:a", 1, "owner/one"),
        _group("cg:b", 2, "owner/one"),
        _group("cg:c", 1, "owner/two"),
    ]
    records = _records_from_groups(groups)
    result = assign_split(records)
    policy = build_eligibility_policy(n_rows=len(records))
    manifest = build_split_manifest(result, feature_jsonl_sha256="deadbeef")
    summary = build_split_summary(records, result)
    first = manifest_json_bytes(manifest)
    second = manifest_json_bytes(build_split_manifest(result, feature_jsonl_sha256="deadbeef"))
    assert first == second
    assert json.dumps(manifest, indent=2, ensure_ascii=True) + "\n" == first.decode("utf-8")
    out = tmp_path / "split.json"
    write_json(out, manifest)
    assert out.read_bytes() == first
    dumped = json.dumps(manifest)
    for key in FORBIDDEN_SPLIT_KEYS:
        assert key not in dumped
    for item in manifest["assignments"]:
        assert tuple(item.keys()) == ASSIGNMENT_FIELDS
        assert "patch" not in item
        assert "problem_statement" not in item
    assert policy["policy_version"] == POLICY_VERSION
    assert policy["eligible_count"] == len(records)
    assert policy["excluded_count"] == 0
    assert policy["excluded_instance_ids"] == []
    assert policy["primary_oracle"] == "base_changed_files"
    assert policy["curriculum"] == "not enabled at dataset stage"
    assert "test" not in {item["split"] for item in manifest["assignments"]}
    assert summary["global"]["correlation_leakage_count"] == 0
    assert DEV_FRACTION_NUM == 1
    assert DEV_FRACTION_DEN == 10


def test_group_size_mismatch_is_a_hard_fail():
    records = [
        _record("a", "owner/name", "cg:a", size=2),
        _record("b", "owner/name", "cg:a", size=9),
    ]
    with pytest.raises(SplitInputError, match="correlation_group_size"):
        collect_correlation_groups(records)


def test_eligibility_policy_is_keep_all():
    policy = build_eligibility_policy(n_rows=2438)
    assert policy["master_pool_size"] == 2438
    assert policy["eligible_count"] == 2438
    assert policy["excluded_count"] == 0
    assert "zero_symbol_oracle" in policy["not_drop_criteria"]
    assert policy["reward"] == "not implemented in M1D-B"
    assert policy["notes"]["not_m1e"] is True
    assert policy["notes"]["no_verl_parquet"] is True
    assert policy["notes"]["no_internal_test"] is True
    assert policy["external_final_test"] == "not from SWE-Gym"
