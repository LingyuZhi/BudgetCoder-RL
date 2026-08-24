"""SWE-Gym M1D-A eligibility and structural-difficulty features.

Evidence-only: every input row is retained. This module does not filter,
split, write keep/drop decisions, or synthesize an easy/medium/hard score.
Derived fields are privileged evaluator metadata and must not enter
``agent_task_view``.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_SHA256,
    HF_REPO_ID,
    HF_REVISION,
    is_null,
    length_stats,
)
from budget_coder_rl.data.swe_gym_audit import normalize_problem_statement
from budget_coder_rl.data.swe_gym_oracle import oracle_jsonl_path
from budget_coder_rl.data.swe_gym_symbol_oracle import symbol_oracle_jsonl_path
from budget_coder_rl.data.swe_gym_tree_stats import (
    is_code_like_path,
    is_docs_path,
    is_python_ast_path,
    lookup_tree_stats,
)

FEATURE_JSONL_RELPATH = "data/interim/swe_gym/m1d_features.jsonl"
FEATURE_SUMMARY_RELPATH = "data/manifests/swe_gym_m1d_feature_summary.json"
PANDAS_CORRELATION_SPOTLIGHT = (
    "pandas-dev__pandas-53805",
    "pandas-dev__pandas-53830",
)

VALIDITY_REASON_ORDER: tuple[str, ...] = (
    "empty_problem_statement",
    "zero_base_changed_files",
    "repo_unresolved",
    "commit_unresolved",
    "base_changed_blob_missing",
    "gold_diff_parse_failure",
    "oracle_artifact_mismatch",
    "tree_stats_unavailable",
)

DERIVED_PRIVILEGED_FEATURE_FIELDS: tuple[str, ...] = (
    "technical_valid",
    "technical_invalid_reasons",
    "search_space",
    "target_scope",
    "hint_strength",
    "modality",
    "correlation_group_id",
    "correlation_group_size",
)

FORBIDDEN_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "patch",
        "test_patch",
        "hints_text",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "gold_edit_files",
        "oracle_symbols",
        "unmapped_sites",
        "file_results",
        "problem_statement",
        "keep",
        "drop",
        "split",
        "difficulty",
        "easy",
        "medium",
        "hard",
    }
)

# Full-path interior: reject being a suffix/prefix of a longer path.
_PATH_BOUNDARY_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_./")
# Basename: '/' before the filename is a real mention (src/foo.py).
_BASENAME_LEFT_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_.")
_BASENAME_RIGHT_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_./")
_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_DENSITY_DECIMALS = 10


def feature_jsonl_path(repo_root: Path) -> Path:
    return Path(repo_root) / FEATURE_JSONL_RELPATH


def feature_summary_path(repo_root: Path) -> Path:
    return Path(repo_root) / FEATURE_SUMMARY_RELPATH


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            records.append(json.loads(text))
    return records


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    weight = idx - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def float_stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(numbers)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 10),
        "mean": round(sum(ordered) / len(ordered), 4),
        "p50": round(_quantile(ordered, 0.50), 10),
        "p90": round(_quantile(ordered, 0.90), 10),
        "p95": round(_quantile(ordered, 0.95), 10),
        "p99": round(_quantile(ordered, 0.99), 10),
        "max": round(ordered[-1], 10),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, _DENSITY_DECIMALS)


def _sorted_reasons(reasons: Iterable[str]) -> list[str]:
    present = {str(name) for name in reasons}
    ordered = [name for name in VALIDITY_REASON_ORDER if name in present]
    extras = sorted(present.difference(VALIDITY_REASON_ORDER))
    return ordered + extras


def _text(value: Any) -> str:
    if is_null(value):
        return ""
    return str(value)


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    return bool(value)


def normalize_repo_path(path: str) -> str:
    return str(path).replace("\\", "/").strip()


def path_basename(path: str) -> str:
    normalized = normalize_repo_path(path)
    return normalized.rsplit("/", 1)[-1]


def _is_dunder(name: str) -> bool:
    return len(name) >= 4 and name.startswith("__") and name.endswith("__")


def _count_mentions(
    haystack: str,
    needle: str,
    *,
    ignore_case: bool,
    left_chars: set[str],
    right_chars: set[str] | None = None,
) -> int:
    """Count conservative boundary-respecting occurrences of ``needle``."""
    if not needle or not haystack:
        return 0
    text = haystack.lower() if ignore_case else haystack
    target = needle.lower() if ignore_case else needle
    right = left_chars if right_chars is None else right_chars
    count = 0
    start = 0
    while True:
        index = text.find(target, start)
        if index < 0:
            return count
        before_ok = index == 0 or text[index - 1] not in left_chars
        end = index + len(target)
        after_ok = end == len(text) or text[end] not in right
        if before_ok and after_ok:
            count += 1
        start = index + 1


def path_is_mentioned(problem_statement: str, path: str) -> bool:
    haystack = normalize_repo_path(problem_statement)
    needle = normalize_repo_path(path)
    return (
        _count_mentions(
            haystack,
            needle,
            ignore_case=True,
            left_chars=_PATH_BOUNDARY_CHARS,
            right_chars=_PATH_BOUNDARY_CHARS,
        )
        > 0
    )


def basename_is_mentioned(problem_statement: str, path: str) -> bool:
    haystack = normalize_repo_path(problem_statement)
    needle = path_basename(path)
    if not needle:
        return False
    return (
        _count_mentions(
            haystack,
            needle,
            ignore_case=True,
            left_chars=_BASENAME_LEFT_CHARS,
            right_chars=_BASENAME_RIGHT_CHARS,
        )
        > 0
    )


def symbol_name_is_mentioned(problem_statement: str, name: str) -> bool:
    if not name or len(name) < 2 or _is_dunder(name):
        return False
    return (
        _count_mentions(
            problem_statement,
            name,
            ignore_case=False,
            left_chars=_IDENT_CHARS,
            right_chars=_IDENT_CHARS,
        )
        > 0
    )


def qualified_symbol_is_mentioned(problem_statement: str, qualname: str) -> bool:
    if "." not in qualname:
        return False
    return (
        _count_mentions(
            problem_statement,
            qualname,
            ignore_case=False,
            left_chars=_IDENT_CHARS,
            right_chars=_IDENT_CHARS,
        )
        > 0
    )


def unqualified_symbol_names(symbols: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in symbols:
        qualname = str(item.get("qualname") or "")
        last = qualname.rsplit(".", 1)[-1]
        if len(last) < 2 or _is_dunder(last) or last in seen:
            continue
        seen.add(last)
        names.append(last)
    names.sort()
    return names


def qualified_symbol_names(symbols: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in symbols:
        qualname = str(item.get("qualname") or "")
        if "." not in qualname or qualname in seen:
            continue
        seen.add(qualname)
        names.append(qualname)
    names.sort()
    return names


def mention_features(
    problem_statement: str,
    base_changed_files: Sequence[str],
    oracle_symbols: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    files = [normalize_repo_path(path) for path in base_changed_files if path]
    n_full = sum(1 for path in files if path_is_mentioned(problem_statement, path))
    n_base = sum(1 for path in files if basename_is_mentioned(problem_statement, path))
    n_symbol = sum(
        1
        for name in unqualified_symbol_names(oracle_symbols)
        if symbol_name_is_mentioned(problem_statement, name)
    )
    n_qualified = sum(
        1
        for name in qualified_symbol_names(oracle_symbols)
        if qualified_symbol_is_mentioned(problem_statement, name)
    )
    return {
        "gold_full_path_mentioned": n_full > 0,
        "gold_full_path_mention_count": n_full,
        "gold_basename_mentioned": n_base > 0,
        "gold_basename_mention_count": n_base,
        "gold_symbol_name_mentioned": n_symbol > 0,
        "gold_symbol_name_mention_count": n_symbol,
        "gold_qualified_symbol_mentioned": n_qualified > 0,
        "gold_qualified_symbol_mention_count": n_qualified,
    }


def classify_modality(
    base_changed_files: Sequence[str],
    oracle_symbol_count: int,
) -> dict[str, bool]:
    paths = [normalize_repo_path(path) for path in base_changed_files if path]
    n_python = sum(1 for path in paths if is_python_ast_path(path))
    n_code = sum(1 for path in paths if is_code_like_path(path))
    n_docs = sum(1 for path in paths if is_docs_path(path))
    n_non_code = len(paths) - n_code
    has_paths = bool(paths)
    return {
        "has_python_ast_target": n_python > 0,
        "has_code_like_target": n_code > 0,
        "symbol_oracle_available": int(oracle_symbol_count) >= 1,
        "docs_only_target": has_paths and n_docs == len(paths),
        "non_code_only_target": has_paths and n_code == 0,
        "mixed_code_noncode_target": n_code > 0 and n_non_code > 0,
    }


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def correlation_assignments(
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, int]]:
    """Connected components: same (repo, base_commit) OR same normalized PS.

    Empty problem statements do not create text edges. Group id is
    ``cg:`` plus the lexicographically smallest instance_id in the component.
    """
    instance_ids = [_text(item.get("instance_id")) for item in identities]
    forest = _UnionFind(instance_ids)
    by_commit: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_problem: dict[str, list[str]] = defaultdict(list)
    for item in identities:
        instance_id = _text(item.get("instance_id"))
        repo = _text(item.get("repo")).strip()
        commit = _text(item.get("base_commit")).strip()
        if repo and commit:
            by_commit[(repo, commit)].append(instance_id)
        normalized = normalize_problem_statement(item.get("problem_statement"))
        if normalized:
            by_problem[normalized].append(instance_id)

    def _union_group(members: Sequence[str]) -> None:
        if len(members) < 2:
            return
        anchor = sorted(members)[0]
        for other in members:
            forest.union(anchor, other)

    for members in by_commit.values():
        _union_group(members)
    for members in by_problem.values():
        _union_group(members)

    components: dict[str, list[str]] = defaultdict(list)
    for instance_id in instance_ids:
        components[forest.find(instance_id)].append(instance_id)

    assigned: dict[str, tuple[str, int]] = {}
    for members in components.values():
        ordered = sorted(members)
        group_id = f"cg:{ordered[0]}"
        size = len(ordered)
        for instance_id in ordered:
            assigned[instance_id] = (group_id, size)
    return assigned


def _operation_flags(gold_edit_files: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    operations = {str(item.get("operation") or "") for item in gold_edit_files}
    return {
        "contains_added_file": "added" in operations,
        "contains_deleted_file": "deleted" in operations,
        "contains_path_changed": "path_changed" in operations,
    }


def _path_list(value: Any) -> list[str]:
    if not value:
        return []
    paths = [normalize_repo_path(str(item)) for item in value]
    return [path for path in paths if path]


def technical_invalid_reasons(
    *,
    problem_statement: str,
    base_changed_files: Sequence[str],
    repo_ok: bool,
    commit_ok: bool,
    n_blob_missing: int,
    parse_ok: bool,
    oracle_mismatch: bool,
    tree_ok: bool,
) -> list[str]:
    reasons: list[str] = []
    if not str(problem_statement).strip():
        reasons.append("empty_problem_statement")
    if len(base_changed_files) < 1:
        reasons.append("zero_base_changed_files")
    if not repo_ok:
        reasons.append("repo_unresolved")
    if not commit_ok:
        reasons.append("commit_unresolved")
    if int(n_blob_missing) > 0:
        reasons.append("base_changed_blob_missing")
    if not parse_ok:
        reasons.append("gold_diff_parse_failure")
    if oracle_mismatch:
        reasons.append("oracle_artifact_mismatch")
    if not tree_ok:
        reasons.append("tree_stats_unavailable")
    return _sorted_reasons(reasons)


def _density_bundle(
    *,
    tree_ok: bool,
    repo_tracked_files: int,
    repo_python_files: int,
    base_changed_file_count: int,
    n_base_changed_python_ast: int,
) -> tuple[float | None, float | None, list[str]]:
    if not tree_ok:
        return None, None, ["tree_stats_unavailable"]
    file_density = _ratio(base_changed_file_count, repo_tracked_files)
    python_density = _ratio(n_base_changed_python_ast, repo_python_files)
    reasons: list[str] = []
    if file_density is None:
        reasons.append("zero_repo_tracked_files")
    if python_density is None:
        reasons.append("zero_repo_python_files")
    return file_density, python_density, reasons


def build_feature_record(
    row: Mapping[str, Any],
    oracle: Mapping[str, Any],
    symbol: Mapping[str, Any],
    tree: Any,
    *,
    correlation_group_id: str,
    correlation_group_size: int,
) -> dict[str, Any]:
    instance_id = _text(row.get("instance_id"))
    repo = _text(row.get("repo")).strip()
    base_commit = _text(row.get("base_commit")).strip()
    problem_statement = _text(row.get("problem_statement"))

    oracle_files = _path_list(oracle.get("base_changed_files"))
    symbol_files = _path_list(symbol.get("base_changed_files"))
    parse_ok = _truthy(oracle.get("parse_ok")) and _truthy(symbol.get("parse_ok"))
    mismatch = oracle_files != symbol_files
    parquet_commit = base_commit
    symbol_commit = _text(symbol.get("base_commit")).strip()
    if symbol_commit and parquet_commit and symbol_commit != parquet_commit:
        mismatch = True

    base_changed_files = oracle_files
    n_blob_missing = int(symbol.get("n_blob_missing") or 0)
    repo_ok = _truthy(symbol.get("repo_ok"))
    commit_ok = _truthy(symbol.get("commit_ok"))
    tree_ok = bool(tree.ok)
    reasons = technical_invalid_reasons(
        problem_statement=problem_statement,
        base_changed_files=base_changed_files,
        repo_ok=repo_ok,
        commit_ok=commit_ok,
        n_blob_missing=n_blob_missing,
        parse_ok=parse_ok,
        oracle_mismatch=mismatch,
        tree_ok=tree_ok,
    )

    n_python_targets = sum(1 for path in base_changed_files if is_python_ast_path(path))
    n_symbols = int(symbol.get("n_oracle_symbols") or 0)
    file_density, python_density, density_reasons = _density_bundle(
        tree_ok=tree_ok,
        repo_tracked_files=int(tree.repo_tracked_files),
        repo_python_files=int(tree.repo_python_files),
        base_changed_file_count=len(base_changed_files),
        n_base_changed_python_ast=n_python_targets,
    )
    gold_edit_files = list(oracle.get("gold_edit_files") or [])
    oracle_symbols = list(symbol.get("oracle_symbols") or [])
    record = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "technical_valid": len(reasons) == 0,
        "technical_invalid_reasons": reasons,
        "search_space": {
            "repo_tracked_files": int(tree.repo_tracked_files) if tree_ok else None,
            "repo_python_files": int(tree.repo_python_files) if tree_ok else None,
            "repo_code_like_files": int(tree.repo_code_like_files) if tree_ok else None,
            "repo_tracked_blob_bytes": (
                tree.repo_tracked_blob_bytes if tree_ok else None
            ),
            "base_changed_file_count": len(base_changed_files),
            "n_base_changed_python_ast": n_python_targets,
            "file_target_density": file_density,
            "python_target_density": python_density,
            "density_undefined_reasons": density_reasons,
        },
        "target_scope": {
            "gold_edit_file_count": int(oracle.get("n_gold_edit_files") or 0),
            "base_changed_file_count": len(base_changed_files),
            "oracle_symbol_count": n_symbols,
            "gold_hunk_count": int(oracle.get("n_hunks") or 0),
            "added_line_count": int(oracle.get("n_added_lines") or 0),
            "removed_line_count": int(oracle.get("n_removed_lines") or 0),
            **_operation_flags(gold_edit_files),
        },
        "hint_strength": mention_features(
            problem_statement, base_changed_files, oracle_symbols
        ),
        "modality": classify_modality(base_changed_files, n_symbols),
        "correlation_group_id": correlation_group_id,
        "correlation_group_size": int(correlation_group_size),
    }
    leaked = FORBIDDEN_OUTPUT_KEYS.intersection(record)
    if leaked:
        raise RuntimeError(
            f"forbidden keys leaked into feature record: {sorted(leaked)}"
        )
    return record


def instance_alignment_errors(
    rows: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    symbol_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    def _ids(
        records: Sequence[Mapping[str, Any]], label: str
    ) -> tuple[list[str], list[str]]:
        values = [_text(item.get("instance_id")) for item in records]
        local_errors: list[str] = []
        empty = sum(1 for item in values if not item)
        if empty:
            local_errors.append(f"{label} has {empty} empty instance_id(s)")
        if len(values) != len(set(values)):
            local_errors.append(f"{label} instance_id is not unique")
        return values, local_errors

    row_ids, errors = _ids(rows, "parquet")
    oracle_ids, oracle_errors = _ids(oracle_records, "m1c_oracle")
    symbol_ids, symbol_errors = _ids(symbol_records, "m1c_symbol_oracle")
    errors.extend(oracle_errors)
    errors.extend(symbol_errors)
    row_set = set(row_ids)
    oracle_set = set(oracle_ids)
    symbol_set = set(symbol_ids)
    if row_set != oracle_set:
        errors.append(
            "parquet vs m1c_oracle instance_id set mismatch: "
            f"only_parquet={len(row_set - oracle_set)} "
            f"only_oracle={len(oracle_set - row_set)}"
        )
    if row_set != symbol_set:
        errors.append(
            "parquet vs m1c_symbol_oracle instance_id set mismatch: "
            f"only_parquet={len(row_set - symbol_set)} "
            f"only_symbol={len(symbol_set - row_set)}"
        )
    if len(rows) != len(oracle_records):
        errors.append(
            f"parquet n={len(rows)} != m1c_oracle n={len(oracle_records)}"
        )
    if len(rows) != len(symbol_records):
        errors.append(
            f"parquet n={len(rows)} != m1c_symbol_oracle n={len(symbol_records)}"
        )
    return errors


def extract_feature_frame(
    rows: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    symbol_records: Sequence[Mapping[str, Any]],
    tree_provider: Any,
) -> list[dict[str, Any]]:
    """Build one feature record per input row. Never drops rows."""
    errors = instance_alignment_errors(rows, oracle_records, symbol_records)
    if errors:
        raise ValueError("M1D-A artifact alignment failed: " + "; ".join(errors))
    oracle_by_id = {_text(item.get("instance_id")): item for item in oracle_records}
    symbol_by_id = {_text(item.get("instance_id")): item for item in symbol_records}
    identities = [
        {
            "instance_id": _text(row.get("instance_id")),
            "repo": _text(row.get("repo")),
            "base_commit": _text(row.get("base_commit")),
            "problem_statement": row.get("problem_statement"),
        }
        for row in rows
    ]
    groups = correlation_assignments(identities)
    records: list[dict[str, Any]] = []
    for row in rows:
        instance_id = _text(row.get("instance_id"))
        repo = _text(row.get("repo")).strip()
        commit = _text(row.get("base_commit")).strip()
        tree = lookup_tree_stats(tree_provider, repo, commit)
        group_id, group_size = groups[instance_id]
        records.append(
            build_feature_record(
                row,
                oracle_by_id[instance_id],
                symbol_by_id[instance_id],
                tree,
                correlation_group_id=group_id,
                correlation_group_size=group_size,
            )
        )
    records.sort(key=lambda item: item["instance_id"])
    return records


def unique_repo_commits(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    keys = {
        (_text(row.get("repo")).strip(), _text(row.get("base_commit")).strip())
        for row in rows
    }
    return sorted(keys)


def _argmax(records: Sequence[Mapping[str, Any]], getter) -> dict[str, Any] | None:
    best_id = None
    best_value = None
    for item in records:
        value = getter(item)
        if value is None:
            continue
        if best_value is None or (value, item["instance_id"]) > (
            best_value,
            best_id or "",
        ):
            best_value = value
            best_id = item["instance_id"]
    if best_id is None:
        return None
    return {"instance_id": best_id, "value": best_value}


def _argmin(records: Sequence[Mapping[str, Any]], getter) -> dict[str, Any] | None:
    best_id = None
    best_value = None
    for item in records:
        value = getter(item)
        if value is None:
            continue
        if best_value is None or (value, item["instance_id"]) < (
            best_value,
            best_id or "",
        ):
            best_value = value
            best_id = item["instance_id"]
    if best_id is None:
        return None
    return {"instance_id": best_id, "value": best_value}


def _count_true(records: Sequence[Mapping[str, Any]], getter) -> int:
    return sum(1 for item in records if getter(item))


def _modality_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "has_python_ast_target",
        "has_code_like_target",
        "symbol_oracle_available",
        "docs_only_target",
        "non_code_only_target",
        "mixed_code_noncode_target",
    )
    return {
        key: _count_true(records, lambda item, name=key: item["modality"][name])
        for key in keys
    }


def _hint_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "gold_full_path_mentioned": _count_true(
            records, lambda item: item["hint_strength"]["gold_full_path_mentioned"]
        ),
        "gold_basename_mentioned": _count_true(
            records, lambda item: item["hint_strength"]["gold_basename_mentioned"]
        ),
        "gold_symbol_name_mentioned": _count_true(
            records, lambda item: item["hint_strength"]["gold_symbol_name_mentioned"]
        ),
        "gold_qualified_symbol_mentioned": _count_true(
            records,
            lambda item: item["hint_strength"]["gold_qualified_symbol_mentioned"],
        ),
    }


def _validity_reason_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in records:
        counts.update(item["technical_invalid_reasons"])
    return {key: int(counts[key]) for key in _sorted_reasons(counts)}


def _group_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in records:
        group_id = str(item["correlation_group_id"])
        bucket = by_id.get(group_id)
        if bucket is None:
            bucket = {
                "correlation_group_id": group_id,
                "size": int(item["correlation_group_size"]),
                "instance_ids": [],
            }
            by_id[group_id] = bucket
        bucket["instance_ids"].append(item["instance_id"])
    groups = []
    for bucket in by_id.values():
        members = sorted(bucket["instance_ids"])
        groups.append(
            {
                "correlation_group_id": bucket["correlation_group_id"],
                "size": len(members),
                "instance_ids": members,
            }
        )
    groups.sort(key=lambda item: (-item["size"], item["correlation_group_id"]))
    return groups


def _pandas_spotlight(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {item["instance_id"]: item for item in records}
    left_id, right_id = PANDAS_CORRELATION_SPOTLIGHT
    left = by_id.get(left_id)
    right = by_id.get(right_id)
    payload: dict[str, Any] = {
        "instance_ids": list(PANDAS_CORRELATION_SPOTLIGHT),
        "present_in_input": left is not None and right is not None,
    }
    if left is None or right is None:
        payload["same_group"] = False
        return payload
    payload["same_group"] = left["correlation_group_id"] == right["correlation_group_id"]
    payload["correlation_group_id"] = left["correlation_group_id"]
    payload["correlation_group_size"] = int(left["correlation_group_size"])
    return payload


def _per_repo(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        by_repo[str(item["repo"])].append(item)
    report: dict[str, Any] = {}
    for repo in sorted(by_repo):
        group = by_repo[repo]
        n = len(group)
        report[repo] = {
            "n": n,
            "denominator": n,
            "repo_tracked_files": length_stats(
                item["search_space"]["repo_tracked_files"]
                for item in group
                if item["search_space"]["repo_tracked_files"] is not None
            ),
            "file_target_density": float_stats(
                item["search_space"]["file_target_density"] for item in group
            ),
            "python_target_density": float_stats(
                item["search_space"]["python_target_density"] for item in group
            ),
            "base_changed_file_count": length_stats(
                item["target_scope"]["base_changed_file_count"] for item in group
            ),
            "oracle_symbol_count": length_stats(
                item["target_scope"]["oracle_symbol_count"] for item in group
            ),
            "modality_counts": _modality_counts(group),
            "hint_counts": _hint_counts(group),
            "n_technical_valid": _count_true(
                group, lambda item: item["technical_valid"]
            ),
            "n_symbol_capable": _count_true(
                group, lambda item: item["modality"]["symbol_oracle_available"]
            ),
            "n_file_only": _count_true(
                group, lambda item: not item["modality"]["symbol_oracle_available"]
            ),
        }
    return report


def build_feature_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_rows = len(records)
    n_valid = _count_true(records, lambda item: item["technical_valid"])
    groups = _group_records(records)
    group_sizes = [item["size"] for item in groups]
    n_singleton = sum(1 for size in group_sizes if size == 1)
    max_group = groups[0] if groups else None
    n_symbol_capable = _count_true(
        records, lambda item: item["modality"]["symbol_oracle_available"]
    )
    n_file_only = n_rows - n_symbol_capable
    n_file_only_python_unmapped = _count_true(
        records,
        lambda item: (not item["modality"]["symbol_oracle_available"])
        and item["modality"]["has_python_ast_target"],
    )
    n_file_only_non_python = _count_true(
        records,
        lambda item: (not item["modality"]["symbol_oracle_available"])
        and (not item["modality"]["has_python_ast_target"]),
    )
    density_undefined = _count_true(
        records, lambda item: bool(item["search_space"]["density_undefined_reasons"])
    )
    n_defined_file_density = _count_true(
        records, lambda item: item["search_space"]["file_target_density"] is not None
    )
    n_defined_python_density = _count_true(
        records, lambda item: item["search_space"]["python_target_density"] is not None
    )
    return {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": HF_REVISION,
        "sha256": EXPECTED_SHA256,
        "n_rows": n_rows,
        "n_rows_written": n_rows,
        "rows_dropped": 0,
        "technical_validity": {
            "n_valid": n_valid,
            "n_invalid": n_rows - n_valid,
            "denominator": n_rows,
            "reason_counts": _validity_reason_counts(records),
        },
        "search_space": {
            "repo_tracked_files": length_stats(
                item["search_space"]["repo_tracked_files"]
                for item in records
                if item["search_space"]["repo_tracked_files"] is not None
            ),
            "repo_python_files": length_stats(
                item["search_space"]["repo_python_files"]
                for item in records
                if item["search_space"]["repo_python_files"] is not None
            ),
            "repo_code_like_files": length_stats(
                item["search_space"]["repo_code_like_files"]
                for item in records
                if item["search_space"]["repo_code_like_files"] is not None
            ),
            "file_target_density": {
                **float_stats(
                    item["search_space"]["file_target_density"] for item in records
                ),
                "n_defined": n_defined_file_density,
                "n_undefined": n_rows - n_defined_file_density,
                "denominator": n_rows,
            },
            "python_target_density": {
                **float_stats(
                    item["search_space"]["python_target_density"] for item in records
                ),
                "n_defined": n_defined_python_density,
                "n_undefined": n_rows - n_defined_python_density,
                "denominator": n_rows,
            },
            "n_density_undefined_rows": density_undefined,
            "denominator": n_rows,
        },
        "target_scope": {
            "base_changed_file_count": length_stats(
                item["target_scope"]["base_changed_file_count"] for item in records
            ),
            "gold_edit_file_count": length_stats(
                item["target_scope"]["gold_edit_file_count"] for item in records
            ),
            "oracle_symbol_count": length_stats(
                item["target_scope"]["oracle_symbol_count"] for item in records
            ),
            "gold_hunk_count": length_stats(
                item["target_scope"]["gold_hunk_count"] for item in records
            ),
            "added_line_count": length_stats(
                item["target_scope"]["added_line_count"] for item in records
            ),
            "removed_line_count": length_stats(
                item["target_scope"]["removed_line_count"] for item in records
            ),
            "n_contains_added_file": _count_true(
                records, lambda item: item["target_scope"]["contains_added_file"]
            ),
            "n_contains_deleted_file": _count_true(
                records, lambda item: item["target_scope"]["contains_deleted_file"]
            ),
            "n_contains_path_changed": _count_true(
                records, lambda item: item["target_scope"]["contains_path_changed"]
            ),
            "denominator": n_rows,
        },
        "modality_counts": {
            **_modality_counts(records),
            "denominator": n_rows,
        },
        "symbol_capability": {
            "n_symbol_capable": n_symbol_capable,
            "n_file_only": n_file_only,
            "n_file_only_python_unmapped": n_file_only_python_unmapped,
            "n_file_only_non_python": n_file_only_non_python,
            "denominator": n_rows,
            "note": (
                "symbol_capable = oracle_symbol_count >= 1. "
                "file_only = oracle_symbol_count == 0. Zero-symbol is not invalid."
            ),
        },
        "hint_strength": {
            **_hint_counts(records),
            "denominator": n_rows,
            "note": (
                "Legitimate issue-text mentions of base_changed_files / oracle "
                "symbols. Not leakage. Counts are instance-level booleans "
                "(distinct target coverage), denominator is n_rows."
            ),
        },
        "correlation": {
            "n_groups": len(groups),
            "n_singleton_groups": n_singleton,
            "group_size": length_stats(group_sizes),
            "max_group": (
                {
                    "correlation_group_id": max_group["correlation_group_id"],
                    "size": max_group["size"],
                    "n_instance_ids_preview": min(12, max_group["size"]),
                    "instance_ids_preview": max_group["instance_ids"][:12],
                }
                if max_group is not None
                else None
            ),
            "pandas_spotlight": _pandas_spotlight(records),
            "rule": (
                "Connected components over same (repo, base_commit) OR same "
                "normalized problem_statement. Not a split."
            ),
        },
        "per_repo": _per_repo(records),
        "outliers": {
            "max_repo_tracked_files": _argmax(
                records, lambda item: item["search_space"]["repo_tracked_files"]
            ),
            "min_file_target_density": _argmin(
                records, lambda item: item["search_space"]["file_target_density"]
            ),
            "max_file_target_density": _argmax(
                records, lambda item: item["search_space"]["file_target_density"]
            ),
            "max_base_changed_file_count": _argmax(
                records, lambda item: item["target_scope"]["base_changed_file_count"]
            ),
            "max_oracle_symbol_count": _argmax(
                records, lambda item: item["target_scope"]["oracle_symbol_count"]
            ),
            "max_gold_edit_file_count": _argmax(
                records, lambda item: item["target_scope"]["gold_edit_file_count"]
            ),
        },
        "codescout_comparison": {
            "status": "skipped",
            "reason": (
                "Optional reference would require a network fetch or vendoring "
                "OpenHands/SWE-Gym-code-search instance IDs. Core M1D-A does "
                "not depend on CodeScout and does not drop instances."
            ),
        },
        "notes": {
            "no_filter": True,
            "no_split": True,
            "no_keep_drop": True,
            "no_composite_difficulty": True,
            "not_m1d_b": True,
            "privileged_derived_fields": list(DERIVED_PRIVILEGED_FEATURE_FIELDS),
            "code_like_extensions": [
                ".c",
                ".cc",
                ".cpp",
                ".cxx",
                ".h",
                ".hh",
                ".hpp",
                ".hxx",
                ".pxd",
                ".pxi",
                ".py",
                ".pyi",
                ".pyx",
            ],
            "js_ts_not_code_like": True,
            "zero_symbol_is_not_invalid": True,
            "added_files_not_base_retrieval_targets": True,
        },
        "jsonl": FEATURE_JSONL_RELPATH,
    }


def format_feature_report(summary: Mapping[str, Any]) -> str:
    validity = summary["technical_validity"]
    search = summary["search_space"]
    target = summary["target_scope"]
    modality = summary["modality_counts"]
    symbols = summary["symbol_capability"]
    hints = summary["hint_strength"]
    corr = summary["correlation"]
    den = summary["n_rows"]

    def _q(name: str, stats: Mapping[str, Any]) -> str:
        return (
            f"  {name}: n={stats['n']} min={stats['min']} mean={stats['mean']} "
            f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
            f"p99={stats['p99']} max={stats['max']}"
        )

    lines = [
        "SWE-Gym M1D-A eligibility / structural-difficulty audit",
        f"revision: {summary['revision']}",
        f"sha256: {summary['sha256']}",
        f"rows: {summary['n_rows']} (dropped: {summary['rows_dropped']})",
        (
            f"technical_valid: {validity['n_valid']}/{validity['denominator']} "
            f"(invalid {validity['n_invalid']})"
        ),
        "",
        "search-space quantiles:",
        _q("repo_tracked_files", search["repo_tracked_files"]),
        _q("repo_python_files", search["repo_python_files"]),
        _q("repo_code_like_files", search["repo_code_like_files"]),
        _q("file_target_density", search["file_target_density"]),
        _q("python_target_density", search["python_target_density"]),
        (
            "  density undefined: "
            f"file={search['file_target_density']['n_undefined']}/{den} "
            f"python={search['python_target_density']['n_undefined']}/{den}"
        ),
        "",
        "target-scope quantiles:",
        _q("base_changed_files", target["base_changed_file_count"]),
        _q("oracle_symbols", target["oracle_symbol_count"]),
        "",
        f"modality (denominator={den}):",
    ]
    for key in (
        "has_python_ast_target",
        "has_code_like_target",
        "symbol_oracle_available",
        "docs_only_target",
        "non_code_only_target",
        "mixed_code_noncode_target",
    ):
        lines.append(f"  {key}: {modality[key]}/{den}")
    lines.extend(
        [
            "",
            (
                f"symbol_capable: {symbols['n_symbol_capable']}/{den}; "
                f"file_only: {symbols['n_file_only']}/{den} "
                f"(python_unmapped={symbols['n_file_only_python_unmapped']}, "
                f"non_python={symbols['n_file_only_non_python']})"
            ),
            "",
            f"issue hints (denominator={den}):",
            f"  full_path: {hints['gold_full_path_mentioned']}/{den}",
            f"  basename: {hints['gold_basename_mentioned']}/{den}",
            f"  symbol_name: {hints['gold_symbol_name_mentioned']}/{den}",
            f"  qualified_symbol: {hints['gold_qualified_symbol_mentioned']}/{den}",
            "",
            "correlation groups:",
            (
                f"  n_groups={corr['n_groups']} singletons={corr['n_singleton_groups']} "
                f"max={corr['max_group']['size'] if corr['max_group'] else None}"
            ),
            _q("group_size", corr["group_size"]),
        ]
    )
    spotlight = corr["pandas_spotlight"]
    lines.append(
        "  pandas spotlight "
        f"{spotlight['instance_ids']}: present={spotlight.get('present_in_input')} "
        f"same_group={spotlight.get('same_group')} "
        f"id={spotlight.get('correlation_group_id')} "
        f"size={spotlight.get('correlation_group_size')}"
    )
    lines.append("")
    lines.append("per-repo n:")
    for repo, info in summary["per_repo"].items():
        lines.append(f"  {repo}: {info['n']}")
    lines.append("")
    lines.append(f"CodeScout comparison: {summary['codescout_comparison']['status']}")
    lines.append("M1D-B: not started (no filter, no split, no keep/drop).")
    return "\n".join(lines) + "\n"


ORACLE_JSONL_PATH = oracle_jsonl_path
SYMBOL_JSONL_PATH = symbol_oracle_jsonl_path
