"""SWE-Gym field visibility / leakage contract (M1B / M1E).

Python constants are canonical. The tracked M1B JSON manifest must match
the 11-field official partition. M1E policy/runtime parquet uses the
recursive forbidden-key validator defined here; that list is not part of
the official 11-field JSON (derived fields are not SWE-Gym columns).

This module does not build an RL dataset or expose privileged fields to
the agent task view.
"""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym import HF_REVISION, REQUIRED_COLUMNS

AGENT_TASK_INPUT_FIELDS: tuple[str, ...] = ("problem_statement",)
RUNTIME_IDENTITY_FIELDS: tuple[str, ...] = (
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "created_at",
)
PRIVILEGED_FIELDS: tuple[str, ...] = (
    "hints_text",
    "patch",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)
# Derived evaluator metadata. Not official SWE-Gym columns and not part of
# the 11-field visibility partition. Must not enter agent_task_view.
DERIVED_PRIVILEGED_ORACLE_FIELDS: tuple[str, ...] = (
    "gold_edit_files",
    "base_changed_files",
    "oracle_symbols",
    "unmapped_sites",
    "file_results",
)
# M1D-A nested feature blobs and gold-derived difficulty / density keys.
# Duplicated here (not imported from swe_gym_features) to avoid a cycle.
# ``split`` is allowed on M1E policy extra_info and is intentionally omitted.
POLICY_FORBIDDEN_FEATURE_FIELDS: tuple[str, ...] = (
    "technical_valid",
    "technical_invalid_reasons",
    "search_space",
    "target_scope",
    "hint_strength",
    "modality",
    "correlation_group_id",
    "correlation_group_size",
    "file_target_density",
    "python_target_density",
    "gold_full_path_mentioned",
    "gold_basename_mentioned",
    "gold_qualified_symbol_mentioned",
    "gold_symbol_name_mentioned",
    "oracle_symbol_count",
    "base_changed_file_count",
    "gold_edit_file_count",
    "n_oracle_symbols",
    "n_base_changed_files",
    "n_gold_edit_files",
    "n_gold_edit_file_count",
    "repo_tracked_files",
    "repo_python_files",
    "repo_code_like_files",
    "symbol_oracle_available",
    "symbol_applicable",
    "difficulty",
    "easy",
    "medium",
    "hard",
    "keep",
    "drop",
)
POLICY_FORBIDDEN_DERIVED_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *PRIVILEGED_FIELDS,
            *DERIVED_PRIVILEGED_ORACLE_FIELDS,
            *POLICY_FORBIDDEN_FEATURE_FIELDS,
        )
    )
)
POLICY_FORBIDDEN_DERIVED_FIELD_SET: frozenset[str] = frozenset(
    POLICY_FORBIDDEN_DERIVED_FIELDS
)

_POLICY_NOTES: dict[str, str] = {
    "hints_text": (
        "Stage-1 default: do not expose hints_text to the agent."
    ),
    "gold_patch_and_tests": (
        "patch, test_patch, FAIL_TO_PASS, and PASS_TO_PASS are privileged "
        "evaluator information and must not enter the agent task view."
    ),
    "external_lookup": (
        "Future rollout must forbid external web/GitHub lookup of the issue "
        "or repository. M1B records this contract only; no network sandbox "
        "is implemented here."
    ),
}


def field_policy_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "manifests" / "swe_gym_field_policy.json"


def field_policy_record() -> dict[str, Any]:
    return {
        "dataset": "SWE-Gym",
        "revision": HF_REVISION,
        "official_fields": list(REQUIRED_COLUMNS),
        "agent_task_input": list(AGENT_TASK_INPUT_FIELDS),
        "runtime_identity_metadata": list(RUNTIME_IDENTITY_FIELDS),
        "privileged_policy_hidden": list(PRIVILEGED_FIELDS),
        "notes": dict(_POLICY_NOTES),
    }


def validate_field_policy_partition(
    *,
    official_fields: Sequence[str] | None = None,
    agent_task_input: Sequence[str] = AGENT_TASK_INPUT_FIELDS,
    runtime_identity_metadata: Sequence[str] = RUNTIME_IDENTITY_FIELDS,
    privileged_policy_hidden: Sequence[str] = PRIVILEGED_FIELDS,
) -> list[str]:
    """Return hard-fail reasons if the three groups are not a partition of official fields."""
    official = list(
        official_fields if official_fields is not None else REQUIRED_COLUMNS
    )
    groups: tuple[tuple[str, Sequence[str]], ...] = (
        ("agent_task_input", agent_task_input),
        ("runtime_identity_metadata", runtime_identity_metadata),
        ("privileged_policy_hidden", privileged_policy_hidden),
    )
    errors: list[str] = []
    seen: list[str] = []
    seen_set: set[str] = set()
    for group_name, fields in groups:
        if len(set(fields)) != len(fields):
            errors.append(f"{group_name} has duplicate fields")
        for name in fields:
            if name in seen_set:
                errors.append(
                    f"field {name!r} appears in multiple visibility groups"
                )
            seen.append(name)
            seen_set.add(name)
    official_set = set(official)
    partitioned = set(seen)
    missing = [name for name in official if name not in partitioned]
    extra = [name for name in seen if name not in official_set]
    if missing:
        errors.append(f"partition missing official fields: {missing}")
    if extra:
        errors.append(f"partition has fields not in official schema: {extra}")
    if len(official) != len(official_set):
        errors.append("official field list has duplicates")
    return errors


def agent_task_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Policy-visible task input. Privileged fields are never copied."""
    return {name: row[name] for name in AGENT_TASK_INPUT_FIELDS if name in row}


def committed_field_policy_errors(repo_root: Path) -> list[str]:
    """Hard-fail reasons if the tracked JSON does not match the Python contract."""
    errors = validate_field_policy_partition()
    path = field_policy_path(repo_root)
    if not path.is_file():
        errors.append(f"missing field policy manifest: {path}")
        return errors
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"field policy manifest is not valid JSON: {exc}")
        return errors
    expected = field_policy_record()
    if loaded != expected:
        errors.append(
            f"field policy manifest {path} does not match field_policy_record()"
        )
    return errors


def _is_plain_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if isinstance(value, MappingABC):
        return False
    if isinstance(value, (list, tuple)):
        return True
    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return True
    return False


def _sequence_items(value: Any) -> list[Any]:
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, str, bytes)):
        try:
            converted = value.tolist()
        except (TypeError, ValueError):
            converted = list(value)
        if isinstance(converted, list):
            return converted
        return [converted]
    return list(value)


def collect_forbidden_policy_keys(
    obj: Any,
    *,
    forbidden: frozenset[str] | None = None,
    path: str = "",
) -> list[str]:
    """Return human-readable locations of forbidden keys at any nesting depth."""
    names = POLICY_FORBIDDEN_DERIVED_FIELD_SET if forbidden is None else forbidden
    errors: list[str] = []
    if isinstance(obj, MappingABC):
        for key, val in obj.items():
            key_s = str(key)
            child = f"{path}.{key_s}" if path else key_s
            if key_s in names:
                errors.append(f"forbidden key {key_s!r} at {child}")
            errors.extend(
                collect_forbidden_policy_keys(val, forbidden=names, path=child)
            )
        return errors
    if _is_plain_sequence(obj):
        for index, item in enumerate(_sequence_items(obj)):
            child = f"{path}[{index}]" if path else f"[{index}]"
            errors.extend(
                collect_forbidden_policy_keys(item, forbidden=names, path=child)
            )
    return errors


def _prompt_messages(row: Mapping[str, Any]) -> list[Any] | None:
    prompt = row.get("prompt")
    if prompt is None:
        return None
    if _is_plain_sequence(prompt):
        return _sequence_items(prompt)
    return None


def validate_policy_row_prompt(
    row: Mapping[str, Any],
    *,
    problem_statement: str,
    instance_id: str | None = None,
) -> list[str]:
    """Hard-fail reasons if prompt is not exactly the raw problem_statement."""
    prefix = f"{instance_id}: " if instance_id else ""
    errors: list[str] = []
    messages = _prompt_messages(row)
    if messages is None:
        errors.append(f"{prefix}prompt is missing or not a list")
        return errors
    if len(messages) != 1:
        errors.append(
            f"{prefix}prompt must contain exactly one message, got {len(messages)}"
        )
        return errors
    message = messages[0]
    if not isinstance(message, MappingABC):
        errors.append(f"{prefix}prompt[0] is not a mapping")
        return errors
    role = message.get("role")
    content = message.get("content")
    if role != "user":
        errors.append(f"{prefix}prompt[0].role is {role!r}, expected 'user'")
    if not isinstance(content, str):
        errors.append(f"{prefix}prompt[0].content is not a str")
        return errors
    if content != problem_statement:
        errors.append(
            f"{prefix}prompt[0].content does not equal raw problem_statement"
        )
    extra_keys = sorted(str(key) for key in message.keys() if str(key) not in {"role", "content"})
    if extra_keys:
        errors.append(f"{prefix}prompt[0] has unexpected keys: {extra_keys}")
    return errors


def validate_policy_row_leakage(
    row: Mapping[str, Any],
    *,
    problem_statement: str | None = None,
    instance_id: str | None = None,
) -> list[str]:
    """Recursive forbidden-key check plus optional prompt exactness."""
    prefix = f"{instance_id}: " if instance_id else ""
    errors = [
        f"{prefix}{item}" if prefix and not item.startswith(prefix) else item
        for item in collect_forbidden_policy_keys(row)
    ]
    if problem_statement is not None:
        errors.extend(
            validate_policy_row_prompt(
                row,
                problem_statement=problem_statement,
                instance_id=instance_id,
            )
        )
    return errors


def validate_policy_rows_leakage(
    rows: Sequence[Mapping[str, Any]],
    *,
    problem_statements: Mapping[str, str] | None = None,
) -> list[str]:
    """Validate every policy/runtime row. Empty list means pass."""
    errors: list[str] = []
    for row in rows:
        extra = row.get("extra_info") if isinstance(row.get("extra_info"), MappingABC) else {}
        instance_id = None
        if isinstance(extra, MappingABC) and extra.get("instance_id") is not None:
            instance_id = str(extra.get("instance_id"))
        expected_ps = None
        if problem_statements is not None and instance_id is not None:
            if instance_id not in problem_statements:
                errors.append(f"{instance_id}: missing problem_statement for prompt check")
                continue
            expected_ps = problem_statements[instance_id]
        errors.extend(
            validate_policy_row_leakage(
                row,
                problem_statement=expected_ps,
                instance_id=instance_id,
            )
        )
    return errors
