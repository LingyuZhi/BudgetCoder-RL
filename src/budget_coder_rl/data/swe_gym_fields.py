"""SWE-Gym field visibility / leakage contract (M1B).

Python constants are canonical. The tracked JSON manifest must match.
This module does not build an RL dataset or expose privileged fields to
the agent task view.
"""

from __future__ import annotations

import json
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
