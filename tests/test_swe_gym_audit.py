"""Unit tests for M1B SWE-Gym field policy and integrity audit.

Uses crafted fixtures only. Does not read the official raw parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from budget_coder_rl.data.swe_gym import REQUIRED_COLUMNS
from budget_coder_rl.data.swe_gym_audit import (
    audit_frame,
    inspect_selector_controls,
    parse_test_list,
    selector_anomaly_tags,
)
from budget_coder_rl.data.swe_gym_fields import (
    AGENT_TASK_INPUT_FIELDS,
    PRIVILEGED_FIELDS,
    RUNTIME_IDENTITY_FIELDS,
    agent_task_view,
    committed_field_policy_errors,
    field_policy_path,
    field_policy_record,
    validate_field_policy_partition,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "swe_gym_m1b_audit.json"


def _load_fixture() -> pd.DataFrame:
    rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def _records_by_id(frame: pd.DataFrame) -> dict[str, dict]:
    records, _summary = audit_frame(frame)
    return {row["instance_id"]: row for row in records}


def test_field_partition_is_disjoint_and_complete():
    errors = validate_field_policy_partition()
    assert errors == []
    union = (
        set(AGENT_TASK_INPUT_FIELDS)
        | set(RUNTIME_IDENTITY_FIELDS)
        | set(PRIVILEGED_FIELDS)
    )
    assert union == set(REQUIRED_COLUMNS)
    assert set(AGENT_TASK_INPUT_FIELDS).isdisjoint(RUNTIME_IDENTITY_FIELDS)
    assert set(AGENT_TASK_INPUT_FIELDS).isdisjoint(PRIVILEGED_FIELDS)
    assert set(RUNTIME_IDENTITY_FIELDS).isdisjoint(PRIVILEGED_FIELDS)


def test_committed_field_policy_matches_python_contract():
    assert committed_field_policy_errors(REPO_ROOT) == []
    loaded = json.loads(field_policy_path(REPO_ROOT).read_text(encoding="utf-8"))
    assert loaded == field_policy_record()


def test_agent_task_view_excludes_privileged_fields():
    frame = _load_fixture()
    row = frame.loc[1].to_dict()
    assert row["hints_text"]
    assert row["patch"]
    assert row["FAIL_TO_PASS"]
    view = agent_task_view(row)
    assert list(view.keys()) == ["problem_statement"]
    assert set(view).isdisjoint(PRIVILEGED_FIELDS)
    for name in PRIVILEGED_FIELDS:
        assert name not in view
    assert view["problem_statement"] == row["problem_statement"]
    leaked = json.dumps(view)
    assert row["patch"] not in leaked
    assert row["hints_text"] not in leaked
    assert "FAIL_TO_PASS" not in leaked


def test_balanced_pytest_selector_is_not_a_structural_error():
    tags, observations, shape = selector_anomaly_tags(
        "tests/test_foo.py::test_bar[param]"
    )
    assert tags == []
    assert observations == []
    assert shape == "pytest_nodeid"


def test_unbalanced_bracket_malformed_selector_is_flagged():
    tags, _observations, shape = selector_anomaly_tags(
        "conans/test/unittests/tools/cmake/test_cmake_test.py::test_run_tests[Ninja"
    )
    assert "unbalanced_brackets" in tags
    assert "embedded_newline" not in tags
    assert shape == "pytest_nodeid"


def test_fixture_unbalanced_row_uses_generic_heuristic():
    by_id = _records_by_id(_load_fixture())
    record = by_id["owner_a__unbalanced-2"]
    assert "f2p_unbalanced_brackets" in record["flags"]
    assert "conan-io__conan-11594" not in json.dumps(record)


def test_clean_fixture_row_has_no_anomaly_flags():
    by_id = _records_by_id(_load_fixture())
    assert by_id["owner_a__clean-1"]["flags"] == []
    assert "f2p_unbalanced_brackets" not in by_id["owner_a__clean-1"]["flags"]


def test_duplicate_test_selector_is_flagged():
    by_id = _records_by_id(_load_fixture())
    flags = by_id["owner_b__dup_selector-3"]["flags"]
    assert "f2p_duplicate_entries" in flags
    assert "f2p_p2p_overlap" not in flags


def test_f2p_p2p_overlap_is_flagged():
    by_id = _records_by_id(_load_fixture())
    record = by_id["owner_b__overlap-4"]
    assert "f2p_p2p_overlap" in record["flags"]
    assert "f2p_selector_verbatim_in_problem_statement" in record["observations"]


def test_empty_p2p_is_dataset_property_not_heuristic_flag():
    records, summary = audit_frame(_load_fixture())
    by_id = {row["instance_id"]: row for row in records}
    record = by_id["owner_a__empty_p2p-5"]
    assert "p2p_empty" in record["dataset_properties"]
    assert "p2p_empty" not in record["flags"]
    assert record["flags"] == []
    assert "owner_a__empty_p2p-5" not in {
        row["instance_id"] for row in records if row["flags"]
    }
    assert "patch_not_unified_diff" in by_id["owner_a__bad_patch-6"]["flags"]
    assert summary["n_heuristic_suspicion_rows"] == sum(1 for row in records if row["flags"])


def test_duplicate_problem_statement_is_dataset_property():
    records, summary = audit_frame(_load_fixture())
    by_id = {row["instance_id"]: row for row in records}
    assert "duplicate_problem_statement" in by_id["owner_c__dup_ps-7"]["dataset_properties"]
    assert "duplicate_problem_statement" in by_id["owner_c__dup_ps-8"]["dataset_properties"]
    assert "duplicate_problem_statement" not in by_id["owner_c__dup_ps-7"]["flags"]
    assert summary["duplicates"]["problem_statement"]["n_groups"] == 1
    assert summary["duplicates"]["problem_statement"]["n_rows"] == 2
    assert summary["dataset_property_counts"]["duplicate_problem_statement"] == 2
    assert summary["dataset_property_counts"]["p2p_empty"] == 1
    assert summary["rows_dropped"] == 0
    assert summary["n_rows"] == len(records) == 8
    assert "n_flagged_rows" not in summary
    assert summary["heuristic_suspicion_is_not_confirmed_malformed"] is True


def test_audit_does_not_drop_rows():
    frame = _load_fixture()
    records, summary = audit_frame(frame)
    assert len(records) == len(frame)
    assert summary["n_rows"] == len(frame)
    assert summary["rows_dropped"] == 0
    assert summary["n_rows_written"] == len(frame)
    assert "n_heuristic_suspicion_rows" in summary


def test_parse_test_list_preserves_non_string_types():
    assert parse_test_list([1, "ok"]) == [1, "ok"]
    tags, _obs, shape = selector_anomaly_tags(1)
    assert tags == ["non_string_entry"]
    assert shape is None


def test_whitespace_padded_selector():
    tags, _obs, _shape = selector_anomaly_tags(" tests/a.py::test_x")
    assert "whitespace_padded" in tags
    assert "unbalanced_brackets" not in tags


def test_actual_control_char_vs_literal_escape_sequence():
    actual, literal = inspect_selector_controls("id[\x00uuid]")
    assert actual["NUL"] == 1
    assert literal == {}
    actual, literal = inspect_selector_controls(r"id[\x00uuid]")
    assert actual == {}
    assert literal.get(r"\x00") == 1
