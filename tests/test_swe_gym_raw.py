"""Unit tests for M1A SWE-Gym validate/profile helpers.

Uses the tiny fixture only. Does not download Hugging Face data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from budget_coder_rl.data.swe_gym import (
    EXPECTED_N_REPOS,
    EXPECTED_N_ROWS,
    PREVIEW_MAX_CHARS,
    parse_string_list,
    profile_frame,
    truncate_preview,
    validate_schema_and_cardinality,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "swe_gym_tiny.json"


def _load_fixture() -> pd.DataFrame:
    rows = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def test_fixture_passes_with_fixture_cardinality():
    frame = _load_fixture()
    assert validate_schema_and_cardinality(
        frame, expected_n_rows=3, expected_n_repos=2
    ) == []


def test_official_cardinality_rejects_fixture():
    frame = _load_fixture()
    errors = validate_schema_and_cardinality(
        frame,
        expected_n_rows=EXPECTED_N_ROWS,
        expected_n_repos=EXPECTED_N_REPOS,
    )
    assert any("row count" in err for err in errors)
    assert any("unique repo count" in err for err in errors)


def test_missing_column_fails():
    frame = _load_fixture().drop(columns=["patch"])
    errors = validate_schema_and_cardinality(
        frame, expected_n_rows=3, expected_n_repos=2
    )
    assert errors
    assert "patch" in errors[0]


def test_duplicate_instance_id_fails():
    frame = _load_fixture()
    frame.loc[1, "instance_id"] = frame.loc[0, "instance_id"]
    errors = validate_schema_and_cardinality(
        frame, expected_n_rows=3, expected_n_repos=2
    )
    assert any("not unique" in err for err in errors)


def test_extra_column_is_not_a_hard_fail():
    frame = _load_fixture()
    frame["debug_only"] = "x"
    assert validate_schema_and_cardinality(
        frame, expected_n_rows=3, expected_n_repos=2
    ) == []
    profile = profile_frame(frame)
    assert "debug_only" in profile["extra_columns"]


def test_json_string_lists_and_empty_fail_to_pass():
    frame = _load_fixture()
    assert parse_string_list(frame.loc[2, "PASS_TO_PASS"]) == []
    assert parse_string_list(frame.loc[2, "FAIL_TO_PASS"]) == ["one", "two"]
    profile = profile_frame(frame)
    assert profile["empty_list_counts"]["FAIL_TO_PASS"] == 1
    assert profile["empty_list_counts"]["PASS_TO_PASS"] == 1
    assert profile["list_length"]["FAIL_TO_PASS"]["max"] == 2
    assert profile["list_length"]["PASS_TO_PASS"]["max"] == 3


def test_text_length_stats_and_preview_truncation():
    frame = _load_fixture()
    profile = profile_frame(frame)
    patch_stats = profile["text_length"]["patch"]
    assert patch_stats["min"] == len("p3")
    assert patch_stats["max"] == len(frame.loc[1, "patch"])
    assert patch_stats["n"] == 3

    samples = {row["preview_reason"]: row for row in profile["samples"]}
    assert "largest_repo" in samples
    assert samples["largest_repo"]["repo"] == "owner-a/repo-a"
    assert samples["smallest_repo"]["repo"] == "owner-b/repo-b"
    assert samples["longest_problem_statement"]["instance_id"] == "owner_a__repo-a-3"

    long_patch = frame.loc[1, "patch"]
    preview_patch = samples["smallest_repo"]["patch"]
    assert preview_patch is not None
    assert long_patch not in preview_patch
    assert "truncated" in preview_patch
    assert preview_patch.startswith(long_patch[:PREVIEW_MAX_CHARS])
    assert truncate_preview(long_patch) == preview_patch


def test_null_and_empty_hints_are_counted_separately():
    frame = _load_fixture()
    profile = profile_frame(frame)
    hints = profile["field_missing"]["hints_text"]
    assert hints["empty"] == 1
    assert hints["null"] == 1
    assert hints["present"] == 1
