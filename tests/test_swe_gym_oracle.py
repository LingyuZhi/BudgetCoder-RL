"""Unit tests for M1C-A gold-patch oracle extraction.

Uses crafted unified-diff fixtures only. Does not read the official raw parquet.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from budget_coder_rl.data.swe_gym_oracle import (
    build_oracle_summary,
    extract_oracle_frame,
    extract_oracle_from_patch,
    is_test_like_path,
    normalize_diff_path,
    oracle_record_from_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "swe_gym_m1c_oracle.json"


def _load_cases() -> dict[str, dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _record(name: str) -> dict:
    return oracle_record_from_row(_load_cases()[name])


def test_modified_single_file():
    record = _record("modified_single")
    assert record["parse_ok"] is True
    assert record["oracle_source"] == "patch"
    assert record["n_gold_edit_files"] == 1
    file_rec = record["gold_edit_files"][0]
    assert file_rec["path"] == "src/foo.py"
    assert file_rec["source_path"] == "src/foo.py"
    assert file_rec["target_path"] == "src/foo.py"
    assert file_rec["operation"] == "modified"
    assert file_rec["num_hunks"] == 1
    assert file_rec["n_added_lines"] == 1
    assert file_rec["n_removed_lines"] == 1
    hunk = file_rec["hunks"][0]
    assert hunk["source_start"] == 1
    assert hunk["target_start"] == 1
    assert hunk["removed_source_lines"] == [2]
    assert hunk["added_target_lines"] == [2]
    assert "oracle_lines" not in record
    assert "oracle_lines" not in file_rec


def test_multiple_files():
    record = _record("multiple_files")
    assert record["parse_ok"] is True
    assert record["n_gold_edit_files"] == 2
    paths = [item["path"] for item in record["gold_edit_files"]]
    assert paths == ["src/bar.py", "src/foo.py"]
    assert {item["operation"] for item in record["gold_edit_files"]} == {"modified"}


def test_multiple_hunks():
    record = _record("multiple_hunks")
    assert record["parse_ok"] is True
    file_rec = record["gold_edit_files"][0]
    assert file_rec["num_hunks"] == 2
    assert record["n_hunks"] == 2
    starts = [hunk["source_start"] for hunk in file_rec["hunks"]]
    assert starts == [2, 20]
    assert file_rec["hunks"][0]["removed_source_lines"] == [3]
    assert file_rec["hunks"][1]["removed_source_lines"] == [21]


def test_added_file_dev_null():
    record = _record("added_dev_null")
    assert record["parse_ok"] is True
    file_rec = record["gold_edit_files"][0]
    assert file_rec["operation"] == "added"
    assert file_rec["path"] == "src/new.py"
    assert file_rec["source_path"] is None
    assert file_rec["target_path"] == "src/new.py"
    assert file_rec["hunks"][0]["source_start"] == 0
    assert file_rec["hunks"][0]["source_length"] == 0
    assert file_rec["hunks"][0]["added_target_lines"] == [1, 2]
    assert file_rec["hunks"][0]["removed_source_lines"] == []
    dumped = json.dumps(record)
    assert "/dev/null" not in dumped


def test_deleted_file_dev_null():
    record = _record("deleted_dev_null")
    assert record["parse_ok"] is True
    file_rec = record["gold_edit_files"][0]
    assert file_rec["operation"] == "deleted"
    assert file_rec["path"] == "src/old.py"
    assert file_rec["source_path"] == "src/old.py"
    assert file_rec["target_path"] is None
    assert file_rec["hunks"][0]["target_start"] == 0
    assert file_rec["hunks"][0]["target_length"] == 0
    assert file_rec["hunks"][0]["removed_source_lines"] == [1, 2]
    assert file_rec["hunks"][0]["added_target_lines"] == []
    dumped = json.dumps(record)
    assert "/dev/null" not in dumped


def test_added_vs_removed_line_coordinates():
    record = _record("line_coordinates")
    hunk = record["gold_edit_files"][0]["hunks"][0]
    assert hunk["source_start"] == 10
    assert hunk["source_length"] == 5
    assert hunk["target_start"] == 10
    assert hunk["target_length"] == 6
    assert hunk["removed_source_lines"] == [11, 12]
    assert hunk["added_target_lines"] == [11, 12, 13]
    assert hunk["removed_source_lines"] != hunk["added_target_lines"]
    assert "oracle_lines" not in record["gold_edit_files"][0]


def test_test_patch_does_not_enter_oracle():
    cases = _load_cases()
    row = cases["test_patch_isolation"]
    assert "tests/test_foo.py" in row["test_patch"]
    record = oracle_record_from_row(row)
    paths = [item["path"] for item in record["gold_edit_files"]]
    assert paths == ["src/foo.py"]
    assert "tests/test_foo.py" not in paths
    dumped = json.dumps(record)
    assert "tests/test_foo.py" not in dumped
    assert "secret_hint.py" not in dumped

    mutated = dict(row)
    mutated["test_patch"] = cases["added_dev_null"]["patch"]
    mutated["hints_text"] = "src/leaked.py"
    mutated["FAIL_TO_PASS"] = ["src/leaked.py::test_x"]
    mutated["PASS_TO_PASS"] = ["src/leaked.py::test_y"]
    assert oracle_record_from_row(mutated)["gold_edit_files"] == record["gold_edit_files"]


def test_extract_oracle_from_patch_signature_is_patch_only():
    parameters = inspect.signature(extract_oracle_from_patch).parameters
    assert list(parameters) == ["patch"]


def test_deterministic_oracle_files_order():
    record = _record("deterministic_order")
    paths = [item["path"] for item in record["gold_edit_files"]]
    assert paths == ["a.py", "z.py"]
    again = _record("deterministic_order")
    assert [item["path"] for item in again["gold_edit_files"]] == paths


def test_malformed_diff_is_explicit_failure():
    record = _record("malformed")
    assert record["parse_ok"] is False
    assert record["parse_error"]
    assert record["gold_edit_files"] == []
    assert record["base_changed_files"] == []
    assert record["n_gold_edit_files"] == 0
    assert record["instance_id"] == "owner__malformed-9"


def test_empty_patch_is_explicit_failure():
    record = _record("empty_patch")
    assert record["parse_ok"] is False
    assert "empty patch" in record["parse_error"]
    assert record["gold_edit_files"] == []
    assert record["base_changed_files"] == []


def test_hunkless_rename():
    record = _record("hunkless_rename")
    assert record["parse_ok"] is True
    file_rec = record["gold_edit_files"][0]
    assert file_rec["operation"] == "path_changed"
    assert file_rec["path"] == "new_name.py"
    assert file_rec["source_path"] == "old_name.py"
    assert file_rec["target_path"] == "new_name.py"
    assert file_rec["num_hunks"] == 0
    assert file_rec["hunks"] == []
    assert file_rec["n_added_lines"] == 0
    assert file_rec["n_removed_lines"] == 0


def test_test_like_gold_path_is_observed_not_dropped():
    record = _record("test_like_observed")
    assert record["parse_ok"] is True
    paths = [item["path"] for item in record["gold_edit_files"]]
    assert "src/foo.py" in paths
    assert "tests/test_foo.py" in paths
    assert record["test_like_gold_paths"] == ["tests/test_foo.py"]
    assert is_test_like_path("tests/test_foo.py")
    assert is_test_like_path("test/helper.py")
    assert is_test_like_path("pkg/test_api.py")
    assert is_test_like_path("pkg/api_test.py")
    assert not is_test_like_path("src/foo.py")


def test_normalize_strips_git_prefixes_and_rejects_dev_null():
    assert normalize_diff_path("a/src/foo.py") == "src/foo.py"
    assert normalize_diff_path("b/src/foo.py") == "src/foo.py"
    assert normalize_diff_path("/dev/null") is None
    assert normalize_diff_path("a/foo.py") == "foo.py"


def test_frame_does_not_drop_failed_rows():
    cases = _load_cases()
    frame = pd.DataFrame(list(cases.values()))
    records = extract_oracle_frame(frame)
    summary = build_oracle_summary(records)
    assert len(records) == len(frame)
    assert summary["n_rows"] == len(frame)
    assert summary["rows_dropped"] == 0
    assert summary["parse_failure_count"] == 2
    assert "owner__malformed-9" in summary["parse_failure_instance_ids"]
    assert "owner__empty-10" in summary["parse_failure_instance_ids"]
    assert summary["successfully_parsed"] == len(frame) - 2
    assert "n_flagged_rows" not in summary
    assert summary["instance_file_views"]["n_added_only"] == 1
    assert summary["instance_file_views"]["n_zero_base_changed_files"] == 1
    assert "oracle_files" not in records[0]


def test_added_only_has_gold_edit_but_empty_base_view():
    record = _record("added_dev_null")
    assert [item["path"] for item in record["gold_edit_files"]] == ["src/new.py"]
    assert record["n_gold_edit_files"] == 1
    assert record["base_changed_files"] == []
    assert record["n_base_changed_files"] == 0


def test_modified_file_is_in_both_views():
    record = _record("modified_single")
    assert [item["path"] for item in record["gold_edit_files"]] == ["src/foo.py"]
    assert record["base_changed_files"] == ["src/foo.py"]


def test_deleted_file_base_view_uses_source_path():
    record = _record("deleted_dev_null")
    assert [item["path"] for item in record["gold_edit_files"]] == ["src/old.py"]
    assert record["gold_edit_files"][0]["source_path"] == "src/old.py"
    assert record["base_changed_files"] == ["src/old.py"]


def test_path_changed_base_uses_source_gold_edit_uses_target():
    record = _record("hunkless_rename")
    file_rec = record["gold_edit_files"][0]
    assert file_rec["operation"] == "path_changed"
    assert file_rec["path"] == "new_name.py"
    assert file_rec["source_path"] == "old_name.py"
    assert record["base_changed_files"] == ["old_name.py"]
    assert "new_name.py" not in record["base_changed_files"]
