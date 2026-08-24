"""Unit tests for M1C-B base-repository symbol oracles.

Uses synthetic sources and unified diffs. Does not read the official parquet
or the full SWE-Gym Git mirrors.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from budget_coder_rl.data.swe_gym_fields import (
    DERIVED_PRIVILEGED_ORACLE_FIELDS,
    PRIVILEGED_FIELDS,
    agent_task_view,
)
from budget_coder_rl.data.swe_gym_oracle import extract_oracle_from_patch
from budget_coder_rl.data.swe_gym_repos import (
    OfflineGitError,
    blob_exists,
    cache_key,
    clone_or_update_mirror,
    github_remote,
    github_ssh_remote,
    is_safe_repo_path,
    read_blob,
    resolve_commit,
    run_git,
)
from budget_coder_rl.data.swe_gym_symbol_oracle import (
    build_symbol_summary,
    extract_symbol_frame,
    extract_symbol_record,
)
from budget_coder_rl.data.swe_gym_symbols import (
    EVIDENCE_ADDITION_ONE_SIDED,
    EVIDENCE_ADDITION_SAME,
    EVIDENCE_REMOVED,
    REASON_AMBIGUOUS,
    REASON_AST,
    REASON_MODULE_LEVEL,
    REASON_UNSUPPORTED,
    extract_base_symbols,
    map_sources_and_patch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "swe_gym_m1c_symbol_oracle.json"


def _cases() -> dict[str, dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _map(name: str) -> dict:
    case = _cases()[name]
    sources = {}
    if case["source"] != "" or case["path"] in {
        "src/foo.py",
        "src/foo.pyx",
    }:
        if name != "added_file_new_function":
            sources[case["path"]] = case["source"].encode("utf-8")
    return map_sources_and_patch(case["patch"], sources)


def _qualnames(result: dict) -> list[str]:
    return [item["qualname"] for item in result["oracle_symbols"]]


def test_top_level_function_body_modification():
    result = _map("top_level_function")
    assert result["parse_ok"] is True
    assert _qualnames(result) == ["foo"]
    symbol = result["oracle_symbols"][0]
    assert symbol["kind"] == "function"
    assert symbol["path"] == "src/foo.py"
    assert symbol["start_line"] == 1
    assert symbol["end_line"] == 3
    assert EVIDENCE_REMOVED in symbol["evidence"]
    assert result["counts"]["n_removed_line_mapped"] == 1


def test_class_method_modification():
    result = _map("class_method")
    assert _qualnames(result) == ["Foo.bar"]
    symbol = result["oracle_symbols"][0]
    assert symbol["kind"] == "function"
    assert symbol["depth"] == 1
    assert EVIDENCE_REMOVED in symbol["evidence"]


def test_nested_function_innermost_symbol():
    result = _map("nested_innermost")
    assert _qualnames(result) == ["outer.inner"]
    symbol = result["oracle_symbols"][0]
    assert symbol["depth"] == 1
    assert "outer" not in _qualnames(result) or _qualnames(result) == ["outer.inner"]


def test_class_body_attribute_maps_to_class():
    result = _map("class_body_attribute")
    assert _qualnames(result) == ["Foo"]
    assert result["oracle_symbols"][0]["kind"] == "class"
    assert "Foo.bar" not in _qualnames(result)


def test_decorated_function_includes_decorator_line():
    result = _map("decorated_function")
    assert _qualnames(result) == ["foo"]
    symbol = result["oracle_symbols"][0]
    assert symbol["start_line"] == 4
    assert EVIDENCE_REMOVED in symbol["evidence"]


def test_decorated_class_includes_decorator_line():
    result = _map("decorated_class")
    assert _qualnames(result) == ["Foo"]
    assert result["oracle_symbols"][0]["start_line"] == 4


def test_async_function():
    result = _map("async_function")
    assert _qualnames(result) == ["foo"]
    assert result["oracle_symbols"][0]["kind"] == "async_function"


def test_pure_addition_inside_same_function():
    result = _map("addition_inside_function")
    assert _qualnames(result) == ["foo"]
    symbol = result["oracle_symbols"][0]
    assert symbol["evidence"] == [EVIDENCE_ADDITION_SAME]
    assert result["counts"]["n_addition_anchor_same_symbol"] == 1
    assert result["counts"]["n_addition_anchor_one_sided"] == 0
    assert result["unmapped_sites"] == []


def test_addition_between_top_level_symbols_is_unmapped():
    result = _map("addition_between_symbols")
    assert result["oracle_symbols"] == []
    assert "bar" not in _qualnames(result)
    assert result["counts"]["n_ambiguous_addition_anchor"] == 1
    assert result["unmapped_sites"][0]["reason"] == REASON_AMBIGUOUS


def test_module_level_constant_has_no_symbol():
    result = _map("module_level_constant")
    assert result["oracle_symbols"] == []
    assert result["counts"]["n_module_level"] >= 1
    assert all(site["reason"] == REASON_MODULE_LEVEL for site in result["unmapped_sites"])


def test_newly_added_function_is_not_a_base_oracle_symbol():
    result = _map("added_file_new_function")
    assert result["base_changed_files"] == []
    assert result["n_skipped_added_files"] == 1
    assert result["oracle_symbols"] == []
    assert "bar" not in _qualnames(result)


def test_unsupported_pyx_keeps_file_oracle_without_symbols():
    result = _map("unsupported_pyx")
    assert result["file_results"][0]["status"] == REASON_UNSUPPORTED
    assert result["file_results"][0]["eligible"] is False
    assert result["oracle_symbols"] == []
    assert result["unmapped_sites"][0]["reason"] == REASON_UNSUPPORTED


def test_ast_syntax_error_is_explicit_failure():
    result = _map("syntax_error")
    assert result["file_results"][0]["status"] == REASON_AST
    assert result["file_results"][0]["error"]
    assert "SyntaxError" in result["file_results"][0]["error"]
    assert result["oracle_symbols"] == []
    assert result["unmapped_sites"][0]["reason"] == REASON_AST


def test_deterministic_symbol_ordering_and_identity():
    result = _map("deterministic_order")
    names = _qualnames(result)
    assert names == ["zzz", "aaa"]
    ids = [item["symbol_id"] for item in result["oracle_symbols"]]
    assert ids == sorted(
        ids,
        key=lambda _id: (
            result["oracle_symbols"][ids.index(_id)]["start_line"],
            result["oracle_symbols"][ids.index(_id)]["qualname"],
        ),
    )
    again = _map("deterministic_order")
    assert [item["symbol_id"] for item in again["oracle_symbols"]] == ids
    assert result["oracle_symbols"][0]["symbol_id"] != result["oracle_symbols"][1]["symbol_id"]


def test_one_sided_addition_is_weaker_and_not_conservative():
    result = _map("one_sided_addition")
    assert _qualnames(result) == ["foo"]
    assert result["oracle_symbols"][0]["evidence"] == [EVIDENCE_ADDITION_ONE_SIDED]
    assert result["counts"]["n_addition_anchor_same_symbol"] == 0
    assert result["counts"]["n_addition_anchor_one_sided"] == 1


def test_extract_oracle_from_patch_signature_is_still_patch_only():
    parameters = inspect.signature(extract_oracle_from_patch).parameters
    assert list(parameters) == ["patch"]


def test_agent_task_view_excludes_derived_symbol_fields():
    row = {
        "problem_statement": "fix the bug",
        "patch": "secret-patch",
        "hints_text": "secret-hint",
        "oracle_symbols": [{"qualname": "Foo.bar"}],
        "unmapped_sites": [{"reason": "module_level"}],
        "gold_edit_files": [{"path": "src/foo.py"}],
        "base_changed_files": ["src/foo.py"],
        "file_results": [{"path": "src/foo.py"}],
    }
    view = agent_task_view(row)
    assert list(view.keys()) == ["problem_statement"]
    leaked = json.dumps(view)
    for name in DERIVED_PRIVILEGED_ORACLE_FIELDS:
        assert name not in view
        assert name not in leaked
    for name in PRIVILEGED_FIELDS:
        assert name not in view
    assert "Foo.bar" not in leaked
    assert "secret-patch" not in leaked


def test_frame_does_not_drop_rows():
    cases = _cases()
    rows = []
    sources_by_instance = {}
    for index, (name, case) in enumerate(cases.items()):
        instance_id = f"owner__{name}-{index}"
        rows.append(
            {
                "instance_id": instance_id,
                "repo": "owner/repo",
                "base_commit": "abc",
                "patch": case["patch"],
                "problem_statement": name,
            }
        )
        if name != "added_file_new_function":
            sources_by_instance[instance_id] = {
                case["path"]: case["source"].encode("utf-8")
            }
    frame = pd.DataFrame(rows)
    records = extract_symbol_frame(frame, None, sources_by_instance=sources_by_instance)
    summary = build_symbol_summary(records)
    assert len(records) == len(frame)
    assert summary["n_rows"] == len(frame)
    assert summary["rows_dropped"] == 0
    assert summary["notes"]["not_m1d"] is True
    assert "coverage" not in summary


def test_extract_symbol_record_reads_only_needed_fields():
    case = _cases()["top_level_function"]
    record = extract_symbol_record(
        {
            "instance_id": "owner__foo-1",
            "repo": "owner/repo",
            "base_commit": "abc",
            "patch": case["patch"],
            "test_patch": "diff --git a/tests/test_foo.py b/tests/test_foo.py\n",
            "hints_text": "do not use",
        },
        None,
        sources={case["path"]: case["source"].encode("utf-8")},
    )
    dumped = json.dumps(record)
    assert "do not use" not in dumped
    assert "tests/test_foo.py" not in dumped
    assert record["n_oracle_symbols"] == 1


def test_no_fake_module_symbol_is_emitted():
    symbols, n_missing = extract_base_symbols("X = 1\n", "src/foo.py")
    assert symbols == []
    assert n_missing == 0
    assert all(item.qualname != "<module>" for item in symbols)


def test_github_remote_and_cache_key_are_derived():
    assert github_remote("pandas-dev/pandas") == "https://github.com/pandas-dev/pandas.git"
    assert github_ssh_remote("pandas-dev/pandas") == "git@github.com:pandas-dev/pandas.git"
    assert cache_key("Project-MONAI/MONAI") == "Project-MONAI__MONAI.git"
    assert is_safe_repo_path("src/foo.py")
    assert not is_safe_repo_path("../etc/passwd")
    assert not is_safe_repo_path("/abs/path.py")


def test_git_blob_helpers_use_local_mirror(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "pkg.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["add", "pkg.py"], cwd=src, allow_network=True)
    run_git(["commit", "-m", "init"], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None

    mirror = tmp_path / "owner__repo.git"
    action = clone_or_update_mirror(str(src), mirror, allow_network=True)
    assert action == "cloned"
    assert resolve_commit(mirror, sha) == sha
    assert blob_exists(mirror, sha, "pkg.py")
    assert read_blob(mirror, sha, "pkg.py").startswith(b"def foo")
    assert blob_exists(mirror, sha, "missing.py") is False
    with pytest.raises(OfflineGitError):
        run_git(["fetch", "origin"], cwd=mirror, allow_network=False)


def test_bare_repo_fetches_derived_sha_from_local_origin(tmp_path: Path):
    from budget_coder_rl.data.swe_gym_repos import ensure_bare_repo, fetch_commits

    src = tmp_path / "src"
    src.mkdir()
    (src / "pkg.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["add", "pkg.py"], cwd=src, allow_network=True)
    run_git(["commit", "-m", "init"], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None

    dest = tmp_path / "owner__repo.git"
    ensure_bare_repo(dest, str(src), allow_network=True)
    resolved, missing, fetched = fetch_commits(dest, [sha], [str(src)], allow_network=True)
    assert missing == []
    assert sha in resolved
    assert fetched == [sha]
    assert blob_exists(dest, sha, "pkg.py")
    assert read_blob(dest, sha, "pkg.py").startswith(b"def foo")
