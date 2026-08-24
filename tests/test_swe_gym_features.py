"""Unit tests for M1D-A eligibility / structural-difficulty features.

Uses crafted fixtures and a tiny git repo. Does not read the official parquet
or the full SWE-Gym Git mirrors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from budget_coder_rl.data.swe_gym_fields import PRIVILEGED_FIELDS, agent_task_view
from budget_coder_rl.data.swe_gym_features import (
    DERIVED_PRIVILEGED_FEATURE_FIELDS,
    FORBIDDEN_OUTPUT_KEYS,
    basename_is_mentioned,
    build_feature_summary,
    classify_modality,
    correlation_assignments,
    extract_feature_frame,
    path_is_mentioned,
    qualified_symbol_is_mentioned,
    symbol_name_is_mentioned,
)
from budget_coder_rl.data.swe_gym_repos import OfflineGitError, resolve_commit, run_git
from budget_coder_rl.data.swe_gym_tree_stats import (
    CODE_LIKE_EXTENSIONS,
    PYTHON_AST_EXTENSIONS,
    TreeStatStore,
    TreeStats,
    is_code_like_path,
    is_docs_path,
    is_python_ast_path,
    parse_ls_tree_output,
    run_ls_tree,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "data" / "fixtures" / "swe_gym_m1d_features.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_code_like_extension_classification():
    assert is_python_ast_path("src/foo.py")
    assert is_python_ast_path("src/foo.pyi")
    assert is_code_like_path("src/foo.pyx")
    assert is_code_like_path("src/foo.c")
    assert is_code_like_path("src/foo.h")
    assert not is_code_like_path("docs/readme.rst")
    assert not is_code_like_path("ui/app.js")
    assert not is_code_like_path("config.yaml")
    assert not is_code_like_path("notebook.ipynb")
    assert is_docs_path("docs/readme.rst")
    assert is_docs_path("README.md")
    assert ".js" not in CODE_LIKE_EXTENSIONS
    assert PYTHON_AST_EXTENSIONS <= CODE_LIKE_EXTENSIONS


def test_ls_tree_parsing_counts_blobs_and_skips_gitlink():
    stats = parse_ls_tree_output(_fixture()["ls_tree_sample"])
    assert stats.ok is True
    assert stats.repo_tracked_files == 7
    assert stats.repo_python_files == 3
    assert stats.repo_code_like_files == 4
    assert stats.repo_tracked_blob_bytes == 120 + 40 + 80 + 16 + 8 + 32 + 11


def test_tree_stats_store_caches_by_repo_commit(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "pkg.py").write_text("x = 1\n", encoding="utf-8")
    (src / "ext.pyx").write_text("cdef int y\n", encoding="utf-8")
    (src / "README.md").write_text("# hi\n", encoding="utf-8")
    (src / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["add", "."], cwd=src, allow_network=True)
    run_git(["commit", "-m", "init"], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None

    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / "owner__repo.git"
    run_git(["clone", "--bare", str(src), str(dest)], cwd=repos_root, allow_network=True)
    store = TreeStatStore(repos_root)
    first = store.stats("owner/repo", sha)
    second = store.stats("owner/repo", sha)
    assert first is second
    assert store.cache_size() == 1
    assert first.ok is True
    assert first.repo_tracked_files == 4
    assert first.repo_python_files == 1
    assert first.repo_code_like_files == 2
    missing = store.stats("owner/repo", "deadbeef")
    assert missing.ok is False
    assert missing.error == "commit not resolved"


def test_ls_tree_allowed_offline_fetch_still_rejected(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "pkg.py").write_text("x = 1\n", encoding="utf-8")
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["add", "pkg.py"], cwd=src, allow_network=True)
    run_git(["commit", "-m", "init"], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None
    output = run_ls_tree(src, sha)
    assert "pkg.py" in output
    with pytest.raises(OfflineGitError):
        run_git(["fetch", "origin"], cwd=src, allow_network=False)


def test_docs_only_vs_mixed_modality():
    docs = classify_modality(["docs/readme.rst"], 0)
    assert docs["docs_only_target"] is True
    assert docs["non_code_only_target"] is True
    assert docs["has_code_like_target"] is False
    assert docs["has_python_ast_target"] is False
    assert docs["symbol_oracle_available"] is False
    assert docs["mixed_code_noncode_target"] is False
    assert "keep" not in docs

    mixed = classify_modality(["src/foo.py", "docs/readme.rst"], 1)
    assert mixed["has_python_ast_target"] is True
    assert mixed["has_code_like_target"] is True
    assert mixed["symbol_oracle_available"] is True
    assert mixed["docs_only_target"] is False
    assert mixed["non_code_only_target"] is False
    assert mixed["mixed_code_noncode_target"] is True

    empty = classify_modality([], 0)
    assert empty["docs_only_target"] is False
    assert empty["non_code_only_target"] is False


def test_full_path_and_basename_mentions():
    cases = _fixture()["mentions"]
    assert path_is_mentioned(cases["full_path_issue"], "src/foo.py") is True
    assert basename_is_mentioned(cases["full_path_issue"], "src/foo.py") is True
    assert path_is_mentioned(cases["basename_only_issue"], "src/foo.py") is False
    assert basename_is_mentioned(cases["basename_only_issue"], "src/foo.py") is True
    assert path_is_mentioned("Please inspect SRC\\FOO.PY today.", "src/foo.py") is True


def test_symbol_and_qualified_mentions():
    cases = _fixture()["mentions"]
    assert symbol_name_is_mentioned(cases["symbol_issue"], "invalidate") is True
    assert qualified_symbol_is_mentioned(cases["qualified_issue"], "Cache.invalidate") is True
    assert qualified_symbol_is_mentioned(cases["symbol_issue"], "foo") is False
    assert symbol_name_is_mentioned("call __init__ please", "__init__") is False
    assert symbol_name_is_mentioned("use x here", "x") is False


def test_substring_false_positives():
    cases = _fixture()["false_positives"]
    assert path_is_mentioned(cases["foo_pyi"], "src/foo.py") is False
    assert basename_is_mentioned(cases["foo_pyi"], "src/foo.py") is False
    assert basename_is_mentioned(cases["myfoo"], "src/foo.py") is False
    assert symbol_name_is_mentioned(cases["partial_symbol"], "invalidate") is False
    assert path_is_mentioned("see pkg/src/foo.py", "src/foo.py") is False


def test_correlation_transitive_connected_component():
    identities = [
        {
            "instance_id": "a",
            "repo": "r/x",
            "base_commit": "c1",
            "problem_statement": "issue one",
        },
        {
            "instance_id": "b",
            "repo": "r/x",
            "base_commit": "c1",
            "problem_statement": "issue two",
        },
        {
            "instance_id": "c",
            "repo": "r/y",
            "base_commit": "c2",
            "problem_statement": "issue two",
        },
        {
            "instance_id": "d",
            "repo": "r/z",
            "base_commit": "c3",
            "problem_statement": "unique",
        },
    ]
    assigned = correlation_assignments(identities)
    assert assigned["a"][0] == assigned["b"][0] == assigned["c"][0]
    assert assigned["a"][1] == 3
    assert assigned["d"][0] == "cg:d"
    assert assigned["d"][1] == 1
    assert assigned["a"][0] == "cg:a"


def test_pandas_duplicate_pair_style_grouping():
    identities = [
        {
            "instance_id": "pandas-dev__pandas-53805",
            "repo": "pandas-dev/pandas",
            "base_commit": "54bf475fd4d38a08a353a47e44dfecce24cdfb4b",
            "problem_statement": "same issue text",
        },
        {
            "instance_id": "pandas-dev__pandas-53830",
            "repo": "pandas-dev/pandas",
            "base_commit": "54bf475fd4d38a08a353a47e44dfecce24cdfb4b",
            "problem_statement": "same issue text",
        },
    ]
    assigned = correlation_assignments(identities)
    left = assigned["pandas-dev__pandas-53805"]
    right = assigned["pandas-dev__pandas-53830"]
    assert left[0] == right[0] == "cg:pandas-dev__pandas-53805"
    assert left[1] == right[1] == 2


def _minimal_oracle(instance_id: str, files: list[str]) -> dict:
    return {
        "instance_id": instance_id,
        "parse_ok": True,
        "base_changed_files": files,
        "gold_edit_files": [{"path": path, "operation": "modified"} for path in files],
        "n_gold_edit_files": len(files),
        "n_hunks": 1,
        "n_added_lines": 1,
        "n_removed_lines": 1,
    }


def _minimal_symbol(
    instance_id: str,
    files: list[str],
    *,
    base_commit: str = "abc",
    parse_ok: bool = True,
    repo_ok: bool = True,
    commit_ok: bool = True,
    n_blob_missing: int = 0,
    oracle_symbols: list | None = None,
    n_oracle_symbols: int | None = None,
) -> dict:
    symbols = oracle_symbols or []
    count = len(symbols) if n_oracle_symbols is None else n_oracle_symbols
    return {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "parse_ok": parse_ok,
        "repo_ok": repo_ok,
        "commit_ok": commit_ok,
        "n_blob_missing": n_blob_missing,
        "base_changed_files": files,
        "oracle_symbols": symbols,
        "n_oracle_symbols": count,
    }


def test_extract_frame_does_not_drop_and_serializes_deterministically():
    rows = [
        {
            "instance_id": "repo__2",
            "repo": "owner/name",
            "base_commit": "c1",
            "problem_statement": "fix src/foo.py Cache.invalidate",
            "patch": "SECRET_PATCH",
        },
        {
            "instance_id": "repo__1",
            "repo": "owner/name",
            "base_commit": "c1",
            "problem_statement": "another issue",
            "patch": "SECRET_PATCH",
        },
        {
            "instance_id": "repo__3",
            "repo": "owner/other",
            "base_commit": "c2",
            "problem_statement": "docs only",
            "patch": "SECRET_PATCH",
        },
    ]
    files_a = ["src/foo.py"]
    files_b = ["docs/readme.rst"]
    oracle = [
        _minimal_oracle("repo__2", files_a),
        _minimal_oracle("repo__1", files_a),
        _minimal_oracle("repo__3", files_b),
    ]
    symbols = [
        _minimal_symbol(
            "repo__2",
            files_a,
            base_commit="c1",
            oracle_symbols=[{"qualname": "Cache.invalidate"}],
        ),
        _minimal_symbol("repo__1", files_a, base_commit="c1", n_oracle_symbols=0),
        _minimal_symbol("repo__3", files_b, base_commit="c2", n_oracle_symbols=0),
    ]
    tree = {
        ("owner/name", "c1"): TreeStats(True, None, 100, 80, 85, 1000),
        ("owner/other", "c2"): TreeStats(True, None, 50, 10, 10, 200),
    }
    records = extract_feature_frame(rows, oracle, symbols, tree)
    assert len(records) == 3
    assert [item["instance_id"] for item in records] == ["repo__1", "repo__2", "repo__3"]
    dumped = json.dumps(records[0], sort_keys=True, ensure_ascii=True)
    assert dumped == json.dumps(records[0], sort_keys=True, ensure_ascii=True)
    assert "SECRET_PATCH" not in dumped
    assert "keep" not in dumped
    assert "split" not in dumped
    assert "difficulty" not in dumped
    for key in FORBIDDEN_OUTPUT_KEYS:
        assert key not in records[0]
    assert records[0]["correlation_group_id"] == records[1]["correlation_group_id"]
    assert records[0]["correlation_group_size"] == 2
    assert records[2]["modality"]["docs_only_target"] is True
    assert records[1]["hint_strength"]["gold_full_path_mentioned"] is True
    assert records[1]["hint_strength"]["gold_qualified_symbol_mentioned"] is True
    assert records[1]["search_space"]["file_target_density"] == 0.01
    summary = build_feature_summary(records)
    assert summary["rows_dropped"] == 0
    assert summary["n_rows"] == 3
    assert summary["notes"]["not_m1d_b"] is True
    assert summary["codescout_comparison"]["status"] == "skipped"
    assert summary["notes"]["no_split"] is True
    assert summary["notes"]["no_keep_drop"] is True
    assert "split" not in summary
    assert "keep" not in summary


def test_zero_density_and_invalid_reasons_are_recorded_not_dropped():
    rows = [
        {
            "instance_id": "bad__1",
            "repo": "owner/name",
            "base_commit": "c1",
            "problem_statement": "   ",
        }
    ]
    oracle = [_minimal_oracle("bad__1", [])]
    symbols = [
        _minimal_symbol(
            "bad__1",
            [],
            base_commit="c1",
            parse_ok=False,
            repo_ok=False,
            commit_ok=False,
            n_blob_missing=1,
        )
    ]
    tree = {("owner/name", "c1"): TreeStats(False, "boom", 0, 0, 0, None)}
    records = extract_feature_frame(rows, oracle, symbols, tree)
    assert len(records) == 1
    record = records[0]
    assert record["technical_valid"] is False
    assert "empty_problem_statement" in record["technical_invalid_reasons"]
    assert "zero_base_changed_files" in record["technical_invalid_reasons"]
    assert "tree_stats_unavailable" in record["technical_invalid_reasons"]
    assert record["search_space"]["file_target_density"] is None


def test_agent_task_view_excludes_m1d_derived_fields():
    row = {
        "problem_statement": "fix the bug",
        "patch": "secret-patch",
        "hints_text": "secret-hint",
        "search_space": {"repo_tracked_files": 9},
        "target_scope": {"oracle_symbol_count": 2},
        "hint_strength": {"gold_full_path_mentioned": True},
        "modality": {"has_python_ast_target": True},
        "correlation_group_id": "cg:x",
        "technical_valid": True,
    }
    view = agent_task_view(row)
    assert list(view.keys()) == ["problem_statement"]
    leaked = json.dumps(view)
    for name in DERIVED_PRIVILEGED_FEATURE_FIELDS:
        assert name not in view
        assert name not in leaked
    for name in PRIVILEGED_FIELDS:
        assert name not in view
    assert "secret-patch" not in leaked
    assert "keep" not in view
