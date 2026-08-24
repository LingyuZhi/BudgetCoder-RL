"""Exploration tool contract tests on synthetic M2A snapshots."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from budget_coder_rl.data.swe_gym_repos import cache_key, resolve_commit, run_git
from budget_coder_rl.env import (
    ExplorationTools,
    RepoEnvironment,
    RepoFileView,
    TaskRef,
    ToolError,
)
from budget_coder_rl.env.tools import READ_MAX_LINES, SEARCH_MAX_RESULTS


def _init_src(src: Path) -> None:
    src.mkdir()
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["config", "commit.gpgsign", "false"], cwd=src, allow_network=True)


def _commit_all(src: Path, message: str) -> str:
    run_git(["commit", "-m", message], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None
    return sha


def _workspace(tmp_path: Path) -> tuple[ExplorationTools, Path]:
    src = tmp_path / "src"
    _init_src(src)
    (src / "pkg.py").write_text("version = 1\nneedle line\n", encoding="utf-8")
    nested = src / "nested"
    nested.mkdir()
    (nested / "keep.txt").write_text("stable needle\n", encoding="utf-8")
    (src / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (src / "ignored.txt").write_text("IGNORED_UNIQUE\n", encoding="utf-8")
    (src / ".secret.txt").write_text("HIDDEN_UNIQUE\n", encoding="utf-8")
    (src / "binary.bin").write_bytes(b"hello\x00world")
    (src / "inside_link").symlink_to("pkg.py")
    (src / "escape").symlink_to("/etc/passwd")
    (src / "dirlink").symlink_to("nested")
    run_git(["add", "-A"], cwd=src, allow_network=True)
    run_git(["add", "-f", "ignored.txt"], cwd=src, allow_network=True)
    sha = _commit_all(src, "base")
    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(["clone", "--bare", str(src), str(dest)], cwd=repos_root, allow_network=True)
    env = RepoEnvironment(repos_root=repos_root, snapshots_root=tmp_path / "snapshots")
    ws = env.prepare(TaskRef("owner__repo-1", "owner/repo", sha))
    return ExplorationTools(ws.view()), ws.repo_root


def test_tree_is_deterministic_and_does_not_follow_symlinks(tmp_path: Path):
    tools, _root = _workspace(tmp_path)
    first = tools.tree(".", 2)
    second = tools.tree(".", 2)
    assert first == second
    assert first.startswith("# bcrl-obs-v1\n")
    assert "status: ok\n" in first
    assert "l escape\n" in first
    assert "l inside_link\n" in first
    assert "l dirlink\n" in first
    assert "f nested/keep.txt\n" in first
    assert "d nested\n" in first
    # dirlink must not expand nested contents a second time.
    assert first.count("nested/keep.txt") == 1
    lines = [line for line in first.splitlines() if line[:2] in {"d ", "f ", "l "}]
    rels = [line[2:] for line in lines]
    assert rels == sorted(rels)


def test_search_uses_snapshot_view_not_gitignore(tmp_path: Path):
    tools, _root = _workspace(tmp_path)
    ignored = tools.search("IGNORED_UNIQUE")
    assert "ignored.txt:1:IGNORED_UNIQUE" in ignored
    hidden = tools.search("HIDDEN_UNIQUE")
    assert ".secret.txt:1:HIDDEN_UNIQUE" in hidden
    version = tools.search("version")
    assert "pkg.py:" in version
    assert "inside_link:" not in version
    escape_hits = tools.search("root:")
    assert "escape:" not in escape_hits
    assert "/etc/" not in escape_hits
    first = tools.search("needle")
    second = tools.search("needle")
    assert first == second
    assert "pkg.py:2:needle line" in first
    assert "nested/keep.txt:1:stable needle" in first


def test_search_skips_binary_and_rejects_empty_query(tmp_path: Path):
    tools, _root = _workspace(tmp_path)
    hits = tools.search("hello")
    assert "binary.bin" not in hits
    with pytest.raises(ToolError) as exc:
        tools.search("")
    assert exc.value.code == "invalid_argument"
    with pytest.raises(ToolError) as exc:
        tools.search("x", max_results=0)
    assert exc.value.code == "invalid_argument"
    with pytest.raises(ToolError) as exc:
        tools.search("x", max_results=SEARCH_MAX_RESULTS + 1)
    assert exc.value.code == "invalid_argument"


def test_search_truncation_and_path_errors(tmp_path: Path):
    tools, _root = _workspace(tmp_path)
    truncated = tools.search("needle", max_results=1)
    assert "truncated: true\n" in truncated
    assert "match_count: 1\n" in truncated
    with pytest.raises(ToolError) as exc:
        tools.search("needle", path="../etc")
    assert exc.value.code == "path_escape"
    with pytest.raises(ToolError) as exc:
        tools.search("needle", path="missing")
    assert exc.value.code == "path_not_found"


def test_read_bounded_and_deterministic(tmp_path: Path):
    tools, _root = _workspace(tmp_path)
    first = tools.read("pkg.py", 1, 2)
    second = tools.read("pkg.py", 1, 2)
    assert first == second
    assert "     1|version = 1" in first
    assert "     2|needle line" in first
    assert "truncated: false\n" in first
    # end past EOF clamps.
    clamped = tools.read("pkg.py", 1, 99)
    assert "end_line: 2\n" in clamped
    assert "truncated: false\n" in clamped
    followed = tools.read("inside_link", 1, 1)
    assert "version = 1" in followed


def test_read_errors_and_truncation(tmp_path: Path):
    tools, _root = _workspace(tmp_path)
    with pytest.raises(ToolError) as exc:
        tools.read("escape", 1, 1)
    assert exc.value.code == "path_escape"
    with pytest.raises(ToolError) as exc:
        tools.read("nope.py", 1, 1)
    assert exc.value.code == "path_not_found"
    with pytest.raises(ToolError) as exc:
        tools.read("nested", 1, 1)
    assert exc.value.code == "not_a_file"
    with pytest.raises(ToolError) as exc:
        tools.read("pkg.py", 0, 1)
    assert exc.value.code == "invalid_range"
    with pytest.raises(ToolError) as exc:
        tools.read("pkg.py", 3, 1)
    assert exc.value.code == "invalid_range"
    with pytest.raises(ToolError) as exc:
        tools.read("pkg.py", 9, 10)
    assert exc.value.code == "invalid_range"
    with pytest.raises(ToolError) as exc:
        tools.read("binary.bin", 1, 1)
    assert exc.value.code == "undecodable"
    long_range = tools.read("pkg.py", 1, READ_MAX_LINES + 50)
    assert "truncated: false\n" in long_range
    assert "end_line: 2\n" in long_range


def _entry_rels(observation: str) -> list[str]:
    _, body = observation.split("---\n", 1)
    return [line[2:] for line in body.splitlines() if line[:2] in {"d ", "f ", "l "}]


def _search_paths(observation: str) -> list[str]:
    _, body = observation.split("---\n", 1)
    return [line.split(":", 1)[0] for line in body.splitlines() if line]


def test_tree_truncates_sorted_prefix_not_walk_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src = tmp_path / "src"
    _init_src(src)
    (src / "zzz.txt").write_text("z\n", encoding="utf-8")
    (src / "aaa.txt").write_text("a\n", encoding="utf-8")
    nested = src / "nested"
    nested.mkdir()
    (nested / "mmm.txt").write_text("m\n", encoding="utf-8")
    (nested / "bbb.txt").write_text("b\n", encoding="utf-8")
    run_git(["add", "-A"], cwd=src, allow_network=True)
    sha = _commit_all(src, "ordered")
    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(["clone", "--bare", str(src), str(dest)], cwd=repos_root, allow_network=True)
    env = RepoEnvironment(repos_root=repos_root, snapshots_root=tmp_path / "snapshots")
    tools = ExplorationTools(env.prepare(TaskRef("owner__repo-1", "owner/repo", sha)).view())
    monkeypatch.setattr("budget_coder_rl.env.tools.TREE_MAX_ENTRIES", 4)
    observation = tools.tree(".", 2)
    assert "truncated: true\n" in observation
    rels = _entry_rels(observation)
    assert rels == [".", "aaa.txt", "nested", "nested/bbb.txt"]
    assert "zzz.txt" not in rels


def test_search_truncates_sorted_file_order(tmp_path: Path):
    src = tmp_path / "src"
    _init_src(src)
    (src / "zzz.py").write_text("TOKEN\n", encoding="utf-8")
    (src / "aaa.py").write_text("TOKEN\n", encoding="utf-8")
    run_git(["add", "-A"], cwd=src, allow_network=True)
    sha = _commit_all(src, "search-order")
    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(["clone", "--bare", str(src), str(dest)], cwd=repos_root, allow_network=True)
    env = RepoEnvironment(repos_root=repos_root, snapshots_root=tmp_path / "snapshots")
    tools = ExplorationTools(env.prepare(TaskRef("owner__repo-1", "owner/repo", sha)).view())
    observation = tools.search("TOKEN", max_results=1)
    assert "truncated: true\n" in observation
    assert _search_paths(observation) == ["aaa.py"]


def _locked_view(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "view"
    root.mkdir()
    (root / "ok.py").write_text("needle\n", encoding="utf-8")
    locked = root / "locked"
    locked.mkdir()
    (locked / "hid.py").write_text("needle\n", encoding="utf-8")
    os.chmod(locked, 0o000)
    try:
        os.listdir(locked)
    except OSError:
        return root, locked
    os.chmod(locked, 0o755)
    pytest.skip("process can list chmod 000 directories")


def test_search_walk_oserror_is_not_ignored(tmp_path: Path):
    root, locked = _locked_view(tmp_path)
    tools = ExplorationTools(RepoFileView(root))
    try:
        with pytest.raises(OSError):
            tools.search("needle")
    finally:
        os.chmod(locked, 0o755)


def test_tree_scandir_oserror_is_not_ignored(tmp_path: Path):
    root, locked = _locked_view(tmp_path)
    tools = ExplorationTools(RepoFileView(root))
    try:
        with pytest.raises(OSError):
            tools.tree(".", 2)
    finally:
        os.chmod(locked, 0o755)
