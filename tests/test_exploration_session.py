"""Mock session transition tests: protocol/tool errors do not crash."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from budget_coder_rl.data.swe_gym_repos import cache_key, resolve_commit, run_git
from budget_coder_rl.env import ExplorationSession, RepoEnvironment, TaskRef


def _init_src(src: Path) -> None:
    src.mkdir()
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["config", "commit.gpgsign", "false"], cwd=src, allow_network=True)


def _session(tmp_path: Path) -> ExplorationSession:
    src = tmp_path / "src"
    _init_src(src)
    (src / "pkg.py").write_text("version = 1\n", encoding="utf-8")
    (src / "nested").mkdir()
    (src / "nested" / "keep.txt").write_text("stable\n", encoding="utf-8")
    run_git(["add", "."], cwd=src, allow_network=True)
    run_git(["commit", "-m", "base"], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None
    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(["clone", "--bare", str(src), str(dest)], cwd=repos_root, allow_network=True)
    env = RepoEnvironment(repos_root=repos_root, snapshots_root=tmp_path / "snapshots")
    ws = env.prepare(TaskRef("owner__repo-1", "owner/repo", sha))
    return ExplorationSession(ws)


def _tool(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


def test_scripted_tree_search_read_finish(tmp_path: Path):
    session = _session(tmp_path)
    tree = session.step(_tool("tree", {"path": ".", "depth": 2}))
    assert tree.error_kind is None
    assert "f pkg.py" in tree.observation
    search = session.step(_tool("search", {"query": "version"}))
    assert "pkg.py:1:version = 1" in search.observation
    read = session.step(_tool("read", {"path": "pkg.py", "start_line": 1, "end_line": 1}))
    assert "version = 1" in read.observation
    finish = session.step(
        _final({"locations": [{"path": "pkg.py", "symbol": "Scripted.placeholder"}]})
    )
    assert finish.terminal is True
    assert finish.termination == "finish"
    assert finish.submission == {
        "locations": [{"path": "pkg.py", "symbol": "Scripted.placeholder"}]
    }
    assert finish.observation.startswith("# bcrl-obs-v1\n")
    assert session.events[-1]["termination"] == "finish"
    with pytest.raises(RuntimeError, match="already terminated"):
        session.step(_tool("tree", {}))


def test_protocol_and_tool_failures_are_observations(tmp_path: Path):
    session = _session(tmp_path)
    cases = [
        ("<tool_call>{bad}</tool_call>", "protocol", "malformed_json"),
        (
            _tool("tree", {}) + _tool("search", {"query": "x"}),
            "protocol",
            "multiple_actions",
        ),
        (_tool("grep", {"query": "x"}), "protocol", "unknown_tool"),
        (_tool("read", {"path": "../etc/passwd", "start_line": 1, "end_line": 1}), "tool", "path_escape"),
        (_tool("read", {"path": "missing.py", "start_line": 1, "end_line": 1}), "tool", "path_not_found"),
        (_tool("read", {"path": "pkg.py", "start_line": 5, "end_line": 6}), "tool", "invalid_range"),
        (_final({"locations": [{"path": "../x.py"}]}), "protocol", "malformed_final"),
    ]
    for raw, kind, code in cases:
        result = session.step(raw)
        assert result.terminal is False
        assert result.submission is None
        assert result.error_kind == kind
        assert f"code: {code}\n" in result.observation
        assert "status: error\n" in result.observation
        assert result.observation.startswith("# bcrl-obs-v1\n")
    assert session.turn == len(cases)
    finish = session.step(_final({"locations": [{"path": "nested/keep.txt"}]}))
    assert finish.terminal is True
    assert finish.submission == {"locations": [{"path": "nested/keep.txt"}]}
