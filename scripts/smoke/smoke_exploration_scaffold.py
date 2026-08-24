#!/usr/bin/env python
"""CPU-only smoke: M1 task + M2A workspace + scripted M2B scaffold.

Runs tree -> search -> read -> finish without Qwen, veRL, budget, or reward.
Expected agent mistakes must become error observations, not crashes.

Usage (pinned RL conda env):

    python scripts/smoke/smoke_exploration_scaffold.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_repo_workspace import (  # noqa: E402
    PREFERRED_REPOS,
    load_task_rows,
)
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    cache_path_for_repo,
    is_git_dir,
    swe_gym_repos_root,
)
from budget_coder_rl.env import ExplorationSession, RepoEnvironment, TaskRef  # noqa: E402
from budget_coder_rl.protocol import OBS_VERSION  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--repos-root", type=Path, default=None)
    parser.add_argument("--snapshots-root", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--raw-parquet", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke" / "m2b_scaffold_trace.json",
    )
    return parser.parse_args(argv)


def select_smoke_task(rows: list[TaskRef]) -> TaskRef:
    by_repo: dict[str, list[TaskRef]] = {}
    for row in rows:
        if row.repo and row.instance_id and row.base_commit:
            by_repo.setdefault(row.repo, []).append(row)
    for repo in PREFERRED_REPOS:
        if repo in by_repo:
            return by_repo[repo][0]
    if not by_repo:
        raise SystemExit("no M1 tasks with instance_id/repo/base_commit")
    repo = sorted(by_repo)[0]
    return by_repo[repo][0]


def _tool(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict[str, Any]) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


def first_tree_file(observation: str) -> str:
    if "---\n" not in observation:
        raise SystemExit("tree observation missing body")
    body = observation.split("---\n", 1)[1]
    files: list[str] = []
    for line in body.splitlines():
        if line.startswith("f "):
            files.append(line[2:])
    if not files:
        raise SystemExit("tree observation has no regular file")
    for rel in files:
        if rel.endswith(".py"):
            return rel
    return files[0]


def require_ok(result, label: str) -> None:
    if result.error_kind is not None or not result.observation.startswith(f"# {OBS_VERSION}\n"):
        raise SystemExit(f"{label} failed:\n{result.observation}")
    if "status: error\n" in result.observation:
        raise SystemExit(f"{label} returned an error observation:\n{result.observation}")


def require_error(result, label: str, code: str) -> None:
    if result.terminal or result.submission is not None:
        raise SystemExit(f"{label}: error case terminated or submitted")
    if f"code: {code}\n" not in result.observation:
        raise SystemExit(f"{label}: expected code {code}:\n{result.observation}")
    if "status: error\n" not in result.observation:
        raise SystemExit(f"{label}: expected status error:\n{result.observation}")


def run_happy_path(session: ExplorationSession) -> dict[str, Any]:
    tree = session.step(_tool("tree", {"path": ".", "depth": 2}))
    require_ok(tree, "tree")
    rel = first_tree_file(tree.observation)
    filename = rel.rsplit("/", 1)[-1]
    query = filename[:-3] if filename.endswith(".py") else filename
    if not query:
        raise SystemExit("could not derive search query from tree file")
    search = session.step(_tool("search", {"query": query, "path": "."}))
    require_ok(search, "search")
    read = session.step(_tool("read", {"path": rel, "start_line": 1, "end_line": 20}))
    require_ok(read, "read")
    finish = session.step(
        _final({"locations": [{"path": rel, "symbol": "Scripted.placeholder"}]})
    )
    require_ok(finish, "finish")
    if not finish.terminal or finish.termination != "finish":
        raise SystemExit("finish did not terminate the session")
    if finish.submission is None:
        raise SystemExit("finish missing structured submission")
    if finish.submission["locations"][0]["path"] != rel:
        raise SystemExit("finish path was rewritten")
    if finish.submission["locations"][0]["symbol"] != "Scripted.placeholder":
        raise SystemExit("finish symbol was rewritten")
    return {
        "tree_file": rel,
        "query": query,
        "submission": finish.submission,
        "events": session.events,
    }


def run_failure_cases(session: ExplorationSession) -> list[str]:
    cases = [
        ("malformed_json", "<tool_call>{bad}</tool_call>", "malformed_json"),
        (
            "multiple_actions",
            _tool("tree", {}) + _tool("search", {"query": "x"}),
            "multiple_actions",
        ),
        ("unknown_tool", _tool("bash", {"cmd": "ls"}), "unknown_tool"),
        (
            "unsafe_path",
            _tool("read", {"path": "../etc/passwd", "start_line": 1, "end_line": 1}),
            "path_escape",
        ),
        (
            "missing_path",
            _tool("read", {"path": "no-such-file.py", "start_line": 1, "end_line": 1}),
            "path_not_found",
        ),
        (
            "invalid_range",
            _tool("read", {"path": "no-such-file.py", "start_line": 0, "end_line": 1}),
            "invalid_range",
        ),
        (
            "malformed_final",
            _final({"locations": [{"path": "../x.py"}]}),
            "malformed_final",
        ),
    ]
    codes: list[str] = []
    for label, raw, code in cases:
        result = session.step(raw)
        require_error(result, label, code)
        codes.append(code)
    return codes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(args.data_root)
    )
    env = RepoEnvironment(
        repos_root=repos_root,
        snapshots_root=(
            args.snapshots_root.expanduser() if args.snapshots_root is not None else None
        ),
        data_root=args.data_root,
    )
    rows = load_task_rows(repo_root, args.train, args.raw_parquet)
    task = select_smoke_task(rows)
    store = cache_path_for_repo(task.repo, repos_root)
    if not is_git_dir(store):
        print(f"HARD FAIL: local object store missing: {store}", file=sys.stderr)
        return 1

    print("SWE-Gym M2B exploration scaffold smoke")
    print(f"task: {task.instance_id} {task.repo} {task.base_commit}")
    workspace = env.prepare(task)
    workspace.validate()
    if (workspace.repo_root / ".git").exists():
        raise SystemExit("snapshot contains .git")

    happy = ExplorationSession(workspace)
    trace = run_happy_path(happy)
    failures = ExplorationSession(workspace)
    codes = run_failure_cases(failures)

    payload = {
        "instance_id": workspace.instance_id,
        "repo": workspace.repo,
        "base_commit": workspace.base_commit,
        "tree_file": trace["tree_file"],
        "query": trace["query"],
        "final_submission": trace["submission"],
        "termination": "finish",
        "events": trace["events"],
        "failure_codes": codes,
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS")
    print(f"  finish path={trace['tree_file']}")
    print(f"  failure codes={codes}")
    print(f"  trace={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
