#!/usr/bin/env python
"""CPU-only smoke: M1 task metadata -> exact read-only repo snapshot.

Uses local M1 bare object stores. Does not clone/fetch, start Ray/vLLM/GPU,
or implement exploration tools / AgentLoop / reward.

Usage (pinned RL conda env):

    python scripts/smoke/smoke_repo_workspace.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym import parquet_path  # noqa: E402
from budget_coder_rl.data.swe_gym_materialize import train_parquet_path  # noqa: E402
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    BlobStore,
    cache_path_for_repo,
    is_git_dir,
    swe_gym_repos_root,
)
from budget_coder_rl.env import (  # noqa: E402
    CommitNotFoundError,
    InvalidCommitRefError,
    RepoEnvironment,
    RepoUnavailableError,
    TaskRef,
)

PREFERRED_REPOS = (
    "pydantic/pydantic",
    "facebookresearch/hydra",
    "conan-io/conan",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--repos-root", type=Path, default=None)
    parser.add_argument("--snapshots-root", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--raw-parquet", type=Path, default=None)
    return parser.parse_args(argv)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to load M1 task metadata but is not importable. "
            "Use the pinned RL conda env."
        ) from exc
    return pd


def load_task_rows(repo_root: Path, train: Path | None, raw: Path | None) -> list[TaskRef]:
    pd = _require_pandas()
    train_path = train.resolve() if train is not None else train_parquet_path(repo_root)
    raw_path = raw.resolve() if raw is not None else parquet_path(repo_root)
    rows: list[TaskRef] = []
    if train_path.is_file():
        frame = pd.read_parquet(train_path, columns=["extra_info"])
        for record in frame.to_dict(orient="records"):
            extra = _as_mapping(record.get("extra_info"))
            rows.append(TaskRef.from_extra_info(extra))
        return rows
    if raw_path.is_file():
        frame = pd.read_parquet(
            raw_path, columns=["instance_id", "repo", "base_commit"]
        )
        for record in frame.to_dict(orient="records"):
            rows.append(
                TaskRef(
                    instance_id=str(record.get("instance_id") or "").strip(),
                    repo=str(record.get("repo") or "").strip(),
                    base_commit=str(record.get("base_commit") or "").strip(),
                )
            )
        return rows
    raise SystemExit(
        "no M1 task metadata found. Expected "
        f"{train_path} or {raw_path}. Run M1E materialize or M1A download."
    )


def select_smoke_tasks(rows: list[TaskRef]) -> list[TaskRef]:
    by_repo: dict[str, list[TaskRef]] = defaultdict(list)
    for row in rows:
        if row.repo and row.instance_id and row.base_commit:
            by_repo[row.repo].append(row)
    preferred = [repo for repo in PREFERRED_REPOS if repo in by_repo]
    if len(preferred) < 2:
        preferred = sorted(by_repo, key=lambda repo: len(by_repo[repo]))
    if len(preferred) < 2:
        raise SystemExit("need at least two repos in task metadata for smoke")

    selected: list[TaskRef] = []
    pair_repo = None
    for repo in preferred:
        commits = {item.base_commit for item in by_repo[repo]}
        if len(commits) >= 2:
            pair_repo = repo
            break
    if pair_repo is None:
        raise SystemExit("need one preferred repo with two distinct base_commits")

    seen_commits: set[str] = set()
    for item in by_repo[pair_repo]:
        if item.base_commit in seen_commits:
            continue
        selected.append(item)
        seen_commits.add(item.base_commit)
        if len(selected) == 2:
            break

    other = next(repo for repo in preferred if repo != pair_repo)
    selected.append(by_repo[other][0])
    if len(selected) < 3:
        raise SystemExit("failed to select 3 smoke tasks")
    return selected


def _sample_rel_files(repo_root: Path, limit: int = 3) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            found.append(path.relative_to(repo_root).as_posix())
            if len(found) >= limit:
                return found
    return found


def check_workspace(
    env: RepoEnvironment,
    store: BlobStore,
    task: TaskRef,
) -> dict[str, str]:
    ws = env.prepare(task)
    ws.validate()
    view = ws.view()
    samples = _sample_rel_files(ws.repo_root)
    if not samples:
        raise SystemExit(f"{task.instance_id}: snapshot has no regular files")
    for rel in samples:
        snap = view.read_bytes(rel)
        blob = store.read(task.repo, ws.base_commit, rel)
        if snap != blob:
            raise SystemExit(
                f"{task.instance_id}: snapshot != object store for {rel}"
            )
    reset = env.reset(ws)
    reset.validate()
    if reset.base_commit != ws.base_commit:
        raise SystemExit(f"{task.instance_id}: reset changed commit")
    if reset.view().read_bytes(samples[0]) != view.read_bytes(samples[0]):
        raise SystemExit(f"{task.instance_id}: reset was not deterministic")
    if (ws.repo_root / ".git").exists():
        raise SystemExit(f"{task.instance_id}: snapshot contains .git")
    digest = hashlib.sha256(view.read_bytes(samples[0])).hexdigest()[:16]
    return {
        "instance_id": ws.instance_id,
        "repo": ws.repo,
        "base_commit": ws.base_commit,
        "repo_root": str(ws.repo_root),
        "sample": samples[0],
        "sample_sha256_16": digest,
    }


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
    store = BlobStore(repos_root)

    rows = load_task_rows(repo_root, args.train, args.raw_parquet)
    tasks = select_smoke_tasks(rows)
    repos = {task.repo for task in tasks}
    commits = {(task.repo, task.base_commit) for task in tasks}
    if len(repos) < 2 or len(commits) < 3:
        raise SystemExit(
            f"selection too narrow: repos={sorted(repos)} commits={len(commits)}"
        )

    missing_stores = [
        task.repo
        for task in tasks
        if not is_git_dir(cache_path_for_repo(task.repo, repos_root))
    ]
    if missing_stores:
        print("HARD FAIL: local object store missing for:", file=sys.stderr)
        for repo in missing_stores:
            print(f"  {repo} -> {cache_path_for_repo(repo, repos_root)}", file=sys.stderr)
        print("run: python scripts/data/prepare_swe_gym_repos.py", file=sys.stderr)
        return 1

    print("SWE-Gym M2A repository workspace smoke")
    print(f"repos_root: {repos_root}")
    print(f"snapshots_root: {env.snapshots_root}")
    print(f"tasks: {len(tasks)} repos={sorted(repos)}")

    reports = []
    previous_root = None
    previous_commit = None
    for task in tasks:
        print(
            f"preparing {task.instance_id} {task.repo} {task.base_commit[:12]}...",
            flush=True,
        )
        report = check_workspace(env, store, task)
        if previous_root is not None and report["base_commit"] != previous_commit:
            if report["repo_root"] == previous_root:
                raise SystemExit("task switch reused the wrong snapshot root")
        previous_root = report["repo_root"]
        previous_commit = report["base_commit"]
        reports.append(report)
        print(
            f"  ok commit={report['base_commit'][:12]} "
            f"sample={report['sample']} sha={report['sample_sha256_16']}",
            flush=True,
        )

    print("checking invalid inputs...", flush=True)
    good = tasks[0]
    try:
        env.prepare(TaskRef(good.instance_id, good.repo, "HEAD"))
    except InvalidCommitRefError:
        pass
    else:
        raise SystemExit("HEAD was accepted; expected InvalidCommitRefError")
    try:
        env.prepare(TaskRef(good.instance_id, "missing/not-a-repo", good.base_commit))
    except RepoUnavailableError:
        pass
    else:
        raise SystemExit("missing repo was accepted; expected RepoUnavailableError")
    try:
        env.prepare(TaskRef(good.instance_id, good.repo, "ab" * 20))
    except CommitNotFoundError:
        pass
    else:
        raise SystemExit("missing commit was accepted; expected CommitNotFoundError")

    print("PASS")
    for report in reports:
        print(
            f"  {report['instance_id']} {report['repo']} "
            f"{report['base_commit']} {report['sample']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
