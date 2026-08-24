"""M2A read-only repository workspace over M1 bare object stores.

Given ``instance_id + repo + base_commit``, prepare/reset a deterministic
filesystem snapshot of the exact commit. Does not clone, fetch, apply gold
patches, or implement exploration tools.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from budget_coder_rl.data.swe_gym_repos import (
    GitError,
    archive_commit_to_dir,
    bcrl_data_root,
    blob_exists,
    cache_key,
    cache_path_for_repo,
    is_git_dir,
    is_safe_repo_path,
    read_blob,
    resolve_commit,
    resolve_tree,
    swe_gym_repos_root,
)

SNAPSHOT_RELPATH = "repos/swe_gym_snapshots"
SNAPSHOT_SCHEMA = "bcrl-repo-snapshot-v1"
_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SAMPLE_FILE_LIMIT = 3


class RepoWorkspaceError(RuntimeError):
    """Base error for repository workspace failures. No silent fallback."""


class RepoUnavailableError(RepoWorkspaceError):
    """Local object store for ``repo`` is missing or not a git dir."""


class CommitNotFoundError(RepoWorkspaceError):
    """Requested hex SHA is not in the local object store."""


class InvalidCommitRefError(RepoWorkspaceError):
    """``base_commit`` is not a 40-character hex SHA (e.g. HEAD / branch)."""


class WorkspaceIdentityError(RepoWorkspaceError):
    """Snapshot metadata does not match the requested task identity."""


class PathEscapeError(RepoWorkspaceError):
    """A filesystem path escaped the snapshot repo root."""


def swe_gym_snapshots_root(data_root: Path | None = None) -> Path:
    return bcrl_data_root(data_root) / SNAPSHOT_RELPATH


def snapshot_repo_key(repo: str) -> str:
    key = cache_key(repo)
    return key[: -len(".git")] if key.endswith(".git") else key


def require_full_commit_sha(value: str) -> str:
    text = (value or "").strip()
    if not _FULL_SHA.fullmatch(text):
        raise InvalidCommitRefError(
            f"base_commit must be a 40-character hex SHA, got {value!r}"
        )
    return text.lower()


@dataclass(frozen=True)
class TaskRef:
    instance_id: str
    repo: str
    base_commit: str

    @classmethod
    def from_extra_info(cls, extra_info: Mapping[str, Any]) -> TaskRef:
        payload = dict(extra_info)
        return cls(
            instance_id=str(payload.get("instance_id") or "").strip(),
            repo=str(payload.get("repo") or "").strip(),
            base_commit=str(payload.get("base_commit") or "").strip(),
        )


class RepoFileView:
    """Confined, read-only view of a snapshot tree. Not a tool scaffold."""

    def __init__(self, repo_root: Path) -> None:
        self._root = Path(repo_root)

    @property
    def repo_root(self) -> Path:
        return self._root

    def resolve(self, relative_path: str) -> Path:
        rel = (relative_path or "").strip()
        if rel in {".", "./"}:
            candidate = self._root
        else:
            if not is_safe_repo_path(rel):
                raise PathEscapeError(f"unsafe repository path: {relative_path!r}")
            candidate = self._root.joinpath(*rel.replace("\\", "/").split("/"))
        try:
            resolved = candidate.resolve(strict=False)
            root_resolved = self._root.resolve(strict=True)
        except OSError as exc:
            raise PathEscapeError(
                f"cannot resolve path {relative_path!r} under {self._root}"
            ) from exc
        if not _is_relative_to(resolved, root_resolved):
            raise PathEscapeError(
                f"path escaped repo root: {relative_path!r} -> {resolved}"
            )
        return resolved

    def exists(self, relative_path: str) -> bool:
        path = self.resolve(relative_path)
        return path.exists()

    def is_file(self, relative_path: str) -> bool:
        return self.resolve(relative_path).is_file()

    def is_dir(self, relative_path: str) -> bool:
        return self.resolve(relative_path).is_dir()

    def read_bytes(self, relative_path: str) -> bytes:
        path = self.resolve(relative_path)
        if not path.exists():
            raise FileNotFoundError(relative_path)
        if path.is_dir():
            raise IsADirectoryError(relative_path)
        if path.is_symlink() and not _is_relative_to(
            path.resolve(strict=False), self._root.resolve(strict=True)
        ):
            raise PathEscapeError(f"symlink escaped repo root: {relative_path!r}")
        return path.read_bytes()

    def list_dir(self, relative_path: str = ".") -> list[str]:
        path = self.resolve(relative_path)
        if not path.exists():
            raise FileNotFoundError(relative_path)
        if not path.is_dir():
            raise NotADirectoryError(relative_path)
        return sorted(entry.name for entry in path.iterdir())


@dataclass(frozen=True)
class RepoWorkspace:
    instance_id: str
    repo: str
    base_commit: str
    repo_root: Path
    object_store: Path
    meta_path: Path

    def view(self) -> RepoFileView:
        return RepoFileView(self.repo_root)

    def validate(self) -> None:
        _validate_snapshot(
            repo_root=self.repo_root,
            meta_path=self.meta_path,
            object_store=self.object_store,
            repo=self.repo,
            base_commit=self.base_commit,
        )


class RepoEnvironment:
    """Prepare/reset cached read-only snapshots from the M1 object store."""

    def __init__(
        self,
        *,
        repos_root: Path | None = None,
        snapshots_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.repos_root = (
            Path(repos_root) if repos_root is not None else swe_gym_repos_root(data_root)
        )
        self.snapshots_root = (
            Path(snapshots_root)
            if snapshots_root is not None
            else swe_gym_snapshots_root(data_root)
        )

    def prepare(self, task: TaskRef) -> RepoWorkspace:
        instance_id, repo, sha = _normalized_identity(task)
        store = cache_path_for_repo(repo, self.repos_root)
        if not is_git_dir(store):
            raise RepoUnavailableError(
                f"repo object store unavailable: {repo} ({store})"
            )
        resolved = resolve_commit(store, sha)
        if resolved is None:
            raise CommitNotFoundError(
                f"commit not in local object store: {repo} {sha}"
            )
        if resolved.lower() != sha:
            raise WorkspaceIdentityError(
                f"resolved commit {resolved} != requested {sha} for {repo}"
            )
        tree_id = resolve_tree(store, sha)
        if not tree_id:
            raise CommitNotFoundError(f"tree missing for commit: {repo} {sha}")

        snap_dir = self.snapshots_root / snapshot_repo_key(repo) / sha
        lock_path = snap_dir.parent / f"{sha}.lock"
        with _exclusive_lock(lock_path):
            if _snapshot_is_valid(snap_dir, store, repo, sha):
                return _workspace_handle(instance_id, repo, sha, snap_dir, store)
            if snap_dir.exists():
                _remove_snapshot(snap_dir)
            _materialize_snapshot(store, repo, sha, tree_id, snap_dir)
        workspace = _workspace_handle(instance_id, repo, sha, snap_dir, store)
        workspace.validate()
        return workspace

    def prepare_from_extra_info(self, extra_info: Mapping[str, Any]) -> RepoWorkspace:
        return self.prepare(TaskRef.from_extra_info(extra_info))

    def reset(self, workspace: RepoWorkspace) -> RepoWorkspace:
        return self.prepare(
            TaskRef(
                instance_id=workspace.instance_id,
                repo=workspace.repo,
                base_commit=workspace.base_commit,
            )
        )

    def close(
        self, workspace: RepoWorkspace, *, delete_snapshot: bool = False
    ) -> None:
        if not delete_snapshot:
            return
        snap_dir = workspace.repo_root.parent
        lock_path = snap_dir.parent / f"{workspace.base_commit}.lock"
        with _exclusive_lock(lock_path):
            if snap_dir.exists():
                _remove_snapshot(snap_dir)


def _normalized_identity(task: TaskRef) -> tuple[str, str, str]:
    instance_id = (task.instance_id or "").strip()
    repo = (task.repo or "").strip()
    if not instance_id or not repo:
        raise WorkspaceIdentityError(
            f"instance_id and repo are required, got "
            f"instance_id={task.instance_id!r} repo={task.repo!r}"
        )
    sha = require_full_commit_sha(task.base_commit)
    return instance_id, repo, sha


def _workspace_handle(
    instance_id: str,
    repo: str,
    sha: str,
    snap_dir: Path,
    store: Path,
) -> RepoWorkspace:
    return RepoWorkspace(
        instance_id=instance_id,
        repo=repo,
        base_commit=sha,
        repo_root=snap_dir / "tree",
        object_store=store,
        meta_path=snap_dir / "meta.json",
    )


def _snapshot_is_valid(snap_dir: Path, store: Path, repo: str, sha: str) -> bool:
    try:
        handle = _workspace_handle("validate", repo, sha, snap_dir, store)
        handle.validate()
    except (RepoWorkspaceError, GitError, OSError, json.JSONDecodeError):
        return False
    return True


def _validate_snapshot(
    *,
    repo_root: Path,
    meta_path: Path,
    object_store: Path,
    repo: str,
    base_commit: str,
) -> None:
    if not repo_root.is_dir():
        raise WorkspaceIdentityError(f"snapshot tree missing: {repo_root}")
    if (repo_root / ".git").exists():
        raise WorkspaceIdentityError(f"snapshot must not contain .git: {repo_root}")
    if not meta_path.is_file():
        raise WorkspaceIdentityError(f"snapshot meta missing: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkspaceIdentityError(f"snapshot meta is not JSON: {meta_path}") from exc
    if not isinstance(meta, MappingABC):
        raise WorkspaceIdentityError(f"snapshot meta is not an object: {meta_path}")
    if meta.get("schema") != SNAPSHOT_SCHEMA:
        raise WorkspaceIdentityError(
            f"snapshot schema mismatch: {meta.get('schema')!r}"
        )
    if str(meta.get("repo") or "") != repo:
        raise WorkspaceIdentityError(
            f"snapshot repo {meta.get('repo')!r} != {repo!r}"
        )
    if str(meta.get("base_commit") or "").lower() != base_commit.lower():
        raise WorkspaceIdentityError(
            f"snapshot commit {meta.get('base_commit')!r} != {base_commit!r}"
        )
    expected_tree = resolve_tree(object_store, base_commit)
    if not expected_tree:
        raise CommitNotFoundError(f"tree missing for commit: {repo} {base_commit}")
    if str(meta.get("tree_id") or "") != expected_tree:
        raise WorkspaceIdentityError(
            f"snapshot tree_id {meta.get('tree_id')!r} != {expected_tree!r}"
        )
    _check_sample_blobs(repo_root, object_store, base_commit)


def _check_sample_blobs(repo_root: Path, object_store: Path, commit: str) -> None:
    view = RepoFileView(repo_root)
    for rel in _sample_relative_files(repo_root, limit=_SAMPLE_FILE_LIMIT):
        if not is_safe_repo_path(rel):
            continue
        if not blob_exists(object_store, commit, rel):
            raise WorkspaceIdentityError(
                f"snapshot file not a blob at {commit}: {rel}"
            )
        if view.read_bytes(rel) != read_blob(object_store, commit, rel):
            raise WorkspaceIdentityError(
                f"snapshot bytes differ from object store at {commit}: {rel}"
            )


def _sample_relative_files(repo_root: Path, *, limit: int) -> list[str]:
    found: list[str] = []
    root = Path(repo_root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            rel = path.relative_to(root).as_posix()
            found.append(rel)
            if len(found) >= limit:
                return found
    return found


def _materialize_snapshot(
    store: Path,
    repo: str,
    sha: str,
    tree_id: str,
    snap_dir: Path,
) -> None:
    snap_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = snap_dir.parent / f".tmp-{sha}-{os.getpid()}"
    if tmp_dir.exists():
        _remove_snapshot(tmp_dir)
    tree_dir = tmp_dir / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)
    try:
        archive_commit_to_dir(store, sha, tree_dir)
        meta = {
            "schema": SNAPSHOT_SCHEMA,
            "repo": repo,
            "base_commit": sha,
            "tree_id": tree_id,
        }
        (tmp_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _make_tree_readonly(tree_dir)
        os.replace(tmp_dir, snap_dir)
    except Exception:
        if tmp_dir.exists():
            _remove_snapshot(tmp_dir)
        raise


def _remove_snapshot(path: Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    _make_tree_writable(target)
    shutil.rmtree(target)


def _make_tree_readonly(root: Path) -> None:
    root = Path(root)
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                os.chmod(path, 0o444)
            except OSError:
                continue
        try:
            os.chmod(dirpath, 0o555)
        except OSError:
            continue


def _make_tree_writable(root: Path) -> None:
    root = Path(root)
    if not root.exists():
        return
    for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        try:
            os.chmod(dirpath, 0o755)
        except OSError:
            continue
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                os.chmod(path, 0o644)
            except OSError:
                continue


def _exclusive_lock(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class _Lock:
        def __enter__(self) -> Path:
            self.fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            return path

        def __exit__(self, *exc: object) -> None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)

    return _Lock()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
