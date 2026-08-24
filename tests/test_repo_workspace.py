"""Contract tests for M2A read-only repository workspaces.

Uses synthetic two-commit bare repos. Does not read the official parquet
or the full SWE-Gym Git mirrors, except one optional skip-if-missing check
that git archive still diverges from a known pandas M1 commit.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from budget_coder_rl.data.swe_gym_repos import (
    OfflineGitError,
    SAFE_OFFLINE_GIT_VERBS,
    archive_commit_to_dir,
    blob_exists,
    cache_key,
    cache_path_for_repo,
    is_git_dir,
    read_blob,
    resolve_commit,
    run_git,
    swe_gym_repos_root,
)
from budget_coder_rl.env import (
    CommitNotFoundError,
    InvalidCommitRefError,
    PathEscapeError,
    RepoEnvironment,
    RepoUnavailableError,
    TaskRef,
    WorkspaceIdentityError,
    require_full_commit_sha,
)


def _init_src(src: Path) -> None:
    src.mkdir()
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["config", "commit.gpgsign", "false"], cwd=src, allow_network=True)


def _commit_all(src: Path, message: str) -> str:
    run_git(["add", "."], cwd=src, allow_network=True)
    run_git(["commit", "-m", message], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None
    return sha


def _two_commit_bare(tmp_path: Path) -> tuple[Path, Path, str, str]:
    src = tmp_path / "src"
    _init_src(src)
    (src / "pkg.py").write_text("version = 1\n", encoding="utf-8")
    (src / "nested").mkdir()
    (src / "nested" / "keep.txt").write_text("stable\n", encoding="utf-8")
    (src / "inside_link").symlink_to("pkg.py")
    sha1 = _commit_all(src, "base")
    (src / "pkg.py").write_text("version = 2\n", encoding="utf-8")
    (src / "escape").symlink_to("/etc/passwd")
    sha2 = _commit_all(src, "later-like-gold-patch")

    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(
        ["clone", "--bare", str(src), str(dest)],
        cwd=repos_root,
        allow_network=True,
    )
    return repos_root, dest, sha1, sha2


def _env(tmp_path: Path, repos_root: Path) -> RepoEnvironment:
    return RepoEnvironment(
        repos_root=repos_root,
        snapshots_root=tmp_path / "snapshots",
    )


def test_archive_is_offline_safe_and_fetch_still_rejected(tmp_path: Path):
    repos_root, dest, sha1, _sha2 = _two_commit_bare(tmp_path)
    assert "archive" in SAFE_OFFLINE_GIT_VERBS
    out = tmp_path / "archived"
    out.mkdir()
    archive_commit_to_dir(dest, sha1, out, timeout=30)
    assert (out / "pkg.py").read_text(encoding="utf-8") == "version = 1\n"
    assert not (out / ".git").exists()
    with pytest.raises(OfflineGitError):
        run_git(["fetch", "origin"], cwd=dest, allow_network=False)
    with pytest.raises(OfflineGitError):
        run_git(["clone", str(dest), str(tmp_path / "nope")], allow_network=False)
    with pytest.raises(OfflineGitError):
        run_git(["worktree", "list"], cwd=dest, allow_network=False)
    assert cache_path_for_repo("owner/repo", repos_root) == dest


def test_prepare_uses_exact_base_commit_not_later_or_head(tmp_path: Path):
    repos_root, dest, sha1, sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    ws = env.prepare(TaskRef("owner__repo-1", "owner/repo", sha1))
    view = ws.view()
    assert ws.base_commit == sha1
    assert view.read_bytes("pkg.py") == b"version = 1\n"
    assert view.read_bytes("pkg.py") == read_blob(dest, sha1, "pkg.py")
    assert view.read_bytes("pkg.py") != read_blob(dest, sha2, "pkg.py")
    assert view.read_bytes("nested/keep.txt") == b"stable\n"
    assert not (ws.repo_root / ".git").exists()
    assert not hasattr(view, "write_bytes")
    assert not hasattr(view, "write_text")
    with pytest.raises(PermissionError):
        (ws.repo_root / "pkg.py").write_text("hacked\n", encoding="utf-8")


def test_reset_is_deterministic_and_task_switch_changes_commit(tmp_path: Path):
    repos_root, dest, sha1, sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    first = env.prepare(TaskRef("owner__repo-1", "owner/repo", sha1))
    first_root = first.repo_root
    first_bytes = first.view().read_bytes("pkg.py")
    reset = env.reset(first)
    assert reset.base_commit == sha1
    assert reset.repo_root == first_root
    assert reset.view().read_bytes("pkg.py") == first_bytes == b"version = 1\n"

    later = env.prepare(TaskRef("owner__repo-2", "owner/repo", sha2))
    assert later.base_commit == sha2
    assert later.repo_root != first_root
    assert later.view().read_bytes("pkg.py") == b"version = 2\n"
    assert (later.repo_root / "escape").is_symlink()
    with pytest.raises(PathEscapeError):
        later.view().resolve("escape")
    # Original snapshot is unchanged.
    assert Path(first_root, "pkg.py").read_bytes() == b"version = 1\n"
    assert blob_exists(dest, sha2, "pkg.py")


def test_same_commit_reuses_snapshot_for_different_instance(tmp_path: Path):
    repos_root, _dest, sha1, _sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    a = env.prepare(TaskRef("id-a", "owner/repo", sha1))
    b = env.prepare(TaskRef("id-b", "owner/repo", sha1))
    assert a.instance_id == "id-a"
    assert b.instance_id == "id-b"
    assert a.repo_root == b.repo_root
    assert a.base_commit == b.base_commit == sha1


def test_invalid_refs_and_missing_repo_commit_do_not_fallback(tmp_path: Path):
    repos_root, _dest, sha1, _sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    with pytest.raises(InvalidCommitRefError):
        env.prepare(TaskRef("x", "owner/repo", "HEAD"))
    with pytest.raises(InvalidCommitRefError):
        env.prepare(TaskRef("x", "owner/repo", "main"))
    with pytest.raises(InvalidCommitRefError):
        env.prepare(TaskRef("x", "owner/repo", sha1[:12]))
    with pytest.raises(InvalidCommitRefError):
        env.prepare(TaskRef("x", "owner/repo", ""))
    with pytest.raises(InvalidCommitRefError):
        require_full_commit_sha("deadbeef")
    missing_sha = "ab" * 20
    with pytest.raises(CommitNotFoundError):
        env.prepare(TaskRef("x", "owner/repo", missing_sha))
    with pytest.raises(RepoUnavailableError):
        env.prepare(TaskRef("x", "missing/repo", sha1))
    with pytest.raises(WorkspaceIdentityError):
        env.prepare(TaskRef("", "owner/repo", sha1))
    # HEAD on the bare store would resolve; the environment must not use it.
    head = resolve_commit(cache_path_for_repo("owner/repo", repos_root), "HEAD")
    assert head is not None
    assert head != sha1


def test_path_confinement_and_symlink_escape(tmp_path: Path):
    repos_root, _dest, _sha1, sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    ws = env.prepare(TaskRef("owner__repo-2", "owner/repo", sha2))
    view = ws.view()
    with pytest.raises(PathEscapeError):
        view.resolve("../etc/passwd")
    with pytest.raises(PathEscapeError):
        view.resolve("/etc/passwd")
    with pytest.raises(PathEscapeError):
        view.read_bytes("foo/../../etc/passwd")
    with pytest.raises(PathEscapeError):
        view.resolve("escape")
    assert view.read_bytes("inside_link") == b"version = 2\n"
    assert "pkg.py" in view.list_dir(".")
    assert view.exists("nested/keep.txt")
    assert view.is_dir("nested")
    assert view.is_file("pkg.py")
    with pytest.raises(FileNotFoundError):
        view.read_bytes("no-such-file.py")
    with pytest.raises(IsADirectoryError):
        view.read_bytes("nested")


def test_corrupt_snapshot_reset_rebuilds(tmp_path: Path):
    repos_root, _dest, sha1, _sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    ws = env.prepare(TaskRef("owner__repo-1", "owner/repo", sha1))
    meta = ws.meta_path
    original = meta.read_text(encoding="utf-8")
    # meta.json sits beside tree/ and is writable.
    meta.write_text(original.replace(sha1, "aa" * 20), encoding="utf-8")
    with pytest.raises(WorkspaceIdentityError):
        ws.validate()
    rebuilt = env.reset(ws)
    rebuilt.validate()
    assert rebuilt.view().read_bytes("pkg.py") == b"version = 1\n"


def test_close_can_delete_snapshot(tmp_path: Path):
    repos_root, _dest, sha1, _sha2 = _two_commit_bare(tmp_path)
    env = _env(tmp_path, repos_root)
    ws = env.prepare(TaskRef("owner__repo-1", "owner/repo", sha1))
    root = ws.repo_root
    env.close(ws, delete_snapshot=False)
    assert root.is_dir()
    env.close(ws, delete_snapshot=True)
    assert not root.exists()


def _git_archive_file(mirror: Path, sha: str, path: str) -> bytes | None:
    result = run_git(
        ["archive", "--format=tar", sha, path],
        cwd=mirror,
        allow_network=False,
        timeout=60,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tf:
            for member in tf.getmembers():
                handle = tf.extractfile(member)
                if handle is not None:
                    return handle.read()
    except tarfile.TarError:
        return None
    return None


def _git_archive_names(mirror: Path, sha: str) -> set[str]:
    result = run_git(
        ["archive", "--format=tar", sha],
        cwd=mirror,
        allow_network=False,
        timeout=60,
    )
    names: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tf:
        for member in tf.getmembers():
            if member.isfile() or member.issym():
                names.add(member.name)
    return names


def test_snapshot_keeps_export_ignore_and_exact_export_subst_blob(tmp_path: Path):
    src = tmp_path / "src"
    _init_src(src)
    (src / ".gitattributes").write_text(
        "secret.txt export-ignore\nversion.txt export-subst\n",
        encoding="utf-8",
    )
    (src / "secret.txt").write_text("must-remain-in-tree\n", encoding="utf-8")
    (src / "version.txt").write_text("commit $Format:%H$\n", encoding="utf-8")
    sha = _commit_all(src, "attrs")

    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(
        ["clone", "--bare", str(src), str(dest)],
        cwd=repos_root,
        allow_network=True,
    )

    raw_version = read_blob(dest, sha, "version.txt")
    assert b"$Format:%H$" in raw_version
    archive_names = _git_archive_names(dest, sha)
    archived_version = _git_archive_file(dest, sha, "version.txt")
    assert "secret.txt" not in archive_names
    assert archived_version is not None
    assert archived_version != raw_version
    assert sha.encode("ascii") in archived_version or b"$Format:%H$" not in archived_version

    out = tmp_path / "tree"
    out.mkdir()
    archive_commit_to_dir(dest, sha, out, timeout=30)
    assert (out / "secret.txt").read_bytes() == b"must-remain-in-tree\n"
    assert (out / "version.txt").read_bytes() == raw_version
    assert b"$Format:%H$" in (out / "version.txt").read_bytes()

    env = _env(tmp_path, repos_root)
    ws = env.prepare(TaskRef("owner__repo-attrs", "owner/repo", sha))
    assert ws.view().read_bytes("secret.txt") == b"must-remain-in-tree\n"
    assert ws.view().read_bytes("version.txt") == raw_version


def test_pandas_m1_commit_git_archive_diverges_from_tree():
    """Corpus lock: pandas M1 trees still have export-ignore / export-subst.

    This does not materialize a pandas snapshot. It only checks that raw
    ``git archive`` is unsafe as an extractor for this commit.
    """
    sha = "00f10db680c6cf836fad80dda33849081e540230"
    repo = "pandas-dev/pandas"
    mirror = cache_path_for_repo(repo, swe_gym_repos_root())
    if not is_git_dir(mirror) or resolve_commit(mirror, sha) is None:
        pytest.skip("pandas M1 object store / commit not available")
    version_path = "pandas/_version.py"
    ignored_csv = "doc/data/air_quality_long.csv"
    assert blob_exists(mirror, sha, version_path)
    assert blob_exists(mirror, sha, ignored_csv)
    archived_version = _git_archive_file(mirror, sha, version_path)
    archived_csv = _git_archive_file(mirror, sha, ignored_csv)
    raw_version = read_blob(mirror, sha, version_path)
    assert archived_version is not None
    assert archived_version != raw_version
    assert archived_csv is None
