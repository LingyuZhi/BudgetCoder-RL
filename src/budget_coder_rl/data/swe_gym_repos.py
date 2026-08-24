"""SWE-Gym local Git object-store cache (M1C-B).

One full (non-shallow) mirror per unique ``repo``. Instance worktrees are
not created. Preparation may use the network; extraction must not.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_N_REPOS,
    HF_REVISION,
    is_null,
)
from budget_coder_rl.data.swe_gym_oracle import extract_oracle_from_patch

DEFAULT_BCRL_DATA_ROOT = Path.home() / "my_data" / "budget-coder-rl"
CACHE_RELPATH = "repos/swe_gym"
REMOTE_TEMPLATE = "https://github.com/{repo}.git"
CLONE_KIND = "bare"
NETWORK_GIT_VERBS = frozenset({"clone", "fetch", "pull", "push", "remote"})
SAFE_OFFLINE_GIT_VERBS = frozenset({"rev-parse", "cat-file"})


class GitError(RuntimeError):
    """Explicit git subprocess failure."""


class OfflineGitError(GitError):
    """A network git command was attempted while offline."""


def bcrl_data_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    raw = os.environ.get("BCRL_DATA_ROOT")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_BCRL_DATA_ROOT


def swe_gym_repos_root(data_root: Path | None = None) -> Path:
    return bcrl_data_root(data_root) / CACHE_RELPATH


def repo_sources_manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "manifests" / "swe_gym_repo_sources.json"


def cache_key(repo: str) -> str:
    return f"{repo.replace('/', '__')}.git"


def github_remote(repo: str) -> str:
    return REMOTE_TEMPLATE.format(repo=repo)


def github_ssh_remote(repo: str) -> str:
    return f"git@github.com:{repo}.git"


def clone_urls_for_repo(repo: str) -> list[str]:
    """Official GitHub remotes derived from ``owner/name``. HTTPS first, SSH fallback."""
    # SSH first: this cluster's HTTPS proxy often returns 503 to github.com.
    # Canonical remote recorded in the manifest remains the HTTPS GitHub URL.
    return [github_ssh_remote(repo), github_remote(repo)]


def cache_path_for_repo(repo: str, repos_root: Path) -> Path:
    return Path(repos_root) / cache_key(repo)


def is_safe_repo_path(path: str) -> bool:
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if not parts:
        return False
    if any(part == ".." for part in parts):
        return False
    return True


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env.setdefault(
        "GIT_SSH_COMMAND",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
    )
    return env


def run_git(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    allow_network: bool = False,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[bytes]:
    if not args:
        raise GitError("empty git command")
    verb = str(args[0])
    needs_network = verb in NETWORK_GIT_VERBS
    if needs_network and not allow_network:
        raise OfflineGitError(
            f"refusing network git command while offline: git {' '.join(args)}"
        )
    if not allow_network and verb not in SAFE_OFFLINE_GIT_VERBS:
        raise OfflineGitError(
            f"refusing unexpected git command while offline: git {' '.join(args)}"
        )
    command = ["git"]
    if cwd is not None and verb != "clone":
        command.extend(["-C", str(cwd)])
    command.extend(args)
    result = subprocess.run(
        command,
        cwd=None if verb != "clone" else (str(cwd) if cwd is not None else None),
        capture_output=True,
        check=False,
        timeout=timeout,
        env=_git_env(),
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git {args[0]} failed (exit {result.returncode}): {stderr}")
    return result


def is_git_dir(path: Path) -> bool:
    if not Path(path).exists():
        return False
    result = run_git(
        ["rev-parse", "--git-dir"],
        cwd=Path(path),
        allow_network=False,
        check=False,
        timeout=30,
    )
    return result.returncode == 0


def clone_or_update_mirror(
    remote: str,
    dest: Path,
    *,
    allow_network: bool = True,
    timeout: int = 3600,
    fallback_remotes: Sequence[str] | None = None,
) -> str:
    """Clone ``--mirror`` or update an existing mirror. Returns action taken."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and is_git_dir(dest):
        run_git(
            ["remote", "update", "--prune"],
            cwd=dest,
            allow_network=allow_network,
            timeout=timeout,
        )
        return "updated"
    if dest.exists() and not is_git_dir(dest):
        shutil.rmtree(dest)

    remotes = [remote, *[url for url in (fallback_remotes or []) if url != remote]]
    last_error: Exception | None = None
    for url in remotes:
        print(f"  cloning {url}", flush=True)
        try:
            run_git(
                ["clone", "--mirror", url, str(dest)],
                cwd=dest.parent,
                allow_network=allow_network,
                timeout=timeout,
            )
            return "cloned"
        except (GitError, OfflineGitError, OSError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            if dest.exists() and not is_git_dir(dest):
                shutil.rmtree(dest)
            print(f"  clone failed ({type(exc).__name__}: {exc})", flush=True)
    if last_error is not None:
        raise last_error
    raise GitError(f"clone failed with no remotes: {dest}")


def resolve_commit(repo_path: Path, sha: str) -> str | None:
    if not sha or not str(sha).strip():
        return None
    result = run_git(
        ["rev-parse", "--verify", f"{sha}^{{commit}}"],
        cwd=Path(repo_path),
        allow_network=False,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("ascii", errors="replace").strip() or None


def fetch_commit(repo_path: Path, sha: str, *, allow_network: bool = True) -> bool:
    run_git(
        ["fetch", "--no-tags", "origin", sha],
        cwd=Path(repo_path),
        allow_network=allow_network,
        check=False,
        timeout=300,
    )
    return resolve_commit(repo_path, sha) is not None


def _remote_names(repo_path: Path) -> list[str]:
    result = run_git(
        ["remote"],
        cwd=Path(repo_path),
        allow_network=True,
        check=False,
        timeout=30,
    )
    return result.stdout.decode("utf-8", errors="replace").split()


def ensure_bare_repo(
    dest: Path,
    origin_url: str,
    *,
    allow_network: bool = True,
) -> str:
    """Create or reuse a bare repo with ``origin`` set. Not a shallow clone."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    action = "reused"
    if dest.exists() and not is_git_dir(dest):
        shutil.rmtree(dest)
    if not dest.exists():
        run_git(
            ["init", "--bare", str(dest)],
            cwd=dest.parent,
            allow_network=True,
            timeout=60,
        )
        action = "initialized"
    names = _remote_names(dest)
    if "origin" in names:
        run_git(
            ["remote", "set-url", "origin", origin_url],
            cwd=dest,
            allow_network=allow_network,
            timeout=30,
        )
    else:
        run_git(
            ["remote", "add", "origin", origin_url],
            cwd=dest,
            allow_network=allow_network,
            timeout=30,
        )
    return action


def _clear_git_locks(dest: Path) -> None:
    dest = Path(dest)
    if not dest.exists():
        return
    for lock in dest.rglob("*.lock"):
        try:
            lock.unlink()
        except OSError:
            continue


def fetch_origin_refs(
    dest: Path,
    url: str,
    *,
    allow_network: bool = True,
) -> None:
    """One non-shallow fetch of all advertised heads/tags."""
    ensure_bare_repo(dest, url, allow_network=allow_network)
    print(f"  fetching origin heads/tags from {url}", flush=True)
    run_git(
        [
            "fetch",
            "--prune",
            "origin",
            "+refs/heads/*:refs/heads/*",
            "+refs/tags/*:refs/tags/*",
        ],
        cwd=Path(dest),
        allow_network=allow_network,
        check=False,
        timeout=3600,
    )


def fetch_commits(
    dest: Path,
    shas: Sequence[str],
    remotes: Sequence[str],
    *,
    allow_network: bool = True,
    batch_size: int = 32,
) -> tuple[list[str], list[str], list[str]]:
    """Populate a bare repo, then fetch any dataset SHAs still missing."""
    dest = Path(dest)
    resolved: list[str] = []
    missing: list[str] = []
    fetched: list[str] = []
    pending = [sha for sha in shas if sha]
    already = [sha for sha in pending if resolve_commit(dest, sha) is not None]
    resolved.extend(already)
    todo = [sha for sha in pending if sha not in set(already)]
    if not todo:
        return resolved, missing, fetched

    for url in remotes:
        if not todo:
            break
        _clear_git_locks(dest)
        fetch_origin_refs(dest, url, allow_network=allow_network)
        still: list[str] = []
        for sha in todo:
            if resolve_commit(dest, sha) is not None:
                if sha not in resolved:
                    resolved.append(sha)
                    fetched.append(sha)
            else:
                still.append(sha)
        print(
            f"  after origin fetch: {len(pending) - len(still)}/{len(pending)} "
            f"commits present, {len(still)} still missing",
            flush=True,
        )
        if still:
            print(f"  fetching {len(still)} missing commit(s) from {url}", flush=True)
        remaining: list[str] = []
        for start in range(0, len(still), batch_size):
            batch = still[start : start + batch_size]
            print(
                f"  sha batch {start // batch_size + 1}/"
                f"{(len(still) + batch_size - 1) // batch_size} "
                f"({len(batch)} shas)",
                flush=True,
            )
            run_git(
                ["fetch", "--no-tags", "origin", *batch],
                cwd=dest,
                allow_network=allow_network,
                check=False,
                timeout=3600,
            )
            for sha in batch:
                if resolve_commit(dest, sha) is not None:
                    if sha not in resolved:
                        resolved.append(sha)
                        fetched.append(sha)
                else:
                    remaining.append(sha)
        todo = remaining
    missing.extend(todo)
    return resolved, missing, fetched


def blob_exists(repo_path: Path, commit: str, path: str) -> bool:
    if not is_safe_repo_path(path):
        return False
    result = run_git(
        ["cat-file", "-t", f"{commit}:{path}"],
        cwd=Path(repo_path),
        allow_network=False,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        return False
    return result.stdout.decode("ascii", errors="replace").strip() == "blob"


def read_blob(repo_path: Path, commit: str, path: str) -> bytes:
    if not is_safe_repo_path(path):
        raise GitError(f"unsafe repository path: {path!r}")
    result = run_git(
        ["cat-file", "blob", f"{commit}:{path}"],
        cwd=Path(repo_path),
        allow_network=False,
        timeout=60,
    )
    return result.stdout


def directory_size_bytes(path: Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    result = subprocess.run(
        ["du", "-sb", str(root)],
        capture_output=True,
        check=False,
        timeout=300,
    )
    if result.returncode == 0:
        first = result.stdout.decode("ascii", errors="replace").split()[0]
        try:
            return int(first)
        except ValueError:
            pass
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


@dataclass
class RepoCachePlan:
    repo: str
    remote: str
    cache_key: str
    instance_ids: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    blobs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_instances(self) -> int:
        return len(self.instance_ids)

    @property
    def n_unique_base_commits(self) -> int:
        return len(self.commits)


def _row_mapping(frame: Any, index: Any) -> dict[str, Any]:
    row = frame.loc[index]
    return {str(column): row[column] for column in frame.columns}


def collect_repo_cache_plans(frame: Any) -> list[RepoCachePlan]:
    """Derive remotes / commits / blobs from the pinned frame. No hardcoding."""
    by_repo: dict[str, dict[str, Any]] = {}
    for index in frame.index:
        row = _row_mapping(frame, index)
        repo = "" if is_null(row.get("repo")) else str(row["repo"]).strip()
        instance_id = (
            "" if is_null(row.get("instance_id")) else str(row["instance_id"])
        )
        base_commit = (
            "" if is_null(row.get("base_commit")) else str(row["base_commit"]).strip()
        )
        patch = "" if is_null(row.get("patch")) else str(row["patch"])
        if repo not in by_repo:
            by_repo[repo] = {
                "instance_ids": [],
                "commits": set(),
                "blobs": set(),
            }
        bucket = by_repo[repo]
        bucket["instance_ids"].append(instance_id)
        if base_commit:
            bucket["commits"].add(base_commit)
            parsed = extract_oracle_from_patch(patch)
            for path in parsed.base_changed_files:
                bucket["blobs"].add((base_commit, path))

    plans = []
    for repo in sorted(by_repo):
        bucket = by_repo[repo]
        plans.append(
            RepoCachePlan(
                repo=repo,
                remote=github_remote(repo),
                cache_key=cache_key(repo),
                instance_ids=sorted(bucket["instance_ids"]),
                commits=sorted(bucket["commits"]),
                blobs=sorted(bucket["blobs"]),
            )
        )
    return plans


def repo_sources_record(
    plans: Sequence[RepoCachePlan],
    *,
    prepare_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    results = prepare_results or {}
    repos = []
    for plan in plans:
        item = {
            "repo": plan.repo,
            "remote": plan.remote,
            "cache_key": plan.cache_key,
            "clone_kind": CLONE_KIND,
            "n_instances": plan.n_instances,
            "n_unique_base_commits": plan.n_unique_base_commits,
            "n_unique_blobs": len(plan.blobs),
        }
        extra = results.get(plan.repo)
        if extra is not None:
            for key in (
                "action",
                "available",
                "n_commits_resolved",
                "n_commits_missing",
                "n_commits_fetched_by_sha",
                "n_blobs_ok",
                "n_blobs_missing",
            ):
                if key in extra:
                    item[key] = extra[key]
        repos.append(item)
    return {
        "dataset": "SWE-Gym",
        "revision": HF_REVISION,
        "cache_root_env": "BCRL_DATA_ROOT",
        "cache_relpath": CACHE_RELPATH,
        "remote_template": REMOTE_TEMPLATE,
        "clone_kind": CLONE_KIND,
        "n_repos": len(plans),
        "expected_n_repos": EXPECTED_N_REPOS,
        "repos": repos,
        "notes": (
            "Local cache lives under $BCRL_DATA_ROOT/repos/swe_gym. "
            "One bare object store per unique repo; dataset base_commits are "
            "fetched from official GitHub remotes derived from the repo field. "
            "Not a shallow clone and not one worktree per instance. "
            "This manifest stores no host-absolute paths."
        ),
    }


@dataclass
class PrepareReport:
    plans: list[RepoCachePlan]
    per_repo: dict[str, dict[str, Any]]
    missing_commits: list[dict[str, str]]
    missing_blobs: list[dict[str, str]]
    commits_fetched_by_sha: list[dict[str, str]]
    n_repos_available: int
    n_repos_failed: int

    @property
    def ok(self) -> bool:
        return (
            self.n_repos_failed == 0
            and not self.missing_commits
            and not self.missing_blobs
            and self.n_repos_available == EXPECTED_N_REPOS
        )


def prepare_repo_cache(
    frame: Any,
    repos_root: Path,
    *,
    allow_network: bool = True,
    verify_only: bool = False,
) -> PrepareReport:
    plans = collect_repo_cache_plans(frame)
    per_repo: dict[str, dict[str, Any]] = {}
    missing_commits: list[dict[str, str]] = []
    missing_blobs: list[dict[str, str]] = []
    fetched: list[dict[str, str]] = []
    n_available = 0
    n_failed = 0

    for plan in plans:
        dest = cache_path_for_repo(plan.repo, repos_root)
        print(
            f"preparing {plan.repo} ({plan.n_unique_base_commits} commits, "
            f"{len(plan.blobs)} blobs)...",
            flush=True,
        )
        action = "verify_only"
        available = is_git_dir(dest)
        error = None
        fetched_shas: list[str] = []
        if not verify_only:
            try:
                remotes = clone_urls_for_repo(plan.repo)
                action = ensure_bare_repo(
                    dest, remotes[0], allow_network=allow_network
                )
                resolved_shas, still_missing, fetched_shas = fetch_commits(
                    dest,
                    plan.commits,
                    remotes,
                    allow_network=allow_network,
                )
                available = is_git_dir(dest)
                if still_missing:
                    error = (
                        f"{len(still_missing)} commit(s) not in local object store"
                    )
            except (GitError, OfflineGitError, OSError, subprocess.TimeoutExpired) as exc:
                error = f"{type(exc).__name__}: {exc}"
                available = is_git_dir(dest)

        n_resolved = 0
        n_missing_commit = 0
        n_fetched = 0
        n_blob_ok = 0
        n_blob_missing = 0
        if not available:
            n_failed += 1
            n_missing_commit = len(plan.commits)
            n_blob_missing = len(plan.blobs)
            for sha in plan.commits:
                missing_commits.append(
                    {
                        "repo": plan.repo,
                        "base_commit": sha,
                        "reason": error or "repo unavailable",
                    }
                )
            for sha, path in plan.blobs:
                missing_blobs.append(
                    {
                        "repo": plan.repo,
                        "base_commit": sha,
                        "path": path,
                        "reason": error or "repo unavailable",
                    }
                )
        else:
            n_available += 1
            resolved: set[str] = set()
            for sha in fetched_shas:
                fetched.append({"repo": plan.repo, "base_commit": sha})
            n_fetched = len(fetched_shas)
            for sha in plan.commits:
                if resolve_commit(dest, sha) is not None:
                    resolved.add(sha)
                    n_resolved += 1
                    continue
                n_missing_commit += 1
                missing_commits.append(
                    {
                        "repo": plan.repo,
                        "base_commit": sha,
                        "reason": "commit not in local object store",
                    }
                )
            for sha, path in plan.blobs:
                if sha not in resolved:
                    n_blob_missing += 1
                    missing_blobs.append(
                        {
                            "repo": plan.repo,
                            "base_commit": sha,
                            "path": path,
                            "reason": "commit not resolved",
                        }
                    )
                    continue
                if blob_exists(dest, sha, path):
                    n_blob_ok += 1
                else:
                    n_blob_missing += 1
                    missing_blobs.append(
                        {
                            "repo": plan.repo,
                            "base_commit": sha,
                            "path": path,
                            "reason": "blob missing",
                        }
                    )

        per_repo[plan.repo] = {
            "action": action,
            "available": available,
            "error": error,
            "n_commits": len(plan.commits),
            "n_commits_resolved": n_resolved,
            "n_commits_missing": n_missing_commit,
            "n_commits_fetched_by_sha": n_fetched,
            "n_blobs": len(plan.blobs),
            "n_blobs_ok": n_blob_ok,
            "n_blobs_missing": n_blob_missing,
        }

    return PrepareReport(
        plans=plans,
        per_repo=per_repo,
        missing_commits=missing_commits,
        missing_blobs=missing_blobs,
        commits_fetched_by_sha=fetched,
        n_repos_available=n_available,
        n_repos_failed=n_failed,
    )


def format_prepare_report(
    report: PrepareReport, *, repos_root_label: str, disk_bytes: int | None = None
) -> str:
    lines = [
        "SWE-Gym M1C-B repository cache prepare/verify",
        f"repos_root: {repos_root_label}",
        f"repos available: {report.n_repos_available}/{len(report.plans)}",
        f"repos failed: {report.n_repos_failed}",
        f"missing commits: {len(report.missing_commits)}",
        f"missing blobs: {len(report.missing_blobs)}",
        f"commits fetched by sha: {len(report.commits_fetched_by_sha)}",
    ]
    if disk_bytes is not None:
        lines.append(f"cache disk bytes: {disk_bytes}")
    lines.append("")
    for plan in report.plans:
        info = report.per_repo[plan.repo]
        status = "ok" if info["available"] and info["n_commits_missing"] == 0 and info["n_blobs_missing"] == 0 else "FAIL"
        lines.append(
            f"  [{status}] {plan.repo} action={info['action']} "
            f"commits={info['n_commits_resolved']}/{info['n_commits']} "
            f"blobs={info['n_blobs_ok']}/{info['n_blobs']}"
        )
        if info.get("error"):
            lines.append(f"           error: {info['error']}")
    if report.missing_commits:
        lines.append("")
        lines.append("MISSING COMMITS:")
        for item in report.missing_commits[:50]:
            lines.append(
                f"  {item['repo']} {item['base_commit']} ({item['reason']})"
            )
        if len(report.missing_commits) > 50:
            lines.append(f"  ... {len(report.missing_commits) - 50} more")
    if report.missing_blobs:
        lines.append("")
        lines.append("MISSING BLOBS:")
        for item in report.missing_blobs[:50]:
            lines.append(
                f"  {item['repo']} {item['base_commit']}:{item['path']} ({item['reason']})"
            )
        if len(report.missing_blobs) > 50:
            lines.append(f"  ... {len(report.missing_blobs) - 50} more")
    return "\n".join(lines) + "\n"


class BlobStore:
    """Read-only Git object access. Never clones or fetches."""

    def __init__(self, repos_root: Path) -> None:
        self.repos_root = Path(repos_root)
        self._commit_ok: dict[tuple[str, str], bool] = {}
        self._blobs: dict[tuple[str, str, str], bytes | None] = {}
        self._exists: dict[tuple[str, str, str], bool] = {}

    def repo_path(self, repo: str) -> Path | None:
        path = cache_path_for_repo(repo, self.repos_root)
        if is_git_dir(path):
            return path
        return None

    def commit_ok(self, repo: str, commit: str) -> bool:
        key = (repo, commit)
        if key in self._commit_ok:
            return self._commit_ok[key]
        path = self.repo_path(repo)
        ok = path is not None and resolve_commit(path, commit) is not None
        self._commit_ok[key] = ok
        return ok

    def exists(self, repo: str, commit: str, path: str) -> bool:
        key = (repo, commit, path)
        if key in self._exists:
            return self._exists[key]
        repo_path = self.repo_path(repo)
        ok = (
            repo_path is not None
            and self.commit_ok(repo, commit)
            and blob_exists(repo_path, commit, path)
        )
        self._exists[key] = ok
        return ok

    def read(self, repo: str, commit: str, path: str) -> bytes:
        key = (repo, commit, path)
        if key in self._blobs:
            cached = self._blobs[key]
            if cached is None:
                raise GitError(f"blob missing: {repo} {commit}:{path}")
            return cached
        repo_path = self.repo_path(repo)
        if repo_path is None:
            self._blobs[key] = None
            raise GitError(f"repo unavailable: {repo}")
        if not self.commit_ok(repo, commit):
            self._blobs[key] = None
            raise GitError(f"commit missing: {repo} {commit}")
        if not blob_exists(repo_path, commit, path):
            self._blobs[key] = None
            raise GitError(f"blob missing: {repo} {commit}:{path}")
        data = read_blob(repo_path, commit, path)
        self._blobs[key] = data
        return data
