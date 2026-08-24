"""Repository tree statistics for SWE-Gym M1D-A.

Uses ``git ls-tree -r -l`` against local bare mirrors. Never checks out a
worktree, never reads blob contents, and never computes LOC. Unknown
extensions are not treated as code.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from budget_coder_rl.data.swe_gym_oracle import file_extension
from budget_coder_rl.data.swe_gym_repos import (
    GitError,
    cache_path_for_repo,
    is_git_dir,
    resolve_commit,
    run_git,
)
from budget_coder_rl.data.swe_gym_symbols import ELIGIBLE_EXTENSIONS

# Frozen M1D-A classifiers. Unknown extensions are neither python_ast,
# code_like, nor docs.
PYTHON_AST_EXTENSIONS = frozenset(ELIGIBLE_EXTENSIONS)
CODE_LIKE_EXTRA_EXTENSIONS = frozenset(
    {
        ".pyx",
        ".pxd",
        ".pxi",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
    }
)
CODE_LIKE_EXTENSIONS = PYTHON_AST_EXTENSIONS | CODE_LIKE_EXTRA_EXTENSIONS
DOCS_EXTENSIONS = frozenset({".rst", ".md", ".txt", ".html", ".htm", ".adoc"})
EXPLICITLY_NOT_CODE_LIKE_EXTENSIONS = frozenset(
    {
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".ipynb",
        ".sh",
        ".bash",
        ".in",
        ".build",
    }
)

# <mode> SP <type> SP <object> SP <size> TAB <path>
# size is "-" for gitlinks; blob sizes are right-aligned decimal integers.
_LS_TREE_LINE = re.compile(
    r"^(?P<mode>[0-7]{6}) (?P<type>\S+) (?P<object>[0-9a-fA-F]{4,}) "
    r"+(?P<size>-|\d+)\t(?P<path>.*)$"
)


@dataclass(frozen=True)
class TreeStats:
    ok: bool
    error: str | None
    repo_tracked_files: int
    repo_python_files: int
    repo_code_like_files: int
    repo_tracked_blob_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "repo_tracked_files": self.repo_tracked_files,
            "repo_python_files": self.repo_python_files,
            "repo_code_like_files": self.repo_code_like_files,
            "repo_tracked_blob_bytes": self.repo_tracked_blob_bytes,
        }


def unavailable_tree_stats(error: str) -> TreeStats:
    return TreeStats(
        ok=False,
        error=error,
        repo_tracked_files=0,
        repo_python_files=0,
        repo_code_like_files=0,
        repo_tracked_blob_bytes=None,
    )


def is_python_ast_path(path: str) -> bool:
    return file_extension(path) in PYTHON_AST_EXTENSIONS


def is_code_like_path(path: str) -> bool:
    return file_extension(path) in CODE_LIKE_EXTENSIONS


def is_docs_path(path: str) -> bool:
    return file_extension(path) in DOCS_EXTENSIONS


def _unquote_ls_tree_path(raw: str) -> str:
    path = raw.strip()
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        inner = path[1:-1]
        try:
            return bytes(inner, "utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return inner
    return path


def parse_ls_tree_line(line: str) -> dict[str, str] | None:
    """Parse one ``git ls-tree -r -l`` line. Returns None for unusable lines."""
    text = line.rstrip("\n")
    if not text.strip():
        return None
    match = _LS_TREE_LINE.match(text)
    if match is None:
        return None
    return {
        "mode": match.group("mode"),
        "type": match.group("type"),
        "object": match.group("object"),
        "size": match.group("size"),
        "path": _unquote_ls_tree_path(match.group("path")),
    }


def parse_ls_tree_output(text: str) -> TreeStats:
    """Aggregate blob entries from ``git ls-tree -r -l`` output.

    Only ``blob`` entries are counted (including symlink blobs). Gitlink
    ``commit`` entries are ignored. Unknown extensions are not code-like.
    """
    n_files = 0
    n_python = 0
    n_code_like = 0
    n_bytes = 0
    bytes_ok = True
    for raw_line in str(text).splitlines():
        parsed = parse_ls_tree_line(raw_line)
        if parsed is None:
            continue
        if parsed["type"] != "blob":
            continue
        n_files += 1
        path = parsed["path"]
        if is_python_ast_path(path):
            n_python += 1
        if is_code_like_path(path):
            n_code_like += 1
        size_raw = parsed["size"]
        if size_raw == "-":
            bytes_ok = False
        else:
            try:
                n_bytes += int(size_raw)
            except ValueError:
                bytes_ok = False
    return TreeStats(
        ok=True,
        error=None,
        repo_tracked_files=n_files,
        repo_python_files=n_python,
        repo_code_like_files=n_code_like,
        repo_tracked_blob_bytes=n_bytes if bytes_ok else None,
    )


def run_ls_tree(repo_path: Path, commit: str, *, timeout: int = 300) -> str:
    """Read-only recursive listing. Offline: ``ls-tree`` only."""
    result = run_git(
        ["ls-tree", "-r", "-l", commit],
        cwd=Path(repo_path),
        allow_network=False,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(
            f"git ls-tree failed (exit {result.returncode}): {stderr or commit}"
        )
    return result.stdout.decode("utf-8", errors="replace")


class TreeStatStore:
    """Cached ``(repo, base_commit)`` tree statistics from a local mirror root.

    Never clones or fetches. Missing repos/commits become ``ok=False``.
    """

    def __init__(self, repos_root: Path) -> None:
        self.repos_root = Path(repos_root)
        self._cache: dict[tuple[str, str], TreeStats] = {}

    def stats(self, repo: str, commit: str) -> TreeStats:
        key = (repo, commit)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        computed = self._compute(repo, commit)
        self._cache[key] = computed
        return computed

    def cached_get(self, repo: str, commit: str) -> TreeStats | None:
        return self._cache.get((repo, commit))

    def cache_size(self) -> int:
        return len(self._cache)

    def _compute(self, repo: str, commit: str) -> TreeStats:
        if not repo:
            return unavailable_tree_stats("repo unavailable")
        if not commit:
            return unavailable_tree_stats("commit not resolved")
        repo_path = cache_path_for_repo(repo, self.repos_root)
        if not is_git_dir(repo_path):
            return unavailable_tree_stats("repo unavailable")
        if resolve_commit(repo_path, commit) is None:
            return unavailable_tree_stats("commit not resolved")
        try:
            output = run_ls_tree(repo_path, commit)
        except (GitError, OSError, subprocess.TimeoutExpired) as exc:
            return unavailable_tree_stats(f"{type(exc).__name__}: {exc}")
        parsed = parse_ls_tree_output(output)
        return parsed


def lookup_tree_stats(
    provider: Any,
    repo: str,
    commit: str,
) -> TreeStats:
    """Accept a TreeStatStore or ``{(repo, commit): TreeStats}`` mapping."""
    if provider is None:
        return unavailable_tree_stats("tree stats provider missing")
    stats_fn = getattr(provider, "stats", None)
    if callable(stats_fn):
        result = stats_fn(repo, commit)
        if isinstance(result, TreeStats):
            return result
        raise TypeError("tree stats provider.stats must return TreeStats")
    if isinstance(provider, Mapping):
        value = provider.get((repo, commit))
        if value is None:
            return unavailable_tree_stats("tree stats cache miss")
        if isinstance(value, TreeStats):
            return value
        raise TypeError("tree stats mapping values must be TreeStats")
    raise TypeError(f"unsupported tree stats provider: {type(provider)!r}")
