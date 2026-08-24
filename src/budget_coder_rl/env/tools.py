"""Stage-1 exploration tools over an M2A ``RepoFileView``.

Lexical ``tree`` / ``search`` / ``read`` only. Does not follow directory
symlinks, does not honor gitignore/rg config, and does not implement
cumulative budget.
"""

from __future__ import annotations

import os
from pathlib import Path

from budget_coder_rl.data.swe_gym_repos import is_safe_repo_path
from budget_coder_rl.env.repo_workspace import PathEscapeError, RepoFileView
from budget_coder_rl.protocol.observation import (
    format_read,
    format_search,
    format_tree,
)
from budget_coder_rl.protocol.parser import (
    SEARCH_DEFAULT_MAX_RESULTS,
    SEARCH_DEFAULT_PATH,
    TREE_DEFAULT_DEPTH,
    TREE_DEFAULT_PATH,
)

TREE_MAX_DEPTH = 8
TREE_MAX_ENTRIES = 200
SEARCH_MAX_RESULTS = 200
SEARCH_MAX_FILE_BYTES = 1_000_000
SEARCH_BINARY_PROBE_BYTES = 8192
READ_MAX_LINES = 200
READ_MAX_CHARS = 32768
QUERY_MAX_CHARS = 256
READ_LINE_NO_WIDTH = 6


class ToolError(Exception):
    """Expected agent/tool misuse. Session maps this to an error observation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ExplorationTools:
    """Execute structured exploration actions on a confined snapshot view."""

    def __init__(self, view: RepoFileView) -> None:
        self.view = view

    def execute(self, name: str, arguments: dict) -> str:
        if name == "tree":
            return self.tree(
                path=arguments.get("path", TREE_DEFAULT_PATH),
                depth=arguments.get("depth", TREE_DEFAULT_DEPTH),
            )
        if name == "search":
            return self.search(
                query=arguments["query"],
                path=arguments.get("path", SEARCH_DEFAULT_PATH),
                max_results=arguments.get("max_results", SEARCH_DEFAULT_MAX_RESULTS),
            )
        if name == "read":
            return self.read(
                path=arguments["path"],
                start_line=arguments["start_line"],
                end_line=arguments["end_line"],
            )
        raise ToolError("unknown_tool", f"unknown tool {name!r}")

    def tree(self, path: str = TREE_DEFAULT_PATH, depth: int = TREE_DEFAULT_DEPTH) -> str:
        _require_depth(depth)
        start = _safe_start(self.view, path)
        # Full bounded-depth set, then POSIX sort, then truncate.
        entries = _collect_tree_entries(self.view.repo_root, start, depth)
        entries.sort(key=lambda item: item[1])
        truncated = len(entries) > TREE_MAX_ENTRIES
        if truncated:
            entries = entries[:TREE_MAX_ENTRIES]
        return format_tree(
            path=path,
            depth=depth,
            truncated=truncated,
            entries=entries,
        )

    def search(
        self,
        query: str,
        path: str = SEARCH_DEFAULT_PATH,
        max_results: int = SEARCH_DEFAULT_MAX_RESULTS,
    ) -> str:
        if not query:
            raise ToolError("invalid_argument", "query must be nonempty")
        if len(query) > QUERY_MAX_CHARS:
            raise ToolError(
                "invalid_argument",
                f"query exceeds {QUERY_MAX_CHARS} characters",
            )
        _require_max_results(max_results)
        start = _safe_start(self.view, path)
        # Sorted file list first; max_results then cuts that sequence.
        files = _collect_regular_files(self.view.repo_root, start)
        hits: list[tuple[str, int, str]] = []
        truncated = False
        for file_path in files:
            rel = _relpath(file_path, self.view.repo_root)
            text = _read_search_text(file_path)
            if text is None:
                continue
            for line_no, line in enumerate(_split_lines(text), start=1):
                if query not in line:
                    continue
                if len(hits) >= max_results:
                    truncated = True
                    break
                hits.append((rel, line_no, line))
            if truncated:
                break
        return format_search(
            query=query,
            path=path,
            truncated=truncated,
            hits=hits,
        )

    def read(self, path: str, start_line: int, end_line: int) -> str:
        if start_line < 1 or end_line < start_line:
            raise ToolError(
                "invalid_range",
                f"invalid line range: {start_line}-{end_line}",
            )
        try:
            data = self.view.read_bytes(path)
        except PathEscapeError as exc:
            raise ToolError("path_escape", str(exc)) from exc
        except FileNotFoundError:
            raise ToolError("path_not_found", f"path does not exist: {path!r}") from None
        except IsADirectoryError:
            raise ToolError("not_a_file", f"path is a directory: {path!r}") from None
        if b"\x00" in data:
            raise ToolError("undecodable", f"path is binary: {path!r}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("undecodable", f"path is not utf-8: {path!r}") from exc
        lines = _split_lines(text)
        if start_line > len(lines):
            raise ToolError(
                "invalid_range",
                f"start_line {start_line} exceeds file length {len(lines)}",
            )
        last = min(end_line, len(lines))
        truncated = False
        if last - start_line + 1 > READ_MAX_LINES:
            last = start_line + READ_MAX_LINES - 1
            truncated = True
        numbered: list[str] = []
        chars = 0
        actual_end = start_line - 1
        for line_no in range(start_line, last + 1):
            original = lines[line_no - 1]
            remaining = READ_MAX_CHARS - chars
            if remaining <= 0:
                truncated = True
                break
            line = original
            if len(line) > remaining:
                line = line[:remaining]
                truncated = True
            numbered.append(f"{line_no:{READ_LINE_NO_WIDTH}d}|{line}")
            chars += len(line)
            actual_end = line_no
            if len(original) > remaining:
                break
        if not numbered:
            raise ToolError("invalid_range", f"invalid line range: {start_line}-{end_line}")
        return format_read(
            path=path,
            start_line=start_line,
            end_line=actual_end,
            truncated=truncated,
            numbered_lines=numbered,
        )


def _require_depth(depth: int) -> None:
    if depth < 0 or depth > TREE_MAX_DEPTH:
        raise ToolError(
            "invalid_argument",
            f"depth must be between 0 and {TREE_MAX_DEPTH}",
        )


def _require_max_results(max_results: int) -> None:
    if max_results < 1 or max_results > SEARCH_MAX_RESULTS:
        raise ToolError(
            "invalid_argument",
            f"max_results must be between 1 and {SEARCH_MAX_RESULTS}",
        )


def _safe_start(view: RepoFileView, relative_path: str) -> Path:
    root = view.repo_root
    if relative_path in {".", "./"}:
        candidate = root
    else:
        if not is_safe_repo_path(relative_path):
            raise ToolError("path_escape", f"unsafe repository path: {relative_path!r}")
        candidate = root.joinpath(*relative_path.replace("\\", "/").split("/"))
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ToolError(
                "path_escape",
                f"path escaped repo root: {relative_path!r}",
            ) from exc
    if candidate.is_symlink():
        return candidate
    if not candidate.exists():
        raise ToolError("path_not_found", f"path does not exist: {relative_path!r}")
    return candidate


def _collect_tree_entries(root: Path, start: Path, depth: int) -> list[tuple[str, str]]:
    entries = [(_entry_kind(start), _relpath(start, root))]
    if _entry_kind(start) != "d" or depth <= 0:
        return entries
    stack: list[tuple[Path, int]] = [(start, 0)]
    while stack:
        current, current_depth = stack.pop()
        if current_depth >= depth:
            continue
        with os.scandir(current) as handle:
            children = sorted(handle, key=lambda item: item.name)
        for child in children:
            child_path = Path(child.path)
            kind = _direntry_kind(child)
            entries.append((kind, _relpath(child_path, root)))
            if kind == "d":
                stack.append((child_path, current_depth + 1))
    return entries


def _collect_regular_files(root: Path, start: Path) -> list[Path]:
    if start.is_symlink():
        return []
    if start.is_file():
        return [start]
    if not start.is_dir():
        raise ToolError("not_a_directory", f"path is not a directory: {_relpath(start, root)}")
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(
        start,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if path.is_symlink():
                continue
            if path.is_file():
                files.append(path)
    files.sort(key=lambda item: _relpath(item, root))
    return files


def _raise_walk_error(err: OSError) -> None:
    """os.walk defaults to ignoring scandir errors; that violates infra-fail."""
    raise err


def _read_search_text(path: Path) -> str | None:
    with path.open("rb") as handle:
        data = handle.read(SEARCH_MAX_FILE_BYTES)
    probe = data[:SEARCH_BINARY_PROBE_BYTES]
    if b"\x00" in probe:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _split_lines(text: str) -> list[str]:
    if text == "":
        return []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def _entry_kind(path: Path) -> str:
    if path.is_symlink():
        return "l"
    if path.is_dir():
        return "d"
    return "f"


def _direntry_kind(entry: os.DirEntry[str]) -> str:
    if entry.is_symlink():
        return "l"
    if entry.is_dir(follow_symlinks=False):
        return "d"
    return "f"


def _relpath(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "." if rel == "." else rel
