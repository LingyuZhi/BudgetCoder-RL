"""Base-repository Python symbol extraction and change-site mapping (M1C-B).

Pure functions: no Git, no network. Decode uses stdlib encoding detection.
AST uses stdlib ``ast``. Mapping never invents symbols that are absent from
the base source.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_oracle import (
    classify_operation,
    file_extension,
    normalize_diff_path,
)

ELIGIBLE_EXTENSIONS = frozenset({".py", ".pyi"})
SYMBOL_KINDS = ("class", "function", "async_function")
SITE_REMOVED = "removed_source_line"
SITE_ADDITION = "addition_run"
EVIDENCE_REMOVED = "removed_source_line"
EVIDENCE_ADDITION_SAME = "addition_anchor_same_symbol"
EVIDENCE_ADDITION_ONE_SIDED = "addition_anchor_one_sided"
REASON_MODULE_LEVEL = "module_level"
REASON_NO_ENCLOSING = "no_enclosing_symbol"
REASON_AMBIGUOUS = "ambiguous_addition_anchor"
REASON_UNSUPPORTED = "unsupported_extension"
REASON_DECODE = "decode_error"
REASON_AST = "ast_parse_error"
REASON_SPAN = "ast_span_missing"
REASON_MISSING_BLOB = "missing_blob"
REASON_MISSING_COMMIT = "missing_commit"
REASON_MISSING_REPO = "missing_repo"
REASON_INVALID_PATH = "invalid_path"
FILE_STATUS_OK = "ok"


def parser_version_record() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_version_info": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
    }


def is_eligible_path(path: str) -> bool:
    return file_extension(path) in ELIGIBLE_EXTENSIONS


def decode_python_source(blob: bytes) -> tuple[str, str]:
    """Decode source bytes with stdlib encoding detection. Strict."""
    readline = io.BytesIO(blob).readline
    encoding, _consumed = tokenize.detect_encoding(readline)
    return blob.decode(encoding), encoding


@dataclass(frozen=True)
class BaseSymbol:
    path: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    depth: int

    @property
    def symbol_id(self) -> str:
        return (
            f"{self.path}:{self.qualname}:{self.kind}:"
            f"{self.start_line}-{self.end_line}"
        )

    @property
    def span(self) -> int:
        return int(self.end_line) - int(self.start_line)

    def to_record(self, evidence: Sequence[str] | None = None) -> dict[str, Any]:
        record = {
            "path": self.path,
            "qualname": self.qualname,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "depth": self.depth,
            "symbol_id": self.symbol_id,
        }
        if evidence is not None:
            record["evidence"] = list(evidence)
        return record


@dataclass(frozen=True)
class AdditionRun:
    prev_line: int | None
    next_line: int | None
    n_added_lines: int


@dataclass
class FileSymbols:
    path: str
    status: str
    error: str | None = None
    encoding: str | None = None
    symbols: list[BaseSymbol] = field(default_factory=list)
    n_span_missing: int = 0

    @property
    def ok(self) -> bool:
        return self.status == FILE_STATUS_OK


def _kind_for_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_function"
    if isinstance(node, ast.FunctionDef):
        return "function"
    return None


def _decorator_start(node: ast.AST) -> int:
    start = int(getattr(node, "lineno"))
    for decorator in getattr(node, "decorator_list", []) or []:
        lineno = getattr(decorator, "lineno", None)
        if lineno is not None:
            start = min(start, int(lineno))
    return start


def extract_base_symbols(source: str, path: str) -> tuple[list[BaseSymbol], int]:
    """Parse base source and collect class/function spans. No fake module symbol."""
    tree = ast.parse(source, filename=path)
    symbols: list[BaseSymbol] = []
    n_span_missing = 0

    def visit(node: ast.AST, prefix: str, depth: int) -> None:
        nonlocal n_span_missing
        kind = _kind_for_node(node)
        if kind is None:
            for child in ast.iter_child_nodes(node):
                visit(child, prefix, depth)
            return
        name = str(getattr(node, "name"))
        qualname = f"{prefix}.{name}" if prefix else name
        start_line = _decorator_start(node)
        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            n_span_missing += 1
        else:
            symbols.append(
                BaseSymbol(
                    path=path,
                    qualname=qualname,
                    kind=kind,
                    start_line=int(start_line),
                    end_line=int(end_line),
                    depth=depth,
                )
            )
        child_prefix = qualname
        child_depth = depth + 1
        for child in ast.iter_child_nodes(node):
            visit(child, child_prefix, child_depth)

    visit(tree, "", 0)
    symbols.sort(key=_symbol_sort_key)
    return symbols, n_span_missing


def _symbol_sort_key(symbol: BaseSymbol) -> tuple[str, int, int, str, str]:
    return (symbol.path, symbol.start_line, symbol.end_line, symbol.qualname, symbol.kind)


def symbols_from_blob(path: str, blob: bytes) -> FileSymbols:
    if not is_eligible_path(path):
        return FileSymbols(path=path, status=REASON_UNSUPPORTED)
    try:
        source, encoding = decode_python_source(blob)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        return FileSymbols(
            path=path,
            status=REASON_DECODE,
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        symbols, n_span_missing = extract_base_symbols(source, path)
    except SyntaxError as exc:
        return FileSymbols(
            path=path,
            status=REASON_AST,
            error=f"SyntaxError: {exc}",
            encoding=encoding,
        )
    except Exception as exc:  # noqa: BLE001 — report, do not guess symbols
        return FileSymbols(
            path=path,
            status=REASON_AST,
            error=f"{type(exc).__name__}: {exc}",
            encoding=encoding,
        )
    return FileSymbols(
        path=path,
        status=FILE_STATUS_OK,
        encoding=encoding,
        symbols=symbols,
        n_span_missing=n_span_missing,
    )


def innermost_symbol(line: int, symbols: Sequence[BaseSymbol]) -> BaseSymbol | None:
    containing = [
        symbol
        for symbol in symbols
        if symbol.start_line <= line <= symbol.end_line
    ]
    if not containing:
        return None
    return max(
        containing,
        key=lambda item: (
            item.depth,
            -item.span,
            -item.start_line,
            item.qualname,
            item.kind,
        ),
    )


def _require_unidiff():
    try:
        from unidiff import PatchSet
        from unidiff.errors import UnidiffParseError
    except ImportError as exc:
        raise ImportError(
            "unidiff is required for M1C-B addition-anchor mapping. "
            "Install only this package: pip install 'unidiff>=0.7.5,<1'"
        ) from exc
    return PatchSet, UnidiffParseError


def _existing_source_line(line: Any) -> int | None:
    if getattr(line, "is_added", False):
        return None
    number = getattr(line, "source_line_no", None)
    if number is None:
        return None
    return int(number)


def addition_runs_from_patched_file(patched_file: Any) -> list[AdditionRun]:
    runs: list[AdditionRun] = []
    for hunk in patched_file:
        lines = list(hunk)
        last_source: int | None = None
        index = 0
        while index < len(lines):
            line = lines[index]
            if getattr(line, "is_added", False):
                start = index
                while index < len(lines) and getattr(lines[index], "is_added", False):
                    index += 1
                next_source: int | None = None
                for lookahead in lines[index:]:
                    candidate = _existing_source_line(lookahead)
                    if candidate is not None:
                        next_source = candidate
                        break
                runs.append(
                    AdditionRun(
                        prev_line=last_source,
                        next_line=next_source,
                        n_added_lines=index - start,
                    )
                )
                continue
            existing = _existing_source_line(line)
            if existing is not None:
                last_source = existing
            index += 1
    return runs


def addition_runs_by_source_path(patch: str) -> dict[str, list[AdditionRun]]:
    """Walk unified-diff line stream. Added files (no source_path) are skipped."""
    PatchSet, UnidiffParseError = _require_unidiff()
    try:
        patchset = PatchSet(str(patch))
    except UnidiffParseError:
        return {}
    by_path: dict[str, list[AdditionRun]] = {}
    for patched in patchset:
        if getattr(patched, "is_added_file", False):
            continue
        try:
            operation = classify_operation(patched)
        except Exception:  # noqa: BLE001
            operation = None
        if operation == "added":
            continue
        source_path = normalize_diff_path(patched.source_file)
        if source_path is None:
            continue
        by_path.setdefault(source_path, []).extend(
            addition_runs_from_patched_file(patched)
        )
    return by_path


def _site_record(
    *,
    path: str,
    site_kind: str,
    reason: str,
    line: int | None = None,
    prev_line: int | None = None,
    next_line: int | None = None,
    n_added_lines: int | None = None,
) -> dict[str, Any]:
    return {
        "path": path,
        "site_kind": site_kind,
        "line": line,
        "prev_line": prev_line,
        "next_line": next_line,
        "n_added_lines": n_added_lines,
        "reason": reason,
    }


def map_removed_line(
    line: int, symbols: Sequence[BaseSymbol]
) -> tuple[BaseSymbol | None, str | None]:
    symbol = innermost_symbol(line, symbols)
    if symbol is not None:
        return symbol, None
    return None, REASON_MODULE_LEVEL


def map_addition_run(
    run: AdditionRun, symbols: Sequence[BaseSymbol]
) -> tuple[BaseSymbol | None, str | None, str | None]:
    """Return (symbol, evidence, unmapped_reason)."""
    prev_symbol = (
        innermost_symbol(run.prev_line, symbols) if run.prev_line is not None else None
    )
    next_symbol = (
        innermost_symbol(run.next_line, symbols) if run.next_line is not None else None
    )
    prev_exists = run.prev_line is not None
    next_exists = run.next_line is not None

    if prev_exists and next_exists:
        if prev_symbol is not None and next_symbol is not None:
            if prev_symbol.symbol_id == next_symbol.symbol_id:
                return prev_symbol, EVIDENCE_ADDITION_SAME, None
            return None, None, REASON_AMBIGUOUS
        if prev_symbol is None and next_symbol is None:
            return None, None, REASON_MODULE_LEVEL
        return None, None, REASON_AMBIGUOUS

    if prev_exists and not next_exists:
        if prev_symbol is not None:
            return prev_symbol, EVIDENCE_ADDITION_ONE_SIDED, None
        return None, None, REASON_MODULE_LEVEL

    if next_exists and not prev_exists:
        if next_symbol is not None:
            return next_symbol, EVIDENCE_ADDITION_ONE_SIDED, None
        return None, None, REASON_MODULE_LEVEL

    return None, None, REASON_AMBIGUOUS


def _merge_oracle_symbol(
    bucket: dict[str, dict[str, Any]], symbol: BaseSymbol, evidence: str
) -> None:
    current = bucket.get(symbol.symbol_id)
    if current is None:
        bucket[symbol.symbol_id] = symbol.to_record(evidence=[evidence])
        return
    evidence_set = set(current["evidence"])
    evidence_set.add(evidence)
    current["evidence"] = sorted(evidence_set)


def map_file_change_sites(
    *,
    path: str,
    file_symbols: FileSymbols | None,
    removed_lines: Sequence[int],
    addition_runs: Sequence[AdditionRun],
    file_reason: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Map base change sites. file_reason overrides per-site mapping."""
    mapped: dict[str, dict[str, Any]] = {}
    unmapped: list[dict[str, Any]] = []
    counts = {
        "n_removed_line_sites": 0,
        "n_removed_line_mapped": 0,
        "n_addition_run_sites": 0,
        "n_addition_anchor_same_symbol": 0,
        "n_addition_anchor_one_sided": 0,
        "n_module_level": 0,
        "n_no_enclosing_symbol": 0,
        "n_ambiguous_addition_anchor": 0,
    }
    reason = file_reason
    if reason is None and file_symbols is not None and not file_symbols.ok:
        reason = file_symbols.status
    symbols = file_symbols.symbols if file_symbols is not None and file_symbols.ok else []

    for line in removed_lines:
        counts["n_removed_line_sites"] += 1
        if reason is not None:
            unmapped.append(
                _site_record(
                    path=path,
                    site_kind=SITE_REMOVED,
                    reason=reason,
                    line=int(line),
                )
            )
            continue
        symbol, unmapped_reason = map_removed_line(int(line), symbols)
        if symbol is not None:
            _merge_oracle_symbol(mapped, symbol, EVIDENCE_REMOVED)
            counts["n_removed_line_mapped"] += 1
        else:
            site_reason = unmapped_reason or REASON_NO_ENCLOSING
            unmapped.append(
                _site_record(
                    path=path,
                    site_kind=SITE_REMOVED,
                    reason=site_reason,
                    line=int(line),
                )
            )
            if site_reason == REASON_MODULE_LEVEL:
                counts["n_module_level"] += 1
            else:
                counts["n_no_enclosing_symbol"] += 1

    for run in addition_runs:
        counts["n_addition_run_sites"] += 1
        if reason is not None:
            unmapped.append(
                _site_record(
                    path=path,
                    site_kind=SITE_ADDITION,
                    reason=reason,
                    prev_line=run.prev_line,
                    next_line=run.next_line,
                    n_added_lines=run.n_added_lines,
                )
            )
            continue
        symbol, evidence, unmapped_reason = map_addition_run(run, symbols)
        if symbol is not None and evidence is not None:
            _merge_oracle_symbol(mapped, symbol, evidence)
            if evidence == EVIDENCE_ADDITION_SAME:
                counts["n_addition_anchor_same_symbol"] += 1
            elif evidence == EVIDENCE_ADDITION_ONE_SIDED:
                counts["n_addition_anchor_one_sided"] += 1
        else:
            site_reason = unmapped_reason or REASON_AMBIGUOUS
            unmapped.append(
                _site_record(
                    path=path,
                    site_kind=SITE_ADDITION,
                    reason=site_reason,
                    prev_line=run.prev_line,
                    next_line=run.next_line,
                    n_added_lines=run.n_added_lines,
                )
            )
            if site_reason == REASON_MODULE_LEVEL:
                counts["n_module_level"] += 1
            elif site_reason == REASON_NO_ENCLOSING:
                counts["n_no_enclosing_symbol"] += 1
            else:
                counts["n_ambiguous_addition_anchor"] += 1

    oracle = sorted(
        mapped.values(),
        key=lambda item: (
            str(item["path"]),
            int(item["start_line"]),
            int(item["end_line"]),
            str(item["qualname"]),
            str(item["kind"]),
        ),
    )
    return oracle, unmapped, counts


def map_sources_and_patch(
    patch: str,
    sources: Mapping[str, bytes],
    *,
    file_reasons: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Map one patch against in-memory base blobs. Used by tests."""
    from budget_coder_rl.data.swe_gym_oracle import extract_oracle_from_patch

    parsed = extract_oracle_from_patch(patch)
    runs_by_path = addition_runs_by_source_path(patch) if parsed.parse_ok else {}
    reasons = dict(file_reasons or {})
    oracle: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    file_results: list[dict[str, Any]] = []
    totals = {
        "n_removed_line_sites": 0,
        "n_removed_line_mapped": 0,
        "n_addition_run_sites": 0,
        "n_addition_anchor_same_symbol": 0,
        "n_addition_anchor_one_sided": 0,
        "n_module_level": 0,
        "n_no_enclosing_symbol": 0,
        "n_ambiguous_addition_anchor": 0,
    }
    by_path = {item["source_path"]: item for item in parsed.files if item.get("source_path")}
    for path in parsed.base_changed_files:
        file_rec = by_path.get(path, {})
        removed = [
            line
            for hunk in file_rec.get("hunks", [])
            for line in hunk.get("removed_source_lines", [])
        ]
        runs = runs_by_path.get(path, [])
        reason = reasons.get(path)
        file_symbols = None
        if reason is None:
            blob = sources.get(path)
            if blob is None:
                reason = REASON_MISSING_BLOB
            else:
                file_symbols = symbols_from_blob(path, blob)
        mapped, file_unmapped, counts = map_file_change_sites(
            path=path,
            file_symbols=file_symbols,
            removed_lines=removed,
            addition_runs=runs,
            file_reason=reason,
        )
        oracle.extend(mapped)
        unmapped.extend(file_unmapped)
        for key, value in counts.items():
            totals[key] += value
        status = reason or (file_symbols.status if file_symbols is not None else REASON_MISSING_BLOB)
        file_results.append(
            {
                "path": path,
                "status": status,
                "error": None if file_symbols is None else file_symbols.error,
                "extension": file_extension(path),
                "eligible": is_eligible_path(path),
                "encoding": None if file_symbols is None else file_symbols.encoding,
                "n_symbols_extracted": 0 if file_symbols is None else len(file_symbols.symbols),
                "n_oracle_symbols": len(mapped),
                "n_span_missing": 0 if file_symbols is None else file_symbols.n_span_missing,
            }
        )
    oracle.sort(
        key=lambda item: (
            str(item["path"]),
            int(item["start_line"]),
            int(item["end_line"]),
            str(item["qualname"]),
            str(item["kind"]),
        )
    )
    file_results.sort(key=lambda item: str(item["path"]))
    return {
        "parse_ok": parsed.parse_ok,
        "parse_error": parsed.parse_error,
        "oracle_symbols": oracle,
        "unmapped_sites": unmapped,
        "file_results": file_results,
        "counts": totals,
        "base_changed_files": list(parsed.base_changed_files),
        "n_skipped_added_files": sum(
            1 for item in parsed.files if item["operation"] == "added"
        ),
    }
