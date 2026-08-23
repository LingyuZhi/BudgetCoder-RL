"""SWE-Gym gold-patch file / hunk / line-coordinate oracles (M1C-A).

Oracle localization is extracted from gold ``patch`` only. This module does
not read ``test_patch``, ``hints_text``, ``FAIL_TO_PASS``, or
``PASS_TO_PASS`` into ``gold_edit_files``. It does not extract functions,
classes, or AST symbols (M1C-B). It does not drop instances.

Two derived file views:

- ``gold_edit_files``: gold-patch structure (modified / added / deleted /
  path_changed). Not a Stage-1 retrieval reward target by itself.
- ``base_changed_files``: source-side paths that exist at ``base_commit``.
  Pure added files are excluded. This is the candidate oracle for future
  base-repository localization reward. Filtering is left to M1D.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_SHA256,
    HF_REPO_ID,
    HF_REVISION,
    is_null,
    length_stats,
)

ORACLE_SOURCE = "patch"
OPERATIONS: tuple[str, ...] = ("modified", "added", "deleted", "path_changed")
SPOTLIGHT_INSTANCE_ID = "getmoto__moto-7418"
DEV_NULL_PATHS = frozenset({"/dev/null", "dev/null"})
TEST_LIKE_DIR_NAMES = frozenset({"tests", "test"})


class OracleParseError(ValueError):
    """Explicit oracle extraction failure. Not a silent fallback."""


def _require_unidiff():
    try:
        from unidiff import PatchSet
        from unidiff.constants import DEV_NULL
        from unidiff.errors import UnidiffParseError
    except ImportError as exc:
        raise ImportError(
            "unidiff is required for M1C-A oracle extraction. "
            "Install only this package: pip install 'unidiff>=0.7.5,<1'"
        ) from exc
    return PatchSet, UnidiffParseError, DEV_NULL


def oracle_jsonl_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "interim" / "swe_gym" / "m1c_oracle.jsonl"


def oracle_summary_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "manifests" / "swe_gym_m1c_oracle_summary.json"


def normalize_diff_path(raw: str | None) -> str | None:
    """Strip git ``a/`` / ``b/`` prefixes. Return None for ``/dev/null``."""
    if raw is None:
        return None
    path = str(raw).strip()
    if not path:
        return None
    if "\t" in path:
        path = path.split("\t", 1)[0]
    quoted = len(path) >= 2 and path[0] == '"' and path[-1] == '"'
    if quoted:
        path = path[1:-1]
    if path in DEV_NULL_PATHS:
        return None
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if path in DEV_NULL_PATHS or path == "":
        return None
    return path.replace("\\", "/")


def is_test_like_path(path: str) -> bool:
    """Conservative observational matcher. Not a drop filter."""
    normalized = path.replace("\\", "/").lstrip("./")
    parts = [part for part in normalized.split("/") if part]
    if any(part in TEST_LIKE_DIR_NAMES for part in parts):
        return True
    name = parts[-1] if parts else ""
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    return False


def file_extension(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    index = name.rfind(".")
    if index <= 0:
        return "<none>"
    return name[index:].lower()


def classify_operation(patched_file: Any) -> str:
    """Classify using normalized paths. ``path_changed`` is not a git-rename claim.

    ``unidiff.PatchedFile.is_rename`` only compares source vs target paths; it
    does not read ``rename from`` / ``rename to`` metadata. A copy or other
    path-changing edit would look the same, so the label is ``path_changed``.
    """
    if patched_file.is_added_file:
        return "added"
    if patched_file.is_removed_file:
        return "deleted"
    source_path = normalize_diff_path(patched_file.source_file)
    target_path = normalize_diff_path(patched_file.target_file)
    if source_path is not None and target_path is not None and source_path != target_path:
        return "path_changed"
    if patched_file.is_modified_file:
        return "modified"
    raise OracleParseError("unable to classify file operation")


def _assert_no_dev_null(*paths: str | None) -> None:
    for path in paths:
        if path is None:
            continue
        if path in DEV_NULL_PATHS or path.endswith("/dev/null"):
            raise OracleParseError(f"/dev/null leaked into oracle path: {path!r}")


def _hunk_record(hunk: Any) -> dict[str, Any]:
    removed_source_lines: list[int] = []
    added_target_lines: list[int] = []
    for line in hunk:
        if line.is_removed and line.source_line_no is not None:
            removed_source_lines.append(int(line.source_line_no))
        if line.is_added and line.target_line_no is not None:
            added_target_lines.append(int(line.target_line_no))
    return {
        "source_start": int(hunk.source_start),
        "source_length": int(hunk.source_length),
        "target_start": int(hunk.target_start),
        "target_length": int(hunk.target_length),
        "removed_source_lines": removed_source_lines,
        "added_target_lines": added_target_lines,
    }


def _file_record_from_patched(patched_file: Any) -> dict[str, Any]:
    if patched_file.is_binary_file:
        label = normalize_diff_path(patched_file.path) or str(
            patched_file.source_file or patched_file.target_file
        )
        raise OracleParseError(f"binary patch unsupported: {label}")

    operation = classify_operation(patched_file)
    source_path = normalize_diff_path(patched_file.source_file)
    target_path = normalize_diff_path(patched_file.target_file)

    if operation == "added":
        source_path = None
        path = target_path
    elif operation == "deleted":
        target_path = None
        path = source_path
    else:
        path = target_path if target_path is not None else source_path

    if path is None:
        raise OracleParseError(
            f"canonical path resolved to /dev/null or empty ({operation})"
        )
    _assert_no_dev_null(path, source_path, target_path)

    hunks = [_hunk_record(hunk) for hunk in patched_file]
    return {
        "path": path,
        "source_path": source_path,
        "target_path": target_path,
        "operation": operation,
        "num_hunks": len(hunks),
        "n_added_lines": sum(len(hunk["added_target_lines"]) for hunk in hunks),
        "n_removed_lines": sum(len(hunk["removed_source_lines"]) for hunk in hunks),
        "hunks": hunks,
    }


def _file_sort_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record["path"]),
        str(record.get("source_path") or ""),
        str(record.get("target_path") or ""),
        str(record["operation"]),
    )


def _hunk_sort_key(hunk: Mapping[str, Any]) -> tuple[int, int]:
    return (int(hunk["source_start"]), int(hunk["target_start"]))


def _merge_file_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_path: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_path.setdefault(str(record["path"]), []).append(record)

    merged: list[dict[str, Any]] = []
    for path, group in by_path.items():
        identities = {
            (
                item.get("source_path"),
                item.get("target_path"),
                item["operation"],
            )
            for item in group
        }
        if len(identities) != 1:
            raise OracleParseError(
                f"conflicting operations for path {path!r}: {sorted(identities)!r}"
            )
        hunks = [hunk for item in group for hunk in item["hunks"]]
        hunks.sort(key=_hunk_sort_key)
        first = group[0]
        merged.append(
            {
                "path": path,
                "source_path": first.get("source_path"),
                "target_path": first.get("target_path"),
                "operation": first["operation"],
                "num_hunks": len(hunks),
                "n_added_lines": sum(len(hunk["added_target_lines"]) for hunk in hunks),
                "n_removed_lines": sum(
                    len(hunk["removed_source_lines"]) for hunk in hunks
                ),
                "hunks": hunks,
            }
        )
    merged.sort(key=_file_sort_key)
    return merged


def derive_base_changed_files(gold_edit_files: Sequence[Mapping[str, Any]]) -> list[str]:
    """Source-side paths present at base_commit. Pure added files are excluded."""
    paths: list[str] = []
    seen: set[str] = set()
    for item in gold_edit_files:
        source = item.get("source_path")
        if source is None:
            continue
        path = str(source)
        if path in DEV_NULL_PATHS or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return sorted(paths)


def gold_edit_paths(gold_edit_files: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(item["path"]) for item in gold_edit_files})


@dataclass
class OracleParseResult:
    parse_ok: bool
    parse_error: str | None
    files: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_gold_edit_files(self) -> int:
        return len(self.files)

    @property
    def n_hunks(self) -> int:
        return sum(int(item["num_hunks"]) for item in self.files)

    @property
    def n_added_lines(self) -> int:
        return sum(int(item["n_added_lines"]) for item in self.files)

    @property
    def n_removed_lines(self) -> int:
        return sum(int(item["n_removed_lines"]) for item in self.files)

    @property
    def test_like_gold_paths(self) -> list[str]:
        paths = [item["path"] for item in self.files if is_test_like_path(item["path"])]
        return sorted(set(paths))

    @property
    def base_changed_files(self) -> list[str]:
        return derive_base_changed_files(self.files)


def _fail(message: str) -> OracleParseResult:
    return OracleParseResult(parse_ok=False, parse_error=message, files=[])


def extract_oracle_from_patch(patch: str) -> OracleParseResult:
    """Parse one gold unified diff. Does not accept or read test_patch."""
    PatchSet, UnidiffParseError, _dev_null = _require_unidiff()
    if patch is None or str(patch).strip() == "":
        return _fail("empty patch")

    try:
        patchset = PatchSet(str(patch))
        raw_files = [_file_record_from_patched(patched) for patched in patchset]
        if not raw_files:
            raise OracleParseError("patch parsed but contained no file records")
        files = _merge_file_records(raw_files)
    except UnidiffParseError as exc:
        return _fail(f"UnidiffParseError: {exc}")
    except OracleParseError as exc:
        return _fail(f"OracleParseError: {exc}")
    except Exception as exc:  # noqa: BLE001 — report parser errors, do not fallback
        return _fail(f"{type(exc).__name__}: {exc}")

    return OracleParseResult(parse_ok=True, parse_error=None, files=files)


def oracle_record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build one oracle record. Reads only instance_id, repo, and patch."""
    instance_id = "" if is_null(row.get("instance_id")) else str(row["instance_id"])
    repo = "" if is_null(row.get("repo")) else str(row["repo"])
    patch = "" if is_null(row.get("patch")) else str(row["patch"])
    parsed = extract_oracle_from_patch(patch)
    return {
        "instance_id": instance_id,
        "repo": repo,
        "parse_ok": parsed.parse_ok,
        "parse_error": parsed.parse_error,
        "oracle_source": ORACLE_SOURCE,
        "gold_edit_files": parsed.files,
        "base_changed_files": parsed.base_changed_files,
        "n_gold_edit_files": parsed.n_gold_edit_files,
        "n_base_changed_files": len(parsed.base_changed_files),
        "n_hunks": parsed.n_hunks,
        "n_added_lines": parsed.n_added_lines,
        "n_removed_lines": parsed.n_removed_lines,
        "test_like_gold_paths": parsed.test_like_gold_paths,
    }


def _row_mapping(frame: Any, index: Any) -> dict[str, Any]:
    row = frame.loc[index]
    return {str(column): row[column] for column in frame.columns}


def extract_oracle_frame(frame: Any) -> list[dict[str, Any]]:
    """Extract one record per input row. Never drops rows."""
    records = [
        oracle_record_from_row(_row_mapping(frame, index)) for index in frame.index
    ]
    records.sort(key=lambda item: item["instance_id"])
    return records


def _argmax_instance(
    records: Sequence[Mapping[str, Any]], field_name: str
) -> dict[str, Any] | None:
    if not records:
        return None
    best = max(records, key=lambda item: (int(item[field_name]), item["instance_id"]))
    return {
        "instance_id": best["instance_id"],
        field_name: int(best[field_name]),
    }


def _operation_counts(files: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item["operation"]) for item in files)
    return {name: int(counts.get(name, 0)) for name in OPERATIONS}


def _extension_distribution(files: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in files:
        counts[file_extension(str(item["path"]))] += 1
    return {
        key: int(counts[key])
        for key in sorted(counts, key=lambda name: (-counts[name], name))
    }


def _spotlight(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {item["instance_id"]: item for item in records}
    record = by_id.get(SPOTLIGHT_INSTANCE_ID)
    if record is None:
        return {
            "instance_id": SPOTLIGHT_INSTANCE_ID,
            "present_in_input": False,
        }
    return {
        "instance_id": SPOTLIGHT_INSTANCE_ID,
        "present_in_input": True,
        "parse_ok": record["parse_ok"],
        "parse_error": record["parse_error"],
        "n_gold_edit_files": int(record["n_gold_edit_files"]),
        "n_base_changed_files": int(record["n_base_changed_files"]),
        "n_hunks": int(record["n_hunks"]),
        "n_added_lines": int(record["n_added_lines"]),
        "n_removed_lines": int(record["n_removed_lines"]),
        "operations": _operation_counts(record["gold_edit_files"]),
        "test_like_gold_paths": list(record["test_like_gold_paths"]),
    }


def build_oracle_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_rows = len(records)
    ok_records = [item for item in records if item["parse_ok"]]
    fail_records = [item for item in records if not item["parse_ok"]]
    all_files = [file_rec for item in ok_records for file_rec in item["gold_edit_files"]]
    test_like_instances = [
        item for item in ok_records if item["test_like_gold_paths"]
    ]
    n_test_like_files = sum(len(item["test_like_gold_paths"]) for item in ok_records)

    def _ops(item: Mapping[str, Any]) -> set[str]:
        return {str(file_rec["operation"]) for file_rec in item["gold_edit_files"]}

    n_with_added = sum(1 for item in ok_records if "added" in _ops(item))
    n_added_only = sum(1 for item in ok_records if _ops(item) == {"added"})
    n_zero_base = sum(1 for item in ok_records if int(item["n_base_changed_files"]) == 0)
    n_with_deleted = sum(1 for item in ok_records if "deleted" in _ops(item))
    n_with_path_changed = sum(1 for item in ok_records if "path_changed" in _ops(item))
    return {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": HF_REVISION,
        "sha256": EXPECTED_SHA256,
        "oracle_source": ORACLE_SOURCE,
        "n_rows": n_rows,
        "n_rows_written": n_rows,
        "rows_dropped": 0,
        "successfully_parsed": len(ok_records),
        "parse_failure_count": len(fail_records),
        "parse_failure_instance_ids": [item["instance_id"] for item in fail_records],
        "parse_failures": [
            {
                "instance_id": item["instance_id"],
                "parse_error": item["parse_error"],
            }
            for item in fail_records
        ],
        "gold_edit_files_per_instance": length_stats(
            item["n_gold_edit_files"] for item in ok_records
        ),
        "base_changed_files_per_instance": length_stats(
            item["n_base_changed_files"] for item in ok_records
        ),
        "hunks_per_instance": length_stats(item["n_hunks"] for item in ok_records),
        "added_lines_per_instance": length_stats(
            item["n_added_lines"] for item in ok_records
        ),
        "removed_lines_per_instance": length_stats(
            item["n_removed_lines"] for item in ok_records
        ),
        "file_operation_counts": _operation_counts(all_files),
        "file_extension_distribution": _extension_distribution(all_files),
        "instance_file_views": {
            "n_with_added_file": n_with_added,
            "n_added_only": n_added_only,
            "n_zero_base_changed_files": n_zero_base,
            "n_with_deleted_file": n_with_deleted,
            "n_with_path_changed": n_with_path_changed,
            "note": (
                "Observational only. Not a drop filter. "
                "Whether to exclude zero-base-visible or added-heavy "
                "instances is left to M1D."
            ),
        },
        "outliers": {
            "max_gold_edit_files": _argmax_instance(ok_records, "n_gold_edit_files"),
            "max_base_changed_files": _argmax_instance(
                ok_records, "n_base_changed_files"
            ),
            "max_hunks": _argmax_instance(ok_records, "n_hunks"),
            "max_added_lines": _argmax_instance(ok_records, "n_added_lines"),
            "max_removed_lines": _argmax_instance(ok_records, "n_removed_lines"),
        },
        "test_like_gold_paths": {
            "n_instances": len(test_like_instances),
            "n_files": n_test_like_files,
            "note": (
                "Observational only. Test-like gold paths are not dropped "
                "and are not mixed in from test_patch."
            ),
        },
        "spotlight": _spotlight(records),
        "notes": {
            "source_coordinates": "base_commit (removed_source_lines)",
            "target_coordinates": "gold-patched (added_target_lines)",
            "gold_edit_files": (
                "Gold patch structure. May include added target paths that "
                "do not exist at base_commit. Not the Stage-1 retrieval "
                "reward target."
            ),
            "base_changed_files": (
                "Source-side paths present at base_commit. Candidate oracle "
                "for future base-repository localization reward."
            ),
            "path_changed": (
                "source_path != target_path after a/b normalization. "
                "Does not claim Git rename metadata."
            ),
            "no_oracle_lines_field": True,
            "ast_extraction": False,
            "test_patch_used_for_oracle": False,
            "zero_base_visible_filter": "deferred to M1D",
        },
    }
