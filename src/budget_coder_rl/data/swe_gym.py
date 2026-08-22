"""Official SWE-Gym raw dataset pins, validation, and profiling (M1A).

This module does not filter, split, or parse gold patches.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HF_REPO_ID = "SWE-Gym/SWE-Gym"
HF_REVISION = "26a6eae79ae9cb6d4307c3cc99c126fbf23cb3f0"
HF_FILENAME = "data/train-00000-of-00001.parquet"
HF_URL = "https://huggingface.co/datasets/SWE-Gym/SWE-Gym"
EXPECTED_SHA256 = "60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97"
EXPECTED_SIZE_BYTES = 43644473
EXPECTED_N_ROWS = 2438
EXPECTED_N_REPOS = 11
REQUIRED_COLUMNS: tuple[str, ...] = (
    "instance_id",
    "hints_text",
    "patch",
    "test_patch",
    "created_at",
    "problem_statement",
    "repo",
    "base_commit",
    "version",
    "PASS_TO_PASS",
    "FAIL_TO_PASS",
)
TEXT_LENGTH_FIELDS: tuple[str, ...] = (
    "problem_statement",
    "hints_text",
    "patch",
    "test_patch",
)
LIST_LENGTH_FIELDS: tuple[str, ...] = ("FAIL_TO_PASS", "PASS_TO_PASS")
PREVIEW_MAX_CHARS = 200

# Paper (arXiv:2412.21139) short-name frequencies; inspect-only, not a hard gate.
PAPER_REPO_SHORT_COUNTS: dict[str, int] = {
    "pandas": 737,
    "MONAI": 374,
    "moto": 343,
    "mypy": 257,
    "dvc": 225,
    "dask": 145,
    "modin": 107,
    "pydantic": 83,
    "conan": 75,
    "hydra": 66,
    "bokeh": 26,
}


def raw_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "raw" / "swe_gym"


def parquet_path(repo_root: Path) -> Path:
    return raw_dir(repo_root) / HF_FILENAME


def source_json_path(repo_root: Path) -> Path:
    return raw_dir(repo_root) / "SOURCE.json"


def profile_json_path(repo_root: Path) -> Path:
    return raw_dir(repo_root) / "profile.json"


def manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "manifests" / "swe_gym_raw.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parquet_file(path: Path) -> list[str]:
    """Return hard-fail reasons if size or sha256 do not match the official pin."""
    path = Path(path)
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"parquet not found: {path}")
        return errors
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != EXPECTED_SIZE_BYTES:
        errors.append(
            f"parquet size {size} != expected {EXPECTED_SIZE_BYTES}"
        )
    if digest != EXPECTED_SHA256:
        errors.append(
            f"parquet sha256 {digest} != expected {EXPECTED_SHA256}"
        )
    return errors


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        import pandas as pd
    except ImportError:
        return False
    if value is getattr(pd, "NA", object()):
        return True
    # pd.NA / NaT scalars only. Do not call pd.isna on list-like values.
    if isinstance(value, (list, tuple, dict, bytes)):
        return False
    if hasattr(value, "dtype") and getattr(value, "shape", None) not in (None, ()):
        return False
    try:
        result = pd.isna(value)
    except (ValueError, TypeError):
        return False
    return bool(result) if result is not None and not hasattr(result, "__iter__") else False


def parse_string_list(value: Any) -> list[str] | None:
    """Normalize PASS_TO_PASS / FAIL_TO_PASS (list or JSON string) to a list."""
    if is_null(value):
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return parse_string_list(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [value]
        if parsed is None:
            return None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def _column_names(frame: Any) -> list[str]:
    return [str(name) for name in frame.columns]


def _series_values(frame: Any, column: str) -> list[Any]:
    return list(frame[column].tolist())


def validate_schema_and_cardinality(
    frame: Any,
    *,
    expected_n_rows: int = EXPECTED_N_ROWS,
    expected_n_repos: int = EXPECTED_N_REPOS,
    required_columns: Sequence[str] = REQUIRED_COLUMNS,
) -> list[str]:
    """Return hard-fail reasons. An empty list means the frame passed."""
    errors: list[str] = []
    columns = _column_names(frame)
    missing = [name for name in required_columns if name not in columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        return errors

    n_rows = int(len(frame))
    if n_rows != expected_n_rows:
        errors.append(f"row count {n_rows} != expected {expected_n_rows}")

    instance_ids = _series_values(frame, "instance_id")
    null_ids = sum(1 for value in instance_ids if is_null(value) or str(value).strip() == "")
    if null_ids:
        errors.append(f"instance_id has {null_ids} null or empty value(s)")
    non_null_ids = [str(value) for value in instance_ids if not is_null(value)]
    if len(set(non_null_ids)) != len(non_null_ids):
        errors.append("instance_id is not unique")

    repos = _series_values(frame, "repo")
    null_repos = sum(1 for value in repos if is_null(value) or str(value).strip() == "")
    if null_repos:
        errors.append(f"repo has {null_repos} null or empty value(s)")
    unique_repos = {str(value) for value in repos if not is_null(value) and str(value).strip() != ""}
    if len(unique_repos) != expected_n_repos:
        errors.append(
            f"unique repo count {len(unique_repos)} != expected {expected_n_repos}"
        )
    return errors


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    weight = idx - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def length_stats(values: Iterable[int]) -> dict[str, float | int | None]:
    numbers = [int(value) for value in values]
    if not numbers:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(numbers)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 4),
        "p50": round(_quantile(ordered, 0.50), 4),
        "p90": round(_quantile(ordered, 0.90), 4),
        "p95": round(_quantile(ordered, 0.95), 4),
        "p99": round(_quantile(ordered, 0.99), 4),
        "max": ordered[-1],
    }


def truncate_preview(value: Any, max_chars: int = PREVIEW_MAX_CHARS) -> str | None:
    if is_null(value):
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... <truncated, {len(text)} chars total>"


def field_presence(values: Iterable[Any], *, as_list: bool = False) -> dict[str, int]:
    null_count = 0
    empty_count = 0
    present_count = 0
    for value in values:
        if as_list:
            parsed = parse_string_list(value)
            if parsed is None:
                null_count += 1
                continue
            if len(parsed) == 0:
                empty_count += 1
                continue
            present_count += 1
            continue
        if is_null(value):
            null_count += 1
            continue
        if isinstance(value, str) and value == "":
            empty_count += 1
            continue
        present_count += 1
    return {
        "null": null_count,
        "empty": empty_count,
        "present": present_count,
    }


def _repo_counts(frame: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in _series_values(frame, "repo"):
        if is_null(value) or str(value).strip() == "":
            key = "<null>"
        else:
            key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _paper_repo_comparison(repo_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    actual_by_short: dict[str, int] = {}
    for repo, count in repo_counts.items():
        short = repo.rsplit("/", 1)[-1]
        actual_by_short[short] = actual_by_short.get(short, 0) + count
    rows = []
    for short, paper_n in PAPER_REPO_SHORT_COUNTS.items():
        actual_n = actual_by_short.get(short)
        rows.append(
            {
                "short_name": short,
                "paper_count": paper_n,
                "actual_count": actual_n,
                "match": actual_n == paper_n,
            }
        )
    extra = sorted(set(actual_by_short) - set(PAPER_REPO_SHORT_COUNTS))
    return rows if not extra else rows + [{"extra_short_names": extra}]


def _row_preview(frame: Any, index: Any, reason: str) -> dict[str, Any]:
    row = frame.loc[index]
    fail_list = parse_string_list(row["FAIL_TO_PASS"])
    pass_list = parse_string_list(row["PASS_TO_PASS"])
    return {
        "preview_reason": reason,
        "instance_id": None if is_null(row["instance_id"]) else str(row["instance_id"]),
        "repo": None if is_null(row["repo"]) else str(row["repo"]),
        "created_at": None if is_null(row["created_at"]) else str(row["created_at"]),
        "version": None if is_null(row["version"]) else str(row["version"]),
        "problem_statement": truncate_preview(row["problem_statement"]),
        "hints_text": truncate_preview(row["hints_text"]),
        "patch": truncate_preview(row["patch"]),
        "test_patch": truncate_preview(row["test_patch"]),
        "FAIL_TO_PASS_len": None if fail_list is None else len(fail_list),
        "PASS_TO_PASS_len": None if pass_list is None else len(pass_list),
    }


def select_preview_rows(frame: Any) -> list[dict[str, Any]]:
    if len(frame) == 0:
        return []
    repo_counts = _repo_counts(frame)
    ranked_repos = [name for name in repo_counts if name != "<null>"]
    chosen: list[tuple[Any, str]] = []
    seen_indices: set[Any] = set()

    def _add(index: Any, reason: str) -> None:
        if index in seen_indices:
            return
        seen_indices.add(index)
        chosen.append((index, reason))

    if ranked_repos:
        largest = ranked_repos[0]
        smallest = ranked_repos[-1]
        _add(frame.index[frame["repo"].astype(str) == largest][0], "largest_repo")
        _add(frame.index[frame["repo"].astype(str) == smallest][0], "smallest_repo")

    lengths = []
    for index, value in zip(frame.index.tolist(), _series_values(frame, "problem_statement")):
        lengths.append((0 if is_null(value) else len(str(value)), index))
    _add(max(lengths, key=lambda item: item[0])[1], "longest_problem_statement")
    return [_row_preview(frame, index, reason) for index, reason in chosen]


def profile_frame(frame: Any) -> dict[str, Any]:
    columns = _column_names(frame)
    extra_columns = [name for name in columns if name not in REQUIRED_COLUMNS]
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in columns]
    repo_counts = _repo_counts(frame) if "repo" in columns else {}

    field_missing: dict[str, dict[str, int]] = {}
    for name in columns:
        as_list = name in LIST_LENGTH_FIELDS
        field_missing[name] = field_presence(_series_values(frame, name), as_list=as_list)

    text_stats: dict[str, dict[str, float | int | None]] = {}
    for name in TEXT_LENGTH_FIELDS:
        if name not in columns:
            continue
        lengths = [
            len(str(value))
            for value in _series_values(frame, name)
            if not is_null(value)
        ]
        text_stats[name] = length_stats(lengths)

    list_stats: dict[str, dict[str, float | int | None]] = {}
    empty_lists: dict[str, int] = {}
    for name in LIST_LENGTH_FIELDS:
        if name not in columns:
            continue
        parsed_lists = [parse_string_list(value) for value in _series_values(frame, name)]
        empty_lists[name] = sum(1 for item in parsed_lists if item is not None and len(item) == 0)
        list_stats[name] = length_stats(len(item) for item in parsed_lists if item is not None)

    instance_ids = (
        _series_values(frame, "instance_id") if "instance_id" in columns else []
    )
    unique_ids = {
        str(value) for value in instance_ids if not is_null(value) and str(value).strip() != ""
    }

    return {
        "n_rows": int(len(frame)),
        "n_repos": len({k for k in repo_counts if k != "<null>"}),
        "columns": columns,
        "extra_columns": extra_columns,
        "missing_required_columns": missing_columns,
        "instance_id_unique": (
            "instance_id" in columns
            and len(unique_ids) == len(frame)
            and all(not is_null(v) and str(v).strip() != "" for v in instance_ids)
        ),
        "repo_counts": repo_counts,
        "paper_repo_comparison": _paper_repo_comparison(repo_counts) if repo_counts else [],
        "field_missing": field_missing,
        "text_length": text_stats,
        "list_length": list_stats,
        "empty_list_counts": empty_lists,
        "samples": select_preview_rows(frame) if not missing_columns else [],
    }


def manifest_record(*, verified_at: str | None = None) -> dict[str, Any]:
    record = {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "hf_url": HF_URL,
        "revision": HF_REVISION,
        "filename": HF_FILENAME,
        "sha256": EXPECTED_SHA256,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "expected_n_rows": EXPECTED_N_ROWS,
        "expected_n_repos": EXPECTED_N_REPOS,
        "required_columns": list(REQUIRED_COLUMNS),
        "local_dir": "data/raw/swe_gym",
        "notes": (
            "Official SWE-Gym train split. Raw parquet is gitignored under "
            "data/raw/. Do not use SWE-Gym-Lite or SWE-Gym-Raw."
        ),
    }
    if verified_at is not None:
        record["verified_at"] = verified_at
    return record


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
