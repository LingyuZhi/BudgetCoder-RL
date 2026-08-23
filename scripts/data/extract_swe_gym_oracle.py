#!/usr/bin/env python
"""Extract file / hunk / line-coordinate oracles from SWE-Gym gold patches (M1C-A).

Hard-fails on parquet identity / schema / cardinality errors.
Every input row is retained. Parse failures are reported, not filtered.
Does not read test_patch into gold_edit_files. Does not extract AST symbols.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym import (  # noqa: E402
    EXPECTED_N_REPOS,
    EXPECTED_N_ROWS,
    parquet_path,
    validate_schema_and_cardinality,
    verify_parquet_file,
    write_json,
)
from budget_coder_rl.data.swe_gym_audit import write_jsonl  # noqa: E402
from budget_coder_rl.data.swe_gym_fields import committed_field_policy_errors  # noqa: E402
from budget_coder_rl.data.swe_gym_oracle import (  # noqa: E402
    build_oracle_summary,
    extract_oracle_frame,
    oracle_jsonl_path,
    oracle_summary_path,
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to extract SWE-Gym oracles but is not importable. "
            "Use the pinned RL conda env. Do not pip-install packages into that env "
            "from this script."
        ) from exc
    return pd


def _require_unidiff() -> None:
    try:
        import unidiff  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "unidiff is required for M1C-A oracle extraction. "
            "Install only this package: pip install 'unidiff>=0.7.5,<1'. "
            "Do not pip-install other packages from this script."
        ) from exc


def load_parquet(path: Path):
    pd = _require_pandas()
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise SystemExit(
            "reading parquet requires pyarrow (or fastparquet) in the pinned env. "
            "Do not pip-install packages from this script."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"failed to read parquet {path}: {exc}") from exc


def _format_stats(name: str, stats: dict) -> str:
    return (
        f"  {name}: n={stats['n']} min={stats['min']} mean={stats['mean']} "
        f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
        f"p99={stats['p99']} max={stats['max']}"
    )


def format_report(summary: dict) -> str:
    lines = [
        "SWE-Gym M1C-A gold-patch oracle extraction",
        f"revision: {summary['revision']}",
        f"sha256: {summary['sha256']}",
        f"rows: {summary['n_rows']} (dropped: {summary['rows_dropped']})",
        f"parsed: {summary['successfully_parsed']}",
        f"parse failures: {summary['parse_failure_count']}",
        "",
        "per-instance quantiles (parse_ok only):",
        _format_stats("gold_edit_files", summary["gold_edit_files_per_instance"]),
        _format_stats(
            "base_changed_files", summary["base_changed_files_per_instance"]
        ),
        _format_stats("hunks", summary["hunks_per_instance"]),
        _format_stats("added_lines", summary["added_lines_per_instance"]),
        _format_stats("removed_lines", summary["removed_lines_per_instance"]),
        "",
        "file operations:",
    ]
    for name, count in summary["file_operation_counts"].items():
        lines.append(f"  {name}: {count}")

    lines.append("")
    lines.append("file extensions:")
    extensions = summary["file_extension_distribution"]
    for name, count in list(extensions.items())[:15]:
        lines.append(f"  {name}: {count}")
    if len(extensions) > 15:
        lines.append(f"  ... {len(extensions) - 15} more")

    views = summary["instance_file_views"]
    lines.append("")
    lines.append("instance file views (observational, not dropped):")
    lines.append(f"  with_added_file: {views['n_with_added_file']}")
    lines.append(f"  added_only: {views['n_added_only']}")
    lines.append(f"  zero_base_changed_files: {views['n_zero_base_changed_files']}")
    lines.append(f"  with_deleted_file: {views['n_with_deleted_file']}")
    lines.append(f"  with_path_changed: {views['n_with_path_changed']}")

    test_like = summary["test_like_gold_paths"]
    lines.append("")
    lines.append(
        "test-like gold paths (observational, not dropped): "
        f"instances={test_like['n_instances']} files={test_like['n_files']}"
    )

    outliers = summary["outliers"]
    lines.append("")
    lines.append("outliers:")
    for key in (
        "max_gold_edit_files",
        "max_base_changed_files",
        "max_hunks",
        "max_added_lines",
        "max_removed_lines",
    ):
        item = outliers.get(key) or {}
        lines.append(f"  {key}: {item}")

    spotlight = summary["spotlight"]
    lines.append("")
    lines.append(
        f"spotlight {spotlight['instance_id']}: "
        f"present={spotlight.get('present_in_input')} "
        f"parse_ok={spotlight.get('parse_ok')} "
        f"gold_edit={spotlight.get('n_gold_edit_files')} "
        f"base_changed={spotlight.get('n_base_changed_files')} "
        f"hunks={spotlight.get('n_hunks')} "
        f"added={spotlight.get('n_added_lines')} "
        f"removed={spotlight.get('n_removed_lines')} "
        f"operations={spotlight.get('operations')}"
    )

    failures = summary.get("parse_failures") or []
    lines.append("")
    if failures:
        lines.append("PARSE FAILURES:")
        for item in failures:
            lines.append(f"  {item['instance_id']}: {item['parse_error']}")
    else:
        lines.append("PARSE FAILURES: none")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="parquet path (default: data/raw/swe_gym/data/train-00000-of-00001.parquet)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="per-instance oracle JSONL (default: data/interim/swe_gym/m1c_oracle.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="tracked summary JSON (default: data/manifests/swe_gym_m1c_oracle_summary.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    parquet = args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root)
    jsonl_path = (
        args.jsonl.resolve() if args.jsonl is not None else oracle_jsonl_path(repo_root)
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else oracle_summary_path(repo_root)
    )

    _require_unidiff()

    hard_errors: list[str] = []
    if not parquet.is_file():
        hard_errors.append(
            f"parquet not found: {parquet}\nrun: python scripts/data/download_swe_gym.py"
        )
    else:
        hard_errors.extend(verify_parquet_file(parquet))
    hard_errors.extend(committed_field_policy_errors(repo_root))
    if hard_errors:
        print("HARD FAIL:", file=sys.stderr)
        for err in hard_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    frame = load_parquet(parquet)
    schema_errors = validate_schema_and_cardinality(
        frame,
        expected_n_rows=EXPECTED_N_ROWS,
        expected_n_repos=EXPECTED_N_REPOS,
    )
    if schema_errors:
        print("HARD FAIL (schema/cardinality):", file=sys.stderr)
        for err in schema_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    records = extract_oracle_frame(frame)
    summary = build_oracle_summary(records)
    if len(records) != EXPECTED_N_ROWS or int(summary["n_rows"]) != EXPECTED_N_ROWS:
        print("HARD FAIL (oracle dropped or duplicated rows):", file=sys.stderr)
        print(
            f"  - records={len(records)} summary_n_rows={summary['n_rows']}",
            file=sys.stderr,
        )
        return 1
    if int(summary["rows_dropped"]) != 0:
        print("HARD FAIL: rows_dropped != 0", file=sys.stderr)
        return 1

    n_written = write_jsonl(jsonl_path, records)
    if n_written != EXPECTED_N_ROWS:
        print("HARD FAIL (JSONL line count):", file=sys.stderr)
        print(f"  - wrote {n_written} lines, expected {EXPECTED_N_ROWS}", file=sys.stderr)
        return 1
    try:
        summary["jsonl"] = str(jsonl_path.relative_to(repo_root))
    except ValueError:
        summary["jsonl"] = str(jsonl_path)
    write_json(summary_path, summary)

    print(format_report(summary), end="")
    print(f"\nwrote {jsonl_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
