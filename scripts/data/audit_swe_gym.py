#!/usr/bin/env python
"""Audit official SWE-Gym metadata for integrity and leakage (M1B).

Hard-fails on parquet identity / schema / cardinality / field-policy errors.
Heuristic suspicions, dataset properties, and observational signals are
labels only: every input row is retained. Suspicions are not confirmed
malformed rows.
Does not filter, split, extract oracle locations, or create an RL dataset.
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
from budget_coder_rl.data.swe_gym_audit import (  # noqa: E402
    audit_frame,
    audit_jsonl_path,
    audit_summary_path,
    write_jsonl,
)
from budget_coder_rl.data.swe_gym_fields import committed_field_policy_errors  # noqa: E402


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to audit SWE-Gym but is not importable. "
            "Use the pinned RL conda env. Do not pip-install packages into that env "
            "from this script."
        ) from exc
    return pd


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


def format_report(summary: dict) -> str:
    lines = [
        "SWE-Gym M1B integrity / leakage audit",
        f"revision: {summary['revision']}",
        f"rows: {summary['n_rows']} (dropped: {summary['rows_dropped']})",
        f"heuristic suspicion rows: {summary['n_heuristic_suspicion_rows']} "
        "(not confirmed malformed)",
        "",
        "flag classes:",
        f"  heuristic: {summary['flag_class_notes']['heuristic_structural_suspicion']}",
        f"  dataset/correlation: {summary['flag_class_notes']['dataset_correlation_property']}",
        f"  observational: {summary['flag_class_notes']['observational_signal']}",
        "",
        "heuristic structural suspicion counts (instance-level):",
    ]
    heuristic_counts = summary.get("heuristic_suspicion_counts") or {}
    if heuristic_counts:
        for name, count in heuristic_counts.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("dataset / correlation property counts:")
    property_counts = summary.get("dataset_property_counts") or {}
    if property_counts:
        for name, count in property_counts.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("observation counts:")
    obs_counts = summary.get("observation_counts") or {}
    if obs_counts:
        for name, count in obs_counts.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  (none)")

    control = summary.get("selector_control_chars") or {}
    actual = (control.get("actual_control_chars") or {}).get("instance_counts") or {}
    literal = (control.get("literal_escape_sequences") or {}).get("instance_counts") or {}
    lines.append("")
    lines.append(
        "selector control chars: "
        f"actual_c0_instances={control.get('n_instances_with_actual_c0', 0)} "
        f"literal_escape_instances={control.get('n_instances_with_literal_escape_sequence', 0)}"
    )
    if actual:
        parts = ", ".join(f"{key}={value}" for key, value in actual.items())
        lines.append(f"  actual C0 instance counts: {parts}")
    if literal:
        parts = ", ".join(f"{key}={value}" for key, value in literal.items())
        lines.append(f"  literal escape instance counts: {parts}")

    by_repo = summary.get("heuristic_suspicion_by_repo") or {}
    if by_repo:
        lines.append("")
        lines.append("heuristic suspicion by repo:")
        for flag, repo_counts in by_repo.items():
            if not repo_counts:
                continue
            parts = ", ".join(f"{repo}={count}" for repo, count in repo_counts.items())
            lines.append(f"  {flag}: {parts}")

    lines.append("")
    lines.append("selector shape (entry-level):")
    for field, counts in summary["selector_shape"].items():
        parts = ", ".join(f"{key}={value}" for key, value in counts.items())
        lines.append(f"  {field}: {parts}")

    dup = summary["duplicates"]
    lines.append("")
    lines.append("duplicates / correlation:")
    lines.append(
        "  problem_statement groups="
        f"{dup['problem_statement']['n_groups']} rows="
        f"{dup['problem_statement']['n_rows']} extra="
        f"{dup['problem_statement']['n_extra']}"
    )
    lines.append(
        "  patch groups="
        f"{dup['patch']['n_groups']} rows="
        f"{dup['patch']['n_rows']} extra="
        f"{dup['patch']['n_extra']}"
    )
    triple = dup["repo_base_commit_problem_statement"]
    lines.append(
        "  (repo, base_commit, problem_statement) groups="
        f"{triple['n_groups']} rows={triple['n_rows']} extra={triple['n_extra']}"
    )
    repo_commit = summary["repo_base_commit"]
    dist = ", ".join(
        f"size {size}: {count}"
        for size, count in repo_commit["group_size_distribution"].items()
    )
    lines.append(
        f"  (repo, base_commit) groups={repo_commit['n_groups']} "
        f"singletons={repo_commit['n_singletons']} "
        f"max={repo_commit['max_group_size']}"
    )
    lines.append(f"  group size distribution: {dist}")

    known = summary["known_upstream_malformed"]
    lines.append("")
    lines.append(
        f"known upstream malformed {known['instance_id']}: "
        f"present={known['present_in_input']} "
        f"captured_by_generic_rule={known['captured_by_generic_rule']} "
        f"matching_flags={known['matching_flags']}"
    )
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
        help="per-instance audit JSONL (default: data/interim/swe_gym/m1b_audit.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="tracked summary JSON (default: data/manifests/swe_gym_m1b_audit_summary.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    parquet = args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root)
    jsonl_path = args.jsonl.resolve() if args.jsonl is not None else audit_jsonl_path(repo_root)
    summary_path = (
        args.summary.resolve() if args.summary is not None else audit_summary_path(repo_root)
    )

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

    records, summary = audit_frame(frame)
    if len(records) != EXPECTED_N_ROWS or int(summary["n_rows"]) != EXPECTED_N_ROWS:
        print("HARD FAIL (audit dropped or duplicated rows):", file=sys.stderr)
        print(f"  - records={len(records)} summary_n_rows={summary['n_rows']}", file=sys.stderr)
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
