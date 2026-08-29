#!/usr/bin/env python
"""Inspect the official SWE-Gym raw parquet (M1A).

Fails on schema / cardinality / instance_id uniqueness errors.
Does not filter, split, or parse gold patches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym import (  # noqa: E402
    EXPECTED_N_REPOS,
    EXPECTED_N_ROWS,
    LIST_LENGTH_FIELDS,
    TEXT_LENGTH_FIELDS,
    parquet_path,
    profile_frame,
    profile_json_path,
    validate_schema_and_cardinality,
    write_json,
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to inspect SWE-Gym but is not importable. "
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


def format_report(profile: dict) -> str:
    lines = [
        "SWE-Gym raw profile",
        f"rows: {profile['n_rows']}",
        f"unique repos: {profile['n_repos']}",
        f"columns: {profile['columns']}",
        f"extra columns: {profile['extra_columns']}",
        f"instance_id unique: {profile['instance_id_unique']}",
        "",
        "repo counts:",
    ]
    for repo, count in profile["repo_counts"].items():
        lines.append(f"  {repo}: {count}")

    lines.append("")
    lines.append("paper short-name comparison (informational):")
    for row in profile["paper_repo_comparison"]:
        if "extra_short_names" in row:
            lines.append(f"  extra short names: {row['extra_short_names']}")
            continue
        flag = "ok" if row["match"] else "DIFF"
        lines.append(
            f"  {row['short_name']}: actual={row['actual_count']} "
            f"paper={row['paper_count']} [{flag}]"
        )

    lines.append("")
    lines.append("field missing/null/empty:")
    for name, counts in profile["field_missing"].items():
        lines.append(
            f"  {name}: null={counts['null']} empty={counts['empty']} "
            f"present={counts['present']}"
        )

    lines.append("")
    lines.append("text length (chars, non-null):")
    for name in TEXT_LENGTH_FIELDS:
        stats = profile["text_length"].get(name)
        if stats is None:
            continue
        lines.append(
            f"  {name}: n={stats['n']} min={stats['min']} mean={stats['mean']} "
            f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
            f"p99={stats['p99']} max={stats['max']}"
        )

    lines.append("")
    lines.append("list length:")
    for name in LIST_LENGTH_FIELDS:
        stats = profile["list_length"].get(name)
        if stats is None:
            continue
        empty_n = profile["empty_list_counts"].get(name, 0)
        lines.append(
            f"  {name}: n={stats['n']} min={stats['min']} mean={stats['mean']} "
            f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
            f"p99={stats['p99']} max={stats['max']} empty={empty_n}"
        )

    lines.append("")
    lines.append("representative samples (truncated; patches are never printed in full):")
    lines.append(json.dumps(profile["samples"], indent=2, ensure_ascii=True))
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
        "--output",
        type=Path,
        default=None,
        help="profile JSON path (default: data/raw/swe_gym/profile.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    parquet = args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root)
    output = args.output.resolve() if args.output is not None else profile_json_path(repo_root)

    if not parquet.is_file():
        raise SystemExit(
            f"parquet not found: {parquet}\n"
            "run: python scripts/data/prepare_swe_gym.py download"
        )

    frame = load_parquet(parquet)
    errors = validate_schema_and_cardinality(
        frame,
        expected_n_rows=EXPECTED_N_ROWS,
        expected_n_repos=EXPECTED_N_REPOS,
    )
    profile = profile_frame(frame)
    profile["parquet"] = str(parquet)
    write_json(output, profile)
    print(format_report(profile), end="")
    print(f"\nwrote {output}")

    if errors:
        print("HARD FAIL (schema/cardinality):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
