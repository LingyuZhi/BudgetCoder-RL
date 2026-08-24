#!/usr/bin/env python
"""Extract SWE-Gym M1D-A eligibility / structural-difficulty features.

Consumes frozen M1A parquet + M1C-A/B JSONL and local bare Git mirrors.
Every input row is retained. Does not filter, split, or write keep/drop.
Never clones or fetches. Does not re-parse gold patches or re-run AST.
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
from budget_coder_rl.data.swe_gym_features import (  # noqa: E402
    build_feature_summary,
    extract_feature_frame,
    feature_jsonl_path,
    feature_summary_path,
    format_feature_report,
    instance_alignment_errors,
    read_jsonl,
    unique_repo_commits,
)
from budget_coder_rl.data.swe_gym_oracle import oracle_jsonl_path  # noqa: E402
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    CACHE_RELPATH,
    bcrl_data_root,
    swe_gym_repos_root,
)
from budget_coder_rl.data.swe_gym_symbol_oracle import (  # noqa: E402
    symbol_oracle_jsonl_path,
)
from budget_coder_rl.data.swe_gym_tree_stats import TreeStatStore  # noqa: E402


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to extract M1D-A features but is not importable. "
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


def _row_mapping(frame, index) -> dict:
    row = frame.loc[index]
    return {str(column): row[column] for column in frame.columns}


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
        "--oracle-jsonl",
        type=Path,
        default=None,
        help="M1C-A oracle JSONL (default: data/interim/swe_gym/m1c_oracle.jsonl)",
    )
    parser.add_argument(
        "--symbol-jsonl",
        type=Path,
        default=None,
        help="M1C-B symbol JSONL (default: data/interim/swe_gym/m1c_symbol_oracle.jsonl)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="per-instance feature JSONL (default: data/interim/swe_gym/m1d_features.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="tracked summary JSON (default: data/manifests/swe_gym_m1d_feature_summary.json)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="BCRL data root (default: $BCRL_DATA_ROOT or ~/my_data/budget-coder-rl)",
    )
    parser.add_argument(
        "--repos-root",
        type=Path,
        default=None,
        help=f"mirror cache root (default: <data-root>/{CACHE_RELPATH})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse any git clone/fetch (extraction never networks anyway)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    parquet = args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root)
    oracle_path = (
        args.oracle_jsonl.resolve()
        if args.oracle_jsonl is not None
        else oracle_jsonl_path(repo_root)
    )
    symbol_path = (
        args.symbol_jsonl.resolve()
        if args.symbol_jsonl is not None
        else symbol_oracle_jsonl_path(repo_root)
    )
    jsonl_path = (
        args.jsonl.resolve() if args.jsonl is not None else feature_jsonl_path(repo_root)
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else feature_summary_path(repo_root)
    )
    data_root = bcrl_data_root(args.data_root)
    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(data_root)
    )

    # Extraction is always offline: TreeStatStore only uses rev-parse / ls-tree.
    _ = args.offline

    hard_errors: list[str] = []
    if not parquet.is_file():
        hard_errors.append(
            f"parquet not found: {parquet}\nrun: python scripts/data/download_swe_gym.py"
        )
    else:
        hard_errors.extend(verify_parquet_file(parquet))
    hard_errors.extend(committed_field_policy_errors(repo_root))
    if not oracle_path.is_file():
        hard_errors.append(
            f"M1C-A oracle JSONL not found: {oracle_path}\n"
            "run: python scripts/data/extract_swe_gym_oracle.py"
        )
    if not symbol_path.is_file():
        hard_errors.append(
            f"M1C-B symbol JSONL not found: {symbol_path}\n"
            "run: python scripts/data/extract_swe_gym_symbol_oracle.py"
        )
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

    rows = [_row_mapping(frame, index) for index in frame.index]
    oracle_records = read_jsonl(oracle_path)
    symbol_records = read_jsonl(symbol_path)
    align_errors = instance_alignment_errors(rows, oracle_records, symbol_records)
    if align_errors:
        print("HARD FAIL (M1C artifact alignment):", file=sys.stderr)
        for err in align_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    store = TreeStatStore(repos_root)
    keys = unique_repo_commits(rows)
    n_keys = len(keys)
    print(
        f"computing tree stats for {n_keys} unique (repo, base_commit) groups "
        f"from {repos_root} ...",
        flush=True,
    )
    for index, (repo, commit) in enumerate(keys, start=1):
        if index == 1 or index % 50 == 0 or index == n_keys:
            print(
                f"  ls-tree {index}/{n_keys} {repo} {commit[:12]}",
                flush=True,
            )
        store.stats(repo, commit)

    records = extract_feature_frame(rows, oracle_records, symbol_records, store)
    summary = build_feature_summary(records)
    if len(records) != EXPECTED_N_ROWS or int(summary["n_rows"]) != EXPECTED_N_ROWS:
        print("HARD FAIL (feature extract dropped or duplicated rows):", file=sys.stderr)
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
        print(
            f"  - wrote {n_written} lines, expected {EXPECTED_N_ROWS}",
            file=sys.stderr,
        )
        return 1
    try:
        summary["jsonl"] = str(jsonl_path.relative_to(repo_root))
    except ValueError:
        summary["jsonl"] = str(jsonl_path)
    write_json(summary_path, summary)

    print(format_feature_report(summary), end="")
    print(f"\nwrote {jsonl_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
