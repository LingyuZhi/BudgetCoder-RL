#!/usr/bin/env python
"""Extract base-repository function/class symbol oracles (M1C-B).

Reads Git blobs from the prepared SWE-Gym mirror cache. Never clones or
fetches. Every input row is retained. Symbol oracles are privileged
evaluator metadata and are not written into the agent task view.
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
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    CACHE_RELPATH,
    BlobStore,
    bcrl_data_root,
    swe_gym_repos_root,
)
from budget_coder_rl.data.swe_gym_symbol_oracle import (  # noqa: E402
    build_symbol_summary,
    extract_symbol_frame,
    format_symbol_report,
    symbol_oracle_jsonl_path,
    symbol_oracle_summary_path,
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to extract symbol oracles but is not importable. "
            "Use the pinned RL conda env. Do not pip-install packages into that env "
            "from this script."
        ) from exc
    return pd


def _require_unidiff() -> None:
    try:
        import unidiff  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "unidiff is required for M1C-B symbol extraction. "
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
        help="per-instance symbol oracle JSONL "
        "(default: data/interim/swe_gym/m1c_symbol_oracle.jsonl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="tracked summary JSON (default: data/manifests/swe_gym_m1c_symbol_summary.json)",
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
    jsonl_path = (
        args.jsonl.resolve()
        if args.jsonl is not None
        else symbol_oracle_jsonl_path(repo_root)
    )
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else symbol_oracle_summary_path(repo_root)
    )
    data_root = bcrl_data_root(args.data_root)
    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(data_root)
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

    # Extraction is always offline: BlobStore only uses rev-parse / cat-file.
    # --offline is a documented guarantee, not a behavior change.
    _ = args.offline
    store = BlobStore(repos_root)
    records = extract_symbol_frame(frame, store)
    summary = build_symbol_summary(records, repos_root=repos_root)
    if len(records) != EXPECTED_N_ROWS or int(summary["n_rows"]) != EXPECTED_N_ROWS:
        print("HARD FAIL (symbol oracle dropped or duplicated rows):", file=sys.stderr)
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

    print(format_symbol_report(summary), end="")
    print(f"\nwrote {jsonl_path}")
    print(f"wrote {summary_path}")

    missing_commits = int(summary["commits"]["n_missing"])
    missing_blobs = int(summary["blobs"]["n_missing"])
    if missing_commits or missing_blobs:
        print(
            f"HARD FAIL: missing commits={missing_commits} blobs={missing_blobs} "
            "(records were still written; nothing was dropped)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
