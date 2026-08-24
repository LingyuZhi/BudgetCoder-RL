#!/usr/bin/env python
"""Prepare/verify SWE-Gym Git mirrors for M1C-B symbol extraction.

Derives the 11 remotes and all base_commit / blob checks from the pinned
parquet. Does not hardcode instance commits. Does not install target-repo
dependencies or use Docker.
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
from budget_coder_rl.data.swe_gym_fields import committed_field_policy_errors  # noqa: E402
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    CACHE_RELPATH,
    bcrl_data_root,
    directory_size_bytes,
    format_prepare_report,
    prepare_repo_cache,
    repo_sources_manifest_path,
    repo_sources_record,
    swe_gym_repos_root,
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to prepare SWE-Gym repos but is not importable. "
            "Use the pinned RL conda env. Do not pip-install packages into that env "
            "from this script."
        ) from exc
    return pd


def _require_unidiff() -> None:
    try:
        import unidiff  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "unidiff is required to derive base_changed_files during repo prepare. "
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
        "--verify-only",
        action="store_true",
        help="do not clone/fetch; only verify the local object store",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="refuse clone/fetch (implies --verify-only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    parquet = args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root)
    data_root = bcrl_data_root(args.data_root)
    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(data_root)
    )
    verify_only = bool(args.verify_only or args.offline)
    allow_network = not bool(args.offline)

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

    report = prepare_repo_cache(
        frame,
        repos_root,
        allow_network=allow_network,
        verify_only=verify_only,
    )
    disk_bytes = directory_size_bytes(repos_root)
    manifest = repo_sources_record(report.plans, prepare_results=report.per_repo)
    write_json(repo_sources_manifest_path(repo_root), manifest)

    print(
        format_prepare_report(
            report,
            repos_root_label=f"$BCRL_DATA_ROOT/{CACHE_RELPATH}",
            disk_bytes=disk_bytes,
        ),
        end="",
    )
    print(f"wrote {repo_sources_manifest_path(repo_root)}")
    if not report.ok:
        print("HARD FAIL: missing repos, commits, or blobs (listed above)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
