#!/usr/bin/env python
"""Freeze SWE-Gym M1D-B eligibility policy and grouped train/dev split.

Consumes frozen M1D-A feature JSONL. Does not drop instances, create an
internal test split, materialize veRL parquet, re-parse patches, or re-run AST.
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
from budget_coder_rl.data.swe_gym_features import (  # noqa: E402
    feature_jsonl_path,
    feature_summary_path,
    read_jsonl,
)
from budget_coder_rl.data.swe_gym_fields import committed_field_policy_errors  # noqa: E402
from budget_coder_rl.data.swe_gym_split import (  # noqa: E402
    CrossRepoCorrelationError,
    SplitInputError,
    assign_split,
    build_eligibility_policy,
    build_split_manifest,
    build_split_summary,
    feature_jsonl_identity,
    format_split_report,
    manifest_json_bytes,
    policy_path,
    split_path,
    split_summary_path,
    validate_split_invariants,
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to verify parquet instance_id alignment but is "
            "not importable. Use the pinned RL conda env. Do not pip-install "
            "packages into that env from this script."
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
        "--features-jsonl",
        type=Path,
        default=None,
        help="M1D-A feature JSONL (default: data/interim/swe_gym/m1d_features.jsonl)",
    )
    parser.add_argument(
        "--feature-summary",
        type=Path,
        default=None,
        help="M1D-A feature summary (default: data/manifests/swe_gym_m1d_feature_summary.json)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="eligibility policy JSON (default: data/manifests/swe_gym_m1d_policy.json)",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="split manifest JSON (default: data/manifests/swe_gym_m1d_split.json)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="split summary JSON (default: data/manifests/swe_gym_m1d_split_summary.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    parquet = args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root)
    features_path = (
        args.features_jsonl.resolve()
        if args.features_jsonl is not None
        else feature_jsonl_path(repo_root)
    )
    feature_summary = (
        args.feature_summary.resolve()
        if args.feature_summary is not None
        else feature_summary_path(repo_root)
    )
    policy_out = args.policy.resolve() if args.policy is not None else policy_path(repo_root)
    split_out = args.split.resolve() if args.split is not None else split_path(repo_root)
    summary_out = (
        args.summary.resolve() if args.summary is not None else split_summary_path(repo_root)
    )

    hard_errors: list[str] = []
    if not parquet.is_file():
        hard_errors.append(
            f"parquet not found: {parquet}\nrun: python scripts/data/download_swe_gym.py"
        )
    else:
        hard_errors.extend(verify_parquet_file(parquet))
    hard_errors.extend(committed_field_policy_errors(repo_root))
    if not features_path.is_file():
        hard_errors.append(
            f"M1D-A feature JSONL not found: {features_path}\n"
            "run: python scripts/data/extract_swe_gym_m1d_features.py --offline"
        )
    if not feature_summary.is_file():
        hard_errors.append(
            f"M1D-A feature summary not found: {feature_summary}\n"
            "run: python scripts/data/extract_swe_gym_m1d_features.py --offline"
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

    records = read_jsonl(features_path)
    parquet_ids = [str(value) for value in frame["instance_id"].tolist()]
    jsonl_ids = [str(item.get("instance_id") or "") for item in records]
    if len(records) != EXPECTED_N_ROWS:
        print("HARD FAIL (feature JSONL row count):", file=sys.stderr)
        print(f"  - records={len(records)} expected={EXPECTED_N_ROWS}", file=sys.stderr)
        return 1
    if len(jsonl_ids) != len(set(jsonl_ids)):
        print("HARD FAIL: feature JSONL instance_id is not unique", file=sys.stderr)
        return 1
    if set(parquet_ids) != set(jsonl_ids):
        print("HARD FAIL (parquet vs M1D-A JSONL instance_id set):", file=sys.stderr)
        print(
            f"  - only_parquet={len(set(parquet_ids) - set(jsonl_ids))} "
            f"only_jsonl={len(set(jsonl_ids) - set(parquet_ids))}",
            file=sys.stderr,
        )
        return 1

    jsonl_sha256 = feature_jsonl_identity(features_path)
    try:
        result = assign_split(records)
    except CrossRepoCorrelationError as exc:
        print("HARD FAIL (cross-repo correlation groups):", file=sys.stderr)
        print(f"  - {exc}", file=sys.stderr)
        for item in exc.groups:
            print(
                f"  - {item['correlation_group_id']} "
                f"repos={item['repos']} instances={item['instance_ids']}",
                file=sys.stderr,
            )
        print("official split/summary not written; waiting for human review.", file=sys.stderr)
        return 1
    except SplitInputError as exc:
        print("HARD FAIL (split input):", file=sys.stderr)
        print(f"  - {exc}", file=sys.stderr)
        return 1

    invariant_errors = validate_split_invariants(
        records,
        result.assignments,
        expected_n_rows=EXPECTED_N_ROWS,
        expected_n_repos=EXPECTED_N_REPOS,
        require_all_repos_in_dev=True,
    )
    if invariant_errors:
        print("HARD FAIL (split invariants):", file=sys.stderr)
        for err in invariant_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    policy = build_eligibility_policy(n_rows=len(records))
    if int(policy["eligible_count"]) != EXPECTED_N_ROWS or int(policy["excluded_count"]) != 0:
        print("HARD FAIL: eligibility is not 2438/2438 keep-all", file=sys.stderr)
        return 1

    try:
        feature_jsonl_rel = str(features_path.relative_to(repo_root))
        feature_summary_rel = str(feature_summary.relative_to(repo_root))
    except ValueError:
        feature_jsonl_rel = str(features_path)
        feature_summary_rel = str(feature_summary)

    split_manifest = build_split_manifest(
        result,
        feature_jsonl_sha256=jsonl_sha256,
        feature_jsonl=feature_jsonl_rel,
        feature_summary=feature_summary_rel,
    )
    summary = build_split_summary(records, result)

    # In-memory determinism check before writing official artifacts.
    if manifest_json_bytes(split_manifest) != manifest_json_bytes(
        build_split_manifest(
            result,
            feature_jsonl_sha256=jsonl_sha256,
            feature_jsonl=feature_jsonl_rel,
            feature_summary=feature_summary_rel,
        )
    ):
        print("HARD FAIL: split manifest serialization is not deterministic", file=sys.stderr)
        return 1

    write_json(policy_out, policy)
    write_json(split_out, split_manifest)
    write_json(summary_out, summary)

    print(format_split_report(policy, split_manifest, summary), end="")
    print(f"wrote {policy_out}")
    print(f"wrote {split_out}")
    print(f"wrote {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
