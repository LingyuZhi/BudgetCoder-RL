#!/usr/bin/env python
"""Materialize veRL-ready SWE-Gym train/dev parquet and evaluator oracle sidecar.

Consumes frozen M1D split membership and M1C oracles. Does not re-run split,
AST, or unidiff. Does not implement reward, Agent scaffold, or GRPO.
Offline: no network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym import parquet_path  # noqa: E402
from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    MaterializeInputError,
    dataset_manifest_path,
    dev_parquet_path,
    format_materialize_report,
    materialize,
    oracle_parquet_path,
    schema_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_oracle import oracle_jsonl_path  # noqa: E402
from budget_coder_rl.data.swe_gym_split import POLICY_RELPATH, SPLIT_RELPATH  # noqa: E402
from budget_coder_rl.data.swe_gym_symbol_oracle import (  # noqa: E402
    symbol_oracle_jsonl_path,
)


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
        help="raw SWE-Gym parquet (default: data/raw/swe_gym/data/train-00000-of-00001.parquet)",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="frozen M1D split JSON (default: data/manifests/swe_gym_m1d_split.json)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="frozen M1D policy JSON (default: data/manifests/swe_gym_m1d_policy.json)",
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
        "--train-out",
        type=Path,
        default=None,
        help="policy train parquet (default: data/processed/swe_gym/train.parquet)",
    )
    parser.add_argument(
        "--dev-out",
        type=Path,
        default=None,
        help="policy dev parquet (default: data/processed/swe_gym/dev.parquet)",
    )
    parser.add_argument(
        "--oracle-out",
        type=Path,
        default=None,
        help="evaluator sidecar parquet (default: data/processed/swe_gym/evaluator_oracle.parquet)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="schema JSON (default: data/manifests/swe_gym_m1e_schema.json)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="dataset manifest JSON (default: data/manifests/swe_gym_m1e_dataset_manifest.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        result = materialize(
            repo_root=repo_root,
            raw_parquet=args.parquet.resolve() if args.parquet is not None else parquet_path(repo_root),
            split_json=(
                args.split.resolve()
                if args.split is not None
                else repo_root / SPLIT_RELPATH
            ),
            policy_json=(
                args.policy.resolve()
                if args.policy is not None
                else repo_root / POLICY_RELPATH
            ),
            oracle_jsonl=(
                args.oracle_jsonl.resolve()
                if args.oracle_jsonl is not None
                else oracle_jsonl_path(repo_root)
            ),
            symbol_jsonl=(
                args.symbol_jsonl.resolve()
                if args.symbol_jsonl is not None
                else symbol_oracle_jsonl_path(repo_root)
            ),
            train_out=(
                args.train_out.resolve()
                if args.train_out is not None
                else train_parquet_path(repo_root)
            ),
            dev_out=(
                args.dev_out.resolve()
                if args.dev_out is not None
                else dev_parquet_path(repo_root)
            ),
            oracle_out=(
                args.oracle_out.resolve()
                if args.oracle_out is not None
                else oracle_parquet_path(repo_root)
            ),
            schema_out=(
                args.schema.resolve()
                if args.schema is not None
                else schema_path(repo_root)
            ),
            manifest_out=(
                args.manifest.resolve()
                if args.manifest is not None
                else dataset_manifest_path(repo_root)
            ),
        )
    except MaterializeInputError as exc:
        print("HARD FAIL:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"  {line}", file=sys.stderr)
        return 1

    print(format_materialize_report(result["manifest"]), end="")
    print(f"wrote {result['train_path']}")
    print(f"wrote {result['dev_path']}")
    print(f"wrote {result['oracle_path']}")
    print(f"wrote {result['schema_path']}")
    print(f"wrote {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
