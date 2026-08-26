#!/usr/bin/env python
"""Build the scaled-M5 train-candidate manifest (2193 unique + 7 hash pads).

Does not cherry-pick by E014/E015 reward. Does not edit M3C/M5-main manifests.

Usage:

    python scripts/data/build_m5_scaled_train_candidates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_materialize import oracle_parquet_path  # noqa: E402
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m5_scaled import (  # noqa: E402
    build_scaled_train_manifest,
    default_candidate_path,
    load_identity_rows_for_build,
    manifest_errors,
)
from budget_coder_rl.eval.oracle import load_evaluator_oracle  # noqa: E402
from budget_coder_rl.eval.provenance import sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-m1d-fallback", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    identity = load_identity_rows_for_build(
        repo_root, require_parquet=not args.allow_m1d_fallback
    )
    oracle_path = oracle_parquet_path(repo_root)
    oracle = load_evaluator_oracle(oracle_path) if oracle_path.is_file() else None
    if oracle is None:
        print("HARD FAIL: evaluator oracle parquet missing", file=sys.stderr)
        return 1
    payload = build_scaled_train_manifest(
        identity["rows"],
        repo_root=repo_root,
        oracle=oracle,
        identity_source=str(identity.get("source") or "m1e_train_parquet"),
    )
    errors = manifest_errors(payload)
    if errors:
        print(f"HARD FAIL: {errors}", file=sys.stderr)
        return 1
    output = args.output.resolve() if args.output is not None else default_candidate_path(repo_root)
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "n_unique": payload["n_unique"],
                "n_rows": payload["n_rows"],
                "n_pad": payload["n_pad"],
                "optimizer_steps": payload["optimizer_steps"],
                "unique_ids_sha256": payload["unique_ids_sha256"],
                "padded_ids_sha256": payload["padded_ids_sha256"],
                "pad_ids": payload["padding"]["pad_ids"],
                "symbol_applicable_false": payload["symbol_applicable_false"],
                "identity_source": payload["identity_source"],
                "oracle_replayed": payload["oracle_replayed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
