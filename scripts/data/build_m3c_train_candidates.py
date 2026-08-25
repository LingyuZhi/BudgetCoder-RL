#!/usr/bin/env python
"""Build the frozen M4/M5 train-candidate manifest.

Diagnostic n=4 zero-variance is NOT used as a drop rule. Oracle sidecar
is used only as a class filter (symbol_applicable) and is not written
into policy parquet.

Usage:

    python scripts/data/build_m3c_train_candidates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.eval.m3c import (  # noqa: E402
    OVERLONG_INSTANCE_IDS,
    TRAIN_CANDIDATE_TARGET_N,
    build_train_candidate_manifest,
    default_candidate_path,
    load_split_identities,
)
from budget_coder_rl.eval.oracle import load_evaluator_oracle  # noqa: E402

DEFAULT_RULE = (
    "Start from frozen M1E train (2194). Exclude runtime-incompatible prompts "
    "with tokenizer length > 16384 (Project-MONAI__MONAI-6344). Require "
    "evaluator sidecar symbol_applicable=true as the default file+symbol "
    "reward pool. Do not drop a task solely because one n=4 grouped rollout "
    "was zero-variance. Do not cherry-pick by gold patch or per-task reward. "
    "Order remaining identities with repo_name_sort + instance_id_lexicographic "
    "+ repo_round_robin and take the first target_n tasks for repo balance."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--target-n", type=int, default=TRAIN_CANDIDATE_TARGET_N)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rule-text", default=DEFAULT_RULE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    train_path = args.train.resolve() if args.train is not None else train_parquet_path(repo_root)
    oracle_path = (
        args.oracle.resolve() if args.oracle is not None else oracle_parquet_path(repo_root)
    )
    rows = load_split_identities(train_path, expected_split="train")
    oracle = load_evaluator_oracle(oracle_path)
    skipped_overlong: list[str] = []
    skipped_symbol: list[str] = []
    skipped_missing_oracle: list[str] = []
    eligible: list[str] = []
    for row in rows:
        instance_id = row["instance_id"]
        if instance_id in OVERLONG_INSTANCE_IDS:
            skipped_overlong.append(instance_id)
            continue
        if instance_id not in oracle:
            skipped_missing_oracle.append(instance_id)
            continue
        sidecar = oracle.get(instance_id)
        if not sidecar.symbol_applicable:
            skipped_symbol.append(instance_id)
            continue
        eligible.append(instance_id)
    payload = build_train_candidate_manifest(
        rows,
        eligible_ids=eligible,
        skipped={
            "overlong_prompt": skipped_overlong,
            "symbol_unavailable": skipped_symbol,
            "missing_oracle": skipped_missing_oracle,
        },
        rule_text=args.rule_text,
        target_n=args.target_n,
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else default_candidate_path(repo_root)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "n_eligible": payload["n_eligible"],
                "n_selected": payload["n_selected"],
                "skipped": {key: len(value) for key, value in payload["skipped"].items()},
                "ordered_ids_sha256": payload["ordered_ids_sha256"],
                "n_repos": payload["n_repos"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
