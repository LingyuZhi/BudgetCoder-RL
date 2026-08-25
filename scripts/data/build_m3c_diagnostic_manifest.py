#!/usr/bin/env python
"""Build the frozen M3C train diagnostic manifest from M1E train identities.

Does not read gold patches, evaluator oracles, or localization labels.

Usage:

    python scripts/data/build_m3c_diagnostic_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m3c import (  # noqa: E402
    GROUP_N,
    PRIMARY_N,
    build_diagnostic_manifest_from_train_parquet,
    default_diagnostic_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--primary-n", type=int, default=PRIMARY_N)
    parser.add_argument("--group-n", type=int, default=GROUP_N)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    payload = build_diagnostic_manifest_from_train_parquet(
        repo_root,
        parquet_path=args.train.resolve() if args.train is not None else None,
        primary_n=args.primary_n,
        group_n=args.group_n,
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else default_diagnostic_path(repo_root)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "n_universe": payload["n_universe"],
                "n_primary": payload["n_primary"],
                "n_primary_runnable": payload["n_primary_runnable"],
                "skipped_overlong": payload["skipped_overlong"],
                "n_repos_primary": payload["n_repos_primary"],
                "ordered_ids_sha256": payload["ordered_ids_sha256"],
                "primary_ids_sha256": payload["primary_ids_sha256"],
                "oracle_used": payload["oracle_used"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
