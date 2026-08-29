#!/usr/bin/env python
"""Canonical SWE-Gym localization dataset pipeline.

Stages (in order for ``all``):

    download → inspect → audit → extract-oracles → prepare-repos
    → extract-symbol-oracle → extract-features → verify-split → materialize

``verify-split`` checks the frozen train/dev split; it does not re-cut it.
Implementation lives in ``scripts/data/_stages/`` and ``src/budget_coder_rl/data/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES_DIR = Path(__file__).resolve().parent / "_stages"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

STAGE_FILES = {
    "download": "download.py",
    "inspect": "inspect.py",
    "audit": "audit.py",
    "extract-oracles": "extract_oracle.py",
    "prepare-repos": "prepare_repos.py",
    "extract-symbol-oracle": "extract_symbol_oracle.py",
    "extract-features": "extract_features.py",
    "split": "split.py",
    "materialize": "materialize.py",
}

ALL_ORDER = (
    "download",
    "inspect",
    "audit",
    "extract-oracles",
    "prepare-repos",
    "extract-symbol-oracle",
    "extract-features",
    "verify-split",
    "materialize",
)


def _load_stage(name: str):
    filename = STAGE_FILES[name]
    path = STAGES_DIR / filename
    spec = importlib.util.spec_from_file_location(f"bcrl_data_{name.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"missing data stage {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_stage(name: str, argv: list[str] | None = None, *, repo_root: Path | None = None) -> int:
    module = _load_stage(name)
    forwarded = list(argv or [])
    if repo_root is not None:
        forwarded = ["--repo-root", str(repo_root), *forwarded]
    return int(module.main(forwarded))


def _verify_split(repo_root: Path) -> int:
    from budget_coder_rl.data.swe_gym_split import (
        policy_path,
        split_path,
        split_summary_path,
    )

    errors: list[str] = []
    for label, path in (
        ("policy", policy_path(repo_root)),
        ("split", split_path(repo_root)),
        ("summary", split_summary_path(repo_root)),
    ):
        if not path.is_file():
            errors.append(f"missing frozen {label}: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"ok: {label} {path}")
        if label == "split":
            version = str(payload.get("split_version") or payload.get("schema_version") or "")
            if version != "swe-gym-group-repo-v1":
                errors.append(f"split_version {version!r} != swe-gym-group-repo-v1")
            assignments = payload.get("assignments") or []
            if isinstance(assignments, dict):
                n_train = sum(1 for item in assignments.values() if item == "train")
                n_dev = sum(1 for item in assignments.values() if item == "dev")
            else:
                n_train = sum(
                    1
                    for item in assignments
                    if isinstance(item, dict) and item.get("split") == "train"
                )
                n_dev = sum(
                    1
                    for item in assignments
                    if isinstance(item, dict) and item.get("split") == "dev"
                )
            expected_train = int(payload.get("actual_train_rows") or 0)
            expected_dev = int(payload.get("actual_dev_rows") or 0)
            if expected_train and n_train != expected_train:
                errors.append(f"assignment train={n_train} != actual_train_rows={expected_train}")
            if expected_dev and n_dev != expected_dev:
                errors.append(f"assignment dev={n_dev} != actual_dev_rows={expected_dev}")
            print(f"  split_version={version} train={n_train} dev={n_dev}")
    if errors:
        print("HARD FAIL:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("frozen split verified; not re-cut")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: inferred from this script)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download official SWE-Gym parquet")
    sub.add_parser("inspect", help="schema / cardinality inspect")
    sub.add_parser("audit", help="integrity / leakage audit")
    sub.add_parser("extract-oracles", help="extract patch file oracles")
    sub.add_parser("prepare-repos", help="prepare git mirrors / blob cache")
    sub.add_parser("extract-symbol-oracle", help="extract symbol oracles from git blobs")
    sub.add_parser("extract-features", help="extract eligibility / difficulty features")
    sub.add_parser("verify-split", help="verify frozen train/dev split (does not re-cut)")
    sub.add_parser("split", help="re-run split writer (not the default; frozen split is canonical)")
    sub.add_parser("materialize", help="write veRL parquet + evaluator oracle sidecar")
    sub.add_parser("all", help="run the full pipeline; split is verify-only")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, rest = parse_args(argv)
    command = args.command
    repo_root = args.repo_root.resolve()
    if command == "verify-split":
        return _verify_split(repo_root)
    if command == "all":
        for name in ALL_ORDER:
            print(f"==> {name}")
            if name == "verify-split":
                code = _verify_split(repo_root)
            else:
                code = _run_stage(name, rest, repo_root=repo_root)
            if code:
                return code
        return 0
    return _run_stage(command, rest, repo_root=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
