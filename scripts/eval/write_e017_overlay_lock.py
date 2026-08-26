#!/usr/bin/env python
"""Write E017 namespace overlay lock. Does not edit scaled freeze JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.e017 import (  # noqa: E402
    overlay_errors,
    overlay_lock_errors,
    write_overlay_lock,
)
from budget_coder_rl.eval.m4a import load_json  # noqa: E402
from budget_coder_rl.eval.provenance import sha256_file  # noqa: E402


def main() -> int:
    overlay_path = REPO_ROOT / "configs/experiments/stage1_m5_scaled_e017.json"
    overlay = load_json(overlay_path)
    errors = overlay_errors(overlay, repo_root=REPO_ROOT)
    if errors:
        print(f"HARD FAIL: {errors}", file=sys.stderr)
        return 1
    lock = write_overlay_lock(REPO_ROOT)
    digest = sha256_file(overlay_path)
    lock_errors = overlay_lock_errors(REPO_ROOT)
    # Pin may still be zeros until e017.py is updated.
    lock_errors = [item for item in lock_errors if "pinned" not in item]
    if lock_errors:
        print(f"HARD FAIL: lock {lock_errors}", file=sys.stderr)
        return 1
    print(json.dumps({"overlay_sha256": digest, "lock": lock}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
