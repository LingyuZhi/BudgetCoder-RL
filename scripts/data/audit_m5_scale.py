#!/usr/bin/env python
"""CPU-only M5 scale-correction audit.

Reconstructs M1E train 2194 → M3C eligible ~2048 → first-RL 256, classifies
every exclusion, and writes a hashed scaled-pool proposal.

Does not start GPU / Ray / vLLM, and does not rewrite frozen M3C/M5/E014/E015
artifacts.

Usage (pinned RL conda env if tokenizing):

    python scripts/data/audit_m5_scale.py
    python scripts/data/audit_m5_scale.py --skip-tokenize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "smoke") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke"))

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    EXPECTED_TRAIN_ROWS,
    train_parquet_path,
)
from budget_coder_rl.eval.m3b import extra_mapping  # noqa: E402
from budget_coder_rl.eval.m5_scale_audit import (  # noqa: E402
    audit_dir,
    run_audit,
    write_audit_artifacts,
)
from budget_coder_rl.protocol.prompt import (  # noqa: E402
    build_stage1_messages,
    extract_issue_text,
    policy_safe_repo,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="do not encode prompts even if a local tokenizer exists",
    )
    parser.add_argument(
        "--require-live-oracle",
        action="store_true",
        help="hard-fail if evaluator_oracle.parquet is missing",
    )
    return parser.parse_args(argv)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    return extra_mapping(value)


def tokenize_train_prompts(
    *,
    repo_root: Path,
    tokenizer_path: Path | None,
) -> dict[str, int] | None:
    try:
        from smoke_rlhf_dataset import resolve_tokenizer_path
    except ImportError:
        return None
    resolved = resolve_tokenizer_path(tokenizer_path)
    if resolved is None:
        return None
    parquet = train_parquet_path(repo_root)
    if not parquet.is_file():
        return None
    try:
        from transformers import AutoTokenizer

        from budget_coder_rl.agent_loop.tokenization import encode_chat_messages
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    import pandas as pd

    frame = pd.read_parquet(parquet, columns=["prompt", "extra_info"])
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise SystemExit(
            f"HARD FAIL: train parquet n={len(frame)} expected {EXPECTED_TRAIN_ROWS}"
        )
    lengths: dict[str, int] = {}
    for record in frame.to_dict(orient="records"):
        extra = _as_mapping(record.get("extra_info"))
        instance_id = str(extra.get("instance_id") or "").strip()
        issue = extract_issue_text(record.get("prompt"))
        messages = build_stage1_messages(issue, repo=policy_safe_repo(extra))
        lengths[instance_id] = len(encode_chat_messages(tokenizer, messages))
    return lengths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    prompt_tokens = None
    if not args.skip_tokenize:
        print("tokenizing train prompts if tokenizer is available...")
        prompt_tokens = tokenize_train_prompts(
            repo_root=repo_root,
            tokenizer_path=args.tokenizer_path,
        )
        if prompt_tokens is None:
            print("tokenizer unavailable; audit will use M2C stats / overlong jsonl")
        else:
            print(f"tokenized {len(prompt_tokens)} train prompts")

    payload = run_audit(
        repo_root,
        prompt_tokens_by_id=prompt_tokens,
        require_live_oracle=bool(args.require_live_oracle),
    )
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else audit_dir(repo_root)
    )
    written = write_audit_artifacts(payload, output_dir)
    ready = bool(payload.get("READY_FOR_SCALED_M5_DESIGN"))
    primary = payload.get("primary_pool") or {}
    print(
        json.dumps(
            {
                "READY_FOR_SCALED_M5_DESIGN": ready,
                "identity_source": payload.get("identity_source"),
                "oracle_replayed": (payload.get("m3c_replay") or {}).get("oracle_replayed"),
                "n_primary_unique": primary.get("n_unique"),
                "primary_ordered_ids_sha256": (primary.get("stats") or {}).get(
                    "ordered_ids_sha256"
                ),
                "primary_padded_ids_sha256": (primary.get("pad") or {}).get(
                    "padded_ids_sha256"
                ),
                "n_pad": (primary.get("pad") or {}).get("n_pad"),
                "optimizer_steps": (primary.get("pad") or {}).get("optimizer_steps"),
                "errors": payload.get("errors"),
                "warnings": payload.get("warnings"),
                "output_dir": str(output_dir),
                "written": written,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
