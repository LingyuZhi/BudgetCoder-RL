#!/usr/bin/env python
"""Audit Stage-1 runtime initial-prompt token lengths on frozen train/dev.

Uses the real Qwen tokenizer and the same ``build_stage1_messages`` +
``encode_chat_messages`` path as AgentLoop. Does not filter/split M1 data
and does not truncate prompts.

Usage (pinned RL conda env):

    python scripts/data/audit_runtime_prompt_length.py
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

from smoke_rlhf_dataset import resolve_tokenizer_path  # noqa: E402

from budget_coder_rl.agent_loop.tokenization import encode_chat_messages  # noqa: E402
from budget_coder_rl.data.swe_gym import length_stats, write_json  # noqa: E402
from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    EXPECTED_DEV_ROWS,
    EXPECTED_TRAIN_ROWS,
    dev_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.protocol.prompt import (  # noqa: E402
    build_stage1_messages,
    policy_safe_repo,
)

CANDIDATE_LIMITS = (1024, 2048, 4096, 8192, 16384, 32768)
STATS_RELPATH = "data/stats/swe_gym_m2c_prompt_length.json"
OVERLONG_RELPATH = "data/interim/swe_gym/m2c_prompt_overlong.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overlong-jsonl", type=Path, default=None)
    return parser.parse_args(argv)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def _issue_text(prompt: Any) -> str:
    if prompt is None:
        return ""
    items = list(prompt)
    if not items:
        return ""
    first = items[0]
    mapping = _as_mapping(first)
    return str(mapping.get("content") or "")


def audit_split(parquet_path: Path, tokenizer, *, expected_n: int) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path, columns=["prompt", "extra_info"])
    if len(frame) != expected_n:
        raise SystemExit(
            f"{parquet_path}: n={len(frame)} expected {expected_n}"
        )
    lengths: list[int] = []
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        extra = _as_mapping(record.get("extra_info"))
        issue = _issue_text(record.get("prompt"))
        messages = build_stage1_messages(issue, repo=policy_safe_repo(extra))
        n_tokens = len(encode_chat_messages(tokenizer, messages))
        lengths.append(n_tokens)
        records.append(
            {
                "instance_id": extra.get("instance_id"),
                "repo": extra.get("repo"),
                "n_tokens": n_tokens,
            }
        )
    over_limits = {}
    for limit in CANDIDATE_LIMITS:
        over = [item for item in records if int(item["n_tokens"]) > limit]
        over_limits[str(limit)] = {
            "n_over": len(over),
            "fraction": round(len(over) / len(records), 6) if records else 0.0,
        }
    return {
        "n": len(records),
        "stats": length_stats(lengths),
        "over_limits": over_limits,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    if tokenizer_path is None:
        print("HARD FAIL: no local Qwen tokenizer; set BCRL_TOKENIZER_PATH", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_path = args.train.resolve() if args.train is not None else train_parquet_path(repo_root)
    dev_path = args.dev.resolve() if args.dev is not None else dev_parquet_path(repo_root)
    if not train_path.is_file() or not dev_path.is_file():
        print("HARD FAIL: missing M1E parquet. Run materialize_swe_gym_m1e.py", file=sys.stderr)
        return 1

    print(f"tokenizer: {tokenizer_path}")
    print("auditing train...")
    train = audit_split(train_path, tokenizer, expected_n=EXPECTED_TRAIN_ROWS)
    print("auditing dev...")
    dev = audit_split(dev_path, tokenizer, expected_n=EXPECTED_DEV_ROWS)

    combined = train["records"] + dev["records"]
    overlong_jsonl = (
        args.overlong_jsonl
        if args.overlong_jsonl is not None
        else repo_root / OVERLONG_RELPATH
    )
    overlong_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with overlong_jsonl.open("w", encoding="utf-8") as handle:
        for item in combined:
            row = dict(item)
            row["over_limits"] = [
                limit for limit in CANDIDATE_LIMITS if int(item["n_tokens"]) > limit
            ]
            if row["over_limits"]:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "schema_version": "swe-gym-m2c-runtime-prompt-v1",
        "tokenizer": "Qwen/Qwen3-4B-Instruct-2507",
        "tokenizer_path_note": "resolved locally; host path not pinned",
        "encoding": (
            "build_stage1_messages + encode_chat_messages "
            "(veRL apply_chat_template, tools=None, add_generation_prompt=True)"
        ),
        "silent_truncation": False,
        "m1_split_changed": False,
        "candidate_limits": list(CANDIDATE_LIMITS),
        "train": {"n": train["n"], "stats": train["stats"], "over_limits": train["over_limits"]},
        "dev": {"n": dev["n"], "stats": dev["stats"], "over_limits": dev["over_limits"]},
        "m2d_runtime_policy_notes": [
            "Pinned RLHFDataset defaults filter_overlong_prompts=True and max_prompt_length=1024; training must set filter_overlong_prompts=false.",
            "AgentLoop hard-fails when len(prompt_ids) > rollout.prompt_length; do not silent-truncate.",
            "Choose rollout.prompt_length (and later max_model_len) from these runtime-prompt stats before M2D real Qwen generate.",
            "Samples still over the chosen limit should be skipped or failed explicitly, not filtered by rewriting M1.",
        ],
        "overlong_jsonl": str(OVERLONG_RELPATH),
    }
    output = args.output if args.output is not None else repo_root / STATS_RELPATH
    write_json(output, summary)

    def _line(name: str, payload: dict[str, Any]) -> str:
        stats = payload["stats"]
        over = payload["over_limits"]
        over_bits = " ".join(
            f">{limit}={over[str(limit)]['n_over']}" for limit in CANDIDATE_LIMITS
        )
        return (
            f"{name}: n={payload['n']} min={stats['min']} mean={stats['mean']} "
            f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
            f"p99={stats['p99']} max={stats['max']} {over_bits}"
        )

    print(_line("train", summary["train"]))
    print(_line("dev", summary["dev"]))
    print(f"summary: {output}")
    print(f"overlong jsonl: {overlong_jsonl}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
