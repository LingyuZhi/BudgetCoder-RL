#!/usr/bin/env python
"""CPU-only smoke: load M1E policy parquet with pinned veRL RLHFDataset.

Does not start Ray, vLLM, or GPU rollout. Does not implement reward or Agent
scaffold.

Usage (pinned RL conda env):

    python scripts/smoke/smoke_rlhf_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym import (  # noqa: E402
    EXPECTED_SHA256,
    length_stats,
    parquet_path,
    sha256_file,
    verify_parquet_file,
)
from budget_coder_rl.data.swe_gym_fields import (  # noqa: E402
    collect_forbidden_policy_keys,
)
from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    EXPECTED_DEV_ROWS,
    EXPECTED_TRAIN_ROWS,
    to_jsonable,
    train_parquet_path,
    dev_parquet_path,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--raw-parquet", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--skip-token-stats",
        action="store_true",
        help="do not tokenize prompts even if a local tokenizer exists",
    )
    return parser.parse_args(argv)


def resolve_tokenizer_path(cli_path: Path | None) -> str | None:
    if cli_path is not None:
        return str(cli_path)
    env_path = os.environ.get("BCRL_TOKENIZER_PATH") or os.environ.get("BCRL_MODEL_PATH")
    if env_path:
        return env_path
    data_root = Path(
        os.environ.get("BCRL_DATA_ROOT", os.path.expanduser("~/my_data/budget-coder-rl"))
    )
    preferred = data_root / "models" / "Qwen3-4B-Instruct-2507"
    candidates: list[Path] = []
    if preferred.is_dir():
        candidates.append(preferred)
    if (data_root / "models").is_dir():
        candidates.extend(sorted((data_root / "models").glob("*")))
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    for repo_dir in sorted(hub.glob("models--Qwen--*")):
        candidates.extend(sorted((repo_dir / "snapshots").glob("*")))
    for cand in candidates:
        if (Path(cand) / "tokenizer_config.json").exists():
            return str(cand)
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def _as_messages(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    items = list(value)
    messages: list[dict[str, str]] = []
    for item in items:
        mapping = _as_mapping(item)
        messages.append(
            {
                "role": str(mapping.get("role") or ""),
                "content": str(mapping.get("content") or ""),
            }
        )
    return messages


def load_problem_statements(raw_parquet: Path) -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(raw_parquet, columns=["instance_id", "problem_statement"])
    out: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        instance_id = str(record["instance_id"])
        out[instance_id] = str(record["problem_statement"])
    return out


def build_dataset(data_file: Path, tokenizer: Any, cache_dir: Path):
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "return_raw_chat": True,
            "filter_overlong_prompts": False,
            "cache_dir": str(cache_dir),
            "max_prompt_length": 131072,
        }
    )
    return RLHFDataset(
        data_files=str(data_file),
        tokenizer=tokenizer,
        config=config,
    )


def check_dataset(
    dataset: Any,
    *,
    expected_len: int,
    split: str,
    problem_statements: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if len(dataset) != expected_len:
        errors.append(f"{split}: len={len(dataset)} expected={expected_len}")
        return errors
    seen: set[str] = set()
    for index in range(len(dataset)):
        item = dataset[index]
        extra = _as_mapping(item.get("extra_info"))
        instance_id = str(extra.get("instance_id") or "")
        if not instance_id:
            errors.append(f"{split}[{index}]: extra_info.instance_id missing")
            continue
        if instance_id in seen:
            errors.append(f"{split}: duplicate instance_id {instance_id}")
        seen.add(instance_id)
        if instance_id not in problem_statements:
            errors.append(f"{split} {instance_id}: not in raw parquet")
            continue
        expected = [
            {"role": "user", "content": problem_statements[instance_id]}
        ]
        raw_prompt = _as_messages(item.get("raw_prompt"))
        if raw_prompt != expected:
            errors.append(f"{split} {instance_id}: raw_prompt != problem_statement")
        prompt = _as_messages(item.get("prompt"))
        if prompt != expected:
            errors.append(f"{split} {instance_id}: prompt != problem_statement")
        if str(extra.get("split") or "") != split:
            errors.append(
                f"{split} {instance_id}: extra_info.split={extra.get('split')!r}"
            )
        if not extra.get("repo") or not extra.get("base_commit"):
            errors.append(f"{split} {instance_id}: repo/base_commit missing")
        reward = _as_mapping(item.get("reward_model"))
        if str(reward.get("ground_truth") or "") != instance_id:
            errors.append(
                f"{split} {instance_id}: ground_truth={reward.get('ground_truth')!r}"
            )
        leaked = collect_forbidden_policy_keys(
            {
                "prompt": to_jsonable(item.get("prompt")),
                "reward_model": to_jsonable(item.get("reward_model")),
                "extra_info": to_jsonable(extra),
                "data_source": item.get("data_source"),
            }
        )
        if leaked:
            errors.append(f"{split} {instance_id}: leakage {leaked[:4]}")
        if extra.get("instance_id") != instance_id:
            errors.append(f"{split} {instance_id}: extra_info identity mismatch")
    return errors


def prompt_token_lengths(dataset: Any, tokenizer: Any) -> dict[str, Any]:
    lengths: list[int] = []
    for index in range(len(dataset)):
        item = dataset[index]
        messages = _as_messages(item.get("raw_prompt"))
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        lengths.append(len(token_ids))
    return length_stats(lengths)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    train_path = args.train.resolve() if args.train else train_parquet_path(repo_root)
    dev_path = args.dev.resolve() if args.dev else dev_parquet_path(repo_root)
    raw_parquet = (
        args.raw_parquet.resolve() if args.raw_parquet else parquet_path(repo_root)
    )

    errors: list[str] = []
    if not train_path.is_file() or not dev_path.is_file():
        print("HARD FAIL: materialized parquet missing. Run:", file=sys.stderr)
        print("  python scripts/data/materialize_swe_gym_m1e.py", file=sys.stderr)
        return 1
    if not raw_parquet.is_file():
        print(f"HARD FAIL: raw parquet not found: {raw_parquet}", file=sys.stderr)
        return 1
    errors.extend(verify_parquet_file(raw_parquet))

    try:
        from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: F401
    except ImportError as exc:
        print(f"HARD FAIL: cannot import pinned RLHFDataset: {exc}", file=sys.stderr)
        return 1

    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    tokenizer: Any
    token_stats_status = "deferred: no local tokenizer"
    if tokenizer_path is None:
        tokenizer = type("StubTokenizer", (), {})()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        token_stats_status = f"local tokenizer: {tokenizer_path}"

    cache_dir = repo_root / "outputs" / "smoke" / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(train_path, tokenizer, cache_dir / "train")
    dev_ds = build_dataset(dev_path, tokenizer, cache_dir / "dev")
    problem_statements = load_problem_statements(raw_parquet)
    errors.extend(
        check_dataset(
            train_ds,
            expected_len=EXPECTED_TRAIN_ROWS,
            split="train",
            problem_statements=problem_statements,
        )
    )
    errors.extend(
        check_dataset(
            dev_ds,
            expected_len=EXPECTED_DEV_ROWS,
            split="dev",
            problem_statements=problem_statements,
        )
    )

    token_stats: dict[str, Any] | None = None
    if errors:
        print("HARD FAIL:", file=sys.stderr)
        for item in errors[:30]:
            print(f"  - {item}", file=sys.stderr)
        if len(errors) > 30:
            print(f"  ... {len(errors) - 30} more", file=sys.stderr)
        return 1

    if not args.skip_token_stats and tokenizer_path is not None:
        train_stats = prompt_token_lengths(train_ds, tokenizer)
        dev_stats = prompt_token_lengths(dev_ds, tokenizer)
        token_stats = {"train": train_stats, "dev": dev_stats}
        token_stats_status = "computed"
    elif args.skip_token_stats:
        token_stats_status = "skipped"

    sample = train_ds[0]
    extra = _as_mapping(sample.get("extra_info"))
    report = {
        "train_len": len(train_ds),
        "dev_len": len(dev_ds),
        "raw_parquet_sha256": sha256_file(raw_parquet),
        "expected_raw_sha256": EXPECTED_SHA256,
        "sample_instance_id": extra.get("instance_id"),
        "sample_raw_prompt": _as_messages(sample.get("raw_prompt")),
        "runtime_fields": {
            "instance_id": extra.get("instance_id"),
            "repo": extra.get("repo"),
            "base_commit": extra.get("base_commit"),
        },
        "token_stats_status": token_stats_status,
        "token_stats": token_stats,
        "filter_overlong_prompts": False,
        "return_raw_chat": True,
        "agent_name_in_sample": "agent_name" in sample,
    }
    print(json.dumps(report, indent=2, ensure_ascii=True, default=str))
    print("RLHFDataset smoke PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
