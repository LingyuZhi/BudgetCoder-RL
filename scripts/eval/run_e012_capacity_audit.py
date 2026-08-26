#!/usr/bin/env python
"""E012 CPU capacity audit: prompt ranking + smallest defensible envelope.

Does not run GPU, GRPO, or vLLM. Does not edit E011 artifacts or stage1_m5_main.json.

Usage:

    python scripts/eval/run_e012_capacity_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym_materialize import train_parquet_path  # noqa: E402
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.e012 import (  # noqa: E402
    E011_FAILURE_MAX_SEQ,
    ENVELOPE_CANDIDATES,
    EXPECTED_E011_OVERLAY_SHA256,
    PINNED_CAPACITY_NOTES,
    REQUIRED_TASK,
    decide_capacity,
    default_e011_runtime_path,
    default_e012_output_dir,
    load_episode_proxy_rows,
    tokenize_prompt_rows,
    write_overlay_and_lock,
)
from budget_coder_rl.eval.m4a import (  # noqa: E402
    default_candidate_path,
    load_candidate_ordered_ids,
)
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m5a import default_output_dir  # noqa: E402
from budget_coder_rl.eval.m5b import EXPECTED_MAIN_SHA256  # noqa: E402
from budget_coder_rl.eval.provenance import sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--write-overlay", action="store_true", default=True)
    parser.add_argument("--skip-overlay", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    data_root = Path(args.data_root) if args.data_root else bcrl_data_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e012_output_dir(repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    main_path = repo_root / "configs/experiments/stage1_m5_main.json"
    e011_overlay = default_e011_runtime_path(repo_root)
    if sha256_file(main_path) != EXPECTED_MAIN_SHA256:
        print("HARD FAIL: stage1_m5_main.json hash changed", file=sys.stderr)
        return 1
    if sha256_file(e011_overlay) != EXPECTED_E011_OVERLAY_SHA256:
        print("HARD FAIL: E011 overlay hash changed", file=sys.stderr)
        return 1

    candidate_path = default_candidate_path(repo_root)
    ordered_ids = load_candidate_ordered_ids(candidate_path)
    parquet_path = train_parquet_path(repo_root)
    e011_episodes = default_output_dir(repo_root, "E011") / "episodes.jsonl"
    e010_episodes = default_output_dir(repo_root, "E010") / "episodes.jsonl"
    episode_rows = load_episode_proxy_rows(e011_episodes) + load_episode_proxy_rows(e010_episodes)
    episode_prompt_by_id: dict[str, int] = {}
    for row in episode_rows:
        iid = str(row.get("instance_id") or "")
        if not iid:
            continue
        prompt_n = int(row["prompt_token_count"])
        prev = episode_prompt_by_id.get(iid)
        if prev is None or prompt_n > prev:
            episode_prompt_by_id[iid] = prompt_n

    model_path = args.model_path or (data_root / "models" / "Qwen3-4B-Instruct-2507")
    if not Path(model_path).exists():
        print(f"HARD FAIL: missing tokenizer/model {model_path}", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    ranked = tokenize_prompt_rows(
        parquet_path=parquet_path,
        ordered_ids=ordered_ids,
        tokenizer=tokenizer,
        episode_prompt_by_id=episode_prompt_by_id,
    )
    decision = decide_capacity(ranked_prompts=ranked, episode_rows=episode_rows)
    if REQUIRED_TASK not in decision["selected_instance_ids"]:
        print(f"HARD FAIL: {REQUIRED_TASK} not selected", file=sys.stderr)
        return 1

    overlay_info: dict[str, Any] = {}
    if args.write_overlay and not args.skip_overlay:
        overlay_info = write_overlay_and_lock(
            repo_root,
            ppo_max_token_len_per_gpu=int(decision["chosen_envelope"]),
        )

    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E012",
        "pinned_capacity_notes": PINNED_CAPACITY_NOTES,
        "e011_failure": {
            "max_seq_len": E011_FAILURE_MAX_SEQ,
            "task": REQUIRED_TASK,
            "prompt": 12811,
        },
        "n_ranked": len(ranked),
        "ranked_top": sorted(
            ranked, key=lambda row: -float(row["prompt_token_count"])
        )[:16],
        "n_episode_rows": len(episode_rows),
        "decision": decision,
        "candidates": list(ENVELOPE_CANDIDATES),
        "overlay": overlay_info.get("lock") or {},
        "parent_sha256": EXPECTED_MAIN_SHA256,
        "e011_sha256": EXPECTED_E011_OVERLAY_SHA256,
        "research_freeze_unmodified": True,
    }
    write_json(output_dir / "capacity_audit.json", audit)
    print(json.dumps(
        {
            "chosen_envelope": decision["chosen_envelope"],
            "needed": decision["needed"],
            "selected_instance_ids": decision["selected_instance_ids"],
            "overlay_sha256": overlay_info.get("sha256"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
