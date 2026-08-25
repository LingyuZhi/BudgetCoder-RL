#!/usr/bin/env python
"""Write the Stage-1 M3C experiment freeze JSON. No GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m3c import default_candidate_path, default_freeze_path  # noqa: E402
from budget_coder_rl.eval.provenance import sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--regimes",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments" / "E006" / "e006_regimes.json",
    )
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument(
        "--e007-groups",
        type=Path,
        default=REPO_ROOT / "outputs" / "experiments" / "E007" / "m3c_groups.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not args.regimes.is_file():
        print(f"HARD FAIL: missing regimes {args.regimes}", file=sys.stderr)
        return 1
    regimes = json.loads(args.regimes.read_text(encoding="utf-8"))
    candidate_path = (
        args.candidates.resolve()
        if args.candidates is not None
        else default_candidate_path(repo_root)
    )
    if not candidate_path.is_file():
        print(f"HARD FAIL: missing candidate manifest {candidate_path}", file=sys.stderr)
        return 1
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    group_summary = {}
    if args.e007_groups.is_file():
        group_payload = json.loads(args.e007_groups.read_text(encoding="utf-8"))
        group_summary = group_payload.get("group_summary") or {}
    mixed = group_summary.get("mixed_fraction")
    freeze = {
        "schema_version": "bcrl-stage1-m3c-freeze-v1",
        "milestone": "M3C",
        "not_trained": True,
        "reward_loop_wired": False,
        "grpo_optimizer": False,
        "lora_update": False,
        "primary_training_B_obs": regimes.get("primary_training_B_obs"),
        "budget_regimes": {
            "tight": regimes.get("tight"),
            "medium": regimes.get("medium"),
            "loose": regimes.get("loose"),
        },
        "final_evaluation_budget_set": regimes.get("eval_budget_set"),
        "budget_visible": True,
        "budget_accounting_version": "bcrl-bobs-v2",
        "sampling": dict(QWEN3_SAMPLING),
        "validate": False,
        "vllm_rollout_n": 1,
        "proposed_grpo_rollout_n": 4,
        "max_turns": 6,
        "max_new_tokens_per_turn": 2048,
        "envelope": {
            "prompt_length": 16384,
            "response_length": 16384,
            "max_model_len": 32768,
        },
        "localization_reward": {
            "formula": "0.5 * file_f1 + 0.5 * symbol_f1",
            "symbol_unavailable": "file-only localization_score; symbol_status=unavailable",
            "invalid_parse": 0,
            "budget_exhausted_without_valid_final": 0,
            "read_search_bonus": False,
            "token_penalty": False,
        },
        "train_candidate_manifest": {
            "path": "data/manifests/m3c_train_candidates.json",
            "schema_version": candidates.get("schema_version"),
            "n_selected": candidates.get("n_selected"),
            "ordered_ids_sha256": candidates.get("ordered_ids_sha256"),
            "rule_text_sha256": candidates.get("rule_text_sha256"),
            "file_sha256": sha256_file(candidate_path),
        },
        "overlong_train_sample_policy": {
            "filter_overlong_prompts": False,
            "truncation": "error",
            "agent_loop": "PromptTooLongError; no silent truncate",
            "excluded_instance_ids": ["Project-MONAI__MONAI-6344"],
        },
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "verl": {
            "version": "0.8.0.dev0",
            "fork_commit": "8481f9f9880d0f46a75b3db0329d3de8abad3d81",
            "core_commit": "60546ef2a7464a158cd170f58f852a62a4e552ba",
        },
        "m1e_dataset_manifest_sha256": (
            "5b1606760c4864cafb8c4d421472c51ff5f8582e0d6dae9185902095fc17da0c"
        ),
        "e007_group_signal": {
            "mixed_fraction": mixed,
            "zero_variance_fraction": group_summary.get("zero_variance_fraction"),
            "all_zero_fraction": group_summary.get("all_zero_fraction"),
            "mean_group_std": group_summary.get("mean_group_std"),
            "median_group_std": group_summary.get("median_group_std"),
            "needs_n8_probe": group_summary.get("needs_n8_probe"),
            "grpo_signal_plausible": bool(mixed is not None and float(mixed) > 0.05),
        },
        "notes": [
            "Do not silently edit this freeze after M3C. M4/M5 consume it.",
            "Reward, tools, and prompt were not changed to chase trajectories.",
            "vLLM n stays 1; GRPO group size is proposed_grpo_rollout_n.",
        ],
    }
    output = args.output.resolve() if args.output is not None else default_freeze_path(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "primary_training_B_obs": freeze["primary_training_B_obs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
