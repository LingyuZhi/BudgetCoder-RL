#!/usr/bin/env python
"""M4A GPU smoke: RewardLoop + GRPO group uid/advantage. No optimizer.

Trainer-level ``rollout.n=4`` via DataProto.repeat + shared uid.
vLLM per-request n stays 1. Does not call actor.update_actor.

Usage (compute node n30158, conda env ``verl``):

    python scripts/eval/run_m4a_gpu.py --experiment-id E008
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT / "scripts" / "smoke") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke"))

from gpu_runtime import (  # noqa: E402
    M3C_AGENT_LOOP_CONFIG_RELPATH,
    MAX_MODEL_LEN,
    PROMPT_LENGTH,
    RESPONSE_LENGTH,
    apply_reward_loop_config,
    as_mapping,
    assert_sampling_config,
    build_batch,
    build_config,
    get_verl_info,
    init_agent_loop_manager,
    nt,
    object_array,
    pick_free_gpu,
    resolve_model_path,
)
from smoke_rlhf_dataset import resolve_tokenizer_path  # noqa: E402

from budget_coder_rl.agent_loop.rollout_verify import (  # noqa: E402
    verify_padded_sample,
)
from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root, swe_gym_repos_root  # noqa: E402
from budget_coder_rl.env import RepoEnvironment  # noqa: E402
from budget_coder_rl.eval.episode import build_episode_record  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4a import (  # noqa: E402
    BUDGET_VISIBLE,
    EXPERIMENT_ID,
    GROUP_N,
    MILESTONE,
    OBS_TOKENS_LIMIT,
    REWARD_NUM_WORKERS,
    assemble_group_evidence,
    artifact_hashes,
    default_candidate_path,
    default_e007_groups_path,
    default_freeze_path,
    freeze_contract_errors,
    leakage_errors,
    load_candidate_ordered_ids,
    load_e007_groups,
    load_json,
    scalar_advantage,
    select_smoke_instance_ids,
)
from budget_coder_rl.eval.provenance import collect_run_provenance  # noqa: E402
from budget_coder_rl.ray_tmpdir import (  # noqa: E402
    cleanup_our_tmp_ray,
    ray_init_kwargs,
    short_temp_root,
)

TRACE_NOTE = (
    "Research/debug artifact. AgentLoopOutput / DataProto token arrays are the "
    "training truth. Do not rebuild RL token trajectories from this JSONL. "
    "M4A does not run actor.update_actor."
)
EXTRA_FIELD_KEYS = (
    "instance_id",
    "repo",
    "base_commit",
    "split",
    "final_submission",
    "termination",
    "segments",
    "events",
    "prompt_token_count",
    "policy_token_count",
    "observation_token_count",
    "tool_observation_token_count",
    "repo_observation_tokens",
    "budget_metadata_tokens",
    "total_env_tokens",
    "obs_tokens_used",
    "obs_tokens_limit",
    "obs_tokens_remaining",
    "budget_accounting_version",
    "budget_visible",
    "budget_exhausted",
    "sampling_params",
    "sampling_seed",
    "max_turns",
    "max_new_tokens_per_turn",
    "model_name_or_path",
    "trace_role",
    "unpadded_prompt_ids",
    "reward_extra_info",
)
REWARD_FN_RELPATH = "src/budget_coder_rl/reward/localization_score.py"
VERL_PATH_TEXT = """# M4A pinned veRL group/reward path

Checkout: `{commit}` (`0.8.0.dev0`, fork LingyuZhi/rtrl-verl).

## Group identity

- Trainer assigns `non_tensor_batch["uid"] = uuid4()` per logical prompt row
  (`verl/trainer/ppo/ray_trainer.py` ~1346-1349).
- `DataProto.repeat(repeat_times=rollout.n, interleave=True)` uses `np.repeat`,
  so siblings share that uid (`verl/protocol.py` ~971-1007).
- `compute_grpo_outcome_advantage(..., index=data.non_tensor_batch["uid"])`
  (`verl/trainer/ppo/core_algos.py` ~268-331; called from `ray_trainer.py` ~183-195).
- AgentLoop runs **one trajectory per row**. vLLM async `SamplingParams` does
  **not** set `n`. `actor_rollout_ref.rollout.n` is trainer group size.

## Reward

- Leave `AgentLoopOutput.reward_score` unset so RewardLoop runs
  (`verl/experimental/agent_loop/agent_loop.py` `_compute_score` ~806-868).
- `NaiveRewardManager.run_single` merges `tool_extra_fields` into `extra_info`
  and calls custom `compute_score` (`reward_manager/naive.py` ~34-99).
- Scalar is written to `rm_scores[last valid response token]`, then
  `token_level_scores` / `token_level_rewards` with `use_kl_in_reward=false`.
- `_postprocess` does **not** copy input `uid` when RewardLoop handles are set;
  M4A keeps the pre-rollout repeated batch and reattaches `uid`.

## Size-1 groups

Pinned GRPO sets mean=0 and std=1 for singleton uids, so advantage equals the
raw reward (not zero). Expanding the dataset with `rollout.n=1` does **not**
produce group-relative advantages.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--n-tasks", type=int, default=GROUP_N)
    parser.add_argument("--skip-gpu-pick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance-ids", default=None)
    return parser.parse_args(argv)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=True) + "\n")


def extra_from_result(result, index: int) -> dict[str, Any]:
    extra_keys = result.non_tensor_batch
    fake: dict[str, Any] = {}
    for key in EXTRA_FIELD_KEYS:
        payload = extra_keys.get(key)
        if payload is not None:
            fake[key] = payload[index]
    return fake


def index_dataset(dataset) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        extra = as_mapping(item.get("extra_info"))
        instance_id = str(extra.get("instance_id") or "")
        if not instance_id:
            raise SystemExit(f"dataset[{index}] missing extra_info.instance_id")
        indexed[instance_id] = item
    return indexed


def patch_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(extra)
    out["obs_tokens_limit"] = OBS_TOKENS_LIMIT
    out["budget_visible"] = BUDGET_VISIBLE
    out.pop("sampling_seed", None)
    return out


def decoded_obs_texts(tokenizer, segments: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for item in segments:
        if item.get("kind") != "observation":
            continue
        ids = list(item.get("token_ids") or [])
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def write_summary(
    path: Path,
    *,
    status: str,
    groups: list[dict[str, Any]],
    wiring_ok: bool,
    tito_errors: list[str],
    leak_errors: list[str],
    elapsed_s: float,
    instance_ids: list[str],
) -> None:
    passed = [item["instance_id"] for item in groups if item.get("gate_pass")]
    lines = [
        "# M4A / E008 Reward & GRPO Group Semantics",
        "",
        f"- status: **{status}**",
        f"- wiring_ok: {wiring_ok}",
        f"- elapsed_s: {elapsed_s:.1f}",
        f"- instance_ids: {', '.join(instance_ids)}",
        f"- gate_pass groups: {passed or ['(none)']}",
        f"- tito_errors: {len(tito_errors)}",
        f"- leak_errors: {len(leak_errors)}",
        "",
        "PASS requires at least one group with same task, 4 siblings, same uid,",
        "mixed deterministic rewards, non-zero relative advantages, and",
        "`rm_scores.sum == localization_score`. No optimizer step ran.",
        "",
    ]
    for group in groups:
        lines.append(f"## {group.get('instance_id')}")
        lines.append("")
        lines.append(f"- uid: `{group.get('uid')}`")
        lines.append(f"- rewards: {group.get('rewards')}")
        lines.append(f"- advantages: {group.get('advantages')}")
        lines.append(f"- mixed: {group.get('mixed')}")
        lines.append(f"- nonzero_advantage: {group.get('nonzero_advantage')}")
        lines.append(f"- gate_pass: {group.get('gate_pass')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "outputs" / "experiments" / args.experiment_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = default_freeze_path(repo_root)
    freeze = load_json(freeze_path)
    freeze_errors = freeze_contract_errors(freeze)
    if freeze_errors:
        print(f"HARD FAIL: freeze contract {freeze_errors}", file=sys.stderr)
        return 1
    candidate_path = default_candidate_path(repo_root)
    e007_path = default_e007_groups_path(repo_root)
    ordered_ids = load_candidate_ordered_ids(candidate_path)
    if args.instance_ids:
        instance_ids = [item.strip() for item in args.instance_ids.split(",") if item.strip()]
    else:
        instance_ids = select_smoke_instance_ids(
            ordered_ids,
            load_e007_groups(e007_path),
            n=int(args.n_tasks),
        )
    parquet_path = (
        args.parquet.resolve() if args.parquet is not None else train_parquet_path(repo_root)
    )
    oracle_path = oracle_parquet_path(repo_root).resolve()
    agent_loop_config = repo_root / M3C_AGENT_LOOP_CONFIG_RELPATH
    reward_fn_path = repo_root / REWARD_FN_RELPATH
    if not parquet_path.is_file():
        print(f"HARD FAIL: missing parquet {parquet_path}", file=sys.stderr)
        return 1
    if not oracle_path.is_file():
        print(f"HARD FAIL: missing oracle {oracle_path}", file=sys.stderr)
        return 1
    if not reward_fn_path.is_file():
        print(f"HARD FAIL: missing reward fn {reward_fn_path}", file=sys.stderr)
        return 1
    if not agent_loop_config.is_file():
        print(f"HARD FAIL: missing {agent_loop_config}", file=sys.stderr)
        return 1

    os.environ["BCRL_ORACLE_PARQUET"] = str(oracle_path)
    data_root = args.data_root or bcrl_data_root()
    tmp_cleanup = cleanup_our_tmp_ray()
    tmp_root = short_temp_root()
    gpu_info = (
        {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "source": "skipped"}
        if args.skip_gpu_pick
        else pick_free_gpu()
    )
    model_path = resolve_model_path(args.model_path)
    if not Path(model_path).exists():
        print(f"HARD FAIL: model path does not exist: {model_path}", file=sys.stderr)
        return 1
    tokenizer_path = resolve_tokenizer_path(None) or model_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset

    cache_dir = output_dir / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "return_raw_chat": True,
            "filter_overlong_prompts": False,
            "cache_dir": str(cache_dir),
            "max_prompt_length": 131072,
        }
    )
    dataset = RLHFDataset(
        data_files=str(parquet_path),
        tokenizer=tokenizer,
        config=data_cfg,
    )
    indexed = index_dataset(dataset)
    env = RepoEnvironment(
        repos_root=swe_gym_repos_root(args.data_root),
        data_root=args.data_root,
    )
    provenance = collect_run_provenance(
        repo_root,
        agent_loop_config=agent_loop_config,
        model_path=model_path,
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
    )
    provenance["experiment_id"] = args.experiment_id
    provenance["milestone"] = MILESTONE
    provenance["optimizer"] = False
    provenance["lora_update"] = False
    provenance["instance_ids"] = instance_ids
    provenance["selection"] = {
        "universe": "m3c_train_candidates.ordered_ids",
        "filter": "E007 mixed=true, keep candidate order",
        "gold_used_for_cherry_pick": False,
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["envelope"] = {
        "prompt_length": PROMPT_LENGTH,
        "response_length": RESPONSE_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "obs_tokens_limit": OBS_TOKENS_LIMIT,
        "budget_visible": BUDGET_VISIBLE,
        "group_n": GROUP_N,
        "vllm_rollout_n": 1,
        "actor_rollout_ref.rollout.n": GROUP_N,
    }
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "candidates": candidate_path,
            "e007_groups": e007_path,
            "oracle": oracle_path,
            "agent_loop_config": agent_loop_config,
            "reward_fn": reward_fn_path,
        }
    )
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["gpu"] = gpu_info
    provenance["host"] = os.uname().nodename if hasattr(os, "uname") else ""
    provenance["data_root"] = str(data_root)
    write_json(output_dir / "provenance.json", provenance)
    (output_dir / "m4a_verl_path.md").write_text(
        VERL_PATH_TEXT.format(commit=provenance.get("verl", {}).get("commit") or "unknown"),
        encoding="utf-8",
    )

    python_path = os.environ.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    merged_pythonpath = (
        src_path if not python_path else src_path + os.pathsep + python_path
    )
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
            "BCRL_DATA_ROOT": str(data_root),
            "BCRL_ORACLE_PARQUET": str(oracle_path),
            "PYTHONPATH": merged_pythonpath,
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
        }
    }

    import ray
    from verl.experimental.reward_loop import RewardLoopManager
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    ray.init(runtime_env=runtime_env, **ray_init_kwargs(tmp_root))
    provenance["verl_runtime"] = get_verl_info()
    write_json(output_dir / "provenance.json", provenance)

    episodes_path = output_dir / "episodes.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    groups: list[dict[str, Any]] = []
    tito_errors: list[str] = []
    leak_errors: list[str] = []
    wiring_ok = True
    started = time.time()
    stop_reason = "completed"
    try:
        config = build_config(
            model_path,
            n_gpus=args.n_gpus,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            agent_loop_config=str(agent_loop_config),
            rollout_n=GROUP_N,
        )
        apply_reward_loop_config(
            config,
            reward_fn_path=str(reward_fn_path),
            reward_fn_name="compute_score",
            num_workers=REWARD_NUM_WORKERS,
        )
        sampling_recorded = assert_sampling_config(config, require_rollout_n=GROUP_N)
        provenance["sampling_rollout"] = sampling_recorded
        write_json(output_dir / "provenance.json", provenance)
        reward_loop_manager = RewardLoopManager(config)
        manager = init_agent_loop_manager(
            config,
            reward_loop_worker_handles=reward_loop_manager.reward_loop_workers,
        )

        for task_index, instance_id in enumerate(instance_ids):
            if instance_id not in indexed:
                raise SystemExit(f"instance not in parquet: {instance_id}")
            source = indexed[instance_id]
            extra = patch_extra(as_mapping(source.get("extra_info")))
            env.prepare_from_extra_info(extra)
            item = dict(source)
            item["extra_info"] = extra
            item["raw_prompt"] = source.get("raw_prompt")
            item["data_source"] = source.get("data_source")
            item["reward_model"] = source.get("reward_model")
            logical = build_batch([item], validate=False)
            uid = str(uuid.uuid4())
            logical.non_tensor_batch["uid"] = object_array([uid])
            gen_input = logical.repeat(repeat_times=GROUP_N, interleave=True)
            uids = list(gen_input.non_tensor_batch["uid"])
            if len(set(str(value) for value in uids)) != 1:
                raise SystemExit("HARD FAIL: DataProto.repeat did not share uid")
            batch_t0 = time.time()
            gen_output = manager.generate_sequences(prompts=gen_input)
            batch_dt = time.time() - batch_t0
            if len(gen_output) != GROUP_N:
                raise SystemExit(
                    f"generate_sequences returned {len(gen_output)} for group size {GROUP_N}"
                )
            if "rm_scores" not in gen_output.batch.keys():
                raise SystemExit(
                    "HARD FAIL: rm_scores missing; RewardLoop did not assign reward_score"
                )
            gen_output.non_tensor_batch["uid"] = np.array(uids, dtype=object)
            rm_scores = gen_output.batch["rm_scores"]
            response_mask = gen_output.batch["response_mask"]
            advantages, _returns = compute_grpo_outcome_advantage(
                token_level_rewards=rm_scores,
                response_mask=response_mask,
                index=gen_output.non_tensor_batch["uid"],
            )
            members: list[dict[str, Any]] = []
            episode_rows: list[dict[str, Any]] = []
            for sibling in range(GROUP_N):
                extra_fields = extra_from_result(gen_output, sibling)
                sampling = extra_fields.get("sampling_params") or {}
                if sampling.get("temperature") in {0, 0.0}:
                    raise SystemExit(
                        f"HARD FAIL: greedy sampling on {instance_id} sibling {sibling}"
                    )
                if "do_sample" in sampling:
                    raise SystemExit("HARD FAIL: do_sample leaked into sampling_params")
                if sampling.get("seed") is not None:
                    raise SystemExit(
                        "HARD FAIL: sampling seed set; GRPO siblings would collapse"
                    )
                segments = list(extra_fields.get("segments") or [])
                unpadded = extra_fields.get("unpadded_prompt_ids") or []
                tito = verify_padded_sample(
                    prompt_width=PROMPT_LENGTH,
                    response_width=RESPONSE_LENGTH,
                    prompts_row=gen_output.batch["prompts"][sibling],
                    responses_row=gen_output.batch["responses"][sibling],
                    response_mask_row=response_mask[sibling],
                    attention_mask_row=gen_output.batch["attention_mask"][sibling],
                    unpadded_prompt_ids=unpadded,
                    segments=segments,
                )
                if tito:
                    tito_errors.extend(
                        f"{instance_id}[{sibling}]: {err}" for err in tito
                    )
                    wiring_ok = False
                prompt_text = tokenizer.decode(
                    list(unpadded), skip_special_tokens=True
                )
                leaks = leakage_errors(
                    decoded_prompt=prompt_text,
                    decoded_observations=decoded_obs_texts(tokenizer, segments),
                    extra_field_keys=list(extra_fields.keys()),
                )
                if leaks:
                    leak_errors.extend(
                        f"{instance_id}[{sibling}]: {err}" for err in leaks
                    )
                    wiring_ok = False
                reward_extra = extra_fields.get("reward_extra_info") or {}
                if not isinstance(reward_extra, Mapping):
                    reward_extra = {}
                loc_score = reward_extra.get("score")
                if loc_score is None:
                    loc_score = nt(gen_output, "score", sibling)
                rm_score = float(rm_scores[sibling].sum().item())
                adv_scalar = scalar_advantage(
                    advantages[sibling].tolist(),
                    response_mask[sibling].tolist(),
                )
                loc_score_f = float(loc_score) if loc_score is not None else None
                if loc_score_f is None or abs(rm_score - loc_score_f) > 1e-5:
                    wiring_ok = False
                    tito_errors.append(
                        f"{instance_id}[{sibling}]: rm_score={rm_score} "
                        f"!= localization score={loc_score_f}"
                    )
                record = build_episode_record(extra_fields, provenance=provenance)
                record["trace_note"] = TRACE_NOTE
                record["experiment_id"] = args.experiment_id
                record["group"] = {
                    "uid": str(uids[sibling]),
                    "group_n": GROUP_N,
                    "group_index": sibling,
                    "logical_task_index": task_index,
                }
                record["grpo"] = {
                    "uid": str(uids[sibling]),
                    "rm_score": rm_score,
                    "advantage_scalar": adv_scalar,
                    "localization_score": loc_score_f,
                }
                if reward_extra:
                    record["localization"] = {
                        key: reward_extra[key]
                        for key in reward_extra
                        if key not in {"instance_id", "solution_str_chars"}
                    }
                    record["localization"]["localization_score"] = loc_score_f
                episode_rows.append(record)
                members.append(
                    {
                        "instance_id": instance_id,
                        "uid": str(uids[sibling]),
                        "rm_score": rm_score,
                        "localization_score": loc_score_f if loc_score_f is not None else rm_score,
                        "advantage_scalar": adv_scalar,
                        "termination": extra_fields.get("termination"),
                        "rollout_n": sibling,
                    }
                )
            evidence = assemble_group_evidence(members)
            evidence["generate_s"] = batch_dt
            groups.append(evidence)
            if not (evidence["same_task"] and evidence["same_uid"] and evidence["n_members"] == GROUP_N):
                wiring_ok = False
            append_jsonl(episodes_path, episode_rows)
            print(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "uid": uid,
                        "rewards": evidence["rewards"],
                        "mixed": evidence["mixed"],
                        "gate_pass": evidence["gate_pass"],
                        "batch_s": round(batch_dt, 1),
                    }
                ),
                flush=True,
            )
            write_json(output_dir / "group_evidence.json", {"groups": groups})
    except Exception:
        stop_reason = "error"
        write_json(
            output_dir / "run_error.json",
            {
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

    elapsed = time.time() - started
    any_gate = any(item.get("gate_pass") for item in groups)
    if stop_reason == "error":
        status = "FAIL"
    elif tito_errors or leak_errors or not wiring_ok:
        status = "FAIL_WIRING"
    elif any_gate:
        status = "PASS"
    else:
        status = "WIRING_OK_NO_MIXED"
    payload = {
        "status": status,
        "stop_reason": stop_reason,
        "experiment_id": args.experiment_id,
        "milestone": MILESTONE,
        "optimizer": False,
        "wiring_ok": wiring_ok and not tito_errors and not leak_errors,
        "any_gate_pass": any_gate,
        "instance_ids": instance_ids,
        "n_groups": len(groups),
        "elapsed_s": elapsed,
        "tito_errors": tito_errors,
        "leak_errors": leak_errors,
        "groups": groups,
        "sampling": QWEN3_SAMPLING,
        "validate": False,
        "obs_tokens_limit": OBS_TOKENS_LIMIT,
        "budget_visible": BUDGET_VISIBLE,
        "group_n": GROUP_N,
        "vllm_rollout_n": 1,
        "actor_rollout_ref.rollout.n": GROUP_N,
    }
    write_json(output_dir / "group_evidence.json", payload)
    write_json(output_dir / "run_status.json", {k: payload[k] for k in payload if k != "groups"})
    write_summary(
        output_dir / "SUMMARY.md",
        status=status,
        groups=groups,
        wiring_ok=payload["wiring_ok"],
        tito_errors=tito_errors,
        leak_errors=leak_errors,
        elapsed_s=elapsed,
        instance_ids=instance_ids,
    )
    print(json.dumps({k: payload[k] for k in payload if k != "groups"}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
