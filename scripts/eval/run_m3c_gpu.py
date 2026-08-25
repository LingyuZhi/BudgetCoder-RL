#!/usr/bin/env python
"""M3C GPU measurement runner: budget calibration or grouped rollouts.

Uses the M2D AgentLoopManager path. Does not run GRPO, LoRA, or RewardLoop.
vLLM rollout.n stays 1; grouped n is dataset expansion with distinct seeds.

Usage (compute node n30158, pinned conda env ``verl``):

    python scripts/eval/run_m3c_gpu.py --experiment-id E006 --mode calibration
    python scripts/eval/run_m3c_gpu.py --experiment-id E007 --mode grouped \\
        --obs-tokens-limit 4096
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    as_mapping,
    assert_sampling_config,
    build_batch,
    build_config,
    get_verl_info,
    init_agent_loop_manager,
    pick_free_gpu,
    resolve_model_path,
)
from smoke_rlhf_dataset import build_dataset, resolve_tokenizer_path  # noqa: E402

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    dev_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root, swe_gym_repos_root  # noqa: E402
from budget_coder_rl.env import RepoEnvironment  # noqa: E402
from budget_coder_rl.eval.episode import build_episode_record  # noqa: E402
from budget_coder_rl.eval.m3b import (  # noqa: E402
    QWEN3_SAMPLING,
    default_manifest_path as m3b_manifest_path,
    load_manifest as load_m3b_manifest,
)
from budget_coder_rl.eval.m3c import (  # noqa: E402
    CALIBRATION_GPU_BUDGETS,
    GROUP_N,
    PRIMARY_N,
    default_diagnostic_path,
    load_diagnostic_manifest,
    with_group_fields,
)
from budget_coder_rl.eval.provenance import collect_run_provenance  # noqa: E402
from budget_coder_rl.ray_tmpdir import (  # noqa: E402
    cleanup_our_tmp_ray,
    ray_init_kwargs,
    short_temp_root,
)

TRACE_NOTE = (
    "Research/debug artifact. AgentLoopOutput / DataProto token arrays are the "
    "training truth. Do not rebuild RL token trajectories from this JSONL."
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
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=("calibration", "grouped"), required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--primary-n", type=int, default=PRIMARY_N)
    parser.add_argument("--group-n", type=int, default=GROUP_N)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-gpu-pick", action="store_true")
    parser.add_argument("--include-remainder", action="store_true")
    parser.add_argument(
        "--obs-tokens-limit",
        type=int,
        default=None,
        help="grouped-mode hard B_obs; calibration uses --limits",
    )
    parser.add_argument(
        "--limits",
        default=",".join(str(item) for item in CALIBRATION_GPU_BUDGETS),
        help="comma-separated calibration budgets",
    )
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="optional comma-separated subset (E007b)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, np_ndarray()):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def np_ndarray():
    import numpy as np

    return np.ndarray


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


def resume_key(
    instance_id: str,
    *,
    limit: int,
    visible: bool,
    group_index: int,
) -> tuple[str, int, bool, int]:
    return (instance_id, int(limit), bool(visible), int(group_index))


def load_completed(path: Path) -> set[tuple[str, int, bool, int]]:
    done: set[tuple[str, int, bool, int]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            identity = record.get("identity") or {}
            instance_id = str(identity.get("instance_id") or "")
            condition = record.get("condition") or {}
            budget = record.get("budget") or {}
            group = record.get("group") or {}
            limit = condition.get("obs_tokens_limit")
            if limit is None:
                limit = budget.get("obs_tokens_limit")
            visible = condition.get("budget_visible")
            if visible is None:
                visible = budget.get("budget_visible")
            group_index = group.get("group_index")
            if group_index is None:
                group_index = condition.get("group_index")
            if instance_id and limit is not None and visible is not None and group_index is not None:
                done.add(resume_key(instance_id, limit=int(limit), visible=bool(visible), group_index=int(group_index)))
    return done


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


def extra_from_result(result, index: int) -> dict[str, Any]:
    extra_keys = result.non_tensor_batch
    fake: dict[str, Any] = {}
    for key in EXTRA_FIELD_KEYS:
        payload = extra_keys.get(key)
        if payload is not None:
            fake[key] = payload[index]
    return fake


def operational_record(
    job: Mapping[str, Any],
    *,
    error: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "bcrl-episode-v1",
        "trace_note": TRACE_NOTE,
        "identity": {
            "instance_id": job["instance_id"],
            "repo": job.get("repo"),
            "base_commit": None,
            "split": job.get("split"),
        },
        "condition": {
            "budget_visible": True,
            "obs_tokens_limit": job["obs_tokens_limit"],
            "sampling_seed": job.get("sampling_seed"),
            "group_index": job.get("group_index"),
        },
        "group": {
            "group_index": job.get("group_index"),
            "group_n": job.get("group_n"),
        },
        "termination": "operational_error",
        "error": error,
        "provenance": dict(provenance),
    }


def calibration_jobs(manifest: Mapping[str, Any], limits: Sequence[int]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        for limit in limits:
            jobs.append(
                {
                    "instance_id": task["instance_id"],
                    "repo": task["repo"],
                    "split": "dev",
                    "obs_tokens_limit": int(limit),
                    "sampling_seed": int(task["sampling_seed"]),
                    "group_index": 0,
                    "group_n": 1,
                    "skipped_overlong": False,
                }
            )
    return jobs


def grouped_jobs(
    manifest: Mapping[str, Any],
    *,
    limit: int,
    group_n: int,
    primary_n: int,
    include_remainder: bool,
    instance_ids: set[str] | None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        if task.get("skipped_overlong"):
            continue
        if not include_remainder and task.get("set") != "primary":
            continue
        if task.get("set") == "primary" and int(task["task_index"]) >= primary_n:
            continue
        if instance_ids is not None and task["instance_id"] not in instance_ids:
            continue
        seeds = list(task.get("group_seeds") or [])
        if len(seeds) < group_n:
            raise SystemExit(
                f"{task['instance_id']}: group_seeds {len(seeds)} < group_n {group_n}"
            )
        for group_index in range(group_n):
            jobs.append(
                {
                    "instance_id": task["instance_id"],
                    "repo": task["repo"],
                    "split": "train",
                    "obs_tokens_limit": int(limit),
                    "sampling_seed": int(seeds[group_index]),
                    "group_index": int(group_index),
                    "group_n": int(group_n),
                    "skipped_overlong": False,
                }
            )
    return jobs


def parse_limits(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    started = time.time()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "outputs" / "experiments" / args.experiment_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = bcrl_data_root(args.data_root)
    trace_dir = Path(data_root) / "trajectories" / "m3c" / args.experiment_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = trace_dir / "episodes.jsonl"
    heartbeat_path = output_dir / "heartbeat.json"
    status_path = output_dir / "run_status.json"

    agent_loop_config = repo_root / M3C_AGENT_LOOP_CONFIG_RELPATH
    if not agent_loop_config.is_file():
        print(f"HARD FAIL: missing {agent_loop_config}", file=sys.stderr)
        return 1

    if args.mode == "calibration":
        manifest_path = (
            args.manifest.resolve() if args.manifest is not None else m3b_manifest_path(repo_root)
        )
        manifest = load_m3b_manifest(manifest_path)
        parquet_path = args.parquet.resolve() if args.parquet is not None else dev_parquet_path(repo_root)
        limits = parse_limits(args.limits)
        jobs = calibration_jobs(manifest, limits)
        group_n_run = 1
    else:
        if args.obs_tokens_limit is None:
            print("HARD FAIL: grouped mode requires --obs-tokens-limit", file=sys.stderr)
            return 1
        manifest_path = (
            args.manifest.resolve()
            if args.manifest is not None
            else default_diagnostic_path(repo_root)
        )
        manifest = load_diagnostic_manifest(manifest_path)
        parquet_path = (
            args.parquet.resolve() if args.parquet is not None else train_parquet_path(repo_root)
        )
        limits = [int(args.obs_tokens_limit)]
        id_filter = (
            {part.strip() for part in args.instance_ids.split(",") if part.strip()}
            if args.instance_ids
            else None
        )
        jobs = grouped_jobs(
            manifest,
            limit=int(args.obs_tokens_limit),
            group_n=int(args.group_n),
            primary_n=int(args.primary_n),
            include_remainder=bool(args.include_remainder),
            instance_ids=id_filter,
        )
        group_n_run = int(args.group_n)

    if not parquet_path.is_file():
        print(f"HARD FAIL: missing parquet {parquet_path}", file=sys.stderr)
        return 1
    if not manifest_path.is_file():
        print(f"HARD FAIL: missing manifest {manifest_path}", file=sys.stderr)
        return 1

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
    cache_dir = output_dir / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(parquet_path, tokenizer, cache_dir)
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
    provenance["mode"] = args.mode
    provenance["manifest"] = {
        "path": str(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "ordered_ids_sha256": manifest.get("ordered_ids_sha256"),
        "primary_ids_sha256": manifest.get("primary_ids_sha256"),
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["envelope"] = {
        "prompt_length": PROMPT_LENGTH,
        "response_length": RESPONSE_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "obs_tokens_limits": limits,
        "budget_visible": True,
        "group_n": group_n_run,
        "vllm_rollout_n": 1,
    }
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["gpu"] = gpu_info
    provenance["host"] = os.uname().nodename if hasattr(os, "uname") else ""
    write_json(output_dir / "provenance.json", provenance)

    completed = load_completed(episodes_path)
    queue = [
        job
        for job in jobs
        if resume_key(
            job["instance_id"],
            limit=job["obs_tokens_limit"],
            visible=True,
            group_index=job["group_index"],
        )
        not in completed
    ]
    if args.probe_only:
        queue = queue[:1]

    import ray

    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
            "BCRL_DATA_ROOT": str(data_root),
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
        }
    }
    ray_kwargs = ray_init_kwargs(tmp_root)
    ray.init(runtime_env=runtime_env, **ray_kwargs)
    provenance["verl_runtime"] = get_verl_info()
    write_json(output_dir / "provenance.json", provenance)

    n_written = 0
    n_error = 0
    first_batch_seconds = None
    stop_reason = "completed"
    try:
        config = build_config(
            model_path,
            n_gpus=args.n_gpus,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            agent_loop_config=str(agent_loop_config),
        )
        sampling_recorded = assert_sampling_config(config)
        provenance["sampling_rollout"] = sampling_recorded
        write_json(output_dir / "provenance.json", provenance)
        manager = init_agent_loop_manager(config)
        batch_index = 0
        while queue:
            elapsed_h = (time.time() - started) / 3600.0
            if elapsed_h >= args.max_hours:
                stop_reason = "max_hours"
                break
            batch_jobs = queue[: args.batch_size]
            del queue[: args.batch_size]
            items: list[dict[str, Any]] = []
            item_meta: list[dict[str, Any]] = []
            for job in batch_jobs:
                instance_id = job["instance_id"]
                key = resume_key(
                    instance_id,
                    limit=job["obs_tokens_limit"],
                    visible=True,
                    group_index=job["group_index"],
                )
                if key in completed:
                    continue
                if instance_id not in indexed:
                    record = operational_record(
                        job,
                        error=f"instance not in parquet: {instance_id}",
                        provenance=provenance,
                    )
                    append_jsonl(episodes_path, [record])
                    completed.add(key)
                    n_error += 1
                    n_written += 1
                    continue
                source = indexed[instance_id]
                extra = as_mapping(source.get("extra_info"))
                try:
                    env.prepare_from_extra_info(extra)
                except Exception as exc:
                    record = operational_record(
                        job,
                        error=f"snapshot failed: {exc}",
                        provenance=provenance,
                    )
                    append_jsonl(episodes_path, [record])
                    completed.add(key)
                    n_error += 1
                    n_written += 1
                    continue
                patched = dict(source)
                patched["extra_info"] = with_group_fields(
                    extra,
                    visible=True,
                    limit=int(job["obs_tokens_limit"]),
                    seed=int(job["sampling_seed"]),
                    group_index=int(job["group_index"]),
                    group_n=int(job["group_n"]),
                )
                items.append(patched)
                item_meta.append(job)
            if not items:
                continue
            batch_t0 = time.time()
            batch = build_batch(items, validate=False)
            result = manager.generate_sequences(prompts=batch)
            if len(result) != len(items):
                raise SystemExit(
                    f"generate_sequences returned {len(result)} for {len(items)} inputs"
                )
            rows: list[dict[str, Any]] = []
            for index, meta in enumerate(item_meta):
                extra = extra_from_result(result, index)
                sampling = extra.get("sampling_params") or {}
                temperature = sampling.get("temperature")
                if temperature in {0, 0.0}:
                    raise SystemExit(
                        "HARD FAIL: episode sampling temperature is 0 "
                        f"(instance={meta['instance_id']}). validate=True greedy trap?"
                    )
                if "do_sample" in sampling:
                    raise SystemExit("HARD FAIL: do_sample leaked into sampling_params")
                record = build_episode_record(extra, provenance=provenance)
                record["trace_note"] = TRACE_NOTE
                record["experiment_id"] = args.experiment_id
                record["condition"]["group_index"] = int(meta["group_index"])
                record["condition"]["group_n"] = int(meta["group_n"])
                record["group"] = {
                    "group_index": int(meta["group_index"]),
                    "group_n": int(meta["group_n"]),
                }
                rows.append(record)
                completed.add(
                    resume_key(
                        meta["instance_id"],
                        limit=meta["obs_tokens_limit"],
                        visible=True,
                        group_index=meta["group_index"],
                    )
                )
            append_jsonl(episodes_path, rows)
            n_written += len(rows)
            batch_dt = time.time() - batch_t0
            if first_batch_seconds is None:
                first_batch_seconds = batch_dt
            batch_index += 1
            write_json(
                heartbeat_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "batch_index": batch_index,
                    "n_written": n_written,
                    "n_error": n_error,
                    "queue_remaining": len(queue),
                    "elapsed_s": time.time() - started,
                    "last_batch_s": batch_dt,
                },
            )
            print(
                json.dumps(
                    {
                        "batch_index": batch_index,
                        "wrote": len(rows),
                        "batch_s": round(batch_dt, 1),
                        "queue": len(queue),
                    }
                ),
                flush=True,
            )
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
        import ray as _ray

        if _ray.is_initialized():
            _ray.shutdown()

    status = {
        "status": "PASS" if stop_reason != "error" else "FAIL",
        "stop_reason": stop_reason,
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "episodes_path": str(episodes_path),
        "n_written": n_written,
        "n_error": n_error,
        "n_jobs": len(jobs),
        "n_completed": len(completed),
        "elapsed_s": time.time() - started,
        "first_batch_s": first_batch_seconds,
        "sampling": QWEN3_SAMPLING,
        "validate": False,
        "obs_tokens_limits": limits,
        "budget_visible": True,
        "group_n": group_n_run,
        "vllm_rollout_n": 1,
    }
    write_json(status_path, status)
    print(json.dumps(status, indent=2))
    return 0 if stop_reason != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
