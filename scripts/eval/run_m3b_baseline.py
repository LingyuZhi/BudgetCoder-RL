#!/usr/bin/env python
"""M3B frozen base-policy baseline: B0 hidden vs B1 visible, n=1 sampling.

Uses the M2D AgentLoopManager path. Does not run GRPO, LoRA, or RewardLoop.

Usage (compute node n30158, pinned conda env ``verl``):

    python scripts/eval/run_m3b_baseline.py --experiment-id E001
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
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT / "scripts" / "smoke") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke"))

from gpu_runtime import (  # noqa: E402
    AGENT_LOOP_CONFIG_RELPATH,
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
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root, swe_gym_repos_root  # noqa: E402
from budget_coder_rl.env import RepoEnvironment  # noqa: E402
from budget_coder_rl.eval.episode import build_episode_record  # noqa: E402
from budget_coder_rl.eval.m3b import (  # noqa: E402
    PRIMARY_N,
    PROVISIONAL_OBS_TOKENS_LIMIT,
    QWEN3_SAMPLING,
    default_manifest_path,
    load_manifest,
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
    parser.add_argument("--experiment-id", default="E001")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--primary-n", type=int, default=PRIMARY_N)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--no-remainder", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-gpu-pick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
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


def load_completed(path: Path) -> set[tuple[str, bool]]:
    done: set[tuple[str, bool]] = set()
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
            visible = (record.get("condition") or {}).get("budget_visible")
            if instance_id and visible is not None:
                done.add((instance_id, bool(visible)))
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


def with_condition(
    item: Mapping[str, Any],
    *,
    visible: bool,
    limit: int,
    seed: int,
) -> dict[str, Any]:
    extra = dict(as_mapping(item.get("extra_info")))
    extra["budget_visible"] = visible
    extra["obs_tokens_limit"] = limit
    extra["sampling_seed"] = int(seed)
    out = dict(item)
    out["extra_info"] = extra
    return out


def extra_from_result(result, index: int) -> dict[str, Any]:
    extra_keys = result.non_tensor_batch
    fake: dict[str, Any] = {}
    for key in EXTRA_FIELD_KEYS:
        payload = extra_keys.get(key)
        if payload is not None:
            fake[key] = payload[index]
    return fake


def operational_record(
    task: Mapping[str, Any],
    *,
    visible: bool,
    error: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "bcrl-episode-v1",
        "trace_note": TRACE_NOTE,
        "identity": {
            "instance_id": task["instance_id"],
            "repo": task.get("repo"),
            "base_commit": None,
            "split": "dev",
        },
        "condition": {
            "budget_visible": visible,
            "obs_tokens_limit": PROVISIONAL_OBS_TOKENS_LIMIT,
            "sampling_seed": task.get("sampling_seed"),
        },
        "termination": "operational_error",
        "error": error,
        "provenance": dict(provenance),
    }


def heartbeat(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


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
    trace_dir = Path(data_root) / "trajectories" / "m3b" / args.experiment_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = trace_dir / "episodes.jsonl"
    heartbeat_path = output_dir / "heartbeat.json"
    status_path = output_dir / "run_status.json"

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else default_manifest_path(repo_root)
    )
    if not manifest_path.is_file():
        print(f"HARD FAIL: missing manifest {manifest_path}", file=sys.stderr)
        return 1
    manifest = load_manifest(manifest_path)
    agent_loop_config = repo_root / AGENT_LOOP_CONFIG_RELPATH
    if not agent_loop_config.is_file():
        print(f"HARD FAIL: missing {agent_loop_config}", file=sys.stderr)
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
    dev_path = args.dev.resolve() if args.dev is not None else dev_parquet_path(repo_root)
    if not dev_path.is_file():
        print(f"HARD FAIL: missing M1E dev parquet {dev_path}", file=sys.stderr)
        return 1
    cache_dir = output_dir / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(dev_path, tokenizer, cache_dir)
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
    provenance["manifest"] = {
        "path": str(manifest_path),
        "ordered_ids_sha256": manifest["ordered_ids_sha256"],
        "primary_ids_sha256": manifest["primary_ids_sha256"],
        "n_primary": manifest["n_primary"],
        "n_universe": manifest["n_universe"],
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["envelope"] = {
        "prompt_length": PROMPT_LENGTH,
        "response_length": RESPONSE_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "obs_tokens_limit": PROVISIONAL_OBS_TOKENS_LIMIT,
        "obs_tokens_limit_note": manifest["obs_tokens_limit_note"],
    }
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["gpu"] = gpu_info
    provenance["host"] = os.uname().nodename if hasattr(os, "uname") else ""
    write_json(output_dir / "provenance.json", provenance)

    completed = load_completed(episodes_path)
    primary_tasks = [task for task in manifest["tasks"] if task["set"] == "primary"]
    remainder_tasks = [task for task in manifest["tasks"] if task["set"] == "remainder"]
    if args.primary_n < len(primary_tasks):
        primary_tasks = primary_tasks[: args.primary_n]

    def pair_done(task: Mapping[str, Any]) -> bool:
        instance_id = task["instance_id"]
        return (instance_id, False) in completed and (instance_id, True) in completed

    pending_primary = [task for task in primary_tasks if not pair_done(task)]
    queue = list(pending_primary)
    remainder_enabled = not args.no_remainder
    remainder_started = False

    if args.probe_only:
        queue = queue[:1] or primary_tasks[:1]
        remainder_enabled = False

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
            batch_tasks = queue[: args.batch_size]
            del queue[: args.batch_size]
            items: list[dict[str, Any]] = []
            item_meta: list[dict[str, Any]] = []
            for task in batch_tasks:
                instance_id = task["instance_id"]
                if instance_id not in indexed:
                    for visible in (False, True):
                        if (instance_id, visible) in completed:
                            continue
                        record = operational_record(
                            task,
                            visible=visible,
                            error=f"instance not in dev parquet: {instance_id}",
                            provenance=provenance,
                        )
                        append_jsonl(episodes_path, [record])
                        completed.add((instance_id, visible))
                        n_error += 1
                    continue
                source = indexed[instance_id]
                extra = as_mapping(source.get("extra_info"))
                try:
                    env.prepare_from_extra_info(extra)
                except Exception as exc:
                    for visible in (False, True):
                        if (instance_id, visible) in completed:
                            continue
                        record = operational_record(
                            task,
                            visible=visible,
                            error=f"snapshot failed: {exc}",
                            provenance=provenance,
                        )
                        append_jsonl(episodes_path, [record])
                        completed.add((instance_id, visible))
                        n_error += 1
                    n_written += 2
                    continue
                seed = int(task["sampling_seed"])
                for visible in (False, True):
                    if (instance_id, visible) in completed:
                        continue
                    items.append(
                        with_condition(
                            source,
                            visible=visible,
                            limit=PROVISIONAL_OBS_TOKENS_LIMIT,
                            seed=seed,
                        )
                    )
                    item_meta.append(
                        {
                            "instance_id": instance_id,
                            "budget_visible": visible,
                            "sampling_seed": seed,
                        }
                    )
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
                rows.append(record)
                completed.add((meta["instance_id"], bool(meta["budget_visible"])))
            append_jsonl(episodes_path, rows)
            n_written += len(rows)
            batch_dt = time.time() - batch_t0
            if first_batch_seconds is None:
                first_batch_seconds = batch_dt
            batch_index += 1
            heartbeat(
                heartbeat_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "batch_index": batch_index,
                    "n_written": n_written,
                    "n_error": n_error,
                    "queue_remaining": len(queue),
                    "elapsed_s": time.time() - started,
                    "last_batch_s": batch_dt,
                    "remainder_started": remainder_started,
                },
            )
            if (
                remainder_enabled
                and not remainder_started
                and not queue
                and not args.probe_only
            ):
                pending_remainder = [
                    task for task in remainder_tasks if not pair_done(task)
                ]
                if pending_remainder and first_batch_seconds:
                    per_pair = first_batch_seconds / max(1, args.batch_size)
                    eta_h = (len(pending_remainder) * per_pair) / 3600.0
                    remaining_budget = args.max_hours - (time.time() - started) / 3600.0
                    if eta_h <= remaining_budget:
                        queue.extend(pending_remainder)
                        remainder_started = True
                    else:
                        stop_reason = "primary_only_eta"
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

    n_pairs = sum(
        1
        for task in manifest["tasks"]
        if (task["instance_id"], False) in completed
        and (task["instance_id"], True) in completed
    )
    status = {
        "status": "PASS" if stop_reason != "error" else "FAIL",
        "stop_reason": stop_reason,
        "experiment_id": args.experiment_id,
        "episodes_path": str(episodes_path),
        "n_written": n_written,
        "n_error": n_error,
        "n_completed_pairs": n_pairs,
        "n_primary_target": len(primary_tasks),
        "remainder_started": remainder_started,
        "elapsed_s": time.time() - started,
        "first_batch_s": first_batch_seconds,
        "sampling": QWEN3_SAMPLING,
        "validate": False,
        "obs_tokens_limit": PROVISIONAL_OBS_TOKENS_LIMIT,
    }
    write_json(status_path, status)
    print(json.dumps(status, indent=2))
    return 0 if stop_reason != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
