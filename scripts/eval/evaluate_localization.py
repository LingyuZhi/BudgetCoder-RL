#!/usr/bin/env python
"""Canonical held-out localization evaluation.

Evaluate a base policy and/or a LoRA checkpoint across frozen budgets.
Does not run GRPO.

Usage (compute node)::

    python scripts/eval/evaluate_localization.py --phase all
    python scripts/eval/evaluate_localization.py --dry-run
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

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    dev_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root, swe_gym_repos_root  # noqa: E402
from budget_coder_rl.train.gpu_runtime import is_login_host  # noqa: E402
from budget_coder_rl.eval.e018 import (  # noqa: E402
    AGENT_LOOP_CONFIG_RELPATH,
    CANONICAL_RL_STEP,
    EVAL_NAME,
    EXPERIMENT_ID,
    LORA_ALPHA,
    LORA_RANK,
    MILESTONE,
    N_GPUS,
    OUTPUT_ENV,
    TENSOR_MODEL_PARALLEL_SIZE,
    WANDB_EXPERIMENT_NAME,
    WANDB_PROJECT,
    actor_dir_errors,
    checkpoint_path_errors,
    consume_e018_overlay,
    default_e018_output_dir,
    default_rl_actor_dir,
    default_trace_dir,
    forbidden_output_dir_errors,
    jobs_for_phase,
    latest_iteration_errors,
    load_completed,
    overlay_lock_errors,
    resume_key,
    reuse_base_audit,
    shard_file_fingerprints,
    split_jobs_by_policy,
    treatment_integrity_errors,
)
from budget_coder_rl.eval.episode import build_episode_record  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4b import PINNED_VERL_COMMIT, write_json  # noqa: E402
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID, persist_lora_fingerprint  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    default_isolated_verl_root,
    default_verl_source,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    prepend_isolated_verl,
)
from budget_coder_rl.eval.m5b import redact_env  # noqa: E402
from budget_coder_rl.eval.m6 import (  # noqa: E402
    build_policy_extra_info,
    extra_info_leakage_errors,
    load_tasks,
)
from budget_coder_rl.eval.provenance import collect_run_provenance, sha256_file  # noqa: E402
from budget_coder_rl.ray_tmpdir import (  # noqa: E402
    cleanup_our_tmp_ray,
    ray_init_kwargs,
    short_temp_root,
)
from smoke_rlhf_dataset import build_dataset, resolve_tokenizer_path  # noqa: E402

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
    "vllm_lora_int_id",
    "vllm_lora_request_attached",
    "vllm_listed_lora_ids",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/localization.json",
        help="canonical evaluation config",
    )
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--phase", choices=("smoke", "base", "rl", "all"), default="all")
    parser.add_argument("--budget", type=int, action="append", default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-actor-dir", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=N_GPUS)
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=TENSOR_MODEL_PARALLEL_SIZE,
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--skip-gpu-idle", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reuse-base", action="store_true")
    parser.add_argument("--no-reuse-base", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and verify frozen eval contract; do not launch GPU eval.",
    )
    return parser.parse_args(argv)


def _expand_data_root(template: str, data_root: Path) -> Path:
    return Path(os.path.expandvars(str(template).replace("$BCRL_DATA_ROOT", str(data_root))))


def resolve_rl_actor_dir(
    *,
    canonical: Mapping[str, Any],
    data_root: Path,
    cli_path: Path | None,
) -> Path:
    if cli_path is not None:
        return cli_path.resolve()
    rl = canonical.get("rl_checkpoint") or {}
    actor_subdir = str(rl.get("actor_subdir") or "actor")
    candidates: list[Path] = []
    for key in ("path_template", "historical_path_template"):
        template = rl.get(key)
        if template:
            candidates.append(_expand_data_root(str(template), data_root) / actor_subdir)
    candidates.append(default_rl_actor_dir(data_root))
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=True) + "\n")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def index_dataset(dataset) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        extra = _as_dict(item.get("extra_info"))
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


def operational_record(job: Mapping[str, Any], *, error: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "bcrl-episode-v1",
        "trace_note": TRACE_NOTE,
        "experiment_id": EXPERIMENT_ID,
        "identity": {
            "instance_id": job["instance_id"],
            "repo": job.get("repo"),
            "base_commit": None,
            "split": "dev",
        },
        "condition": {
            "condition_id": job["condition_id"],
            "policy": job["policy"],
            "budget_visible": job["budget_visible"],
            "obs_tokens_limit": job["obs_tokens_limit"],
            "sampling_seed": job.get("sampling_seed"),
        },
        "termination": "operational_error",
        "error": error,
        "provenance": dict(provenance),
    }


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if is_login_host(host):
        raise SystemExit(f"HARD FAIL: do not run eval GPU on login node ({host})")
    try:
        import subprocess

        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
    except Exception as exc:
        raise SystemExit(f"HARD FAIL: nvidia-smi failed on {host}: {exc}") from exc
    return host


def _apply_eval_kv_budget(config: Any) -> None:
    from omegaconf import open_dict

    with open_dict(config):
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5


def _shutdown_ray() -> None:
    import ray

    if ray.is_initialized():
        ray.shutdown()
    time.sleep(5)


def _maybe_wandb(output_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"enabled": False}
    try:
        import wandb
    except Exception as exc:
        payload["error"] = str(exc)
        write_json(output_dir / "wandb_run.json", payload)
        return payload
    os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_EXPERIMENT_NAME,
        dir=str(output_dir / "wandb"),
        config={
            "experiment_id": EXPERIMENT_ID,
            "canonical_rl_step": CANONICAL_RL_STEP,
            "eval_only": True,
        },
        resume="allow",
    )
    payload = {
        "enabled": True,
        "id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
        "name": WANDB_EXPERIMENT_NAME,
        "project": WANDB_PROJECT,
    }
    write_json(output_dir / "wandb_run.json", payload)
    try:
        wandb.config.update(dict(provenance), allow_val_change=True)
    except Exception:
        pass
    return payload


def _read_generate_evidence(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "vllm_generate_evidence.jsonl"
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compact_adapter(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        "digest": payload.get("digest"),
        "n_adapter_tensors": payload.get("n_adapter_tensors") or payload.get("n_lora_tensors"),
        "n_lora_tensors": payload.get("n_lora_tensors") or payload.get("n_adapter_tensors"),
        "n_trainable": payload.get("n_trainable"),
        "peft_config_present": payload.get("peft_config_present"),
        "lora_b_max_abs": payload.get("lora_b_max_abs"),
        "adapter_nonzero": payload.get("adapter_nonzero"),
        "error": payload.get("error"),
    }


def init_e018_rl_manager(config, *, checkpoint_actor_dir: str, output_dir: Path):
    import ray

    from budget_coder_rl.train.eval_actor import E018ActorWorker
    from budget_coder_rl.train.eval_vllm_server import register_e018_vllm_replica
    from budget_coder_rl.train.m4b_trainer import _unwrap_snapshot
    from budget_coder_rl.train.gpu_runtime import query_vllm_lora_ids
    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import omega_conf_to_dataclass
    from verl.utils.device import get_device_name
    from verl.workers.rollout.llm_server import LLMServerManager

    errors = checkpoint_path_errors(checkpoint_actor_dir)
    errors.extend(actor_dir_errors(Path(checkpoint_actor_dir)))
    errors.extend(latest_iteration_errors(Path(checkpoint_actor_dir)))
    if errors:
        raise SystemExit(f"HARD FAIL: E018 checkpoint gate {errors}")

    register_e018_vllm_replica()
    assert config.actor_rollout_ref.rollout.mode == "async"
    global_pool_id = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        },
        mapping={Role.ActorRollout: global_pool_id},
    )
    resource_pool_manager.create_resource_pool()
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ray.remote(E018ActorWorker),
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    worker_dict_cls = create_colocated_worker_cls(
        class_dict={"actor_rollout": actor_rollout_cls}
    )
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool_manager.get_resource_pool(Role.ActorRollout),
        ray_cls_with_init=worker_dict_cls,
        device_name=get_device_name(),
    )
    actor_rollout_wg = wg_dict.spawn(prefix_set={"actor_rollout"})["actor_rollout"]
    actor_rollout_wg.init_model()
    shard_files = shard_file_fingerprints(Path(checkpoint_actor_dir))
    actor_rollout_wg.load_checkpoint(str(checkpoint_actor_dir))
    load_ok = True
    try:
        snapshot = persist_lora_fingerprint(_unwrap_snapshot(actor_rollout_wg.snapshot_trainable_params()))
    except Exception:
        snapshot = {"digest": "", "error": traceback.format_exc()}
    write_json(output_dir / "lora_load_fingerprint.json", snapshot)

    llm_server_manager = LLMServerManager.create(
        config=config, worker_group=actor_rollout_wg
    )
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=None,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()
    update_ok = True
    sync_payload = {}
    sync_path = output_dir / "vllm_sync_payload.json"
    rank0 = output_dir / "vllm_sync_payload_rank0.json"
    source = rank0 if rank0.is_file() else sync_path
    if source.is_file():
        text = source.read_text(encoding="utf-8")
        try:
            sync_payload = json.loads(text)
        except json.JSONDecodeError:
            sync_payload, _ = json.JSONDecoder().raw_decode(text)

    http_probe = query_vllm_lora_ids(llm_server_manager)
    listed: list[int] = []
    lora_as_adapter: list[Any] = []
    for replica in llm_server_manager.get_replicas():
        handle = getattr(replica, "_server_handle", None)
        if handle is None:
            lora_as_adapter.append(None)
            continue
        try:
            state = ray.get(handle.e018_engine_lora_state.remote())
            listed.extend(int(item) for item in (state.get("listed_lora_ids") or []))
            lora_as_adapter.append(bool(state.get("lora_as_adapter")))
        except Exception as exc:
            lora_as_adapter.append(f"error:{exc}")
    integrity = {
        "checkpoint_actor_dir": str(checkpoint_actor_dir),
        "load_ok": load_ok,
        "update_weights_ok": update_ok,
        "shard_files": shard_files,
        "load_fingerprint": _compact_adapter(snapshot),
        "sync_payload": _compact_adapter(sync_payload),
        "listed_lora_ids": sorted(set(listed)),
        "lora_as_adapter": all(item is True for item in lora_as_adapter if item is not None)
        and any(item is True for item in lora_as_adapter),
        "lora_as_adapter_raw": lora_as_adapter,
        "http_probe": http_probe,
        "http_saw_adapter": int(VLLM_LORA_INT_ID)
        in set(int(item) for item in (http_probe.get("lora_int_ids") or [])),
        "lora_request_attached": False,
        "used_output_difference_as_proof": False,
        "require_load_digest": True,
    }
    pre_errors = treatment_integrity_errors(integrity, require_generate=False)
    integrity["pre_generate_errors"] = pre_errors
    write_json(output_dir / "treatment_integrity.json", integrity)
    if pre_errors:
        raise SystemExit(
            "HARD FAIL: E018 treatment-integrity (pre-generate): " + "; ".join(pre_errors)
        )
    return {
        "agent_loop_manager": agent_loop_manager,
        "llm_server_manager": llm_server_manager,
        "actor_rollout_wg": actor_rollout_wg,
        "integrity": integrity,
    }


def _finalize_generate_integrity(output_dir: Path, integrity: dict[str, Any]) -> dict[str, Any]:
    rows = _read_generate_evidence(output_dir)
    attached = any(row.get("lora_request_attached") for row in rows)
    listed = []
    lora_int_id = None
    for row in rows:
        listed.extend(int(item) for item in (row.get("listed_lora_ids") or []))
        if row.get("lora_int_id") is not None:
            lora_int_id = int(row["lora_int_id"])
    integrity["generate_rows"] = len(rows)
    integrity["lora_request_attached"] = bool(attached)
    if listed:
        integrity["listed_lora_ids"] = sorted(set(listed) | set(integrity.get("listed_lora_ids") or []))
    if lora_int_id is not None:
        integrity["lora_int_id"] = lora_int_id
    errors = treatment_integrity_errors(integrity, require_generate=True)
    integrity["errors"] = errors
    integrity["pass"] = not errors
    write_json(output_dir / "treatment_integrity.json", integrity)
    if errors:
        raise SystemExit(
            "HARD FAIL: E018 treatment-integrity (generate): " + "; ".join(errors)
        )
    return integrity


def _run_session(
    *,
    session_name: str,
    jobs: Sequence[Mapping[str, Any]],
    use_lora: bool,
    checkpoint_actor_dir: Path | None,
    config_factory,
    runtime_env: dict[str, Any],
    tmp_root: Path,
    indexed: Mapping[str, Mapping[str, Any]],
    env: RepoEnvironment,
    episodes_path: Path,
    output_dir: Path,
    provenance: dict[str, Any],
    batch_size: int,
    max_hours: float,
    started: float,
    completed: set[tuple[str, str, int]],
) -> dict[str, Any]:
    from budget_coder_rl.train.gpu_runtime import (
        assert_sampling_config,
        build_batch,
        init_eval_agent_loop_manager,
        as_mapping,
    )

    import ray

    pending = [job for job in jobs if resume_key(job) not in completed]
    stats = {
        "session": session_name,
        "n_jobs": len(jobs),
        "n_pending": len(pending),
        "n_written": 0,
        "n_error": 0,
        "stop_reason": "completed",
        "treatment_integrity_pass": None,
    }
    if not pending:
        print(json.dumps({"session": session_name, "skipped": "already_complete"}))
        return stats

    ray.init(runtime_env=runtime_env, **ray_init_kwargs(tmp_root))
    integrity: dict[str, Any] | None = None
    try:
        from budget_coder_rl.train.gpu_runtime import get_verl_info

        provenance[f"verl_runtime_{session_name}"] = get_verl_info()
        write_json(output_dir / "provenance.json", provenance)
        config = config_factory()
        sampling_recorded = assert_sampling_config(config)
        provenance[f"sampling_rollout_{session_name}"] = sampling_recorded
        write_json(output_dir / "provenance.json", provenance)
        if use_lora:
            bundle = init_e018_rl_manager(
                config,
                checkpoint_actor_dir=str(checkpoint_actor_dir),
                output_dir=output_dir,
            )
            integrity = bundle["integrity"]
        else:
            bundle = init_eval_agent_loop_manager(
                config,
                checkpoint_actor_dir=None,
                require_lora_id=None,
            )
        manager = bundle["agent_loop_manager"]
        queue = list(pending)
        batch_index = 0
        while queue:
            elapsed_h = (time.time() - started) / 3600.0
            if elapsed_h >= max_hours:
                stats["stop_reason"] = "max_hours"
                break
            batch_jobs = queue[:batch_size]
            del queue[:batch_size]
            items: list[dict[str, Any]] = []
            item_meta: list[dict[str, Any]] = []
            for job in batch_jobs:
                key = resume_key(job)
                if key in completed:
                    continue
                instance_id = str(job["instance_id"])
                if instance_id not in indexed:
                    record = operational_record(
                        job,
                        error=f"instance not in parquet: {instance_id}",
                        provenance=provenance,
                    )
                    append_jsonl(episodes_path, [record])
                    completed.add(key)
                    stats["n_error"] += 1
                    stats["n_written"] += 1
                    continue
                source = dict(indexed[instance_id])
                extra = as_mapping(source.get("extra_info"))
                try:
                    env.prepare_from_extra_info(extra)
                    patched_extra = build_policy_extra_info(extra, job)
                    leaks = extra_info_leakage_errors(patched_extra)
                    if leaks:
                        raise ValueError(str(leaks))
                except Exception as exc:
                    record = operational_record(
                        job,
                        error=f"prepare/extra_info failed: {exc}",
                        provenance=provenance,
                    )
                    append_jsonl(episodes_path, [record])
                    completed.add(key)
                    stats["n_error"] += 1
                    stats["n_written"] += 1
                    continue
                patched = dict(source)
                patched["extra_info"] = patched_extra
                items.append(patched)
                item_meta.append(dict(job))
            if not items:
                continue
            batch_t0 = time.time()
            batch = build_batch(items, validate=False)
            result = manager.generate_sequences(prompts=batch)
            if use_lora and batch_index == 0:
                _finalize_generate_integrity(output_dir, integrity or {})
                stats["treatment_integrity_pass"] = True
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
                        f"(instance={meta['instance_id']})"
                    )
                if "do_sample" in sampling:
                    raise SystemExit("HARD FAIL: do_sample leaked into sampling_params")
                record = build_episode_record(extra, provenance=provenance)
                record["trace_note"] = TRACE_NOTE
                record["experiment_id"] = EXPERIMENT_ID
                record["eval_name"] = EVAL_NAME
                record["source_experiment"] = EXPERIMENT_ID
                record["condition"]["condition_id"] = meta["condition_id"]
                record["condition"]["policy"] = meta["policy"]
                record["condition"]["budget_visible"] = meta["budget_visible"]
                record["condition"]["obs_tokens_limit"] = meta["obs_tokens_limit"]
                record["condition"]["sampling_seed"] = meta["sampling_seed"]
                rows.append(record)
                completed.add(resume_key(meta))
            append_jsonl(episodes_path, rows)
            stats["n_written"] += len(rows)
            batch_dt = time.time() - batch_t0
            batch_index += 1
            write_json(
                output_dir / "heartbeat.json",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session": session_name,
                    "batch_index": batch_index,
                    "n_written_session": stats["n_written"],
                    "queue_remaining": len(queue),
                    "elapsed_s": time.time() - started,
                    "last_batch_s": batch_dt,
                },
            )
            print(
                json.dumps(
                    {
                        "session": session_name,
                        "batch_index": batch_index,
                        "wrote": len(rows),
                        "batch_s": round(batch_dt, 1),
                        "queue": len(queue),
                    }
                ),
                flush=True,
            )
    except Exception:
        stats["stop_reason"] = "error"
        write_json(
            output_dir / f"run_error_{session_name}.json",
            {
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    finally:
        _shutdown_ray()
    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    started = time.time()
    if args.experiment_id != EXPERIMENT_ID:
        print(f"HARD FAIL: experiment_id must be {EXPERIMENT_ID}", file=sys.stderr)
        return 1
    lock_msgs = overlay_lock_errors(repo_root)
    if lock_msgs:
        print(f"HARD FAIL: E018 overlay lock {lock_msgs}", file=sys.stderr)
        return 1
    consume_e018_overlay(repo_root=repo_root)
    canonical_path = args.config.resolve() if args.config.is_absolute() else (repo_root / args.config)
    if not canonical_path.is_file():
        print(f"HARD FAIL: missing eval config {canonical_path}", file=sys.stderr)
        return 1
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    if args.dry_run:
        print(f"config: {canonical_path}")
        print(f"n_tasks: {canonical.get('n_tasks')} budgets: {canonical.get('budgets')}")
        print(f"agent_loop: {canonical.get('agent_loop_config')}")
        print("dry-run: eval contract loaded; not launching GPU eval")
        return 0
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e018_output_dir(repo_root)
    )
    forbidden = forbidden_output_dir_errors(output_dir, repo_root)
    if forbidden:
        print(f"HARD FAIL: {forbidden}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[OUTPUT_ENV] = str(output_dir)
    os.environ["BCRL_M6_OUTPUT_DIR"] = str(output_dir)

    data_root = Path(args.data_root) if args.data_root is not None else bcrl_data_root()
    audit_path = output_dir / "reuse_audit.json"
    if args.reuse_base and args.no_reuse_base:
        print("HARD FAIL: --reuse-base and --no-reuse-base are mutually exclusive", file=sys.stderr)
        return 1
    if args.reuse_base:
        reuse_base = True
        audit = {"allow_reuse": True, "decision": "forced_reuse", "reasons": ["cli --reuse-base"]}
    elif args.no_reuse_base:
        reuse_base = False
        audit = {"allow_reuse": False, "decision": "forced_rerun", "reasons": ["cli --no-reuse-base"]}
    elif audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        reuse_base = bool(audit.get("allow_reuse"))
    else:
        audit = reuse_base_audit(repo_root, data_root=data_root)
        reuse_base = bool(audit.get("allow_reuse"))
    write_json(audit_path, audit)

    isolated_root = default_isolated_verl_root(data_root)
    verl_info = ensure_isolated_verl_checkout(
        isolated_root=isolated_root,
        source_git=default_verl_source(repo_root),
        pinned_commit=PINNED_VERL_COMMIT,
        create=True,
    )
    merged_pythonpath = prepend_isolated_verl(isolated_root, repo_root)
    verl_import_errors, verl_runtime = imported_verl_errors(isolated_root=isolated_root)
    if verl_import_errors:
        print(f"HARD FAIL: {verl_import_errors}", file=sys.stderr)
        return 1

    from budget_coder_rl.train.gpu_runtime import (  # noqa: E402
        M3C_AGENT_LOOP_CONFIG_RELPATH,
        apply_eval_lora_config,
        build_config,
        require_visible_gpus,
        resolve_model_path,
    )

    host = _assert_compute_node()
    if int(args.n_gpus) != N_GPUS:
        print(f"HARD FAIL: localization eval requires n_gpus={N_GPUS}", file=sys.stderr)
        return 1
    gpu_info = require_visible_gpus(N_GPUS, idle=not args.skip_gpu_idle)
    model_path = resolve_model_path(args.model_path)
    if not Path(model_path).exists():
        print(f"HARD FAIL: model path does not exist: {model_path}", file=sys.stderr)
        return 1

    agent_loop_config = repo_root / AGENT_LOOP_CONFIG_RELPATH
    if not agent_loop_config.is_file():
        print(f"HARD FAIL: missing {agent_loop_config}", file=sys.stderr)
        return 1
    if M3C_AGENT_LOOP_CONFIG_RELPATH not in str(agent_loop_config):
        print("HARD FAIL: eval must use configs/agent/repo_exploration.yaml", file=sys.stderr)
        return 1

    checkpoint_actor = resolve_rl_actor_dir(
        canonical=canonical,
        data_root=data_root,
        cli_path=args.checkpoint_actor_dir,
    )
    if "stage1_m5_scaled_e017" in str(checkpoint_actor):
        ckpt_errors = checkpoint_path_errors(checkpoint_actor)
        if ckpt_errors:
            print(f"HARD FAIL: {ckpt_errors}", file=sys.stderr)
            return 1
    elif "global_step_275" not in str(checkpoint_actor):
        print("HARD FAIL: RL checkpoint must be global_step_275", file=sys.stderr)
        return 1
    if args.phase in {"rl", "all", "smoke"} and not Path(checkpoint_actor).is_dir():
        print(f"HARD FAIL: missing RL actor dir {checkpoint_actor}", file=sys.stderr)
        return 1

    tasks = load_tasks(repo_root)
    jobs = jobs_for_phase(tasks, args.phase, reuse_base=reuse_base)
    if args.probe_only:
        jobs = jobs[:1]
    base_jobs, rl_jobs = split_jobs_by_policy(jobs)

    tmp_cleanup = cleanup_our_tmp_ray()
    tmp_root = short_temp_root()
    tokenizer_path = resolve_tokenizer_path(None) or model_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    parquet_path = args.dev.resolve() if args.dev is not None else dev_parquet_path(repo_root)
    if not parquet_path.is_file():
        print(f"HARD FAIL: missing M1E dev parquet {parquet_path}", file=sys.stderr)
        return 1
    cache_dir = output_dir / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(parquet_path, tokenizer, cache_dir)
    indexed = index_dataset(dataset)
    env = RepoEnvironment(
        repos_root=swe_gym_repos_root(args.data_root),
        data_root=args.data_root,
    )

    trace_dir = default_trace_dir(data_root)
    trace_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = trace_dir / "episodes.jsonl"
    completed = load_completed(episodes_path)

    provenance = collect_run_provenance(
        repo_root,
        verl_source=isolated_root,
        agent_loop_config=agent_loop_config,
        model_path=model_path,
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
    )
    provenance.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "eval_name": EVAL_NAME,
            "phase": args.phase,
            "reuse_base": reuse_base,
            "reuse_audit": audit,
            "canonical_rl_step": CANONICAL_RL_STEP,
            "split_kind": "held-out-task",
            "not_held_out_repository_test": True,
            "overlay_sha256": sha256_file(repo_root / "configs/historical/stage1_m6_e018.json"),
            "parent_eval_sha256": sha256_file(repo_root / "configs/historical/stage1_m6_eval.json"),
            "n_jobs": len(jobs),
            "n_base_jobs": len(base_jobs),
            "n_rl_jobs": len(rl_jobs),
            "n_already_completed": len(completed),
            "sampling_intended": dict(QWEN3_SAMPLING),
            "validate": False,
            "vllm_rollout_n": 1,
            "isolated_verl": verl_info,
            "verl_runtime": verl_runtime,
            "gpu": gpu_info,
            "host": host,
            "data_root": str(data_root),
            "checkpoint_actor_dir": str(checkpoint_actor),
            "episodes_path": str(episodes_path),
        }
    )
    write_json(output_dir / "provenance.json", provenance)
    _maybe_wandb(output_dir, provenance)

    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
            "BCRL_DATA_ROOT": str(data_root),
            OUTPUT_ENV: str(output_dir),
            "BCRL_M6_OUTPUT_DIR": str(output_dir),
            "PYTHONPATH": merged_pythonpath,
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
            "http_proxy": os.environ.get("http_proxy") or "",
            "https_proxy": os.environ.get("https_proxy") or "",
            "HTTPS_PROXY": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "",
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY") or "",
        }
    }
    write_json(output_dir / "runtime_env_redacted.json", redact_env(runtime_env["env_vars"]))

    def base_config():
        config = build_config(
            model_path,
            n_gpus=N_GPUS,
            tensor_model_parallel_size=int(args.tensor_model_parallel_size),
            agent_loop_config=str(agent_loop_config),
            sampling=QWEN3_SAMPLING,
            rollout_n=1,
        )
        _apply_eval_kv_budget(config)
        return config

    def rl_config():
        config = base_config()
        apply_eval_lora_config(config, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
        return config

    session_stats: list[dict[str, Any]] = []
    stop_reason = "completed"
    try:
        if base_jobs:
            session_stats.append(
                _run_session(
                    session_name="base",
                    jobs=base_jobs,
                    use_lora=False,
                    checkpoint_actor_dir=None,
                    config_factory=base_config,
                    runtime_env=runtime_env,
                    tmp_root=tmp_root,
                    indexed=indexed,
                    env=env,
                    episodes_path=episodes_path,
                    output_dir=output_dir,
                    provenance=provenance,
                    batch_size=int(args.batch_size),
                    max_hours=float(args.max_hours),
                    started=started,
                    completed=completed,
                )
            )
        if rl_jobs:
            if any(item.get("stop_reason") == "max_hours" for item in session_stats):
                stop_reason = "max_hours"
            else:
                session_stats.append(
                    _run_session(
                        session_name="rl",
                        jobs=rl_jobs,
                        use_lora=True,
                        checkpoint_actor_dir=checkpoint_actor,
                        config_factory=rl_config,
                        runtime_env=runtime_env,
                        tmp_root=tmp_root,
                        indexed=indexed,
                        env=env,
                        episodes_path=episodes_path,
                        output_dir=output_dir,
                        provenance=provenance,
                        batch_size=int(args.batch_size),
                        max_hours=float(args.max_hours),
                        started=started,
                        completed=completed,
                    )
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
        _shutdown_ray()

    if any(item.get("stop_reason") == "max_hours" for item in session_stats):
        stop_reason = "max_hours"
    if any(item.get("stop_reason") == "error" for item in session_stats):
        stop_reason = "error"

    n_written = sum(int(item.get("n_written") or 0) for item in session_stats)
    n_error = sum(int(item.get("n_error") or 0) for item in session_stats)
    status = {
        "status": "PASS" if stop_reason != "error" else "FAIL",
        "stop_reason": stop_reason,
        "experiment_id": EXPERIMENT_ID,
        "eval_name": EVAL_NAME,
        "phase": args.phase,
        "reuse_base": reuse_base,
        "episodes_path": str(episodes_path),
        "n_jobs": len(jobs),
        "n_written": n_written,
        "n_error": n_error,
        "n_completed": len(completed),
        "elapsed_s": time.time() - started,
        "sampling": QWEN3_SAMPLING,
        "validate": False,
        "canonical_rl_step": CANONICAL_RL_STEP,
        "sessions": session_stats,
        "split_kind": "held-out-task",
        "tmp_cleanup": tmp_cleanup,
    }
    write_json(output_dir / f"run_status_{args.phase}.json", status)
    write_json(output_dir / "run_status.json", status)
    print(json.dumps(json_safe(status), indent=2))
    return 0 if stop_reason != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
