#!/usr/bin/env python
"""M7D factorial cells. Inference only. No GRPO / optimizer.step.

Cell A: M7C standalone eval AgentLoopManager, Base, G=1.
Cells B/C: E017 trainer construction (uid + _get_gen_batch + DataProto.repeat).
Cell D: same as C with fresh zero-init LoRA in a new Ray session.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT / "scripts" / "smoke") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke"))

from budget_coder_rl.data.swe_gym_materialize import train_parquet_path  # noqa: E402
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root, swe_gym_repos_root  # noqa: E402
from budget_coder_rl.env import RepoEnvironment  # noqa: E402
from budget_coder_rl.eval.e014 import is_login_host  # noqa: E402
from budget_coder_rl.eval.episode import build_episode_record  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4b import PINNED_VERL_COMMIT, write_json  # noqa: E402
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID, persist_lora_fingerprint  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    SHARED_VERL_ROOT,
    default_isolated_verl_root,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    prepend_isolated_verl,
)
from budget_coder_rl.eval.m5b import redact_env  # noqa: E402
from budget_coder_rl.eval.m6 import extra_info_leakage_errors  # noqa: E402
from budget_coder_rl.eval.m7c import AGENT_LOOP_CONFIG_RELPATH  # noqa: E402
from budget_coder_rl.eval.m7d import (  # noqa: E402
    CELL_SPECS,
    CELLS,
    E017_CHECKPOINT_MARKER,
    EXPERIMENT_ID,
    GROUP_N,
    LORA_ALPHA,
    LORA_RANK,
    MILESTONE,
    N_GPUS,
    N_SUBSET,
    OBS_TOKENS_LIMIT,
    SEED_POLICY,
    TENSOR_MODEL_PARALLEL_SIZE,
    VALIDATE,
    VLLM_ROLLOUT_N,
    build_first_request_record,
    build_unseeded_extra_info,
    canonicalize_sampling_params,
    default_m7d_output_dir,
    default_trace_dir,
    first_generation_from_episode,
    forbidden_output_dir_errors,
    lora_runtime_metadata,
    sampling_contract,
    subset_tasks,
    trajectory_info_from_index,
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
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=N_GPUS)
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=TENSOR_MODEL_PARALLEL_SIZE,
    )
    parser.add_argument("--logical-batch-size", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--n-subset", type=int, default=N_SUBSET)
    parser.add_argument("--cells", default="A,B,C,D")
    parser.add_argument("--skip-gpu-idle", action="store_true")
    parser.add_argument("--skip-audit-gate", action="store_true")
    return parser.parse_args(argv)


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


def resume_key(job: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(job["cell"]), str(job["instance_id"]), int(job["sibling_index"]))


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            m7d = row.get("m7d") or {}
            identity = row.get("identity") or {}
            cell = str(m7d.get("cell") or "")
            instance_id = str(identity.get("instance_id") or "")
            sibling = m7d.get("sibling_index")
            if cell and instance_id and sibling is not None:
                done.add((cell, instance_id, int(sibling)))
    return done


def operational_record(job: Mapping[str, Any], *, error: str, provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "bcrl-episode-v1",
        "trace_note": TRACE_NOTE,
        "experiment_id": EXPERIMENT_ID,
        "identity": {
            "instance_id": job["instance_id"],
            "repo": job.get("repo"),
            "base_commit": None,
            "split": "train",
        },
        "m7d": {
            "cell": job["cell"],
            "sibling_index": job.get("sibling_index", 0),
            "uid": job.get("uid"),
            "path": CELL_SPECS[str(job["cell"])]["path"],
        },
        "condition": {
            "budget_visible": True,
            "obs_tokens_limit": OBS_TOKENS_LIMIT,
            "sampling_seed": None,
        },
        "termination": "operational_error",
        "error": error,
        "provenance": dict(provenance),
    }


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if is_login_host(host):
        raise SystemExit(f"HARD FAIL: do not run M7D GPU on login node ({host})")
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
        config.actor_rollout_ref.rollout.n = 1
        config.trainer.val_before_train = False


def _shutdown_ray() -> None:
    import ray

    if ray.is_initialized():
        ray.shutdown()
    time.sleep(5)


def expand_like_e017(logical, group_n: int):
    """E017 fit() construction: uid + _get_gen_batch + DataProto.repeat."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    n = len(logical)
    logical.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(n)], dtype=object
    )
    gen_batch = RayPPOTrainer._get_gen_batch(None, logical)
    gen_batch.meta_info["global_steps"] = 0
    gen_batch.meta_info["validate"] = False
    expanded = gen_batch.repeat(repeat_times=int(group_n), interleave=True)
    if len(expanded) != n * int(group_n):
        raise SystemExit(
            f"HARD FAIL: DataProto.repeat produced {len(expanded)} rows, expected {n * int(group_n)}"
        )
    return expanded


def jobs_from_subset(subset: Mapping[str, Any], cells: Sequence[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cell in cells:
        spec = CELL_SPECS[cell]
        for task in list(subset.get("train_tasks") or []):
            jobs.append(
                {
                    "cell": cell,
                    "instance_id": task["instance_id"],
                    "repo": task.get("repo"),
                    "split": "train",
                    "subset_index": int(task["subset_index"]),
                    "group_n": int(spec["group_n"]),
                    "path": spec["path"],
                    "lora": spec["lora"],
                }
            )
    return jobs


def _unwrap_snapshot(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        if not result:
            return {}
        first = result[0]
        return dict(first) if isinstance(first, Mapping) else {}
    if isinstance(result, Mapping):
        return dict(result)
    return {}


def init_session(
    *,
    config,
    with_lora: bool,
    output_dir: Path,
) -> dict[str, Any]:
    from gpu_runtime import init_eval_agent_loop_manager, query_vllm_lora_ids

    if with_lora:
        bundle = init_eval_agent_loop_manager(
            config,
            checkpoint_actor_dir=None,
            require_lora_id=int(VLLM_LORA_INT_ID),
        )
    else:
        bundle = init_eval_agent_loop_manager(
            config,
            checkpoint_actor_dir=None,
            require_lora_id=None,
        )
    probe = dict(bundle.get("lora_probe") or {})
    listed = list(probe.get("lora_int_ids") or [])
    fingerprint = None
    if with_lora:
        actor = bundle.get("actor_rollout_wg")
        try:
            fingerprint = persist_lora_fingerprint(
                _unwrap_snapshot(actor.snapshot_trainable_params())
            )
            write_json(output_dir / "lora_fresh_fingerprint.json", fingerprint)
        except Exception as exc:
            fingerprint = {"error": f"{type(exc).__name__}: {exc}"}
            write_json(output_dir / "lora_fresh_fingerprint.json", fingerprint)
        lora_b = 0.0
        for name, info in dict((fingerprint or {}).get("lora") or {}).items():
            if "lora_b" in str(name).lower():
                lora_b = max(lora_b, float((info or {}).get("max_abs") or 0.0))
        fingerprint["lora_b_max_abs"] = lora_b
    else:
        live = query_vllm_lora_ids(bundle["llm_server_manager"])
        listed = list(live.get("lora_int_ids") or listed)
        if listed:
            raise SystemExit(f"HARD FAIL: Base session listed residual LoRA ids {listed}")
    bundle["listed_lora_ids"] = listed
    bundle["lora_fingerprint"] = fingerprint
    bundle["with_lora"] = with_lora
    return bundle


def run_logical_batch(
    *,
    cell: str,
    jobs: list[dict[str, Any]],
    indexed: Mapping[str, Mapping[str, Any]],
    env: RepoEnvironment,
    manager,
    tokenizer,
    provenance: Mapping[str, Any],
    lora_meta: Mapping[str, Any],
    episodes_path: Path,
    request_path: Path,
    first_gen_path: Path,
    completed: set[tuple[str, str, int]],
) -> dict[str, int]:
    from gpu_runtime import as_mapping, build_batch

    spec = CELL_SPECS[cell]
    group_n = int(spec["group_n"])
    stats = {"n_written": 0, "n_error": 0}
    items: list[dict[str, Any]] = []
    logical_meta: list[dict[str, Any]] = []
    for job in jobs:
        instance_id = str(job["instance_id"])
        if instance_id not in indexed:
            record = operational_record(
                {**job, "sibling_index": 0},
                error=f"instance not in train parquet: {instance_id}",
                provenance=provenance,
            )
            append_jsonl(episodes_path, [record])
            completed.add(resume_key({**job, "sibling_index": 0}))
            stats["n_error"] += 1
            stats["n_written"] += 1
            continue
        source = dict(indexed[instance_id])
        extra = as_mapping(source.get("extra_info"))
        try:
            env.prepare_from_extra_info(extra)
            patched_extra = build_unseeded_extra_info(extra)
            leaks = extra_info_leakage_errors(patched_extra)
            if leaks:
                raise ValueError(str(leaks))
        except Exception as exc:
            record = operational_record(
                {**job, "sibling_index": 0},
                error=f"prepare/extra_info failed: {exc}",
                provenance=provenance,
            )
            append_jsonl(episodes_path, [record])
            completed.add(resume_key({**job, "sibling_index": 0}))
            stats["n_error"] += 1
            stats["n_written"] += 1
            continue
        patched = dict(source)
        patched["extra_info"] = patched_extra
        if patched.get("raw_prompt") is None:
            patched["raw_prompt"] = patched.get("prompt")
        items.append(patched)
        logical_meta.append(dict(job))
    if not items:
        return stats
    logical = build_batch(items, validate=False)
    if spec["path"] == "trainer_rollout":
        batch = expand_like_e017(logical, group_n)
    else:
        batch = logical
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [None for _ in range(len(batch))], dtype=object
            )
    index_payload = batch.non_tensor_batch.get("index")
    indices = list(index_payload) if index_payload is not None else []
    traj = trajectory_info_from_index(indices, validate=False)
    uid_payload = batch.non_tensor_batch.get("uid")
    uids = [
        None if item is None else str(item)
        for item in (list(uid_payload) if uid_payload is not None else [])
    ]
    sampling = canonicalize_sampling_params(sampling_contract(group_n=group_n, lora=spec["lora"]))
    request_rows: list[dict[str, Any]] = []
    row_meta: list[dict[str, Any]] = []
    for expanded_index in range(len(batch)):
        logical_index = expanded_index // group_n if spec["path"] == "trainer_rollout" else expanded_index
        sibling_index = int(traj[expanded_index]["rollout_n"]) if traj else 0
        job = logical_meta[logical_index]
        meta = {
            **job,
            "sibling_index": sibling_index,
            "uid": uids[expanded_index] if expanded_index < len(uids) else None,
        }
        if resume_key(meta) in completed:
            continue
        extra = as_mapping(batch.non_tensor_batch["extra_info"][expanded_index])
        kwargs = {
            "raw_prompt": batch.non_tensor_batch["raw_prompt"][expanded_index],
            "extra_info": extra,
            "agent_name": "repo_exploration",
        }
        record = build_first_request_record(
            cell=cell,
            logical_task_index=int(job["subset_index"]),
            sibling_index=sibling_index,
            instance_id=str(job["instance_id"]),
            uid=meta["uid"],
            dataset_index=extra.get("index"),
            extra_info=extra,
            kwargs=kwargs,
            tokenizer=tokenizer,
            sampling_params=sampling,
            lora_meta=lora_meta,
            request_kwargs={
                "validate": False,
                "effective_n": 1,
                "trainer_repeat": spec["path"] == "trainer_rollout",
                "group_n": group_n,
            },
        )
        record["source"] = "gpu"
        request_rows.append(record)
        row_meta.append({"expanded_index": expanded_index, **meta})
    append_jsonl(request_path, request_rows)
    result = manager.generate_sequences(prompts=batch)
    if len(result) != len(batch):
        raise SystemExit(
            f"HARD FAIL: generate_sequences returned {len(result)} for {len(batch)} inputs"
        )
    episode_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    for meta in row_meta:
        index = int(meta["expanded_index"])
        extra = extra_from_result(result, index)
        sampling_out = extra.get("sampling_params") or {}
        temperature = sampling_out.get("temperature")
        if temperature in {0, 0.0}:
            raise SystemExit(
                "HARD FAIL: episode sampling temperature is 0 "
                f"(cell={cell} instance={meta['instance_id']})"
            )
        if "do_sample" in sampling_out:
            raise SystemExit("HARD FAIL: do_sample leaked into sampling_params")
        extra.setdefault("split", "train")
        extra.setdefault("instance_id", meta["instance_id"])
        extra.setdefault("obs_tokens_limit", OBS_TOKENS_LIMIT)
        extra.setdefault("budget_visible", True)
        extra["sampling_seed"] = extra.get("sampling_seed")
        record = build_episode_record(extra, sampling=sampling_out, provenance=provenance)
        record["trace_note"] = TRACE_NOTE
        record["experiment_id"] = EXPERIMENT_ID
        record["identity"]["split"] = "train"
        record["condition"]["sampling_seed"] = None
        record["m7d"] = {
            "cell": cell,
            "sibling_index": meta["sibling_index"],
            "uid": meta.get("uid"),
            "logical_task_index": meta["subset_index"],
            "path": spec["path"],
            "group_n": group_n,
            "lora": spec["lora"],
            "seed_policy": SEED_POLICY,
        }
        episode_rows.append(record)
        first = first_generation_from_episode(record)
        first["source"] = "gpu"
        first_rows.append(first)
        completed.add(resume_key(meta))
    append_jsonl(episodes_path, episode_rows)
    append_jsonl(first_gen_path, first_rows)
    stats["n_written"] += len(episode_rows)
    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root) if args.data_root else bcrl_data_root()
    output_dir = Path(args.output_dir) if args.output_dir else default_m7d_output_dir(repo_root)
    blocked = forbidden_output_dir_errors(output_dir, repo_root)
    if blocked:
        print(f"HARD FAIL: {blocked}", file=sys.stderr)
        return 1
    if E017_CHECKPOINT_MARKER in str(data_root / "checkpoints"):
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit_gate:
        gate_path = output_dir / "audit_gate.json"
        if not gate_path.is_file():
            print(f"HARD FAIL: missing first-request gate {gate_path}", file=sys.stderr)
            return 1
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("allow_gpu"):
            print(
                f"HARD FAIL: first-request audit forbade GPU: {gate}",
                file=sys.stderr,
            )
            return 1

    isolated_root = default_isolated_verl_root(data_root)
    verl_info = ensure_isolated_verl_checkout(
        isolated_root=isolated_root,
        source_git=SHARED_VERL_ROOT,
        pinned_commit=PINNED_VERL_COMMIT,
        create=True,
    )
    merged_pythonpath = prepend_isolated_verl(isolated_root, repo_root)
    verl_import_errors, verl_runtime = imported_verl_errors(isolated_root=isolated_root)
    if verl_import_errors:
        print(f"HARD FAIL: {verl_import_errors}", file=sys.stderr)
        return 1

    from gpu_runtime import (  # noqa: E402
        M3C_AGENT_LOOP_CONFIG_RELPATH,
        apply_eval_lora_config,
        assert_sampling_config,
        build_config,
        require_visible_gpus,
        resolve_model_path,
    )

    host = _assert_compute_node()
    if int(args.n_gpus) != N_GPUS:
        print(f"HARD FAIL: M7D requires n_gpus={N_GPUS}", file=sys.stderr)
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
        print("HARD FAIL: M7D must use the M3C AgentLoop YAML", file=sys.stderr)
        return 1

    requested_cells = [item.strip() for item in str(args.cells).split(",") if item.strip()]
    for cell in requested_cells:
        if cell not in CELLS:
            print(f"HARD FAIL: unknown cell {cell}", file=sys.stderr)
            return 1
    subset = subset_tasks(repo_root=repo_root, n=int(args.n_subset))
    write_json(output_dir / "subset_manifest.json", subset)
    jobs = jobs_from_subset(subset, requested_cells)

    tmp_cleanup = cleanup_our_tmp_ray()
    tmp_root = short_temp_root()
    tokenizer_path = resolve_tokenizer_path(None) or model_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cache_dir = output_dir / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(train_parquet_path(repo_root), tokenizer, cache_dir / "train")
    indexed = index_dataset(train_ds)
    env = RepoEnvironment(
        repos_root=swe_gym_repos_root(args.data_root),
        data_root=args.data_root,
    )
    trace_dir = default_trace_dir(data_root)
    trace_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = trace_dir / "episodes.jsonl"
    request_path = output_dir / "first_request_audit_gpu.jsonl"
    first_gen_path = output_dir / "first_generation_outputs.jsonl"
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
            "eval_only": True,
            "not_training": True,
            "no_optimizer_step": True,
            "seed_policy": SEED_POLICY,
            "n_jobs": len(jobs),
            "n_already_completed": len(completed),
            "sampling_intended": dict(QWEN3_SAMPLING),
            "validate": VALIDATE,
            "vllm_rollout_n": VLLM_ROLLOUT_N,
            "isolated_verl": verl_info,
            "verl_runtime": verl_runtime,
            "gpu": gpu_info,
            "host": host,
            "data_root": str(data_root),
            "episodes_path": str(episodes_path),
            "config_sha256": sha256_file(repo_root / "configs/experiments/stage1_m7d.json"),
            "tmp_cleanup": tmp_cleanup,
        }
    )
    write_json(output_dir / "provenance.json", provenance)
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
            "BCRL_DATA_ROOT": str(data_root),
            "PYTHONPATH": merged_pythonpath,
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
            "http_proxy": os.environ.get("http_proxy") or "",
            "https_proxy": os.environ.get("https_proxy") or "",
            "HTTPS_PROXY": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "",
        }
    }
    write_json(output_dir / "runtime_env_redacted.json", redact_env(runtime_env["env_vars"]))

    import ray
    from gpu_runtime import get_verl_info

    stats = {
        "n_jobs": len(jobs),
        "n_written": 0,
        "n_error": 0,
        "stop_reason": "completed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cells": requested_cells,
    }

    def run_session(cells: Sequence[str], *, with_lora: bool) -> None:
        nonlocal stats
        session_jobs = [job for job in jobs if job["cell"] in set(cells)]
        pending_logical = []
        for job in session_jobs:
            spec = CELL_SPECS[job["cell"]]
            needed = [
                resume_key({**job, "sibling_index": sibling})
                for sibling in range(int(spec["group_n"]))
            ]
            if all(key in completed for key in needed):
                continue
            pending_logical.append(job)
        if not pending_logical:
            return
        ray.init(runtime_env=runtime_env, **ray_init_kwargs(tmp_root))
        try:
            provenance["verl_runtime_replay"] = get_verl_info()
            write_json(output_dir / "provenance.json", provenance)
            config = build_config(
                model_path,
                n_gpus=N_GPUS,
                tensor_model_parallel_size=int(args.tensor_model_parallel_size),
                agent_loop_config=str(agent_loop_config),
                sampling=QWEN3_SAMPLING,
                rollout_n=1,
            )
            _apply_eval_kv_budget(config)
            if with_lora:
                apply_eval_lora_config(config, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA)
            sampling_recorded = assert_sampling_config(config, require_rollout_n=1)
            provenance[f"sampling_rollout_{'+'.join(cells)}"] = sampling_recorded
            write_json(output_dir / "provenance.json", provenance)
            bundle = init_session(config=config, with_lora=with_lora, output_dir=output_dir)
            manager = bundle["agent_loop_manager"]
            fingerprint = bundle.get("lora_fingerprint") or {}
            lora_meta = lora_runtime_metadata(
                cell=cells[0],
                attached=with_lora,
                lora_int_id=VLLM_LORA_INT_ID if with_lora else None,
                listed_ids=bundle.get("listed_lora_ids") or [],
                checkpoint_path=None,
                adapter_digest=(fingerprint or {}).get("digest"),
                lora_b_max_abs=(fingerprint or {}).get("lora_b_max_abs"),
            )
            if not lora_meta.get("ok"):
                raise SystemExit(f"HARD FAIL: LoRA metadata {lora_meta.get('errors')}")
            write_json(output_dir / f"lora_meta_{'_'.join(cells)}.json", lora_meta)
            by_cell: dict[str, list[dict[str, Any]]] = {}
            for job in pending_logical:
                by_cell.setdefault(job["cell"], []).append(job)
            for cell, cell_jobs in by_cell.items():
                queue = list(cell_jobs)
                while queue:
                    elapsed_h = (time.time() - started) / 3600.0
                    if elapsed_h >= float(args.max_hours):
                        stats["stop_reason"] = "max_hours"
                        return
                    batch_jobs = queue[: int(args.logical_batch_size)]
                    del queue[: int(args.logical_batch_size)]
                    cell_lora_meta = lora_runtime_metadata(
                        cell=cell,
                        attached=with_lora,
                        lora_int_id=VLLM_LORA_INT_ID if with_lora else None,
                        listed_ids=bundle.get("listed_lora_ids") or [],
                        checkpoint_path=None,
                        adapter_digest=(fingerprint or {}).get("digest"),
                        lora_b_max_abs=(fingerprint or {}).get("lora_b_max_abs"),
                    )
                    batch_stats = run_logical_batch(
                        cell=cell,
                        jobs=batch_jobs,
                        indexed=indexed,
                        env=env,
                        manager=manager,
                        tokenizer=tokenizer,
                        provenance=provenance,
                        lora_meta=cell_lora_meta,
                        episodes_path=episodes_path,
                        request_path=request_path,
                        first_gen_path=first_gen_path,
                        completed=completed,
                    )
                    stats["n_written"] += batch_stats["n_written"]
                    stats["n_error"] += batch_stats["n_error"]
                    print(
                        json.dumps(
                            {
                                "cell": cell,
                                "written": stats["n_written"],
                                "pending_logical": len(queue),
                                "last": [item["instance_id"] for item in batch_jobs],
                            }
                        ),
                        flush=True,
                    )
        finally:
            _shutdown_ray()

    base_cells = [cell for cell in requested_cells if CELL_SPECS[cell]["lora"] is None]
    lora_cells = [cell for cell in requested_cells if CELL_SPECS[cell]["lora"] is not None]
    if base_cells:
        run_session(base_cells, with_lora=False)
    if lora_cells and stats["stop_reason"] == "completed":
        run_session(lora_cells, with_lora=True)

    elapsed = time.time() - started
    status = "PASS" if stats["stop_reason"] == "completed" and stats["n_error"] == 0 else "INCOMPLETE"
    payload = {
        **stats,
        "status": status,
        "elapsed_s": elapsed,
        "episodes_path": str(episodes_path),
        "n_completed": len(completed),
        "optimizer_step": False,
    }
    write_json(output_dir / "run_status.json", payload)
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
