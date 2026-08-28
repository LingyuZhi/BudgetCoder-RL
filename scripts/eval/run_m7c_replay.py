#!/usr/bin/env python
"""M7C matched Base-policy train-vs-dev replay. Inference only. No GRPO.

Requires CPU prompt-path audit allow_replay=true. Does not load LoRA.
Does not write into E017/E018 artifact directories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root, swe_gym_repos_root  # noqa: E402
from budget_coder_rl.env import RepoEnvironment  # noqa: E402
from budget_coder_rl.eval.e014 import is_login_host  # noqa: E402
from budget_coder_rl.eval.episode import build_episode_record  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4b import PINNED_VERL_COMMIT, write_json  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    SHARED_VERL_ROOT,
    default_isolated_verl_root,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    prepend_isolated_verl,
)
from budget_coder_rl.eval.m5b import redact_env  # noqa: E402
from budget_coder_rl.eval.m6 import extra_info_leakage_errors  # noqa: E402
from budget_coder_rl.eval.m7c import (  # noqa: E402
    AGENT_LOOP_CONFIG_RELPATH,
    EXPERIMENT_ID,
    MILESTONE,
    N_GPUS,
    N_SUBSET,
    OBS_TOKENS_LIMIT,
    TENSOR_MODEL_PARALLEL_SIZE,
    VALIDATE,
    VLLM_ROLLOUT_N,
    build_matched_extra_info,
    default_m7c_output_dir,
    default_trace_dir,
    forbidden_output_dir_errors,
    subset_tasks,
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--n-subset", type=int, default=N_SUBSET)
    parser.add_argument("--probe-only", action="store_true")
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


def resume_key(job: Mapping[str, Any]) -> tuple[str, str]:
    return (str(job["instance_id"]), str(job["split"]))


def load_completed(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            identity = row.get("identity") or {}
            instance_id = str(identity.get("instance_id") or "")
            split = str(identity.get("split") or "")
            if instance_id and split:
                done.add((instance_id, split))
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
            "split": job["split"],
        },
        "condition": {
            "budget_visible": True,
            "obs_tokens_limit": OBS_TOKENS_LIMIT,
            "sampling_seed": job.get("sampling_seed"),
        },
        "termination": "operational_error",
        "error": error,
        "provenance": dict(provenance),
    }


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if is_login_host(host):
        raise SystemExit(f"HARD FAIL: do not run M7C GPU on login node ({host})")
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


def jobs_from_subset(subset: Mapping[str, Any], *, probe_only: bool) -> list[dict[str, Any]]:
    jobs = []
    for task in list(subset.get("train_tasks") or []) + list(subset.get("dev_tasks") or []):
        jobs.append(
            {
                "instance_id": task["instance_id"],
                "repo": task.get("repo"),
                "split": task["split"],
                "subset_index": task["subset_index"],
                "sampling_seed": int(task["sampling_seed"]),
                "obs_tokens_limit": OBS_TOKENS_LIMIT,
                "budget_visible": True,
            }
        )
    if probe_only:
        return jobs[:2]
    return jobs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    repo_root = Path(args.repo_root)
    data_root = Path(args.data_root) if args.data_root else bcrl_data_root()
    output_dir = Path(args.output_dir) if args.output_dir else default_m7c_output_dir(repo_root)
    blocked = forbidden_output_dir_errors(output_dir, repo_root)
    if blocked:
        print(f"HARD FAIL: {blocked}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audit_gate:
        gate_path = output_dir / "audit_gate.json"
        if not gate_path.is_file():
            print(f"HARD FAIL: missing prompt-path gate {gate_path}", file=sys.stderr)
            return 1
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("allow_replay"):
            print(
                f"HARD FAIL: prompt-path audit forbade replay: {gate.get('confound_reasons')}",
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
        assert_sampling_config,
        build_batch,
        build_config,
        init_eval_agent_loop_manager,
        require_visible_gpus,
        resolve_model_path,
        as_mapping,
    )

    host = _assert_compute_node()
    if int(args.n_gpus) != N_GPUS:
        print(f"HARD FAIL: M7C requires n_gpus={N_GPUS}", file=sys.stderr)
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
        print("HARD FAIL: M7C must use the M3C AgentLoop YAML", file=sys.stderr)
        return 1

    subset = subset_tasks(repo_root=repo_root, n=int(args.n_subset))
    write_json(output_dir / "subset_manifest.json", subset)
    jobs = jobs_from_subset(subset, probe_only=bool(args.probe_only))

    tmp_cleanup = cleanup_our_tmp_ray()
    tmp_root = short_temp_root()
    tokenizer_path = resolve_tokenizer_path(None) or model_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cache_dir = output_dir / "rlhf_dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_ds = build_dataset(train_parquet_path(repo_root), tokenizer, cache_dir / "train")
    dev_ds = build_dataset(dev_parquet_path(repo_root), tokenizer, cache_dir / "dev")
    indexed = {
        "train": index_dataset(train_ds),
        "dev": index_dataset(dev_ds),
    }
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
            "eval_only": True,
            "lora": None,
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
            "config_sha256": sha256_file(repo_root / "configs/experiments/stage1_m7c.json"),
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

    pending = [job for job in jobs if resume_key(job) not in completed]
    stats = {
        "n_jobs": len(jobs),
        "n_pending": len(pending),
        "n_written": 0,
        "n_error": 0,
        "stop_reason": "completed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not pending:
        write_json(output_dir / "run_status.json", {**stats, "status": "PASS", "skipped": "already_complete"})
        print(json.dumps({"skipped": "already_complete", "episodes": str(episodes_path)}))
        return 0

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
        sampling_recorded = assert_sampling_config(config, require_rollout_n=1)
        provenance["sampling_rollout"] = sampling_recorded
        write_json(output_dir / "provenance.json", provenance)
        bundle = init_eval_agent_loop_manager(
            config,
            checkpoint_actor_dir=None,
            require_lora_id=None,
        )
        manager = bundle["agent_loop_manager"]
        queue = list(pending)
        while queue:
            elapsed_h = (time.time() - started) / 3600.0
            if elapsed_h >= float(args.max_hours):
                stats["stop_reason"] = "max_hours"
                break
            batch_jobs = queue[: int(args.batch_size)]
            del queue[: int(args.batch_size)]
            items: list[dict[str, Any]] = []
            item_meta: list[dict[str, Any]] = []
            for job in batch_jobs:
                key = resume_key(job)
                if key in completed:
                    continue
                split = str(job["split"])
                instance_id = str(job["instance_id"])
                table = indexed[split]
                if instance_id not in table:
                    record = operational_record(
                        job,
                        error=f"instance not in {split} parquet: {instance_id}",
                        provenance=provenance,
                    )
                    append_jsonl(episodes_path, [record])
                    completed.add(key)
                    stats["n_error"] += 1
                    stats["n_written"] += 1
                    continue
                source = dict(table[instance_id])
                extra = as_mapping(source.get("extra_info"))
                try:
                    env.prepare_from_extra_info(extra)
                    patched_extra = build_matched_extra_info(
                        extra,
                        sampling_seed=int(job["sampling_seed"]),
                    )
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
                        f"(instance={meta['instance_id']})"
                    )
                if "do_sample" in sampling:
                    raise SystemExit("HARD FAIL: do_sample leaked into sampling_params")
                extra.setdefault("split", meta["split"])
                extra.setdefault("instance_id", meta["instance_id"])
                extra.setdefault("sampling_seed", meta["sampling_seed"])
                extra.setdefault("obs_tokens_limit", OBS_TOKENS_LIMIT)
                extra.setdefault("budget_visible", True)
                record = build_episode_record(extra, sampling=sampling, provenance=provenance)
                record["trace_note"] = TRACE_NOTE
                record["experiment_id"] = EXPERIMENT_ID
                record["identity"]["split"] = meta["split"]
                record["condition"]["sampling_seed"] = meta["sampling_seed"]
                rows.append(record)
                completed.add(resume_key(meta))
            append_jsonl(episodes_path, rows)
            stats["n_written"] += len(rows)
            print(
                json.dumps(
                    {
                        "written": stats["n_written"],
                        "pending": len(queue),
                        "last": [item["instance_id"] for item in item_meta],
                    }
                ),
                flush=True,
            )
    finally:
        _shutdown_ray()

    elapsed = time.time() - started
    status = "PASS" if stats["stop_reason"] == "completed" and stats["n_error"] == 0 else "INCOMPLETE"
    payload = {
        **stats,
        "status": status,
        "elapsed_s": elapsed,
        "episodes_path": str(episodes_path),
        "n_completed": len(completed),
    }
    write_json(output_dir / "run_status.json", payload)
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
