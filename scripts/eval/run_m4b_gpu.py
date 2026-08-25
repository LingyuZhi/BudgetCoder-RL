#!/usr/bin/env python
"""M4B GPU smoke: one real GRPO optimizer step on the pinned RayPPOTrainer path.

Uses ``verl.trainer.main_ppo.run_ppo`` + LoRA. Does not retune reward/prompt/tools
and does not verify adapter reload (M4C).

Usage (compute node n30158, conda env ``verl``):

    python scripts/eval/run_m4b_gpu.py --experiment-id E003
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
from typing import Any

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
    apply_m4b_train_config,
    apply_reward_loop_config,
    assert_sampling_config,
    build_config,
    get_verl_info,
    pick_free_gpu,
    resolve_model_path,
)

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4a import REWARD_NUM_WORKERS  # noqa: E402
from budget_coder_rl.eval.m4b import (  # noqa: E402
    BUDGET_VISIBLE,
    EXPERIMENT_ID,
    GROUP_N,
    LORA_ALPHA,
    LORA_RANK,
    MILESTONE,
    N_TASKS,
    OBS_TOKENS_LIMIT,
    OUTPUT_ENV,
    REWARD_FN_RELPATH,
    VERL_PATH_TEXT,
    artifact_hashes,
    audit_verl_checkout,
    default_candidate_path,
    default_e007_groups_path,
    default_freeze_path,
    freeze_contract_errors,
    load_candidate_ordered_ids,
    load_e007_groups,
    load_json,
    select_smoke_instance_ids,
    write_json,
    write_smoke_parquet,
    write_summary,
)
from budget_coder_rl.eval.provenance import collect_run_provenance  # noqa: E402
from budget_coder_rl.ray_tmpdir import (  # noqa: E402
    cleanup_our_tmp_ray,
    ray_init_kwargs,
    short_temp_root,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--n-tasks", type=int, default=N_TASKS)
    parser.add_argument("--skip-gpu-pick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance-ids", default=None)
    return parser.parse_args(argv)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "outputs" / "experiments" / args.experiment_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[OUTPUT_ENV] = str(output_dir)
    for stale in (
        "group_evidence.json",
        "loss_mask_evidence.json",
        "lora_delta.json",
        "lora_before.json",
        "lora_after.json",
        "pre_update_status.json",
        "update_actor_error.json",
        "run_error.json",
        "run_status.json",
        "SUMMARY.md",
    ):
        path = output_dir / stale
        if path.is_file():
            path.unlink()

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
    smoke_parquet = output_dir / "train_smoke.parquet"
    for required in (parquet_path, oracle_path, reward_fn_path, agent_loop_config):
        if not Path(required).is_file():
            print(f"HARD FAIL: missing {required}", file=sys.stderr)
            return 1

    parquet_info = write_smoke_parquet(parquet_path, smoke_parquet, instance_ids)
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

    verl_audit = audit_verl_checkout(output_dir=output_dir, require_pin=True)
    provenance = collect_run_provenance(
        repo_root,
        agent_loop_config=agent_loop_config,
        model_path=model_path,
    )
    provenance["experiment_id"] = args.experiment_id
    provenance["milestone"] = MILESTONE
    provenance["optimizer"] = True
    provenance["lora_update"] = True
    provenance["adapter_reload"] = False
    provenance["instance_ids"] = instance_ids
    provenance["selection"] = {
        "universe": "m3c_train_candidates.ordered_ids",
        "filter": "E007 mixed=true, keep candidate order",
        "n_tasks": int(args.n_tasks),
        "gold_used_for_cherry_pick": False,
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["lora"] = {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "target_modules": "all-linear",
        "strategy": "fsdp",
    }
    provenance["envelope"] = {
        "prompt_length": PROMPT_LENGTH,
        "response_length": RESPONSE_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "obs_tokens_limit": OBS_TOKENS_LIMIT,
        "budget_visible": BUDGET_VISIBLE,
        "group_n": GROUP_N,
        "vllm_rollout_n": 1,
        "actor_rollout_ref.rollout.n": GROUP_N,
        "train_batch_size": int(args.n_tasks),
        "total_training_steps": 1,
    }
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "candidates": candidate_path,
            "e007_groups": e007_path,
            "oracle": oracle_path,
            "agent_loop_config": agent_loop_config,
            "reward_fn": reward_fn_path,
            "smoke_parquet": smoke_parquet,
        }
    )
    provenance["smoke_parquet"] = parquet_info
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["gpu"] = gpu_info
    provenance["host"] = os.uname().nodename if hasattr(os, "uname") else ""
    provenance["data_root"] = str(data_root)
    provenance["verl_audit"] = verl_audit
    write_json(output_dir / "provenance.json", provenance)
    (output_dir / "m4b_verl_path.md").write_text(
        VERL_PATH_TEXT.format(commit=verl_audit.get("commit") or "unknown"),
        encoding="utf-8",
    )

    python_path = os.environ.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    merged_pythonpath = src_path if not python_path else src_path + os.pathsep + python_path
    os.environ["PYTHONPATH"] = merged_pythonpath
    os.environ["BCRL_DATA_ROOT"] = str(data_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if host.lower().startswith("sn") or "login" in host.lower():
        print(f"HARD FAIL: do not run M4B GPU on login node ({host})", file=sys.stderr)
        return 1
    try:
        import subprocess

        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
    except Exception as exc:
        print(f"HARD FAIL: nvidia-smi failed on {host}: {exc}", file=sys.stderr)
        return 1
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_USE_V1": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "BCRL_DATA_ROOT": str(data_root),
            "BCRL_ORACLE_PARQUET": str(oracle_path),
            OUTPUT_ENV: str(output_dir),
            "PYTHONPATH": merged_pythonpath,
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
        }
    }

    import ray
    from verl.trainer.main_ppo import run_ppo

    from budget_coder_rl.train.m4b_trainer import M4BTaskRunner

    ray.init(runtime_env=runtime_env, **ray_init_kwargs(tmp_root))
    provenance["verl_runtime"] = get_verl_info()
    write_json(output_dir / "provenance.json", provenance)

    started = time.time()
    stop_reason = "completed"
    status = "FAIL"
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
        apply_m4b_train_config(
            config,
            train_files=str(smoke_parquet),
            val_files=str(smoke_parquet),
            n_tasks=int(args.n_tasks),
            n_gpus=int(args.n_gpus),
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            default_local_dir=str(output_dir / "checkpoints"),
        )
        sampling_recorded = assert_sampling_config(config, require_rollout_n=GROUP_N)
        provenance["sampling_rollout"] = sampling_recorded
        write_json(output_dir / "config_resolved.json", _safe_config(config))
        write_json(output_dir / "provenance.json", provenance)
        run_ppo(
            config,
            task_runner_class=ray.remote(num_cpus=1)(M4BTaskRunner),
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
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        if ray.is_initialized():
            ray.shutdown()

    elapsed = time.time() - started
    groups_payload = _load_optional_json(output_dir / "group_evidence.json")
    groups = list(groups_payload.get("groups") or [])
    learning = dict(groups_payload.get("learning") or {})
    loss_mask = _load_optional_json(output_dir / "loss_mask_evidence.json")
    lora_payload = _load_optional_json(output_dir / "lora_delta.json")
    grad = dict(lora_payload.get("grad") or {})
    pg_loss = dict(lora_payload.get("pg_loss") or {})
    lora = dict(lora_payload.get("lora") or {})
    gate = dict(lora_payload.get("gate") or {})
    if gate.get("pass"):
        status = "PASS"
        if stop_reason == "error":
            stop_reason = "optimizer_passed_poststep_error"
    else:
        status = "FAIL"
        if not gate:
            gate = {"pass": False, "reasons": ["optimizer evidence missing"]}
    write_summary(
        output_dir / "SUMMARY.md",
        status=status,
        groups=groups,
        learning=learning,
        loss_mask=loss_mask,
        grad=grad,
        pg_loss=pg_loss,
        lora=lora,
        gate=gate,
        verl_commit=str(verl_audit.get("commit") or "unknown"),
        instance_ids=instance_ids,
        elapsed_s=elapsed,
    )
    write_json(
        output_dir / "run_status.json",
        {
            "status": status,
            "stop_reason": stop_reason,
            "elapsed_s": elapsed,
            "gate": gate,
            "instance_ids": instance_ids,
        },
    )
    print(json.dumps({"status": status, "stop_reason": stop_reason, "elapsed_s": elapsed, "gate": gate}, indent=2))
    return 0 if status == "PASS" else 1


def _safe_config(config: Any) -> Any:
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_container(config, resolve=True)
    except Exception:
        return {"error": "config resolve failed"}


if __name__ == "__main__":
    raise SystemExit(main())
