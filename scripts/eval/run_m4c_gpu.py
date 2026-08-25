#!/usr/bin/env python
"""M4C GPU smoke: persist updated LoRA, fresh-reload, prove vLLM uses it.

Two Ray sessions on the pinned ``run_ppo`` path:

1. one-step GRPO + official FSDP ``_save_checkpoint``
2. ``resume_path`` reload + ``update_weights`` + one real AgentLoop

Does not retune reward/prompt/tools and does not require reward/text change.

Usage (compute node n30158, conda env ``verl``):

    python scripts/eval/run_m4c_gpu.py --experiment-id E009
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    apply_m4c_reload_config,
    apply_m4c_save_config,
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
    OUTPUT_ENV as M4B_OUTPUT_ENV,
    audit_verl_checkout,
    write_smoke_parquet,
)
from budget_coder_rl.eval.m4c import (  # noqa: E402
    BUDGET_VISIBLE,
    EXPERIMENT_ID,
    GROUP_N,
    LORA_ALPHA,
    LORA_RANK,
    MILESTONE,
    N_TASKS,
    OBS_TOKENS_LIMIT,
    OUTPUT_ENV,
    PHASE_ENV,
    RELOAD_N,
    REWARD_FN_RELPATH,
    VERL_PATH_TEXT,
    VLLM_LORA_INT_ID,
    artifact_hashes,
    build_checkpoint_manifest,
    checkpoint_integrity_errors,
    default_candidate_path,
    default_checkpoint_root,
    default_e007_groups_path,
    default_freeze_path,
    freeze_contract_errors,
    global_step_dir,
    load_candidate_ordered_ids,
    load_e007_groups,
    load_generate_evidence,
    load_json,
    m4c_gate,
    select_smoke_instance_ids,
    vllm_sync_errors,
    write_json,
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


def _safe_config(config: Any) -> Any:
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_container(config, resolve=True)
    except Exception:
        return {"error": "config resolve failed"}


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if host.lower().startswith("sn") or "login" in host.lower():
        raise SystemExit(f"HARD FAIL: do not run M4C GPU on login node ({host})")
    try:
        import subprocess

        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
    except Exception as exc:
        raise SystemExit(f"HARD FAIL: nvidia-smi failed on {host}: {exc}") from exc
    return host


def _run_phase(
    *,
    phase: str,
    config: Any,
    runtime_env: dict[str, Any],
    tmp_root: Path,
    output_dir: Path,
) -> None:
    import ray
    from verl.trainer.main_ppo import run_ppo

    from budget_coder_rl.train.m4c_trainer import M4CTaskRunner

    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars[PHASE_ENV] = phase
    os.environ[PHASE_ENV] = phase
    phase_runtime = dict(runtime_env)
    phase_runtime["env_vars"] = env_vars
    write_json(output_dir / f"config_resolved_{phase}.json", _safe_config(config))
    ray.init(runtime_env=phase_runtime, **ray_init_kwargs(tmp_root))
    try:
        run_ppo(
            config,
            task_runner_class=ray.remote(num_cpus=1)(M4CTaskRunner),
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()


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
    os.environ[M4B_OUTPUT_ENV] = str(output_dir)
    checkpoint_root = default_checkpoint_root(output_dir)
    if checkpoint_root.is_dir():
        shutil.rmtree(checkpoint_root)
    for stale in (
        "group_evidence.json",
        "loss_mask_evidence.json",
        "lora_delta.json",
        "lora_before.json",
        "lora_after.json",
        "lora_theta0.json",
        "lora_theta1.json",
        "lora_theta_reloaded.json",
        "lora_theta_compare_save.json",
        "lora_fingerprint_compare.json",
        "pre_update_status.json",
        "update_actor_error.json",
        "save_checkpoint_evidence.json",
        "checkpoint_manifest.json",
        "vllm_sync_payload.json",
        "vllm_sync_payload_save.json",
        "vllm_sync_payload_reload.json",
        "vllm_generate_evidence.jsonl",
        "reload_rollout_evidence.json",
        "episodes.jsonl",
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

    host = _assert_compute_node()
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
    provenance["adapter_reload"] = True
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
        "save_path": "RayPPOTrainer._save_checkpoint / FSDPCheckpointManager",
        "vllm_lora_int_id": VLLM_LORA_INT_ID,
    }
    provenance["envelope"] = {
        "prompt_length": PROMPT_LENGTH,
        "response_length": RESPONSE_LENGTH,
        "max_model_len": MAX_MODEL_LEN,
        "obs_tokens_limit": OBS_TOKENS_LIMIT,
        "budget_visible": BUDGET_VISIBLE,
        "group_n": GROUP_N,
        "vllm_rollout_n": 1,
        "actor_rollout_ref.rollout.n_save": GROUP_N,
        "actor_rollout_ref.rollout.n_reload": RELOAD_N,
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
    provenance["host"] = host
    provenance["data_root"] = str(data_root)
    provenance["verl_audit"] = verl_audit
    provenance["checkpoint_root"] = str(checkpoint_root)
    write_json(output_dir / "provenance.json", provenance)
    (output_dir / "m4c_verl_path.md").write_text(
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
            M4B_OUTPUT_ENV: str(output_dir),
            PHASE_ENV: "save",
            "PYTHONPATH": merged_pythonpath,
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
        }
    }

    started = time.time()
    stop_reason = "completed"
    status = "FAIL"
    gate: dict[str, Any] = {}
    try:
        provenance["verl_runtime"] = get_verl_info()
        write_json(output_dir / "provenance.json", provenance)

        save_config = build_config(
            model_path,
            n_gpus=args.n_gpus,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            agent_loop_config=str(agent_loop_config),
            rollout_n=GROUP_N,
        )
        apply_reward_loop_config(
            save_config,
            reward_fn_path=str(reward_fn_path),
            reward_fn_name="compute_score",
            num_workers=REWARD_NUM_WORKERS,
        )
        apply_m4c_save_config(
            save_config,
            train_files=str(smoke_parquet),
            val_files=str(smoke_parquet),
            n_tasks=int(args.n_tasks),
            n_gpus=int(args.n_gpus),
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            default_local_dir=str(checkpoint_root),
        )
        provenance["sampling_rollout_save"] = assert_sampling_config(
            save_config, require_rollout_n=GROUP_N
        )
        write_json(output_dir / "provenance.json", provenance)
        _run_phase(
            phase="save",
            config=save_config,
            runtime_env=runtime_env,
            tmp_root=tmp_root,
            output_dir=output_dir,
        )

        lora_payload = _load_optional_json(output_dir / "lora_delta.json")
        optimizer_gate = dict(lora_payload.get("gate") or {})
        if not optimizer_gate.get("pass"):
            raise RuntimeError(
                "HARD FAIL: phase-A optimizer gate failed: "
                + "; ".join(optimizer_gate.get("reasons") or ["missing lora_delta.json"])
            )
        ckpt_errors = checkpoint_integrity_errors(checkpoint_root, expected_step=1)
        manifest = build_checkpoint_manifest(
            checkpoint_root,
            expected_step=1,
            project_commit=str((provenance.get("budget_coder_rl") or {}).get("commit")),
            verl_commit=str(verl_audit.get("commit") or "unknown"),
            extra={"save_checkpoint_evidence": _load_optional_json(output_dir / "save_checkpoint_evidence.json")},
        )
        write_json(output_dir / "checkpoint_manifest.json", manifest)
        if ckpt_errors:
            raise RuntimeError("HARD FAIL: FSDP checkpoint incomplete: " + "; ".join(ckpt_errors))

        resume_from_path = global_step_dir(checkpoint_root, 1)
        reload_config = build_config(
            model_path,
            n_gpus=args.n_gpus,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            agent_loop_config=str(agent_loop_config),
            rollout_n=RELOAD_N,
        )
        apply_reward_loop_config(
            reload_config,
            reward_fn_path=str(reward_fn_path),
            reward_fn_name="compute_score",
            num_workers=REWARD_NUM_WORKERS,
        )
        apply_m4c_reload_config(
            reload_config,
            train_files=str(smoke_parquet),
            val_files=str(smoke_parquet),
            n_tasks=int(args.n_tasks),
            n_gpus=int(args.n_gpus),
            resume_from_path=str(resume_from_path),
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            default_local_dir=str(checkpoint_root),
        )
        provenance["sampling_rollout_reload"] = assert_sampling_config(
            reload_config, require_rollout_n=RELOAD_N
        )
        write_json(output_dir / "provenance.json", provenance)
        _run_phase(
            phase="reload",
            config=reload_config,
            runtime_env=runtime_env,
            tmp_root=tmp_root,
            output_dir=output_dir,
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

    elapsed = time.time() - started
    lora_payload = _load_optional_json(output_dir / "lora_delta.json")
    optimizer_gate = dict(lora_payload.get("gate") or {})
    optimizer_lora = dict(lora_payload.get("lora") or {})
    theta0 = _load_optional_json(output_dir / "lora_theta0.json")
    theta1 = _load_optional_json(output_dir / "lora_theta1.json")
    reloaded = _load_optional_json(output_dir / "lora_theta_reloaded.json")
    reload_rollout = _load_optional_json(output_dir / "reload_rollout_evidence.json")
    generate_rows = [
        row
        for row in load_generate_evidence(output_dir / "vllm_generate_evidence.jsonl")
        if str(row.get("phase") or "") == "reload"
    ]
    vllm_errors = vllm_sync_errors(
        payload=_load_optional_json(output_dir / "vllm_sync_payload_reload.json"),
        generate_rows=generate_rows,
        saved_payload=_load_optional_json(output_dir / "vllm_sync_payload_save.json"),
    )
    ckpt_errors = checkpoint_integrity_errors(checkpoint_root, expected_step=1)
    n_episodes = int(reload_rollout.get("n_episodes") or 0)
    if n_episodes <= 0 and (output_dir / "episodes.jsonl").is_file():
        n_episodes = sum(
            1 for line in (output_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        )
    gate = m4c_gate(
        optimizer_gate=optimizer_gate,
        theta0=theta0,
        theta1=theta1,
        reloaded=reloaded,
        checkpoint_errors=ckpt_errors,
        vllm_errors=vllm_errors,
        n_reload_episodes=n_episodes,
        reload_tito_errors=list(reload_rollout.get("tito_errors") or []),
    )
    status = "PASS" if gate.get("pass") else "FAIL"
    vllm_summary = {
        "lora_int_id": VLLM_LORA_INT_ID,
        "lora_request_attached": any(row.get("lora_request_attached") for row in generate_rows),
        "n_reload_generate_rows": len(generate_rows),
        "errors": vllm_errors,
    }
    write_json(output_dir / "vllm_adapter_evidence.json", vllm_summary)
    write_summary(
        output_dir / "SUMMARY.md",
        status=status,
        gate=gate,
        verl_commit=str(verl_audit.get("commit") or "unknown"),
        instance_ids=instance_ids,
        elapsed_s=elapsed,
        checkpoint_root=str(checkpoint_root),
        n_reload_episodes=n_episodes,
        optimizer=optimizer_lora,
        vllm=vllm_summary,
    )
    write_json(
        output_dir / "run_status.json",
        {
            "status": status,
            "stop_reason": stop_reason if status == "PASS" else (stop_reason or "gate_failed"),
            "elapsed_s": elapsed,
            "gate": gate,
            "instance_ids": instance_ids,
            "vllm": vllm_summary,
        },
    )
    print(
        json.dumps(
            {"status": status, "stop_reason": stop_reason, "elapsed_s": elapsed, "gate": gate},
            indent=2,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
