#!/usr/bin/env python
"""M5A GPU rehearsal: 4 GRPO steps on frozen main-run knobs.

Uses isolated pinned veRL + stock RayPPOTrainer. Pilot weights are disposable.

Usage (compute node n30158, conda env ``verl``):

    python scripts/eval/run_m5a_pilot.py --experiment-id E010 --skip-gpu-pick
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

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4a import REWARD_NUM_WORKERS, artifact_hashes  # noqa: E402
from budget_coder_rl.eval.m4b import write_json, write_smoke_parquet  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    EXPERIMENT_ID,
    GROUP_N,
    LORA_ALPHA,
    LORA_RANK,
    MILESTONE,
    OUTPUT_ENV,
    PILOT_STEPS,
    REWARD_FN_RELPATH,
    SEED,
    SHARED_VERL_ROOT,
    TRAIN_BATCH_SIZE,
    checkpoint_dir_manifest,
    default_isolated_verl_root,
    default_main_config_path,
    default_output_dir,
    default_pilot_config_path,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    load_json,
    m5_freeze_consume_errors,
    m5a_gate,
    prepend_isolated_verl,
    select_prefix_instance_ids,
    summarize_metrics_jsonl,
)
from budget_coder_rl.eval.m4a import (  # noqa: E402
    default_candidate_path,
    default_freeze_path,
    load_candidate_ordered_ids,
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
    parser.add_argument("--n-tasks", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--n-steps", type=int, default=PILOT_STEPS)
    parser.add_argument("--skip-gpu-pick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance-ids", default=None)
    return parser.parse_args(argv)


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if host.lower().startswith("sn") or "login" in host.lower():
        raise SystemExit(f"HARD FAIL: do not run M5A GPU on login node ({host})")
    try:
        import subprocess

        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
    except Exception as exc:
        raise SystemExit(f"HARD FAIL: nvidia-smi failed on {host}: {exc}") from exc
    return host


def _safe_config(config: Any) -> Any:
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_container(config, resolve=True)
    except Exception:
        return {"error": "config resolve failed"}


def write_summary(
    path: Path,
    *,
    status: str,
    evidence: dict[str, Any],
    gate: dict[str, Any],
    elapsed_s: float,
) -> None:
    lines = [
        "# M5A / E010 Main-run preflight rehearsal",
        "",
        f"- status: **{status}**",
        f"- READY_FOR_M5B: **{gate.get('READY_FOR_M5B')}**",
        f"- elapsed_s: {float(elapsed_s):.1f}",
        f"- veRL: `{evidence.get('verl_commit')}` isolated={evidence.get('verl_isolated_clean')}",
        f"- ppo_max_token_len_per_gpu: {evidence.get('ppo_max_token_len_per_gpu')}",
        f"- steps: {evidence.get('n_steps_completed')} nonzero_adv={evidence.get('n_steps_nonzero_advantage')}",
        f"- wandb: {evidence.get('wandb_logged')} metrics_jsonl={evidence.get('metrics_jsonl_present')}",
        f"- gate reasons: {gate.get('reasons') or ['(none)']}",
        "",
        "Pilot weights are disposable and are not M6 candidates.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(repo_root, args.experiment_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[OUTPUT_ENV] = str(output_dir)
    if not os.environ.get("WANDB_API_KEY"):
        print("HARD FAIL: WANDB_API_KEY is not set", file=sys.stderr)
        return 1

    freeze_path = default_freeze_path(repo_root)
    candidate_path = default_candidate_path(repo_root)
    freeze = load_json(freeze_path)
    freeze_errors = m5_freeze_consume_errors(
        freeze, freeze_path=freeze_path, candidate_path=candidate_path
    )
    if freeze_errors:
        print(f"HARD FAIL: freeze contract {freeze_errors}", file=sys.stderr)
        return 1

    main_path = default_main_config_path(repo_root)
    if not main_path.is_file():
        print(f"HARD FAIL: missing {main_path}; run preflight first", file=sys.stderr)
        return 1
    main_cfg = load_json(main_path)
    newly = main_cfg.get("newly_frozen") or {}
    actor = newly.get("actor") or {}
    ppo_max = int(actor.get("ppo_max_token_len_per_gpu") or 0)
    if ppo_max <= 0:
        print("HARD FAIL: main config missing ppo_max_token_len_per_gpu", file=sys.stderr)
        return 1

    data_root = args.data_root or bcrl_data_root()
    isolated_root = default_isolated_verl_root(Path(data_root))
    verl_info = ensure_isolated_verl_checkout(
        isolated_root=isolated_root,
        source_git=SHARED_VERL_ROOT,
        create=True,
    )
    merged_pythonpath = prepend_isolated_verl(isolated_root, repo_root)
    verl_import_errors, verl_runtime = imported_verl_errors(isolated_root=isolated_root)
    if verl_import_errors:
        print(f"HARD FAIL: {verl_import_errors}", file=sys.stderr)
        write_json(output_dir / "verl_import_error.json", {"errors": verl_import_errors, "info": verl_runtime})
        return 1

    from gpu_runtime import (  # noqa: E402
        M3C_AGENT_LOOP_CONFIG_RELPATH,
        apply_m5_train_config,
        apply_reward_loop_config,
        assert_sampling_config,
        build_config,
        get_verl_info,
        pick_free_gpu,
        resolve_model_path,
    )

    ordered_ids = load_candidate_ordered_ids(candidate_path)
    if args.instance_ids:
        instance_ids = [item.strip() for item in args.instance_ids.split(",") if item.strip()]
    else:
        instance_ids = select_prefix_instance_ids(
            ordered_ids, n_tasks=int(args.n_tasks), n_steps=int(args.n_steps)
        )
    parquet_path = (
        args.parquet.resolve() if args.parquet is not None else train_parquet_path(repo_root)
    )
    oracle_path = oracle_parquet_path(repo_root).resolve()
    agent_loop_config = repo_root / M3C_AGENT_LOOP_CONFIG_RELPATH
    reward_fn_path = repo_root / REWARD_FN_RELPATH
    smoke_parquet = output_dir / "train_pilot.parquet"
    for required in (parquet_path, oracle_path, reward_fn_path, agent_loop_config):
        if not Path(required).is_file():
            print(f"HARD FAIL: missing {required}", file=sys.stderr)
            return 1
    parquet_info = write_smoke_parquet(parquet_path, smoke_parquet, instance_ids)
    os.environ["BCRL_ORACLE_PARQUET"] = str(oracle_path)
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

    checkpoint_root = output_dir / "checkpoints"
    provenance = collect_run_provenance(
        repo_root,
        verl_source=isolated_root,
        agent_loop_config=agent_loop_config,
        model_path=model_path,
    )
    provenance["experiment_id"] = args.experiment_id
    provenance["milestone"] = MILESTONE
    provenance["phase"] = "pilot"
    provenance["instance_ids"] = instance_ids
    provenance["selection"] = {
        "universe": "m3c_train_candidates.ordered_ids",
        "filter": "prefix; no mixed cherry-pick",
        "n_tasks": int(args.n_tasks),
        "n_steps": int(args.n_steps),
        "gold_used_for_cherry_pick": False,
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["isolated_verl"] = verl_info
    provenance["verl_runtime"] = verl_runtime
    provenance["gpu"] = gpu_info
    provenance["host"] = host
    provenance["data_root"] = str(data_root)
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "candidates": candidate_path,
            "main_config": main_path,
            "pilot_config": default_pilot_config_path(repo_root),
            "oracle": oracle_path,
            "agent_loop_config": agent_loop_config,
            "reward_fn": reward_fn_path,
            "smoke_parquet": smoke_parquet,
        }
    )
    provenance["smoke_parquet"] = parquet_info
    write_json(output_dir / "provenance.json", provenance)

    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
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
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_PROJECT": "budget-coder-rl",
            "https_proxy": proxy or "",
            "http_proxy": os.environ.get("http_proxy") or "",
            "HTTPS_PROXY": proxy or "",
        }
    }

    import ray
    from verl.trainer.main_ppo import run_ppo

    from budget_coder_rl.train.m5_trainer import M5TaskRunner

    ray.init(runtime_env=runtime_env, **ray_init_kwargs(tmp_root))
    provenance["verl_runtime_after_init"] = get_verl_info()
    write_json(output_dir / "provenance.json", provenance)

    started = time.time()
    stop_reason = "completed"
    oom = False
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
        apply_m5_train_config(
            config,
            train_files=str(smoke_parquet),
            val_files=str(smoke_parquet),
            n_tasks=int(args.n_tasks),
            n_gpus=int(args.n_gpus),
            ppo_max_token_len_per_gpu=ppo_max,
            total_training_steps=int(args.n_steps),
            experiment_name=f"{args.experiment_id}-m5a-pilot",
            default_local_dir=str(checkpoint_root),
            save_freq=1,
            max_actor_ckpt_to_keep=1,
            resume_mode="disable",
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            actor_lr=float((actor.get("optim_lr") if actor else None) or 1e-6),
            seed=SEED,
            calculate_entropy=True,
            wandb=True,
            wandb_proxy=proxy,
        )
        sampling_recorded = assert_sampling_config(config, require_rollout_n=GROUP_N)
        provenance["sampling_rollout"] = sampling_recorded
        write_json(output_dir / "config_resolved.json", _safe_config(config))
        write_json(output_dir / "provenance.json", provenance)
        run_ppo(
            config,
            task_runner_class=ray.remote(num_cpus=1)(M5TaskRunner),
        )
    except Exception as exc:
        stop_reason = "error"
        message = f"{type(exc).__name__}: {exc}"
        if "out of memory" in message.lower() or "oom" in message.lower():
            oom = True
            stop_reason = "oom"
        write_json(
            output_dir / "run_error.json",
            {
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "oom": oom,
            },
        )
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        if ray.is_initialized():
            ray.shutdown()

    elapsed = time.time() - started
    metrics_summary = summarize_metrics_jsonl(output_dir / "metrics.jsonl")
    wandb_run = {}
    wandb_path = output_dir / "wandb_run.json"
    if wandb_path.is_file():
        wandb_run = load_json(wandb_path)
    ckpt = checkpoint_dir_manifest(checkpoint_root)
    n_steps = int(metrics_summary.get("n_steps") or 0)
    n_nonzero = int(metrics_summary.get("n_steps_nonzero_advantage") or 0)
    evidence = {
        "verl_isolated_clean": not verl_import_errors and not verl_info.get("dirty"),
        "verl_commit": verl_runtime.get("commit") or verl_info.get("commit"),
        "m3c_freeze_ok": not freeze_errors,
        "seqlen_characterized": True,
        "ppo_max_token_len_frozen": True,
        "ppo_max_token_len_per_gpu": ppo_max,
        "assert_risk_if_8192": ppo_max > 8192,
        "pilot_completed": stop_reason == "completed" and n_steps >= 2,
        "pilot_oom": oom,
        "n_steps_completed": n_steps,
        "n_steps_nonzero_advantage": n_nonzero,
        "metrics_jsonl_present": (output_dir / "metrics.jsonl").is_file(),
        "wandb_logged": bool(wandb_run.get("id")),
        "main_config_written": main_path.is_file(),
        "checkpoint_policy_frozen": True,
        "checkpoint": ckpt,
        "wandb_run": wandb_run,
        "metrics_summary": metrics_summary,
        "stop_reason": stop_reason,
    }
    gate = m5a_gate(evidence)
    status = "PASS" if gate.get("pass") else "FAIL"
    write_json(output_dir / "READY_FOR_M5B.json", {"READY_FOR_M5B": gate.get("READY_FOR_M5B"), **gate, **evidence})
    write_json(output_dir / "run_status.json", {"status": status, "elapsed_s": elapsed, "gate": gate, **evidence})
    write_summary(
        output_dir / "SUMMARY.md",
        status=status,
        evidence=evidence,
        gate=gate,
        elapsed_s=elapsed,
    )
    print(
        json.dumps(
            {
                "status": status,
                "READY_FOR_M5B": gate.get("READY_FOR_M5B"),
                "stop_reason": stop_reason,
                "elapsed_s": elapsed,
                "gate": gate,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
