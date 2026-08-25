#!/usr/bin/env python
"""M5B / E011 canonical Stage 1 GRPO: 256 tasks, 32 steps, 2xA100.

Consumes frozen stage1_m5_main.json plus the E011 systems overlay.
Does not retune reward / prompt / tools / parser / budget / sampling.

Usage (compute node, conda env ``verl``):

    python scripts/eval/run_m5b_main.py --experiment-id E011
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

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4a import REWARD_NUM_WORKERS, artifact_hashes  # noqa: E402
from budget_coder_rl.eval.m4b import write_json, write_smoke_parquet  # noqa: E402
from budget_coder_rl.eval.m5a import (  # noqa: E402
    GROUP_N,
    LORA_ALPHA,
    LORA_RANK,
    OUTPUT_ENV,
    REWARD_FN_RELPATH,
    SEED,
    SHARED_VERL_ROOT,
    TRAIN_BATCH_SIZE,
    checkpoint_dir_manifest,
    default_isolated_verl_root,
    default_main_config_path,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    load_json,
    m5_freeze_consume_errors,
    prepend_isolated_verl,
    summarize_metrics_jsonl,
)
from budget_coder_rl.eval.m4a import (  # noqa: E402
    default_candidate_path,
    default_freeze_path,
    load_candidate_ordered_ids,
)
from budget_coder_rl.eval.m5b import (  # noqa: E402
    CANONICAL_CHECKPOINT_STEP,
    EXPERIMENT_ID,
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_TP,
    HARD_STOP_ENV,
    MILESTONE,
    PLACEMENT_ENV,
    HardStopError,
    build_training_summary,
    checkpoint_dir_conflict_errors,
    classify_hard_stop_from_text,
    consume_runtime_overlay,
    default_checkpoint_dir,
    default_e011_output_dir,
    disk_capacity_errors,
    expected_hybrid_placement,
    is_login_host,
    project_tree_dirty_errors,
    redact_env,
    research_knob_errors,
    resource_lifecycle,
    selected_m6_candidate,
)
from budget_coder_rl.eval.provenance import collect_run_provenance, sha256_file  # noqa: E402
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-resume", action="store_true")
    return parser.parse_args(argv)


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if is_login_host(host):
        raise SystemExit(f"HARD FAIL: do not run M5B GPU on login node ({host})")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e011_output_dir(repo_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[OUTPUT_ENV] = str(output_dir)
    os.environ[HARD_STOP_ENV] = "1"
    os.environ[PLACEMENT_ENV] = "1"
    os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
    os.environ.setdefault("WANDB_PROJECT", "budget-coder-rl")

    if not args.allow_dirty:
        dirty = project_tree_dirty_errors(repo_root)
        if dirty:
            print(f"HARD FAIL: {dirty}", file=sys.stderr)
            return 1
    if not os.environ.get("WANDB_API_KEY"):
        print("HARD FAIL: WANDB_API_KEY is not set", file=sys.stderr)
        return 1

    freeze_path = default_freeze_path(repo_root)
    candidate_path = default_candidate_path(repo_root)
    main_path = default_main_config_path(repo_root)
    freeze = load_json(freeze_path)
    freeze_errors = m5_freeze_consume_errors(
        freeze, freeze_path=freeze_path, candidate_path=candidate_path
    )
    if freeze_errors:
        print(f"HARD FAIL: freeze contract {freeze_errors}", file=sys.stderr)
        return 1
    if sha256_file(main_path) != EXPECTED_MAIN_SHA256:
        print(
            f"HARD FAIL: stage1_m5_main.json was edited ({sha256_file(main_path)})",
            file=sys.stderr,
        )
        return 1
    main_cfg = load_json(main_path)
    knob_errors = research_knob_errors(main_cfg)
    if knob_errors:
        print(f"HARD FAIL: research knobs {knob_errors}", file=sys.stderr)
        return 1
    try:
        runtime = consume_runtime_overlay(repo_root=repo_root)
    except HardStopError as exc:
        print(f"HARD FAIL: {exc.reason} {exc.details}", file=sys.stderr)
        return 1

    n_gpus = int(runtime["n_gpus"])
    tp = int(runtime["tensor_model_parallel_size"])
    if n_gpus != EXPECTED_N_GPUS or tp != EXPECTED_TP:
        print(
            f"HARD FAIL: overlay placement n_gpus={n_gpus} tp={tp} "
            f"(need {EXPECTED_N_GPUS} / {EXPECTED_TP})",
            file=sys.stderr,
        )
        return 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])

    newly = main_cfg.get("newly_frozen") or {}
    actor = newly.get("actor") or {}
    trainer_cfg = newly.get("trainer") or {}
    ppo_max = int(actor.get("ppo_max_token_len_per_gpu") or 0)
    n_steps = int(trainer_cfg.get("total_training_steps") or 0)
    save_freq = int(trainer_cfg.get("save_freq") or 0)
    keep = int(trainer_cfg.get("max_actor_ckpt_to_keep") or 2)
    resume_mode = str(trainer_cfg.get("resume_mode") or "auto")
    experiment_name = str(trainer_cfg.get("experiment_name") or "E011-m5-main")
    if ppo_max <= 0 or n_steps != 32:
        print("HARD FAIL: frozen trainer steps/token cap missing", file=sys.stderr)
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
        require_visible_gpus,
        resolve_model_path,
    )

    instance_ids = load_candidate_ordered_ids(candidate_path)
    if len(instance_ids) != 256:
        print(f"HARD FAIL: expected 256 candidates, got {len(instance_ids)}", file=sys.stderr)
        return 1
    parquet_path = (
        args.parquet.resolve() if args.parquet is not None else train_parquet_path(repo_root)
    )
    oracle_path = oracle_parquet_path(repo_root).resolve()
    agent_loop_config = repo_root / M3C_AGENT_LOOP_CONFIG_RELPATH
    reward_fn_path = repo_root / REWARD_FN_RELPATH
    train_parquet = output_dir / "train_e011.parquet"
    for required in (parquet_path, oracle_path, reward_fn_path, agent_loop_config):
        if not Path(required).is_file():
            print(f"HARD FAIL: missing {required}", file=sys.stderr)
            return 1
    parquet_info = write_smoke_parquet(parquet_path, train_parquet, instance_ids)
    os.environ["BCRL_ORACLE_PARQUET"] = str(oracle_path)
    tmp_cleanup = cleanup_our_tmp_ray()
    tmp_root = short_temp_root()
    gpu_info = require_visible_gpus(n_gpus, idle=True)
    model_path = resolve_model_path(args.model_path)
    if not Path(model_path).exists():
        print(f"HARD FAIL: model path does not exist: {model_path}", file=sys.stderr)
        return 1
    host = _assert_compute_node()
    lifecycle = resource_lifecycle(host=host)

    checkpoint_root = default_checkpoint_dir(Path(data_root))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_iter = checkpoint_root / "latest_checkpointed_iteration.txt"
    official_resume = resume_mode == "auto" and latest_iter.is_file()
    ckpt_conflicts = checkpoint_dir_conflict_errors(
        checkpoint_root, allow_resume=bool(args.allow_resume) or official_resume
    )
    if ckpt_conflicts:
        print(f"HARD FAIL: {ckpt_conflicts}", file=sys.stderr)
        return 1
    disk_errors = disk_capacity_errors(checkpoint_root, output_dir)
    if disk_errors:
        print(f"HARD FAIL: {disk_errors}", file=sys.stderr)
        return 1

    provenance = collect_run_provenance(
        repo_root,
        verl_source=isolated_root,
        agent_loop_config=agent_loop_config,
        model_path=model_path,
    )
    provenance["experiment_id"] = args.experiment_id
    provenance["milestone"] = MILESTONE
    provenance["phase"] = "main"
    provenance["instance_ids"] = instance_ids
    provenance["selection"] = {
        "universe": "m3c_train_candidates.ordered_ids",
        "filter": "full frozen pool; no cherry-pick; shuffle=false; one pass",
        "n_tasks": len(instance_ids),
        "n_steps": n_steps,
        "gold_used_for_cherry_pick": False,
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["isolated_verl"] = verl_info
    provenance["verl_runtime"] = verl_runtime
    provenance["gpu"] = gpu_info
    provenance["host"] = host
    provenance["lifecycle"] = lifecycle
    provenance["runtime_overlay"] = runtime
    provenance["data_root"] = str(data_root)
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["checkpoint_root"] = str(checkpoint_root)
    provenance["checkpoint_selection"] = {
        "rule": runtime["checkpoint_selection_rule"],
        "canonical_global_step": CANONICAL_CHECKPOINT_STEP,
        "post_hoc_curve_pick": False,
    }
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "candidates": candidate_path,
            "main_config": main_path,
            "overlay": repo_root / "configs/experiments/stage1_m5b_e011_runtime.json",
            "oracle": oracle_path,
            "agent_loop_config": agent_loop_config,
            "reward_fn": reward_fn_path,
            "train_parquet": train_parquet,
        }
    )
    provenance["train_parquet"] = parquet_info
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
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
            "BCRL_DATA_ROOT": str(data_root),
            "BCRL_ORACLE_PARQUET": str(oracle_path),
            OUTPUT_ENV: str(output_dir),
            HARD_STOP_ENV: "1",
            PLACEMENT_ENV: "1",
            "PYTHONPATH": merged_pythonpath,
            "TMPDIR": str(tmp_root),
            "RAY_TMPDIR": str(tmp_root),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_PROJECT": "budget-coder-rl",
            "WANDB_DIR": str(output_dir / "wandb"),
            "https_proxy": proxy or "",
            "http_proxy": os.environ.get("http_proxy") or "",
            "HTTPS_PROXY": proxy or "",
        }
    }
    write_json(
        output_dir / "runtime_env_redacted.json",
        redact_env(runtime_env["env_vars"]),
    )

    import ray
    from verl.trainer.main_ppo import run_ppo

    from budget_coder_rl.train.m5_trainer import M5TaskRunner

    ray.init(runtime_env=runtime_env, **ray_init_kwargs(tmp_root))
    provenance["verl_runtime_after_init"] = get_verl_info()
    try:
        provenance["ray_cluster_resources"] = dict(ray.cluster_resources())
    except Exception as exc:
        provenance["ray_cluster_resources"] = {"error": str(exc)}
    write_json(output_dir / "provenance.json", provenance)

    started = time.time()
    stop_reason = "completed"
    oom = False
    hard_stop = None
    try:
        config = build_config(
            model_path,
            n_gpus=n_gpus,
            tensor_model_parallel_size=tp,
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
            train_files=str(train_parquet),
            val_files=str(train_parquet),
            n_tasks=int(TRAIN_BATCH_SIZE),
            n_gpus=n_gpus,
            ppo_max_token_len_per_gpu=ppo_max,
            total_training_steps=n_steps,
            experiment_name=experiment_name,
            default_local_dir=str(checkpoint_root),
            save_freq=save_freq,
            max_actor_ckpt_to_keep=keep,
            resume_mode=resume_mode,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            actor_lr=float((actor.get("optim_lr") if actor else None) or 1e-6),
            seed=SEED,
            calculate_entropy=True,
            wandb=True,
            wandb_proxy=proxy,
        )
        if int(config.trainer.n_gpus_per_node) != n_gpus:
            raise SystemExit(
                f"HARD FAIL: trainer.n_gpus_per_node={config.trainer.n_gpus_per_node} != {n_gpus}"
            )
        if int(config.actor_rollout_ref.rollout.tensor_model_parallel_size) != tp:
            raise SystemExit(
                "HARD FAIL: rollout.tensor_model_parallel_size="
                f"{config.actor_rollout_ref.rollout.tensor_model_parallel_size} != {tp}"
            )
        sampling_recorded = assert_sampling_config(config, require_rollout_n=GROUP_N)
        provenance["sampling_rollout"] = sampling_recorded
        provenance["expected_placement"] = expected_hybrid_placement(
            n_gpus=n_gpus, tensor_model_parallel_size=tp
        )
        write_json(output_dir / "config_resolved.json", _safe_config(config))
        write_json(output_dir / "provenance.json", provenance)
        run_ppo(
            config,
            task_runner_class=ray.remote(num_cpus=1)(M5TaskRunner),
        )
    except HardStopError as exc:
        stop_reason = exc.reason
        hard_stop = {"reason": exc.reason, "details": exc.details}
        write_json(output_dir / "run_error.json", hard_stop)
        print(f"HARD STOP: {exc.reason}", file=sys.stderr)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        classified = classify_hard_stop_from_text(message)
        stop_reason = classified or "error"
        if classified == "OOM" or "out of memory" in message.lower():
            oom = True
            stop_reason = "OOM"
        write_json(
            output_dir / "run_error.json",
            {
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "oom": oom,
                "classified": classified,
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
    placement = {}
    if (output_dir / "placement.json").is_file():
        placement = load_json(output_dir / "placement.json")
    n_completed = int(metrics_summary.get("n_steps") or 0)
    m6 = selected_m6_candidate(checkpoint_root)
    research_unmodified = (
        sha256_file(main_path) == EXPECTED_MAIN_SHA256 and not knob_errors
    )
    status = "PASS" if stop_reason == "completed" and n_completed >= n_steps else "FAIL"
    evidence = {
        "status": status,
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "project_commit": (provenance.get("budget_coder_rl") or {}).get("commit"),
        "verl_isolated_clean": not verl_import_errors and not verl_info.get("dirty"),
        "verl_commit": verl_runtime.get("commit") or verl_info.get("commit"),
        "main_config_sha256": sha256_file(main_path),
        "overlay_sha256": runtime.get("overlay_sha256"),
        "launch_mechanism": lifecycle.get("mechanism"),
        "n_gpus": n_gpus,
        "tensor_model_parallel_size": tp,
        "n_vllm_replicas": (placement.get("n_vllm_replicas") or runtime.get("n_gpus")),
        "fsdp_world_size": placement.get("actor_rollout_wg.world_size")
        or placement.get("fsdp_world_size"),
        "n_steps_completed": n_completed,
        "n_steps_nonzero_advantage": int(metrics_summary.get("n_steps_nonzero_advantage") or 0),
        "metrics_jsonl_present": (output_dir / "metrics.jsonl").is_file(),
        "wandb_logged": bool(wandb_run.get("id")),
        "wandb_url": wandb_run.get("url"),
        "checkpoint": ckpt,
        "checkpoint_steps": ckpt.get("global_steps"),
        "m6_candidate": m6.get("path"),
        "m6_selection": m6,
        "research_freeze_unmodified": research_unmodified,
        "oom": oom,
        "hard_stop": hard_stop,
        "placement": placement,
        "lifecycle": lifecycle,
        "metrics_summary": metrics_summary,
        "host": host,
    }
    write_json(output_dir / "run_status.json", evidence)
    (output_dir / "SUMMARY.md").write_text(
        build_training_summary(
            output_dir=output_dir,
            checkpoint_root=checkpoint_root,
            evidence=evidence,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "stop_reason": stop_reason, "elapsed_s": elapsed}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
