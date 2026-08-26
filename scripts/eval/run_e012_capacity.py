#!/usr/bin/env python
"""E012 2-GPU capacity stress: 8 long prompts, 2 GRPO updates.

Does not run the 32-step canonical main. Does not edit E011 or stage1_m5_main.json.

Usage (compute node, conda env ``verl``):

    python scripts/eval/run_e012_capacity.py
"""

from __future__ import annotations

import argparse
import json
import math
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
from budget_coder_rl.eval.e012 import (  # noqa: E402
    E011_FAILURE_MAX_SEQ,
    EXPERIMENT_ID,
    EXPECTED_E011_OVERLAY_SHA256,
    MILESTONE,
    REQUIRED_TASK,
    STRESS_STEPS,
    STRESS_UNIQUE_TASKS,
    build_capacity_summary,
    consume_runtime_overlay,
    default_e012_output_dir,
    default_stress_checkpoint_dir,
    is_login_host,
    load_episode_proxy_rows,
    packing_assert_covers,
    ready_payload,
    repeat_ids_for_steps,
    resource_lifecycle,
)
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING  # noqa: E402
from budget_coder_rl.eval.m4a import (  # noqa: E402
    REWARD_NUM_WORKERS,
    artifact_hashes,
    default_candidate_path,
    default_freeze_path,
)
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
    default_isolated_verl_root,
    default_main_config_path,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    load_json,
    m5_freeze_consume_errors,
    prepend_isolated_verl,
    summarize_metrics_jsonl,
)
from budget_coder_rl.eval.m5b import (  # noqa: E402
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_TP,
    HARD_STOP_ENV,
    PLACEMENT_ENV,
    PPO_MAX_ENV,
    HardStopError,
    classify_hard_stop_from_text,
    disk_capacity_errors,
    expected_hybrid_placement,
    project_tree_dirty_errors,
    redact_env,
    research_knob_errors,
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
    return parser.parse_args(argv)


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if is_login_host(host):
        raise SystemExit(f"HARD FAIL: do not run E012 GPU on login node ({host})")
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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _count_dask_episodes(path: Path) -> int:
    n = 0
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        identity = rec.get("identity") if isinstance(rec.get("identity"), dict) else {}
        if str(identity.get("instance_id") or rec.get("instance_id") or "") == REQUIRED_TASK:
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_e012_output_dir(repo_root)
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
        print("HARD FAIL: stage1_m5_main.json was edited", file=sys.stderr)
        return 1
    e011_overlay = repo_root / "configs/experiments/stage1_m5b_e011_runtime.json"
    if sha256_file(e011_overlay) != EXPECTED_E011_OVERLAY_SHA256:
        print("HARD FAIL: E011 overlay was edited", file=sys.stderr)
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
    ppo_max = int(runtime["ppo_max_token_len_per_gpu"])
    if n_gpus != EXPECTED_N_GPUS or tp != EXPECTED_TP:
        print(f"HARD FAIL: overlay placement n_gpus={n_gpus} tp={tp}", file=sys.stderr)
        return 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])
    os.environ[PPO_MAX_ENV] = str(ppo_max)

    audit_path = output_dir / "capacity_audit.json"
    if not audit_path.is_file():
        print("HARD FAIL: capacity_audit.json missing; run run_e012_capacity_audit.py", file=sys.stderr)
        return 1
    audit = load_json(audit_path)
    selected = list((audit.get("decision") or {}).get("selected_instance_ids") or [])
    if len(selected) != STRESS_UNIQUE_TASKS or REQUIRED_TASK not in selected:
        print(f"HARD FAIL: audit selection invalid {selected}", file=sys.stderr)
        return 1
    instance_ids = repeat_ids_for_steps(selected, n_steps=STRESS_STEPS)

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

    packing = packing_assert_covers(max_token_len=ppo_max, max_seq_len=E011_FAILURE_MAX_SEQ)
    write_json(output_dir / "packing_assert.json", packing)
    if not packing.get("covers"):
        print(f"HARD FAIL: packing assert {packing}", file=sys.stderr)
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

    parquet_path = (
        args.parquet.resolve() if args.parquet is not None else train_parquet_path(repo_root)
    )
    oracle_path = oracle_parquet_path(repo_root).resolve()
    agent_loop_config = repo_root / M3C_AGENT_LOOP_CONFIG_RELPATH
    reward_fn_path = repo_root / REWARD_FN_RELPATH
    train_parquet = output_dir / "train_e012_stress.parquet"
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

    checkpoint_root = default_stress_checkpoint_dir(Path(data_root))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    disk_errors = disk_capacity_errors(checkpoint_root, output_dir)
    if disk_errors:
        print(f"HARD FAIL: {disk_errors}", file=sys.stderr)
        return 1

    newly = main_cfg.get("newly_frozen") or {}
    actor = newly.get("actor") or {}

    provenance = collect_run_provenance(
        repo_root,
        verl_source=isolated_root,
        agent_loop_config=agent_loop_config,
        model_path=model_path,
    )
    provenance["experiment_id"] = args.experiment_id
    provenance["milestone"] = MILESTONE
    provenance["phase"] = "capacity_stress"
    provenance["instance_ids"] = selected
    provenance["instance_ids_repeated"] = instance_ids
    provenance["selection"] = {
        "universe": "m3c_train_candidates.ordered_ids",
        "filter": "longest prompts by tokenizer/episode count; required dask__dask-10042",
        "n_unique_tasks": len(selected),
        "n_steps": STRESS_STEPS,
        "gold_used_for_cherry_pick": False,
        "reward_used_for_selection": False,
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
    provenance["not_m6_candidate"] = True
    provenance["packing_assert"] = packing
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "candidates": candidate_path,
            "main_config": main_path,
            "e011_overlay": e011_overlay,
            "e012_overlay": repo_root / "configs/experiments/stage1_m5_e012_runtime.json",
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
            PPO_MAX_ENV: str(ppo_max),
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
    write_json(output_dir / "runtime_env_redacted.json", redact_env(runtime_env["env_vars"]))

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
            total_training_steps=STRESS_STEPS,
            experiment_name="E012-capacity-stress",
            default_local_dir=str(checkpoint_root),
            save_freq=-1,
            max_actor_ckpt_to_keep=0,
            resume_mode="disable",
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            actor_lr=float((actor.get("optim_lr") if actor else None) or 1e-6),
            seed=SEED,
            calculate_entropy=True,
            wandb=True,
            wandb_proxy=proxy,
        )
        applied = int(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu)
        logprob = int(config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu)
        if applied != ppo_max:
            raise SystemExit(f"HARD FAIL: actor ppo_max={applied} != overlay {ppo_max}")
        if logprob < ppo_max:
            raise SystemExit(
                f"HARD FAIL: log_prob_max_token_len_per_gpu={logprob} < envelope {ppo_max}; "
                "matching systems change required"
            )
        if int(config.trainer.n_gpus_per_node) != n_gpus:
            raise SystemExit("HARD FAIL: n_gpus mismatch")
        if int(config.actor_rollout_ref.rollout.tensor_model_parallel_size) != tp:
            raise SystemExit("HARD FAIL: TP mismatch")
        sampling_recorded = assert_sampling_config(config, require_rollout_n=GROUP_N)
        provenance["sampling_rollout"] = sampling_recorded
        provenance["log_prob_max_token_len_per_gpu"] = logprob
        provenance["ref_log_prob_max_token_len_per_gpu"] = int(
            getattr(config.actor_rollout_ref.ref, "log_prob_max_token_len_per_gpu", -1) or -1
        )
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
    if (output_dir / "wandb_run.json").is_file():
        wandb_run = load_json(output_dir / "wandb_run.json")
    placement = {}
    if (output_dir / "placement.json").is_file():
        placement = load_json(output_dir / "placement.json")
    n_completed = int(metrics_summary.get("n_steps") or 0)
    bcrl_rows = []
    step_bcrl = output_dir / "step_bcrl.jsonl"
    if step_bcrl.is_file():
        for line in step_bcrl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                bcrl_rows.append(json.loads(line))
    realized = None
    for row in bcrl_rows:
        metrics = row.get("metrics") or {}
        proxy = metrics.get("bcrl/seq/training_proxy_max")
        if proxy is None:
            continue
        realized = max(float(realized or 0), float(proxy))
    episode_realized = None
    for row in load_episode_proxy_rows(output_dir / "episodes.jsonl"):
        episode_realized = max(float(episode_realized or 0), float(row["training_seq_proxy"]))
    if episode_realized is not None:
        realized = max(float(realized or 0), episode_realized)
    dask_n = _count_dask_episodes(output_dir / "episodes.jsonl")
    pg_ok = True
    grad_ok = True
    if (output_dir / "metrics.jsonl").is_file():
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metrics = json.loads(line).get("metrics") or {}
            if "actor/pg_loss" in metrics and not _finite(metrics.get("actor/pg_loss")):
                pg_ok = False
            if "actor/grad_norm" in metrics and not _finite(metrics.get("actor/grad_norm")):
                grad_ok = False
    research_unmodified = sha256_file(main_path) == EXPECTED_MAIN_SHA256 and not knob_errors
    reasons: list[str] = []
    if not packing.get("covers"):
        reasons.append("packing_assert_16751_failed")
    if n_completed < STRESS_STEPS:
        reasons.append(f"steps {n_completed}<{STRESS_STEPS}")
    if stop_reason != "completed":
        reasons.append(f"stop_reason={stop_reason}")
    if oom:
        reasons.append("OOM")
    if not pg_ok or not grad_ok:
        reasons.append("non_finite_pg_or_grad")
    if dask_n < GROUP_N * STRESS_STEPS:
        reasons.append(f"dask episodes {dask_n} < {GROUP_N * STRESS_STEPS}")
    if not research_unmodified:
        reasons.append("research_freeze_modified")
    if int(placement.get("n_vllm_replicas") or 0) not in {0, 2} and n_completed:
        # placement may be missing n_vllm_replicas key; expected dump fills it
        pass
    if n_completed >= STRESS_STEPS and int(placement.get("n_vllm_replicas") or 2) != 2:
        reasons.append("vllm replicas != 2")
    ready = not reasons
    margin = None if realized is None else ppo_max - float(realized)
    evidence = {
        "status": "PASS" if ready else "FAIL",
        "READY_FOR_E012_MAIN": ready,
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "chosen_envelope": ppo_max,
        "realized_max_seq": realized,
        "margin": margin,
        "packing_covers": packing.get("covers"),
        "dask_in_episodes": dask_n,
        "n_steps_completed": n_completed,
        "overlay_sha256": runtime.get("overlay_sha256"),
        "parent_sha256": EXPECTED_MAIN_SHA256,
        "e011_sha256": EXPECTED_E011_OVERLAY_SHA256,
        "research_freeze_unmodified": research_unmodified,
        "oom": oom,
        "pg_loss_finite": pg_ok,
        "grad_norm_finite": grad_ok,
        "wandb_url": wandb_run.get("url"),
        "placement": placement,
        "hard_stop": hard_stop,
        "host": host,
        "gate_reasons": reasons,
        "not_m6_candidate": True,
        "canonical_32_step_not_started": True,
        "stress_tasks": selected,
        "seed": SEED,
    }
    write_json(output_dir / "run_status.json", evidence)
    write_json(
        output_dir / "READY_FOR_E012_MAIN.json",
        ready_payload(ready=ready, reasons=reasons, extra=evidence),
    )
    (output_dir / "SUMMARY.md").write_text(
        build_capacity_summary(evidence=evidence),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "READY_FOR_E012_MAIN": ready,
            "stop_reason": stop_reason,
            "chosen_envelope": ppo_max,
            "realized_max_seq": realized,
            "margin": margin,
            "elapsed_s": elapsed,
        },
        indent=2,
    ))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
