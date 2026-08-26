#!/usr/bin/env python
"""E016 scaled-M5 preflight: 2 optimizer steps, full topology, no 275-step main.

Consumes frozen stage1_m5_scaled.json. Does not edit M3C/M5-main/E014/E015.
Does not start the scaled main run.

Usage (compute node, conda env ``verl``):

    python scripts/eval/run_e016_preflight.py
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
from budget_coder_rl.eval.e012 import load_episode_proxy_rows  # noqa: E402
from budget_coder_rl.eval.e013 import packing_smokes, peak_gpu_memory_mib  # noqa: E402
from budget_coder_rl.eval.e014 import is_login_host, resource_lifecycle  # noqa: E402
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING, sha256_ids  # noqa: E402
from budget_coder_rl.eval.m4a import (  # noqa: E402
    REWARD_NUM_WORKERS,
    artifact_hashes,
    default_freeze_path,
    load_json,
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
    checkpoint_dir_manifest,
    default_isolated_verl_root,
    ensure_isolated_verl_checkout,
    imported_verl_errors,
    prepend_isolated_verl,
    summarize_metrics_jsonl,
)
from budget_coder_rl.eval.m5b import (  # noqa: E402
    EXPECTED_N_GPUS,
    EXPECTED_TP,
    HARD_STOP_ENV,
    PLACEMENT_ENV,
    PPO_MAX_ENV,
    HardStopError,
    checkpoint_dir_conflict_errors,
    classify_hard_stop_from_text,
    disk_capacity_errors,
    expected_hybrid_placement,
    redact_env,
)
from budget_coder_rl.eval.m5_scaled import (  # noqa: E402
    DISK_MIN_GIB,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    EXPECTED_PAD_IDS,
    EXPECTED_PADDED_IDS_SHA256,
    EXPECTED_UNIQUE_IDS_SHA256,
    MAIN_STEPS,
    N_ROWS,
    N_UNIQUE,
    PPO_MAX_TOKEN_LEN,
    PREFLIGHT_EXPERIMENT_ID,
    PREFLIGHT_MAX_STEPS,
    PREFLIGHT_STEPS,
    SYMBOL_UNAVAILABLE_IN_FIRST_BATCH,
    WANDB_EXPERIMENT_NAME,
    build_preflight_summary,
    checkpoint_audit,
    consume_scaled_errors,
    default_candidate_path,
    default_contract_path,
    default_preflight_checkpoint_dir,
    default_preflight_output_dir,
    default_preflight_path,
    forbidden_output_dir_errors,
    historical_untouched_errors,
    load_padded_ids,
    observe_checkpoint_retention,
    predicted_main_compute,
    preflight_overlay_errors,
    ready_payload,
)
from budget_coder_rl.eval.provenance import collect_run_provenance, sha256_file  # noqa: E402
from budget_coder_rl.ray_tmpdir import (  # noqa: E402
    cleanup_our_tmp_ray,
    ray_init_kwargs,
    short_temp_root,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=PREFLIGHT_EXPERIMENT_ID)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Refuse a dirty worktree. Default allows dirty so freeze files can be uncommitted.",
    )
    parser.add_argument("--n-steps", type=int, default=PREFLIGHT_STEPS)
    return parser.parse_args(argv)


def _assert_compute_node() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else ""
    if is_login_host(host):
        raise SystemExit(f"HARD FAIL: do not run E016 GPU on login node ({host})")
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


def _clip_ratio_positive(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metrics = json.loads(line).get("metrics") or {}
        for key in ("response_length/clip_ratio", "prompt_length/clip_ratio"):
            raw = metrics.get(key)
            try:
                if raw is not None and float(raw) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _symbol_unavailable_seen(episodes_path: Path) -> dict[str, Any]:
    n = 0
    n_unavail = 0
    conan = 0
    if not episodes_path.is_file():
        return {
            "n_episodes": 0,
            "n_symbol_unavailable": 0,
            "conan_10213_n": 0,
            "seen": False,
        }
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        identity = row.get("identity") if isinstance(row.get("identity"), dict) else {}
        loc = row.get("localization") if isinstance(row.get("localization"), dict) else {}
        instance_id = str(identity.get("instance_id") or row.get("instance_id") or "")
        status = loc.get("symbol_status") or row.get("symbol_status")
        if instance_id == SYMBOL_UNAVAILABLE_IN_FIRST_BATCH:
            conan += 1
            if status == "unavailable":
                n_unavail += 1
        elif status == "unavailable":
            n_unavail += 1
    return {
        "n_episodes": n,
        "n_symbol_unavailable": n_unavail,
        "conan_10213_n": conan,
        "seen": n_unavail > 0 or conan > 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    n_steps = int(args.n_steps)
    if n_steps < 1 or n_steps > PREFLIGHT_MAX_STEPS:
        print(
            f"HARD FAIL: preflight steps {n_steps} not in 1..{PREFLIGHT_MAX_STEPS}",
            file=sys.stderr,
        )
        return 1
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_preflight_output_dir(repo_root)
    )
    forbidden = forbidden_output_dir_errors(output_dir, repo_root)
    if forbidden:
        print(f"HARD FAIL: {forbidden}", file=sys.stderr)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[OUTPUT_ENV] = str(output_dir)
    os.environ[HARD_STOP_ENV] = "1"
    os.environ[PLACEMENT_ENV] = "1"
    os.environ.setdefault("WANDB_DIR", str(output_dir / "wandb"))
    os.environ.setdefault("WANDB_PROJECT", "budget-coder-rl")

    consume = consume_scaled_errors(repo_root)
    if consume:
        print(f"HARD FAIL: scaled contract {consume}", file=sys.stderr)
        return 1
    if args.require_clean:
        from budget_coder_rl.eval.m5b import project_tree_dirty_errors

        dirty = project_tree_dirty_errors(repo_root)
        if dirty:
            print(f"HARD FAIL: {dirty}", file=sys.stderr)
            return 1
    if not os.environ.get("WANDB_API_KEY"):
        print("HARD FAIL: WANDB_API_KEY is not set", file=sys.stderr)
        return 1

    candidate_path = default_candidate_path(repo_root)
    contract_path = default_contract_path(repo_root)
    overlay_path = default_preflight_path(repo_root)
    freeze_path = default_freeze_path(repo_root)
    candidate = load_json(candidate_path)
    contract = load_json(contract_path)
    overlay = load_json(overlay_path)
    overlay_errors = preflight_overlay_errors(overlay, contract=contract)
    if overlay_errors:
        print(f"HARD FAIL: preflight overlay {overlay_errors}", file=sys.stderr)
        return 1
    trainer_over = (overlay.get("overrides") or {}).get("trainer") or {}
    overlay_steps = int(trainer_over.get("total_training_steps") or 0)
    if overlay_steps > PREFLIGHT_MAX_STEPS or n_steps > PREFLIGHT_MAX_STEPS:
        print("HARD FAIL: refusing to run more than 2 optimizer steps", file=sys.stderr)
        return 1
    newly = contract.get("newly_frozen") or {}
    actor = newly.get("actor") or {}
    trainer_cfg = newly.get("trainer") or {}
    gpu_cfg = newly.get("gpu") or {}
    if int(trainer_cfg.get("total_training_steps") or 0) != MAIN_STEPS:
        print("HARD FAIL: scaled contract total_training_steps != 275", file=sys.stderr)
        return 1
    if int(actor.get("ppo_max_token_len_per_gpu") or 0) != PPO_MAX_TOKEN_LEN:
        print("HARD FAIL: scaled contract ppo_max != 20480", file=sys.stderr)
        return 1
    n_gpus = int(gpu_cfg.get("n_gpus") or 0)
    tp = int(gpu_cfg.get("tensor_model_parallel_size") or 0)
    ppo_max = PPO_MAX_TOKEN_LEN
    if n_gpus != EXPECTED_N_GPUS or tp != EXPECTED_TP:
        print(f"HARD FAIL: placement n_gpus={n_gpus} tp={tp}", file=sys.stderr)
        return 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_cfg.get("cuda_visible_devices") or "0,1")
    os.environ[PPO_MAX_ENV] = str(ppo_max)

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

    packing = packing_smokes(max_token_len=ppo_max)
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

    instance_ids = load_padded_ids(candidate_path)
    if len(instance_ids) != N_ROWS:
        print(f"HARD FAIL: padded ids {len(instance_ids)} != {N_ROWS}", file=sys.stderr)
        return 1
    if sha256_ids(instance_ids) != EXPECTED_PADDED_IDS_SHA256:
        print("HARD FAIL: padded_ids hash drifted", file=sys.stderr)
        return 1
    unique_ids = [str(item) for item in candidate.get("ordered_ids") or []]
    if sha256_ids(unique_ids) != EXPECTED_UNIQUE_IDS_SHA256:
        print("HARD FAIL: unique_ids hash drifted", file=sys.stderr)
        return 1
    parquet_path = (
        args.parquet.resolve() if args.parquet is not None else train_parquet_path(repo_root)
    )
    oracle_path = oracle_parquet_path(repo_root).resolve()
    agent_loop_config = repo_root / M3C_AGENT_LOOP_CONFIG_RELPATH
    reward_fn_path = repo_root / REWARD_FN_RELPATH
    train_parquet = output_dir / "train_e016.parquet"
    for required in (parquet_path, oracle_path, reward_fn_path, agent_loop_config):
        if not Path(required).is_file():
            print(f"HARD FAIL: missing {required}", file=sys.stderr)
            return 1
    parquet_info = write_smoke_parquet(parquet_path, train_parquet, instance_ids)
    if int(parquet_info.get("n_rows") or 0) != N_ROWS:
        print(f"HARD FAIL: train parquet rows {parquet_info.get('n_rows')} != {N_ROWS}", file=sys.stderr)
        return 1
    predicted_batches = N_ROWS // TRAIN_BATCH_SIZE
    if predicted_batches != MAIN_STEPS:
        print("HARD FAIL: 2200/8 != 275", file=sys.stderr)
        return 1
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
    lifecycle["session_name"] = PREFLIGHT_EXPERIMENT_ID

    checkpoint_root = default_preflight_checkpoint_dir(Path(data_root))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    ckpt_conflicts = checkpoint_dir_conflict_errors(checkpoint_root, allow_resume=False)
    if ckpt_conflicts:
        print(f"HARD FAIL: {ckpt_conflicts}", file=sys.stderr)
        return 1
    disk_errors = disk_capacity_errors(
        checkpoint_root, output_dir, min_gib=DISK_MIN_GIB
    )
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
    provenance["milestone"] = "E016-SCALED-PREFLIGHT"
    provenance["phase"] = "scaled_preflight"
    provenance["do_not_start_275"] = True
    provenance["instance_ids_head"] = instance_ids[:16]
    provenance["n_padded_ids"] = len(instance_ids)
    provenance["selection"] = {
        "universe": "m5_scaled_train_candidates.padded_ids",
        "filter": "full 2200-row padded pool; SequentialSampler; stop after 2 steps",
        "n_unique": N_UNIQUE,
        "n_rows": N_ROWS,
        "n_steps": n_steps,
        "gold_used_for_cherry_pick": False,
        "reward_used_for_selection": False,
        "pad_ids": list(EXPECTED_PAD_IDS),
    }
    provenance["sampling_intended"] = dict(QWEN3_SAMPLING)
    provenance["isolated_verl"] = verl_info
    provenance["verl_runtime"] = verl_runtime
    provenance["gpu"] = gpu_info
    provenance["host"] = host
    provenance["lifecycle"] = lifecycle
    provenance["data_root"] = str(data_root)
    provenance["ray_tmpdir"] = str(tmp_root)
    provenance["tmp_cleanup"] = tmp_cleanup
    provenance["checkpoint_root"] = str(checkpoint_root)
    provenance["resume_mode"] = "disable"
    provenance["packing_assert"] = packing
    provenance["predicted_dataloader_batches"] = predicted_batches
    provenance["artifacts"] = artifact_hashes(
        {
            "freeze": freeze_path,
            "scaled_manifest": candidate_path,
            "scaled_contract": contract_path,
            "preflight_overlay": overlay_path,
            "canonical_envelope": repo_root / "configs/experiments/stage1_canonical_execution_envelope.json",
            "e014_overlay": repo_root / "configs/experiments/stage1_m5_e014_runtime.json",
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
            total_training_steps=n_steps,
            experiment_name=WANDB_EXPERIMENT_NAME,
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
        applied_steps = int(config.trainer.total_training_steps)
        if applied_steps > PREFLIGHT_MAX_STEPS:
            raise SystemExit(
                f"HARD FAIL: applied total_training_steps={applied_steps} > {PREFLIGHT_MAX_STEPS}"
            )
        applied = int(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu)
        logprob = int(config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu)
        if applied != ppo_max:
            raise SystemExit(f"HARD FAIL: actor ppo_max={applied} != {ppo_max}")
        if logprob < ppo_max:
            raise SystemExit(
                f"HARD FAIL: log_prob_max_token_len_per_gpu={logprob} < envelope {ppo_max}"
            )
        if int(config.trainer.n_gpus_per_node) != n_gpus:
            raise SystemExit("HARD FAIL: n_gpus mismatch")
        if int(config.actor_rollout_ref.rollout.tensor_model_parallel_size) != tp:
            raise SystemExit("HARD FAIL: TP mismatch")
        if int(config.data.train_batch_size) != TRAIN_BATCH_SIZE:
            raise SystemExit("HARD FAIL: train_batch_size mismatch")
        if int(config.actor_rollout_ref.rollout.n) != GROUP_N:
            raise SystemExit("HARD FAIL: G != 4")
        sampling_recorded = assert_sampling_config(config, require_rollout_n=GROUP_N)
        provenance["sampling_rollout"] = sampling_recorded
        provenance["log_prob_max_token_len_per_gpu"] = logprob
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
    ckpt = checkpoint_dir_manifest(checkpoint_root)
    retention = observe_checkpoint_retention(checkpoint_root)
    write_json(output_dir / "checkpoint_retention.json", retention)
    realized = None
    step_bcrl = output_dir / "step_bcrl.jsonl"
    if step_bcrl.is_file():
        for line in step_bcrl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metrics = json.loads(line).get("metrics") or {}
            proxy_max = metrics.get("bcrl/seq/training_proxy_max")
            if proxy_max is None:
                continue
            realized = max(float(realized or 0), float(proxy_max))
    for row in load_episode_proxy_rows(output_dir / "episodes.jsonl"):
        realized = max(float(realized or 0), float(row["training_seq_proxy"]))
    pg_ok = True
    grad_ok = True
    invalid_rates: list[float] = []
    if (output_dir / "metrics.jsonl").is_file():
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metrics = json.loads(line).get("metrics") or {}
            if "actor/pg_loss" in metrics and not _finite(metrics.get("actor/pg_loss")):
                pg_ok = False
            if "actor/grad_norm" in metrics and not _finite(metrics.get("actor/grad_norm")):
                grad_ok = False
            raw_invalid = metrics.get("bcrl/invalid_action_rate")
            try:
                if raw_invalid is not None:
                    invalid_rates.append(float(raw_invalid))
            except (TypeError, ValueError):
                pass
    clipped = _clip_ratio_positive(output_dir / "metrics.jsonl")
    gpu_peak = peak_gpu_memory_mib(output_dir / "gpu_sampler.jsonl")
    symbol_obs = _symbol_unavailable_seen(output_dir / "episodes.jsonl")
    historical_ok = not historical_untouched_errors(repo_root)
    reasons: list[str] = []
    if not packing.get("covers"):
        reasons.append("packing_assert_failed")
    if n_completed < n_steps:
        reasons.append(f"steps {n_completed}<{n_steps}")
    if n_completed > PREFLIGHT_MAX_STEPS:
        reasons.append("ran more than 2 steps")
    if stop_reason != "completed":
        reasons.append(f"stop_reason={stop_reason}")
    if oom:
        reasons.append("OOM")
    if not pg_ok or not grad_ok:
        reasons.append("non_finite_pg_or_grad")
    if not historical_ok:
        reasons.append("historical_freeze_mutated")
    if n_completed >= n_steps and int(placement.get("n_vllm_replicas") or 2) != 2:
        reasons.append("vllm replicas != 2")
    if n_completed >= 1 and not symbol_obs.get("seen"):
        reasons.append("symbol_unavailable_path_not_observed")
    if retention.get("n_actor_dirs", 0) < 1 and n_completed >= 1 and stop_reason == "completed":
        reasons.append("checkpoint_actor_missing")
    ready = not reasons
    margin = None if realized is None else ppo_max - float(realized)
    n_traj = n_completed * TRAIN_BATCH_SIZE * GROUP_N
    invalid_mean = (
        sum(invalid_rates) / len(invalid_rates) if invalid_rates else None
    )
    evidence = {
        "status": "PASS" if ready else "FAIL",
        "READY_FOR_SCALED_M5_MAIN": ready,
        "do_not_start_main": True,
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "n_unique": N_UNIQUE,
        "n_rows": N_ROWS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "main_steps": MAIN_STEPS,
        "n_trajectories": N_ROWS * GROUP_N,
        "unique_ids_sha256": EXPECTED_UNIQUE_IDS_SHA256,
        "padded_ids_sha256": EXPECTED_PADDED_IDS_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "pad_ids": list(EXPECTED_PAD_IDS),
        "per_repo_counts": candidate.get("per_repo_counts"),
        "symbol_applicable_true": candidate.get("symbol_applicable_true"),
        "symbol_applicable_false": candidate.get("symbol_applicable_false"),
        "old_256_coverage_n": candidate.get("old_256_coverage_n"),
        "checkpoint_audit": checkpoint_audit(),
        "compute_estimate": predicted_main_compute(),
        "ckpt_retention_observation": retention,
        "preflight_steps": n_steps,
        "n_steps_completed": n_completed,
        "n_trajectories_preflight": n_traj,
        "n_steps_nonzero_advantage": metrics_summary.get("n_steps_nonzero_advantage"),
        "n_steps_mixed": metrics_summary.get("n_steps_mixed"),
        "realized_max_seq": realized,
        "margin": margin,
        "packing_covers": packing.get("covers"),
        "oom": oom,
        "pg_loss_finite": pg_ok,
        "grad_norm_finite": grad_ok,
        "clip_ratio_positive": clipped,
        "invalid_action_rate": invalid_mean,
        "symbol_unavailable_seen": symbol_obs,
        "gpu0_peak_mib": gpu_peak.get("gpu0_peak_mib"),
        "gpu1_peak_mib": gpu_peak.get("gpu1_peak_mib"),
        "gpu_peak_mib": gpu_peak.get("peak_mib"),
        "wandb_url": wandb_run.get("url"),
        "placement": placement,
        "hard_stop": hard_stop,
        "host": host,
        "gate_reasons": reasons,
        "checkpoint": ckpt,
        "checkpoint_steps": ckpt.get("global_steps"),
        "n_gpus": n_gpus,
        "tensor_model_parallel_size": tp,
        "n_vllm_replicas": placement.get("n_vllm_replicas"),
        "fsdp_world_size": placement.get("fsdp_world_size")
        or placement.get("actor_rollout_wg.world_size"),
        "metrics_summary": metrics_summary,
        "seed": SEED,
        "predicted_dataloader_batches": predicted_batches,
    }
    write_json(output_dir / "run_status.json", evidence)
    write_json(
        output_dir / "READY_FOR_SCALED_M5_MAIN.json",
        ready_payload(ready=ready, reasons=reasons, extra=evidence),
    )
    (output_dir / "SUMMARY.md").write_text(
        build_preflight_summary(evidence=evidence),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "READY_FOR_SCALED_M5_MAIN": ready,
                "status": evidence["status"],
                "stop_reason": stop_reason,
                "n_steps_completed": n_completed,
                "do_not_start_main": True,
                "realized_max_seq": realized,
                "elapsed_s": elapsed,
                "gate_reasons": reasons,
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
