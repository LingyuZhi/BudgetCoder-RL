"""M5B / E011 helpers: systems overlay, 2-GPU placement, hard-stop, selection.

Does not edit stage1_m5_main.json or the M3C freeze. Does not retune
reward / prompt / tools / parser / budget / sampling / GRPO group size.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.m4a import GROUP_N, OBS_TOKENS_LIMIT, load_json
from budget_coder_rl.eval.m4b import write_json
from budget_coder_rl.eval.m5a import (
    MAIN_STEPS,
    SEED,
    TRAIN_BATCH_SIZE,
    default_main_config_path,
    default_output_dir,
)
from budget_coder_rl.eval.provenance import git_info, sha256_file

EXPERIMENT_ID = "E011"
MILESTONE = "M5B"
SESSION_NAME = "E011"
OUTPUT_ENV = "BCRL_M5_OUTPUT_DIR"
HARD_STOP_ENV = "BCRL_M5B_HARD_STOP"
PLACEMENT_ENV = "BCRL_M5B_PLACEMENT"
PPO_MAX_ENV = "BCRL_PPO_MAX_TOKEN_LEN"
RUNTIME_CONFIG_RELPATH = "configs/experiments/stage1_m5b_e011_runtime.json"
RUNTIME_LOCK_RELPATH = "configs/experiments/stage1_m5b_e011_runtime.lock.json"
MAIN_LOCK_RELPATH = "configs/experiments/stage1_m5_main.lock.json"
EXPECTED_MAIN_SHA256 = "fac90e49b1c3d6bc42beff57cdc73a407b2d9f88cb83f748dc4670d6dfc9837b"
EXPECTED_M3C_SHA256 = "49084af1c792e2049af72d4c98291dc546b829122034dba9e698cea8f7284185"
EXPECTED_CANDIDATE_SHA256 = "3ece05681486bf28dc99637f98674723ddf4797024e0af856f5725fb71c7e81b"
EXPECTED_N_GPUS = 2
EXPECTED_TP = 1
EXPECTED_NNODES = 1
CANONICAL_CHECKPOINT_STEP = 32
CHECKPOINT_SELECTION_RULE = "terminal_global_step_32"
CHECKPOINT_RELPATH = "checkpoints/stage1_m5_main"
DISK_MIN_GIB = 40.0
ZERO_ADV_STREAK_STOP = 8
PARSE_COLLAPSE_STREAK_STOP = 8
PPO_MAX_TOKEN_LEN = 16384
COMPUTE_HOST_HINT = "n30158"

ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "parent",
        "allowed_override_scope",
        "checkpoint_selection",
        "overrides",
        "notes",
    }
)
ALLOWED_OVERRIDE_SECTIONS = frozenset({"gpu", "launch", "systems"})
ALLOWED_GPU_KEYS = frozenset(
    {
        "n_gpus",
        "nnodes",
        "tensor_model_parallel_size",
        "device",
        "cuda_visible_devices",
    }
)
ALLOWED_LAUNCH_KEYS = frozenset({"session_name", "preferred", "fallback"})
FORBIDDEN_RESEARCH_SECTIONS = frozenset(
    {
        "actor",
        "algorithm",
        "data",
        "trainer",
        "inherited_m3c",
        "inherited_m4_runtime",
        "localization_reward",
        "sampling",
    }
)


class HardStopError(RuntimeError):
    """Abort E011 without editing the frozen contract."""

    def __init__(self, reason: str, details: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.reason = str(reason)
        self.details = dict(details or {})


def default_runtime_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_CONFIG_RELPATH


def default_runtime_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_LOCK_RELPATH


def default_main_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / MAIN_LOCK_RELPATH


def default_e011_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_checkpoint_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / CHECKPOINT_RELPATH


def is_login_host(host: str | None = None) -> bool:
    name = (host or socket.gethostname() or "").lower()
    return name.startswith("sn") or "login" in name


def resource_lifecycle(*, host: str | None = None) -> dict[str, Any]:
    name = host or socket.gethostname()
    slurm_job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    login = is_login_host(name)
    if slurm_job:
        mechanism = "sbatch"
        reason = "SLURM_JOB_ID is set; tmux cannot be assumed to keep the allocation"
    elif login:
        mechanism = "ssh_compute_then_tmux"
        reason = f"login/head node {name}; GPU run must move to {COMPUTE_HOST_HINT}"
    else:
        mechanism = "tmux"
        reason = "persistent compute node without Slurm job; named tmux is sufficient"
    return {
        "hostname": name,
        "login_node": login,
        "slurm_job_id": slurm_job,
        "mechanism": mechanism,
        "session_name": SESSION_NAME,
        "compute_host_hint": COMPUTE_HOST_HINT,
        "reason": reason,
    }


def project_tree_dirty_errors(repo_root: Path) -> list[str]:
    info = git_info(Path(repo_root))
    if info.get("dirty"):
        preview = info.get("dirty_files") or []
        return [
            "BudgetCoder-RL worktree is dirty; E011 must start from a committed state: "
            + ", ".join(str(item) for item in preview[:12])
        ]
    if not info.get("commit"):
        return ["could not resolve BudgetCoder-RL HEAD commit"]
    return []


def parent_freeze_errors(
    *,
    repo_root: Path,
    overlay: Mapping[str, Any],
) -> list[str]:
    main_path = default_main_config_path(Path(repo_root))
    lock_path = default_main_lock_path(Path(repo_root))
    errors: list[str] = []
    if not main_path.is_file():
        return [f"missing immutable main config {main_path}"]
    actual = sha256_file(main_path)
    if actual != EXPECTED_MAIN_SHA256:
        errors.append(
            f"stage1_m5_main.json sha256 {actual} != frozen {EXPECTED_MAIN_SHA256}"
        )
    if lock_path.is_file():
        lock = load_json(lock_path)
        lock_sha = str(lock.get("sha256") or "")
        if lock_sha and lock_sha != actual:
            errors.append(
                f"main lock sha256 {lock_sha} != file sha256 {actual}; freeze was edited"
            )
    parent = overlay.get("parent") if isinstance(overlay.get("parent"), MappingABC) else {}
    parent_sha = str(parent.get("sha256") or "")
    if parent_sha != actual:
        errors.append(
            f"overlay parent sha256 {parent_sha} != current main sha256 {actual}"
        )
    if parent_sha != EXPECTED_MAIN_SHA256:
        errors.append(
            f"overlay parent sha256 {parent_sha} != expected {EXPECTED_MAIN_SHA256}"
        )
    return errors


def overlay_errors(
    overlay: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_OVERLAY_KEYS]
    if extra:
        errors.append(f"runtime overlay has disallowed keys: {extra}")
    if str(overlay.get("schema_version") or "") != "bcrl-stage1-m5b-runtime-v1":
        errors.append("runtime overlay schema_version must be bcrl-stage1-m5b-runtime-v1")
    if str(overlay.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("runtime overlay experiment_id must be E011")
    if str(overlay.get("milestone") or "") != MILESTONE:
        errors.append("runtime overlay milestone must be M5B")
    selection = overlay.get("checkpoint_selection")
    if not isinstance(selection, MappingABC):
        errors.append("runtime overlay missing checkpoint_selection")
    else:
        if str(selection.get("rule") or "") != CHECKPOINT_SELECTION_RULE:
            errors.append(
                f"checkpoint_selection.rule must be {CHECKPOINT_SELECTION_RULE}"
            )
        if int(selection.get("canonical_global_step") or 0) != CANONICAL_CHECKPOINT_STEP:
            errors.append("canonical_global_step must be 32")
    overrides = overlay.get("overrides") if isinstance(overlay.get("overrides"), MappingABC) else {}
    extra_sections = [key for key in overrides.keys() if str(key) not in ALLOWED_OVERRIDE_SECTIONS]
    if extra_sections:
        errors.append(f"runtime overrides has disallowed sections: {extra_sections}")
    research = [key for key in overrides.keys() if str(key) in FORBIDDEN_RESEARCH_SECTIONS]
    if research:
        errors.append(f"runtime overlay must not change research sections: {research}")
    gpu = overrides.get("gpu") if isinstance(overrides.get("gpu"), MappingABC) else {}
    bad_gpu = [key for key in gpu.keys() if str(key) not in ALLOWED_GPU_KEYS]
    if bad_gpu:
        errors.append(f"gpu overrides disallowed: {bad_gpu}")
    if int(gpu.get("n_gpus") or 0) != EXPECTED_N_GPUS:
        errors.append(f"gpu.n_gpus must be {EXPECTED_N_GPUS}")
    if int(gpu.get("nnodes") or EXPECTED_NNODES) != EXPECTED_NNODES:
        errors.append("gpu.nnodes must stay 1")
    if int(gpu.get("tensor_model_parallel_size") or 0) != EXPECTED_TP:
        errors.append(
            "gpu.tensor_model_parallel_size must stay 1 "
            "(do not set TP=2; Hydra default is 2 and would change rollout semantics)"
        )
    launch = overrides.get("launch") if isinstance(overrides.get("launch"), MappingABC) else {}
    bad_launch = [key for key in launch.keys() if str(key) not in ALLOWED_LAUNCH_KEYS]
    if bad_launch:
        errors.append(f"launch overrides disallowed: {bad_launch}")
    if str(launch.get("session_name") or "") != SESSION_NAME:
        errors.append("launch.session_name must be E011")
    if repo_root is not None:
        errors.extend(parent_freeze_errors(repo_root=Path(repo_root), overlay=overlay))
    return errors


def overlay_lock_errors(repo_root: Path) -> list[str]:
    overlay_path = default_runtime_config_path(repo_root)
    lock_path = default_runtime_lock_path(repo_root)
    errors: list[str] = []
    if not overlay_path.is_file():
        return [f"missing runtime overlay {overlay_path}"]
    if not lock_path.is_file():
        return [f"missing runtime lock {lock_path}"]
    actual = sha256_file(overlay_path)
    lock = load_json(lock_path)
    lock_sha = str(lock.get("sha256") or "")
    if lock_sha != actual:
        errors.append(
            f"runtime overlay sha256 {actual} != lock {lock_sha}; overlay was edited"
        )
    parent_sha = str(lock.get("parent_sha256") or "")
    if parent_sha != EXPECTED_MAIN_SHA256:
        errors.append(
            f"runtime lock parent_sha256 {parent_sha} != {EXPECTED_MAIN_SHA256}"
        )
    return errors


def consume_runtime_overlay(
    *,
    repo_root: Path,
    overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = default_runtime_config_path(repo_root)
    payload = overlay if overlay is not None else load_json(path)
    errors = overlay_errors(payload, repo_root=repo_root)
    errors.extend(overlay_lock_errors(repo_root))
    if errors:
        raise HardStopError("runtime overlay contract failed", {"errors": errors})
    overrides = payload.get("overrides") or {}
    gpu = overrides.get("gpu") or {}
    launch = overrides.get("launch") or {}
    selection = payload.get("checkpoint_selection") or {}
    return {
        "n_gpus": int(gpu["n_gpus"]),
        "nnodes": int(gpu.get("nnodes") or 1),
        "tensor_model_parallel_size": int(gpu["tensor_model_parallel_size"]),
        "device": str(gpu.get("device") or "2xA100-40GB"),
        "cuda_visible_devices": str(gpu.get("cuda_visible_devices") or "0,1"),
        "session_name": str(launch.get("session_name") or SESSION_NAME),
        "checkpoint_selection_rule": str(selection.get("rule") or CHECKPOINT_SELECTION_RULE),
        "canonical_global_step": int(
            selection.get("canonical_global_step") or CANONICAL_CHECKPOINT_STEP
        ),
        "overlay_sha256": sha256_file(path),
        "parent_sha256": str((payload.get("parent") or {}).get("sha256") or ""),
        "overlay_path": str(path),
    }


def expected_hybrid_placement(
    *,
    n_gpus: int = EXPECTED_N_GPUS,
    tensor_model_parallel_size: int = EXPECTED_TP,
    data_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> dict[str, Any]:
    replica_world = (
        int(tensor_model_parallel_size)
        * int(data_parallel_size)
        * int(pipeline_model_parallel_size)
    )
    world = int(n_gpus)
    n_replicas = world // replica_world if replica_world else 0
    return {
        "fsdp_world_size": world,
        "vllm_tensor_model_parallel_size": int(tensor_model_parallel_size),
        "vllm_data_parallel_size_config": int(data_parallel_size),
        "vllm_pipeline_model_parallel_size": int(pipeline_model_parallel_size),
        "replica_world_size": replica_world,
        "n_vllm_replicas": n_replicas,
        "note": "TP=1 and 2 GPUs => two TP=1 vLLM replicas colocated with FSDP ranks; not TP=2",
    }


def placement_errors(
    placement: Mapping[str, Any],
    *,
    n_gpus: int = EXPECTED_N_GPUS,
    tensor_model_parallel_size: int = EXPECTED_TP,
) -> list[str]:
    errors: list[str] = []
    if int(placement.get("fsdp_world_size") or 0) != int(n_gpus):
        errors.append(
            f"FSDP world_size {placement.get('fsdp_world_size')} != n_gpus {n_gpus}"
        )
    if int(placement.get("vllm_tensor_model_parallel_size") or 0) != int(
        tensor_model_parallel_size
    ):
        errors.append(
            "rollout tensor_model_parallel_size "
            f"{placement.get('vllm_tensor_model_parallel_size')} != {tensor_model_parallel_size}"
        )
    expected = expected_hybrid_placement(
        n_gpus=n_gpus,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )
    if int(placement.get("n_vllm_replicas") or 0) != int(expected["n_vllm_replicas"]):
        errors.append(
            f"vLLM replicas {placement.get('n_vllm_replicas')} != expected {expected['n_vllm_replicas']}"
        )
    if int(tensor_model_parallel_size) != EXPECTED_TP:
        errors.append("E011 forbids TP!=1")
    return errors


def canonical_m6_checkpoint(checkpoint_root: Path) -> Path:
    return Path(checkpoint_root) / f"global_step_{CANONICAL_CHECKPOINT_STEP}"


def selected_m6_candidate(
    checkpoint_root: Path,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del metrics  # explicitly ignore train/W&B curves
    path = canonical_m6_checkpoint(checkpoint_root)
    return {
        "rule": CHECKPOINT_SELECTION_RULE,
        "global_step": CANONICAL_CHECKPOINT_STEP,
        "path": str(path),
        "exists": path.is_dir(),
        "post_hoc_curve_pick": False,
    }


def disk_free_gib(path: Path) -> float:
    Path(path).mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return float(usage.free) / (1024**3)


def disk_capacity_errors(*paths: Path, min_gib: float = DISK_MIN_GIB) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            free = disk_free_gib(path)
        except OSError as exc:
            errors.append(f"disk check failed for {path}: {exc}")
            continue
        if free < float(min_gib):
            errors.append(f"{path} has {free:.1f} GiB free; need >= {min_gib:.0f} GiB")
    return errors


def checkpoint_dir_conflict_errors(
    checkpoint_root: Path,
    *,
    allow_resume: bool = False,
) -> list[str]:
    root = Path(checkpoint_root)
    if not root.exists():
        return []
    entries = [item for item in root.iterdir() if item.name != "latest_checkpointed_iteration.txt"]
    if not entries:
        return []
    if allow_resume:
        return []
    return [
        f"checkpoint dir {root} is not empty ({[item.name for item in entries[:8]]}); "
        "refusing to start a new E011 over leftover shards. Use official resume or a new experiment id."
    ]


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _is_bad_number(value: Any) -> bool:
    number = _as_float(value)
    if number is None:
        return False
    return not math.isfinite(number)


def classify_hard_stop_from_text(message: str) -> str | None:
    lowered = message.lower()
    if "out of memory" in lowered or "cuda oom" in lowered or lowered.strip() == "oom":
        return "OOM"
    if "nan" in lowered or "inf" in lowered:
        return "NaN_or_Inf"
    if "ppo_max_token_len_per_gpu" in lowered or "sequence length" in lowered and "assert" in lowered:
        return "sequence_over_envelope"
    if "response_mask" in lowered or "tito" in lowered or "token-in/token-out" in lowered:
        return "response_mask_or_tito"
    if any(marker in lowered for marker in ("oracle_symbols", "gold_edit_files", "base_changed_files")):
        return "evaluator_oracle_leakage"
    if "add_lora" in lowered or "update_weights" in lowered or "lora" in lowered and "fail" in lowered:
        return "adapter_weight_sync"
    if "checkpoint" in lowered and any(word in lowered for word in ("corrupt", "failed", "missing")):
        return "corrupted_checkpoint"
    if "ray" in lowered or "vllm" in lowered or "enginecore" in lowered:
        if any(word in lowered for word in ("died", "crashed", "unhealthy", "actor died", "worker died")):
            return "ray_vllm_failure"
    return None


def _state_path(output_dir: Path) -> Path:
    return Path(output_dir) / "hard_stop_state.json"


def _load_state(output_dir: Path) -> dict[str, Any]:
    path = _state_path(output_dir)
    if not path.is_file():
        return {"zero_adv_streak": 0, "parse_collapse_streak": 0}
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return {"zero_adv_streak": 0, "parse_collapse_streak": 0}


def inspect_step_metrics_for_hard_stop(
    metrics: Mapping[str, Any],
    *,
    step: int,
    output_dir: Path,
    ppo_max_token_len: int | None = None,
) -> None:
    payload = dict(metrics or {})
    reasons: list[str] = []
    envelope = int(
        ppo_max_token_len
        if ppo_max_token_len is not None
        else os.environ.get(PPO_MAX_ENV) or PPO_MAX_TOKEN_LEN
    )
    for key in (
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/loss",
        "actor/entropy",
        "bcrl/reward/mean",
    ):
        if _is_bad_number(payload.get(key)):
            reasons.append(f"NaN_or_Inf:{key}={payload.get(key)}")
    clip = _as_float(payload.get("response_length/clip_ratio"))
    prompt_clip = _as_float(payload.get("prompt_length/clip_ratio"))
    if clip is not None and clip > 0:
        reasons.append(f"sequence_over_envelope:response_length/clip_ratio={clip}")
    if prompt_clip is not None and prompt_clip > 0:
        reasons.append(f"sequence_over_envelope:prompt_length/clip_ratio={prompt_clip}")
    proxy_max = _as_float(payload.get("bcrl/seq/training_proxy_max"))
    if proxy_max is not None and proxy_max > envelope:
        reasons.append(
            f"sequence_over_envelope:training_proxy_max={proxy_max}>{envelope}"
        )

    state = _load_state(output_dir)
    nonzero = payload.get("bcrl/any_nonzero_advantage")
    if nonzero in {0, 0.0, False, None}:
        state["zero_adv_streak"] = int(state.get("zero_adv_streak") or 0) + 1
    else:
        state["zero_adv_streak"] = 0
    parse_ok = _as_float(payload.get("bcrl/parse_ok_rate"))
    invalid = _as_float(payload.get("bcrl/invalid_action_rate"))
    collapsed = parse_ok is not None and parse_ok <= 0.0 and invalid is not None and invalid >= 0.95
    if collapsed:
        state["parse_collapse_streak"] = int(state.get("parse_collapse_streak") or 0) + 1
    else:
        state["parse_collapse_streak"] = 0
    state["last_step"] = int(step)
    write_json(_state_path(output_dir), state)
    if int(state["zero_adv_streak"]) >= ZERO_ADV_STREAK_STOP:
        reasons.append(
            f"sustained_all_zero_grpo_signal:{state['zero_adv_streak']} consecutive steps"
        )
    if int(state["parse_collapse_streak"]) >= PARSE_COLLAPSE_STREAK_STOP:
        reasons.append(
            f"protocol_parser_collapse:{state['parse_collapse_streak']} consecutive steps"
        )
    if not reasons:
        return
    details = {
        "step": int(step),
        "reasons": reasons,
        "metrics_subset": {
            key: payload.get(key)
            for key in (
                "actor/pg_loss",
                "actor/grad_norm",
                "response_length/clip_ratio",
                "bcrl/any_nonzero_advantage",
                "bcrl/parse_ok_rate",
                "bcrl/seq/training_proxy_max",
            )
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(Path(output_dir) / "hard_stop.json", details)
    raise HardStopError(reasons[0], details)


def sample_nvidia_gpus() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": int(float(parts[0])),
                "name": parts[1],
                "memory_used_mi": int(float(parts[2])),
                "memory_total_mi": int(float(parts[3])),
                "utilization_gpu": int(float(parts[4])),
            }
        )
    return rows


def append_gpu_sample(path: Path) -> dict[str, Any]:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": sample_nvidia_gpus(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return payload


def redact_env(env: Mapping[str, Any]) -> dict[str, Any]:
    secret_keys = {"WANDB_API_KEY", "WANDB_API_KEY_FILE"}
    out: dict[str, Any] = {}
    for key, value in env.items():
        name = str(key)
        if name in secret_keys or "API_KEY" in name.upper() or "SECRET" in name.upper():
            out[name] = "<redacted>"
        else:
            out[name] = value
    return out


def research_knobs_from_main(main: Mapping[str, Any]) -> dict[str, Any]:
    newly = main.get("newly_frozen") if isinstance(main.get("newly_frozen"), MappingABC) else {}
    inherited = main.get("inherited_m3c") if isinstance(main.get("inherited_m3c"), MappingABC) else {}
    actor = newly.get("actor") if isinstance(newly.get("actor"), MappingABC) else {}
    data = newly.get("data") if isinstance(newly.get("data"), MappingABC) else {}
    algo = newly.get("algorithm") if isinstance(newly.get("algorithm"), MappingABC) else {}
    trainer = newly.get("trainer") if isinstance(newly.get("trainer"), MappingABC) else {}
    return {
        "B_obs": inherited.get("primary_training_B_obs"),
        "budget_visible": inherited.get("budget_visible"),
        "sampling": inherited.get("sampling"),
        "train_batch_size": data.get("train_batch_size"),
        "shuffle": data.get("shuffle"),
        "rollout_n": algo.get("rollout_n"),
        "vllm_n": algo.get("vllm_n"),
        "total_training_steps": trainer.get("total_training_steps"),
        "total_epochs": trainer.get("total_epochs"),
        "save_freq": trainer.get("save_freq"),
        "seed": trainer.get("seed"),
        "ppo_max_token_len_per_gpu": actor.get("ppo_max_token_len_per_gpu"),
        "optim_lr": actor.get("optim_lr"),
        "use_kl_loss": actor.get("use_kl_loss"),
        "entropy_coeff": actor.get("entropy_coeff"),
    }


def research_knob_errors(main: Mapping[str, Any]) -> list[str]:
    knobs = research_knobs_from_main(main)
    errors: list[str] = []
    if knobs.get("B_obs") != OBS_TOKENS_LIMIT:
        errors.append("B_obs != 4096")
    if knobs.get("budget_visible") is not True:
        errors.append("budget_visible is not true")
    if int(knobs.get("train_batch_size") or 0) != TRAIN_BATCH_SIZE:
        errors.append("train_batch_size != 8")
    if knobs.get("shuffle") is not False:
        errors.append("shuffle must stay false")
    if int(knobs.get("rollout_n") or 0) != GROUP_N:
        errors.append("rollout_n / G != 4")
    if int(knobs.get("vllm_n") or 0) != 1:
        errors.append("vllm_n != 1")
    if int(knobs.get("total_training_steps") or 0) != MAIN_STEPS:
        errors.append("total_training_steps != 32")
    if int(knobs.get("total_epochs") or 0) != 1:
        errors.append("total_epochs != 1")
    if int(knobs.get("save_freq") or 0) != 8:
        errors.append("save_freq != 8")
    if int(knobs.get("seed") or 0) != SEED:
        errors.append("seed != 20260826")
    if int(knobs.get("ppo_max_token_len_per_gpu") or 0) != PPO_MAX_TOKEN_LEN:
        errors.append("ppo_max_token_len_per_gpu != 16384")
    sampling = knobs.get("sampling") if isinstance(knobs.get("sampling"), MappingABC) else {}
    if abs(float(sampling.get("temperature") or 0) - 0.7) > 1e-6:
        errors.append("sampling temperature changed")
    if abs(float(sampling.get("top_p") or 0) - 0.8) > 1e-6:
        errors.append("sampling top_p changed")
    if int(sampling.get("top_k") or 0) != 20:
        errors.append("sampling top_k changed")
    return errors


def metric_series(path: Path, key: str) -> list[tuple[int, Any]]:
    rows: list[tuple[int, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            step = payload.get("step") or payload.get("global_steps")
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), MappingABC) else payload
            rows.append((int(step or 0), metrics.get(key)))
    return rows


def summarize_series(pairs: Sequence[tuple[int, Any]]) -> dict[str, Any]:
    values = [_as_float(item) for _, item in pairs]
    present = [item for item in values if item is not None]
    if not present:
        return {"n": 0, "first": None, "last": None, "min": None, "max": None, "mean": None}
    return {
        "n": len(present),
        "first": present[0],
        "last": present[-1],
        "min": min(present),
        "max": max(present),
        "mean": sum(present) / len(present),
    }


def build_training_summary(
    *,
    output_dir: Path,
    checkpoint_root: Path,
    evidence: Mapping[str, Any],
) -> str:
    metrics_path = Path(output_dir) / "metrics.jsonl"
    lines = [
        "# M5B / E011 Canonical Stage 1 GRPO",
        "",
        f"- status: **{evidence.get('status')}**",
        f"- stop_reason: {evidence.get('stop_reason')}",
        f"- project_commit: `{evidence.get('project_commit')}`",
        f"- veRL: `{evidence.get('verl_commit')}` isolated={evidence.get('verl_isolated_clean')}",
        f"- main_config_sha256: `{evidence.get('main_config_sha256')}`",
        f"- overlay_sha256: `{evidence.get('overlay_sha256')}`",
        f"- launch: {evidence.get('launch_mechanism')}",
        f"- placement: n_gpus={evidence.get('n_gpus')} TP={evidence.get('tensor_model_parallel_size')} "
        f"replicas={evidence.get('n_vllm_replicas')} fsdp_world={evidence.get('fsdp_world_size')}",
        f"- steps: {evidence.get('n_steps_completed')}/32",
        f"- wandb: {evidence.get('wandb_url')}",
        f"- canonical M6 candidate: `{evidence.get('m6_candidate')}`",
        f"- research freeze unmodified: **{evidence.get('research_freeze_unmodified')}**",
        "",
        "## Dynamics",
        "",
    ]
    for key, label in (
        ("bcrl/reward/mean", "reward_mean"),
        ("bcrl/group/mixed_fraction", "mixed_group"),
        ("bcrl/group/zero_var_fraction", "zero_var"),
        ("actor/pg_loss", "pg_loss"),
        ("actor/grad_norm", "grad_norm"),
        ("actor/ppo_kl", "ppo_kl"),
        ("actor/entropy", "entropy"),
        ("bcrl/parse_ok_rate", "parse_ok"),
        ("bcrl/budget_exhaustion_rate", "budget_exhaustion"),
        ("timing_s/step", "step_s"),
    ):
        series = summarize_series(metric_series(metrics_path, key))
        lines.append(
            f"- {label}: first={series['first']} last={series['last']} "
            f"min={series['min']} max={series['max']} mean={series['mean']}"
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            f"- root: `{checkpoint_root}`",
            f"- global_steps: {evidence.get('checkpoint_steps')}",
            f"- selection_rule: `{CHECKPOINT_SELECTION_RULE}` (not post-hoc)",
            "",
            "## User status commands",
            "",
            "```bash",
            f"ssh {COMPUTE_HOST_HINT}",
            "cd ~/my_proj/budget-coder-rl",
            "bash scripts/eval/e011_session.sh status",
            "bash scripts/eval/e011_session.sh logs",
            "bash scripts/eval/e011_session.sh attach",
            "nvidia-smi",
            "cat outputs/experiments/E011/wandb_run.json",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
