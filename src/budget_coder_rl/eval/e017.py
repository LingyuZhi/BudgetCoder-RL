"""E017 scaled-M5 main GRPO namespace. Consumes frozen stage1_m5_scaled.json.

Does not edit M3C/M5-main/scaled freeze files, E014, E015, or E016 artifacts.
Fresh launch uses resume_mode=disable on an empty E017 checkpoint directory.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.train.gpu_runtime import is_login_host
from budget_coder_rl.eval.m4a import GROUP_N, load_json
from budget_coder_rl.eval.m4b import write_json
from budget_coder_rl.eval.m5_scaled import (
    CONTRACT_RELPATH,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    EXPECTED_PADDED_IDS_SHA256,
    EXPECTED_UNIQUE_IDS_SHA256,
    MAIN_STEPS,
    MAX_ACTOR_CKPT_TO_KEEP,
    N_ROWS,
    N_TRAJECTORIES,
    N_UNIQUE,
    PPO_MAX_TOKEN_LEN,
    SAVE_FREQ,
    consume_scaled_errors,
    default_preflight_checkpoint_dir,
    default_scaled_checkpoint_dir,
    historical_untouched_errors as scaled_historical_untouched_errors,
)
from budget_coder_rl.eval.m5a import SEED, TRAIN_BATCH_SIZE, default_output_dir
from budget_coder_rl.eval.m5b import (
    EXPECTED_N_GPUS,
    EXPECTED_NNODES,
    EXPECTED_TP,
    disk_capacity_errors,
    expected_hybrid_placement,
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E017"
MILESTONE = "E017-SCALED-M5-MAIN"
SESSION_NAME = "E017"
SCHEMA_VERSION = "bcrl-stage1-m5-scaled-e017-v1"
WANDB_EXPERIMENT_NAME = "E017-scaled-m5-main"
OVERLAY_RELPATH = "configs/historical/stage1_m5_scaled_e017.json"
OVERLAY_LOCK_RELPATH = "configs/historical/stage1_m5_scaled_e017.lock.json"
CHECKPOINT_RELPATH = "checkpoints/stage1_m5_scaled_e017"
TRAJECTORY_RELPATH = "trajectories/stage1_m5_scaled/E017"
COMPUTE_HOST_HINT = "n30158"
EXPECTED_SLURM_JOB = "21702"
LAUNCH_DISK_MIN_GIB = 120.0
MIN_REMAINING_HOURS = 20.0
STATUS_LOCK = threading.Lock()

FORBIDDEN_OUTPUT_IDS = ("E011", "E012", "E013", "E014", "E015", "E016")
FORBIDDEN_CHECKPOINT_MARKERS = (
    "stage1_m5_main",
    "stage1_m5_e014",
    "stage1_m5_scaled_e016",
)
ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "not_preflight",
        "do_not_start_275",
        "inherits",
        "parent",
        "allowed_override_scope",
        "overrides",
        "output_dir",
        "trajectory_dir",
        "notes",
    }
)
ALLOWED_OVERRIDE_SECTIONS = frozenset({"trainer", "checkpoint", "launch"})
ALLOWED_TRAINER_KEYS = frozenset(
    {"experiment_name", "default_local_dir", "resume_mode"}
)
FORBIDDEN_TRAINER_KNOBS = frozenset(
    {
        "total_training_steps",
        "total_epochs",
        "save_freq",
        "max_actor_ckpt_to_keep",
        "val_before_train",
        "test_freq",
        "seed",
        "logger",
        "project_name",
        "critic_enable",
    }
)
FORBIDDEN_RESEARCH_SECTIONS = frozenset(
    {
        "actor",
        "algorithm",
        "data",
        "gpu",
        "inherited_m3c",
        "inherited_m4_runtime",
        "localization_reward",
        "sampling",
        "systems",
    }
)
ALLOWED_CHECKPOINT_KEYS = frozenset({"directory_template"})
ALLOWED_LAUNCH_KEYS = frozenset({"session_name"})
JSONL_LINK_NAMES = ("episodes.jsonl", "metrics.jsonl", "step_bcrl.jsonl")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY_ON_DISK = _REPO_ROOT / "configs/historical/stage1_m5_scaled_e017.json"
EXPECTED_OVERLAY_SHA256 = (
    sha256_file(_OVERLAY_ON_DISK) if _OVERLAY_ON_DISK.is_file() else ("0" * 64)
)


def default_overlay_path(repo_root: Path) -> Path:
    return Path(repo_root) / OVERLAY_RELPATH


def default_overlay_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / OVERLAY_LOCK_RELPATH


def default_e017_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_checkpoint_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / CHECKPOINT_RELPATH


def default_trajectory_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / TRAJECTORY_RELPATH


def default_e016_ready_path(repo_root: Path) -> Path:
    return Path(repo_root) / "outputs" / "experiments" / "E016" / "READY_FOR_SCALED_M5_MAIN.json"


def resource_lifecycle(*, host: str | None = None) -> dict[str, Any]:
    name = host or socket.gethostname()
    slurm_job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    login = is_login_host(name)
    if slurm_job:
        mechanism = "tmux_in_slurm"
        reason = "SLURM_JOB_ID is set; use tmux inside this allocation"
    elif login:
        mechanism = "ssh_compute_then_tmux"
        reason = f"login/head node {name}; GPU run must move to {COMPUTE_HOST_HINT}"
    else:
        mechanism = "tmux"
        reason = "persistent compute node; named tmux is sufficient"
    return {
        "hostname": name,
        "login_node": login,
        "slurm_job_id": slurm_job,
        "mechanism": mechanism,
        "session_name": SESSION_NAME,
        "compute_host_hint": COMPUTE_HOST_HINT,
        "reason": reason,
    }


def parse_slurm_duration(text: str | None) -> float | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in {"UNLIMITED", "INFINITE"}:
        return float("inf")
    if upper in {"INVALID", "UNKNOWN", "N/A", "NONE"}:
        return None
    days = 0
    clock = raw
    if "-" in raw:
        day_part, clock = raw.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = clock.split(":")
    try:
        numbers = [int(item) for item in parts]
    except ValueError:
        return None
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    elif len(numbers) == 1:
        hours, minutes, seconds = 0, 0, numbers[0]
    else:
        return None
    return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)


def _scontrol_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)=(\S+)", raw):
        mapping[match.group(1)] = match.group(2)
    return mapping


def inspect_slurm_job(job_id: str | None = None) -> dict[str, Any]:
    job = str(job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or EXPECTED_SLURM_JOB)
    try:
        raw = subprocess.check_output(
            ["scontrol", "show", "job", job],
            text=True,
            timeout=30,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "job_id": job,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    fields = _scontrol_map(raw)
    time_limit_s = parse_slurm_duration(fields.get("TimeLimit"))
    run_time_s = parse_slurm_duration(fields.get("RunTime"))
    remaining_s = None
    if time_limit_s is not None and run_time_s is not None:
        remaining_s = max(0.0, float(time_limit_s) - float(run_time_s))
    remaining_h = None if remaining_s is None else remaining_s / 3600.0
    state = str(fields.get("JobState") or fields.get("State") or "")
    return {
        "job_id": job,
        "ok": True,
        "raw_excerpt": raw[:2000],
        "JobState": state,
        "TimeLimit": fields.get("TimeLimit"),
        "RunTime": fields.get("RunTime"),
        "EndTime": fields.get("EndTime"),
        "StartTime": fields.get("StartTime"),
        "NodeList": fields.get("NodeList") or fields.get("BatchHost"),
        "NumNodes": fields.get("NumNodes"),
        "time_limit_s": time_limit_s,
        "run_time_s": run_time_s,
        "remaining_s": remaining_s,
        "remaining_h": remaining_h,
        "sufficient": remaining_h is not None
        and remaining_h >= MIN_REMAINING_HOURS
        and state.upper() == "RUNNING",
    }


def allocation_errors(*, job_id: str | None = None, host: str | None = None) -> list[str]:
    errors: list[str] = []
    name = (host or socket.gethostname() or "").lower()
    if is_login_host(name):
        errors.append(f"refusing to launch E017 on login node {name}")
    info = inspect_slurm_job(job_id or EXPECTED_SLURM_JOB)
    if not info.get("ok"):
        errors.append(f"scontrol failed for job {info.get('job_id')}: {info.get('error')}")
        return errors
    if str(info.get("job_id")) != EXPECTED_SLURM_JOB:
        errors.append(
            f"expected Slurm job {EXPECTED_SLURM_JOB}, got {info.get('job_id')}"
        )
    state = str(info.get("JobState") or "").upper()
    if state != "RUNNING":
        errors.append(f"Slurm job {info.get('job_id')} state={state or 'unknown'} (need RUNNING)")
    remaining_h = info.get("remaining_h")
    if remaining_h is None:
        errors.append("could not parse Slurm remaining time")
    elif float(remaining_h) < MIN_REMAINING_HOURS:
        errors.append(
            f"Slurm remaining {float(remaining_h):.1f}h < {MIN_REMAINING_HOURS:.0f}h; "
            "tmux cannot outlive the allocation"
        )
    nodelist = str(info.get("NodeList") or "").lower()
    if nodelist and COMPUTE_HOST_HINT not in nodelist and COMPUTE_HOST_HINT not in name:
        errors.append(f"job nodelist {info.get('NodeList')} is not {COMPUTE_HOST_HINT}")
    return errors


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            return [f"refusing to write into {experiment_id} artifact directory {forbidden}"]
    return []


def checkpoint_path_errors(checkpoint_root: Path, data_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    resolved = Path(checkpoint_root).resolve()
    text = str(resolved)
    if CHECKPOINT_RELPATH.split("/")[-1] not in text:
        errors.append(f"checkpoint path must contain {CHECKPOINT_RELPATH}")
    expected = default_checkpoint_dir(data_root).resolve()
    if resolved != expected:
        errors.append(f"checkpoint path {resolved} != E017 dir {expected}")
    scaled = default_scaled_checkpoint_dir(data_root).resolve()
    if resolved == scaled:
        errors.append(f"refusing generic scaled checkpoint dir {scaled}")
    preflight = default_preflight_checkpoint_dir(data_root).resolve()
    if resolved == preflight:
        errors.append(f"refusing E016 checkpoint dir {preflight}")
    for marker in FORBIDDEN_CHECKPOINT_MARKERS:
        if marker in text and "stage1_m5_scaled_e017" not in Path(text).name:
            errors.append(f"refusing checkpoint path containing {marker}")
    return errors


def e016_ready_errors(repo_root: Path) -> list[str]:
    path = default_e016_ready_path(repo_root)
    if not path.is_file():
        return [f"missing E016 ready artifact {path}"]
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read E016 ready artifact: {exc}"]
    if payload.get("READY_FOR_SCALED_M5_MAIN") is not True:
        return [f"E016 READY_FOR_SCALED_M5_MAIN is {payload.get('READY_FOR_SCALED_M5_MAIN')}"]
    return []


def overlay_errors(
    overlay: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_OVERLAY_KEYS]
    if extra:
        errors.append(f"E017 overlay has disallowed keys: {extra}")
    if str(overlay.get("schema_version") or "") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if str(overlay.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("experiment_id must be E017")
    if str(overlay.get("milestone") or "") != MILESTONE:
        errors.append(f"milestone must be {MILESTONE}")
    if overlay.get("not_preflight") is not True:
        errors.append("not_preflight must be true")
    if overlay.get("do_not_start_275") is not False:
        errors.append("do_not_start_275 must be false for the main run")
    inherited = str(overlay.get("inherits") or "")
    if inherited not in {
        CONTRACT_RELPATH,
        "configs/experiments/stage1_m5_scaled.json",
    }:
        errors.append(f"inherits must be {CONTRACT_RELPATH}")
    parent = overlay.get("parent") if isinstance(overlay.get("parent"), MappingABC) else {}
    if str(parent.get("sha256") or "") != EXPECTED_CONTRACT_SHA256:
        errors.append(
            f"parent.sha256 {parent.get('sha256')} != {EXPECTED_CONTRACT_SHA256}"
        )
    overrides = overlay.get("overrides") if isinstance(overlay.get("overrides"), MappingABC) else {}
    extra_sections = [key for key in overrides.keys() if str(key) not in ALLOWED_OVERRIDE_SECTIONS]
    if extra_sections:
        errors.append(f"overrides has disallowed sections: {extra_sections}")
    research = [key for key in overrides.keys() if str(key) in FORBIDDEN_RESEARCH_SECTIONS]
    if research:
        errors.append(f"must not change research sections: {research}")
    trainer = overrides.get("trainer") if isinstance(overrides.get("trainer"), MappingABC) else {}
    bad_trainer = [key for key in trainer.keys() if str(key) not in ALLOWED_TRAINER_KEYS]
    if bad_trainer:
        errors.append(f"trainer overrides disallowed: {bad_trainer}")
    knobs = [key for key in trainer.keys() if str(key) in FORBIDDEN_TRAINER_KNOBS]
    if knobs:
        errors.append(f"trainer overlay must not set research knobs: {knobs}")
    if str(trainer.get("experiment_name") or "") != WANDB_EXPERIMENT_NAME:
        errors.append(f"experiment_name must be {WANDB_EXPERIMENT_NAME}")
    if str(trainer.get("resume_mode") or "") != "disable":
        errors.append("fresh launch resume_mode must be disable")
    default_local = str(trainer.get("default_local_dir") or "")
    if "stage1_m5_scaled_e017" not in default_local:
        errors.append("default_local_dir must be the E017 checkpoint directory")
    if "stage1_m5_scaled_e016" in default_local:
        errors.append("default_local_dir must not be the E016 checkpoint directory")
    checkpoint = overrides.get("checkpoint") if isinstance(overrides.get("checkpoint"), MappingABC) else {}
    bad_ckpt = [key for key in checkpoint.keys() if str(key) not in ALLOWED_CHECKPOINT_KEYS]
    if bad_ckpt:
        errors.append(f"checkpoint overrides disallowed: {bad_ckpt}")
    template = str(checkpoint.get("directory_template") or "")
    if "stage1_m5_scaled_e017" not in template:
        errors.append("checkpoint.directory_template must be stage1_m5_scaled_e017")
    launch = overrides.get("launch") if isinstance(overrides.get("launch"), MappingABC) else {}
    bad_launch = [key for key in launch.keys() if str(key) not in ALLOWED_LAUNCH_KEYS]
    if bad_launch:
        errors.append(f"launch overrides disallowed: {bad_launch}")
    if str(launch.get("session_name") or "") != SESSION_NAME:
        errors.append("launch.session_name must be E017")
    if repo_root is not None:
        errors.extend(scaled_historical_untouched_errors(Path(repo_root)))
        errors.extend(consume_scaled_errors(Path(repo_root)))
    return errors


def overlay_lock_errors(repo_root: Path) -> list[str]:
    overlay_path = default_overlay_path(repo_root)
    lock_path = default_overlay_lock_path(repo_root)
    if not overlay_path.is_file():
        return [f"missing E017 overlay {overlay_path}"]
    if not lock_path.is_file():
        return [f"missing E017 lock {lock_path}"]
    actual = sha256_file(overlay_path)
    lock = load_json(lock_path)
    errors: list[str] = []
    if str(lock.get("sha256") or "") != actual:
        errors.append(f"E017 overlay sha256 {actual} != lock {lock.get('sha256')}")
    if EXPECTED_OVERLAY_SHA256 != "0" * 64 and actual != EXPECTED_OVERLAY_SHA256:
        errors.append(f"E017 overlay sha256 {actual} != pinned {EXPECTED_OVERLAY_SHA256}")
    if str(lock.get("parent_sha256") or "") != EXPECTED_CONTRACT_SHA256:
        errors.append("E017 lock parent_sha256 mismatch")
    if str(lock.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("E017 lock experiment_id must be E017")
    if int(lock.get("optimizer_steps") or 0) != MAIN_STEPS:
        errors.append("E017 lock optimizer_steps must be 275")
    if lock.get("do_not_start_275") is not False:
        errors.append("E017 lock do_not_start_275 must be false")
    return errors


def consume_e017_overlay(*, repo_root: Path, overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = default_overlay_path(repo_root)
    payload = overlay if overlay is not None else load_json(path)
    errors = overlay_errors(payload, repo_root=repo_root)
    if overlay is None:
        errors.extend(overlay_lock_errors(repo_root))
    if errors:
        from budget_coder_rl.eval.m5b import HardStopError

        raise HardStopError("E017 overlay contract failed", {"errors": errors})
    overrides = payload.get("overrides") or {}
    trainer = overrides.get("trainer") or {}
    launch = overrides.get("launch") or {}
    return {
        "experiment_name": str(trainer.get("experiment_name") or WANDB_EXPERIMENT_NAME),
        "default_local_dir": str(trainer.get("default_local_dir") or default_checkpoint_dir()),
        "resume_mode": str(trainer.get("resume_mode") or "disable"),
        "session_name": str(launch.get("session_name") or SESSION_NAME),
        "output_dir": str(payload.get("output_dir") or default_e017_output_dir(repo_root)),
        "trajectory_dir": str(payload.get("trajectory_dir") or default_trajectory_dir()),
        "overlay_sha256": sha256_file(path) if path.is_file() else None,
        "parent_sha256": str((payload.get("parent") or {}).get("sha256") or ""),
        "n_gpus": EXPECTED_N_GPUS,
        "nnodes": EXPECTED_NNODES,
        "tensor_model_parallel_size": EXPECTED_TP,
        "expected_placement": expected_hybrid_placement(
            n_gpus=EXPECTED_N_GPUS, tensor_model_parallel_size=EXPECTED_TP
        ),
        "n_unique": N_UNIQUE,
        "n_rows": N_ROWS,
        "n_steps": MAIN_STEPS,
        "group_n": GROUP_N,
        "n_trajectories": N_TRAJECTORIES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "save_freq": SAVE_FREQ,
        "max_actor_ckpt_to_keep": MAX_ACTOR_CKPT_TO_KEEP,
        "ppo_max_token_len_per_gpu": PPO_MAX_TOKEN_LEN,
        "seed": SEED,
    }


def latest_metrics_step(metrics_path: Path) -> int | None:
    if not Path(metrics_path).is_file():
        return None
    last: int | None = None
    try:
        with Path(metrics_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                step = payload.get("step")
                if step is None:
                    continue
                last = int(step)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return last
    return last


def wandb_url_from_dir(output_dir: Path) -> str | None:
    path = Path(output_dir) / "wandb_run.json"
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    url = payload.get("url")
    return str(url) if url else None


def merge_run_status(output_dir: Path, **fields: Any) -> dict[str, Any]:
    path = Path(output_dir) / "run_status.json"
    with STATUS_LOCK:
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                payload = load_json(path)
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload.update(fields)
        payload["experiment_id"] = payload.get("experiment_id") or EXPERIMENT_ID
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        step = latest_metrics_step(Path(output_dir) / "metrics.jsonl")
        if step is not None:
            payload["current_step"] = step
        url = wandb_url_from_dir(Path(output_dir))
        if url:
            payload["wandb_url"] = url
        write_json(path, payload)
        return payload


def link_trajectory_jsonl(*, output_dir: Path, trajectory_dir: Path) -> dict[str, Any]:
    traj = Path(trajectory_dir)
    traj.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for name in JSONL_LINK_NAMES:
        src = Path(output_dir) / name
        dst = traj / name
        if dst.exists() or dst.is_symlink():
            linked.append(str(dst))
            continue
        dst.symlink_to(src)
        linked.append(str(dst))
    return {"trajectory_dir": str(traj), "links": linked}


def launch_disk_errors(*paths: Path) -> list[str]:
    return disk_capacity_errors(*paths, min_gib=LAUNCH_DISK_MIN_GIB)


def build_launch_summary(*, evidence: Mapping[str, Any], checkpoint_root: Path) -> str:
    lines = [
        "# E017 scaled-M5 main GRPO",
        "",
        f"- status: **{evidence.get('status')}**",
        f"- stop_reason: `{evidence.get('stop_reason')}`",
        f"- n_steps_completed: {evidence.get('n_steps_completed')} / {MAIN_STEPS}",
        f"- expected trajectories: **{N_TRAJECTORIES}**",
        f"- unique/padded: {N_UNIQUE} / {N_ROWS}",
        f"- G: {GROUP_N}",
        f"- ppo_max: {PPO_MAX_TOKEN_LEN}",
        f"- overlay_sha256: `{evidence.get('overlay_sha256')}`",
        f"- contract_sha256: `{evidence.get('contract_sha256')}`",
        f"- manifest_sha256: `{evidence.get('manifest_sha256')}`",
        f"- resume_mode: `{evidence.get('resume_mode')}`",
        f"- wandb: {evidence.get('wandb_url')}",
        f"- checkpoint_root: `{checkpoint_root}`",
        f"- host: {evidence.get('host')}",
        f"- slurm: {evidence.get('slurm_job_id')}",
        "",
        "Do not auto-start M6. Do not edit frozen scaled contract from this run.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_overlay_lock(repo_root: Path) -> dict[str, Any]:
    overlay_path = default_overlay_path(repo_root)
    digest = sha256_file(overlay_path)
    lock = {
        "path": OVERLAY_RELPATH,
        "sha256": digest,
        "parent_path": CONTRACT_RELPATH,
        "parent_sha256": EXPECTED_CONTRACT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "unique_ids_sha256": EXPECTED_UNIQUE_IDS_SHA256,
        "padded_ids_sha256": EXPECTED_PADDED_IDS_SHA256,
        "experiment_id": EXPERIMENT_ID,
        "optimizer_steps": MAIN_STEPS,
        "group_n": GROUP_N,
        "n_trajectories": N_TRAJECTORIES,
        "save_freq": SAVE_FREQ,
        "ppo_max_token_len_per_gpu": PPO_MAX_TOKEN_LEN,
        "do_not_start_275": False,
        "resume_mode": "disable",
        "note": "Namespace overlay only. Not a scientific-knob change. Not E016.",
    }
    write_json(default_overlay_lock_path(repo_root), lock)
    return lock
