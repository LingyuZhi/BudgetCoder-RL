"""E014 canonical 32-step M5B rerun overlay. Consumes frozen 20480 envelope.

Does not mutate stage1_m5_main.json, E011-E013 overlays/artifacts, or the
canonical execution envelope JSON.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.e012 import EXPECTED_E011_OVERLAY_SHA256
from budget_coder_rl.eval.e013 import (
    CHOSEN_ENVELOPE,
    EXPECTED_E012_ENVELOPE,
    EXPECTED_E012_OVERLAY_SHA256,
)
from budget_coder_rl.eval.m4a import GROUP_N, load_json
from budget_coder_rl.eval.m4b import write_json
from budget_coder_rl.eval.m5a import (
    MAIN_STEPS,
    MAX_MODEL_LEN,
    N_CANDIDATES,
    SEED,
    TRAIN_BATCH_SIZE,
    default_output_dir,
)
from budget_coder_rl.eval.m5b import (
    CANONICAL_CHECKPOINT_STEP,
    CHECKPOINT_SELECTION_RULE,
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_NNODES,
    EXPECTED_TP,
    HardStopError,
    expected_hybrid_placement,
    selected_m6_candidate,
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E014"
MILESTONE = "M5B-E014-canonical"
SESSION_NAME = "E014"
SCHEMA_VERSION = "bcrl-stage1-e014-runtime-v1"
RUNTIME_CONFIG_RELPATH = "configs/experiments/stage1_m5_e014_runtime.json"
RUNTIME_LOCK_RELPATH = "configs/experiments/stage1_m5_e014_runtime.lock.json"
CANONICAL_ENVELOPE_RELPATH = "configs/experiments/stage1_canonical_execution_envelope.json"
E011_RUNTIME_RELPATH = "configs/experiments/stage1_m5b_e011_runtime.json"
E012_RUNTIME_RELPATH = "configs/experiments/stage1_m5_e012_runtime.json"
E013_RUNTIME_RELPATH = "configs/experiments/stage1_m5_e013_runtime.json"
EXPECTED_E013_OVERLAY_SHA256 = (
    "1977b427696c105183c6363e806cba623c7c135210c44b711fd15ee1f516b74d"
)
EXPECTED_CANONICAL_SHA256 = (
    "0b5928dbf28fd3f5949b3f62dcac47b23970b900a42b595c6fee6514c2986f65"
)
REVISION_REASON = "canonical_32_step_rerun"
CHECKPOINT_RELPATH = "checkpoints/stage1_m5_e014"
FORBIDDEN_OUTPUT_IDS = ("E011", "E012", "E013")
COMPUTE_HOST_HINT = "n30158"
WANDB_EXPERIMENT_NAME = "E014-m5-main"

ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "parent",
        "canonical_envelope",
        "e011_runtime",
        "e012_runtime",
        "e013_runtime",
        "revision_reason",
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
ALLOWED_SYSTEMS_KEYS = frozenset({"ppo_max_token_len_per_gpu"})
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


def default_runtime_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_CONFIG_RELPATH


def default_runtime_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_LOCK_RELPATH


def default_canonical_envelope_path(repo_root: Path) -> Path:
    return Path(repo_root) / CANONICAL_ENVELOPE_RELPATH


def default_e014_output_dir(repo_root: Path) -> Path:
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


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            return [f"refusing to write into {experiment_id} artifact directory {forbidden}"]
    return []


def historical_untouched_errors(repo_root: Path) -> list[str]:
    errors: list[str] = []
    checks = (
        (Path(repo_root) / "configs/experiments/stage1_m5_main.json", EXPECTED_MAIN_SHA256, "stage1_m5_main.json"),
        (Path(repo_root) / E011_RUNTIME_RELPATH, EXPECTED_E011_OVERLAY_SHA256, "E011 overlay"),
        (Path(repo_root) / E012_RUNTIME_RELPATH, EXPECTED_E012_OVERLAY_SHA256, "E012 overlay"),
        (Path(repo_root) / E013_RUNTIME_RELPATH, EXPECTED_E013_OVERLAY_SHA256, "E013 overlay"),
        (default_canonical_envelope_path(repo_root), EXPECTED_CANONICAL_SHA256, "canonical envelope"),
    )
    for path, expected, label in checks:
        if not path.is_file():
            errors.append(f"missing {label} {path}")
            continue
        digest = sha256_file(path)
        if digest != expected:
            errors.append(f"{label} sha256 {digest} != frozen {expected}")
    envelope = load_json(default_canonical_envelope_path(repo_root)) if (
        default_canonical_envelope_path(repo_root).is_file()
    ) else {}
    if int(envelope.get("ppo_max_token_len_per_gpu") or 0) != CHOSEN_ENVELOPE:
        errors.append("canonical envelope ppo_max is no longer 20480")
    return errors


def overlay_errors(
    overlay: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_OVERLAY_KEYS]
    if extra:
        errors.append(f"E014 overlay has disallowed keys: {extra}")
    if str(overlay.get("schema_version") or "") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if str(overlay.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("experiment_id must be E014")
    if str(overlay.get("milestone") or "") != MILESTONE:
        errors.append(f"milestone must be {MILESTONE}")
    if str(overlay.get("revision_reason") or "") != REVISION_REASON:
        errors.append(f"revision_reason must be {REVISION_REASON}")
    selection = overlay.get("checkpoint_selection")
    if not isinstance(selection, MappingABC):
        errors.append("missing checkpoint_selection")
    else:
        if str(selection.get("rule") or "") != CHECKPOINT_SELECTION_RULE:
            errors.append(f"checkpoint_selection.rule must be {CHECKPOINT_SELECTION_RULE}")
        if int(selection.get("canonical_global_step") or 0) != CANONICAL_CHECKPOINT_STEP:
            errors.append("canonical_global_step must be 32")
    overrides = overlay.get("overrides") if isinstance(overlay.get("overrides"), MappingABC) else {}
    extra_sections = [key for key in overrides.keys() if str(key) not in ALLOWED_OVERRIDE_SECTIONS]
    if extra_sections:
        errors.append(f"overrides has disallowed sections: {extra_sections}")
    research = [key for key in overrides.keys() if str(key) in FORBIDDEN_RESEARCH_SECTIONS]
    if research:
        errors.append(f"must not change research sections: {research}")
    gpu = overrides.get("gpu") if isinstance(overrides.get("gpu"), MappingABC) else {}
    bad_gpu = [key for key in gpu.keys() if str(key) not in ALLOWED_GPU_KEYS]
    if bad_gpu:
        errors.append(f"gpu overrides disallowed: {bad_gpu}")
    if int(gpu.get("n_gpus") or 0) != EXPECTED_N_GPUS:
        errors.append(f"gpu.n_gpus must be {EXPECTED_N_GPUS}")
    if int(gpu.get("nnodes") or EXPECTED_NNODES) != EXPECTED_NNODES:
        errors.append("gpu.nnodes must stay 1")
    if int(gpu.get("tensor_model_parallel_size") or 0) != EXPECTED_TP:
        errors.append("gpu.tensor_model_parallel_size must stay 1 (not TP=2)")
    launch = overrides.get("launch") if isinstance(overrides.get("launch"), MappingABC) else {}
    bad_launch = [key for key in launch.keys() if str(key) not in ALLOWED_LAUNCH_KEYS]
    if bad_launch:
        errors.append(f"launch overrides disallowed: {bad_launch}")
    if str(launch.get("session_name") or "") != SESSION_NAME:
        errors.append("launch.session_name must be E014")
    systems = overrides.get("systems") if isinstance(overrides.get("systems"), MappingABC) else {}
    if not systems:
        errors.append("overrides.systems is required")
    else:
        bad_sys = [key for key in systems.keys() if str(key) not in ALLOWED_SYSTEMS_KEYS]
        if bad_sys:
            errors.append(f"systems overrides disallowed: {bad_sys}")
        ppo_max = int(systems.get("ppo_max_token_len_per_gpu") or 0)
        if ppo_max != CHOSEN_ENVELOPE:
            errors.append(f"systems.ppo_max_token_len_per_gpu={ppo_max} must be {CHOSEN_ENVELOPE}")
        if ppo_max >= MAX_MODEL_LEN:
            errors.append("E014 envelope must stay below max_model_len=32768")
    parent = overlay.get("parent") if isinstance(overlay.get("parent"), MappingABC) else {}
    if str(parent.get("sha256") or "") != EXPECTED_MAIN_SHA256:
        errors.append(f"parent.sha256 {parent.get('sha256')} != {EXPECTED_MAIN_SHA256}")
    canonical = overlay.get("canonical_envelope") if isinstance(
        overlay.get("canonical_envelope"), MappingABC
    ) else {}
    if str(canonical.get("sha256") or "") != EXPECTED_CANONICAL_SHA256:
        errors.append(
            f"canonical_envelope.sha256 {canonical.get('sha256')} != {EXPECTED_CANONICAL_SHA256}"
        )
    if int(canonical.get("ppo_max_token_len_per_gpu") or 0) != CHOSEN_ENVELOPE:
        errors.append("canonical_envelope.ppo_max_token_len_per_gpu must be 20480")
    e011 = overlay.get("e011_runtime") if isinstance(overlay.get("e011_runtime"), MappingABC) else {}
    if str(e011.get("sha256") or "") != EXPECTED_E011_OVERLAY_SHA256:
        errors.append(f"e011_runtime.sha256 {e011.get('sha256')} != {EXPECTED_E011_OVERLAY_SHA256}")
    e012 = overlay.get("e012_runtime") if isinstance(overlay.get("e012_runtime"), MappingABC) else {}
    if str(e012.get("sha256") or "") != EXPECTED_E012_OVERLAY_SHA256:
        errors.append(f"e012_runtime.sha256 {e012.get('sha256')} != {EXPECTED_E012_OVERLAY_SHA256}")
    if int(e012.get("ppo_max_token_len_per_gpu") or 0) != EXPECTED_E012_ENVELOPE:
        errors.append("e012_runtime.ppo_max_token_len_per_gpu must remain 18432")
    e013 = overlay.get("e013_runtime") if isinstance(overlay.get("e013_runtime"), MappingABC) else {}
    if str(e013.get("sha256") or "") != EXPECTED_E013_OVERLAY_SHA256:
        errors.append(f"e013_runtime.sha256 {e013.get('sha256')} != {EXPECTED_E013_OVERLAY_SHA256}")
    if repo_root is not None:
        errors.extend(historical_untouched_errors(Path(repo_root)))
    return errors


def overlay_lock_errors(repo_root: Path) -> list[str]:
    overlay_path = default_runtime_config_path(repo_root)
    lock_path = default_runtime_lock_path(repo_root)
    if not overlay_path.is_file():
        return [f"missing E014 overlay {overlay_path}"]
    if not lock_path.is_file():
        return [f"missing E014 lock {lock_path}"]
    actual = sha256_file(overlay_path)
    lock = load_json(lock_path)
    errors: list[str] = []
    if str(lock.get("sha256") or "") != actual:
        errors.append(f"E014 overlay sha256 {actual} != lock {lock.get('sha256')}")
    if str(lock.get("parent_sha256") or "") != EXPECTED_MAIN_SHA256:
        errors.append("E014 lock parent_sha256 mismatch")
    if str(lock.get("canonical_sha256") or "") != EXPECTED_CANONICAL_SHA256:
        errors.append("E014 lock canonical_sha256 mismatch")
    if int(lock.get("ppo_max_token_len_per_gpu") or 0) != CHOSEN_ENVELOPE:
        errors.append("E014 lock envelope must be 20480")
    return errors


def consume_runtime_overlay(
    *,
    repo_root: Path,
    overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = default_runtime_config_path(repo_root)
    payload = overlay if overlay is not None else load_json(path)
    errors = overlay_errors(payload, repo_root=repo_root)
    if overlay is None:
        errors.extend(overlay_lock_errors(repo_root))
    if errors:
        raise HardStopError("E014 runtime overlay contract failed", {"errors": errors})
    overrides = payload.get("overrides") or {}
    gpu = overrides.get("gpu") or {}
    launch = overrides.get("launch") or {}
    systems = overrides.get("systems") or {}
    selection = payload.get("checkpoint_selection") or {}
    return {
        "n_gpus": int(gpu["n_gpus"]),
        "nnodes": int(gpu.get("nnodes") or 1),
        "tensor_model_parallel_size": int(gpu["tensor_model_parallel_size"]),
        "device": str(gpu.get("device") or "2xA100-40GB"),
        "cuda_visible_devices": str(gpu.get("cuda_visible_devices") or "0,1"),
        "session_name": str(launch.get("session_name") or SESSION_NAME),
        "ppo_max_token_len_per_gpu": int(systems["ppo_max_token_len_per_gpu"]),
        "checkpoint_selection_rule": str(selection.get("rule") or CHECKPOINT_SELECTION_RULE),
        "canonical_global_step": int(
            selection.get("canonical_global_step") or CANONICAL_CHECKPOINT_STEP
        ),
        "revision_reason": str(payload.get("revision_reason") or REVISION_REASON),
        "overlay_sha256": sha256_file(path) if path.is_file() else None,
        "parent_sha256": str((payload.get("parent") or {}).get("sha256") or ""),
        "canonical_sha256": str((payload.get("canonical_envelope") or {}).get("sha256") or ""),
        "e011_sha256": str((payload.get("e011_runtime") or {}).get("sha256") or ""),
        "e012_sha256": str((payload.get("e012_runtime") or {}).get("sha256") or ""),
        "e013_sha256": str((payload.get("e013_runtime") or {}).get("sha256") or ""),
        "overlay_path": str(path),
        "expected_placement": expected_hybrid_placement(
            n_gpus=int(gpu["n_gpus"]),
            tensor_model_parallel_size=int(gpu["tensor_model_parallel_size"]),
        ),
        "train_batch_size": TRAIN_BATCH_SIZE,
        "n_candidates": N_CANDIDATES,
        "n_steps": MAIN_STEPS,
        "group_n": GROUP_N,
        "seed": SEED,
    }


def m6_candidate(checkpoint_root: Path) -> dict[str, Any]:
    return selected_m6_candidate(checkpoint_root)


def build_training_summary(*, evidence: Mapping[str, Any], checkpoint_root: Path) -> str:
    lines = [
        "# E014 canonical 32-step M5B GRPO",
        "",
        f"- READY_FOR_M6: **{evidence.get('READY_FOR_M6')}**",
        f"- status: **{evidence.get('status')}**",
        f"- stop_reason: {evidence.get('stop_reason')}",
        f"- chosen_envelope: `{evidence.get('chosen_envelope')}`",
        f"- overlay_sha256: `{evidence.get('overlay_sha256')}`",
        f"- canonical_sha256: `{evidence.get('canonical_sha256')}`",
        f"- parent_sha256: `{evidence.get('parent_sha256')}`",
        f"- realized_max_seq: {evidence.get('realized_max_seq')}",
        f"- margin: {evidence.get('margin')}",
        f"- OOM: {evidence.get('oom')}",
        f"- packing_covers: {evidence.get('packing_covers')}",
        f"- n_steps_completed: {evidence.get('n_steps_completed')}/{MAIN_STEPS}",
        f"- m6_candidate: `{evidence.get('m6_candidate')}`",
        f"- m6_exists: {evidence.get('m6_exists')}",
        f"- research_freeze_unmodified: **{evidence.get('research_freeze_unmodified')}**",
        f"- wandb: {evidence.get('wandb_url')}",
        f"- checkpoint_root: `{checkpoint_root}`",
        "",
        "Do not treat intermediate save_freq=8 shards as M6 candidates.",
        "",
    ]
    return "\n".join(lines) + "\n"


def ready_payload(
    *, ready: bool, reasons: Sequence[str], extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = {
        "READY_FOR_M6": bool(ready),
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "group_n": GROUP_N,
        "n_candidates": N_CANDIDATES,
        "n_steps": MAIN_STEPS,
        "chosen_envelope": CHOSEN_ENVELOPE,
        "checkpoint_selection": CHECKPOINT_SELECTION_RULE,
        "canonical_global_step": CANONICAL_CHECKPOINT_STEP,
    }
    if extra:
        payload.update(dict(extra))
    return payload
