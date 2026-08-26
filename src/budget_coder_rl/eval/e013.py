"""E013 systems-only 20480 envelope headroom. Does not mutate E012/E011/main."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.e012 import (
    ENVELOPE_CANDIDATES,
    EXPECTED_E011_OVERLAY_SHA256,
    packing_assert_covers,
    select_long_prompt_instance_ids,
)
from budget_coder_rl.eval.m4a import GROUP_N, load_json
from budget_coder_rl.eval.m4b import write_json
from budget_coder_rl.eval.m5a import (
    MAX_MODEL_LEN,
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
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E013"
MILESTONE = "M5B-E013-envelope-headroom"
SESSION_NAME = "E013"
SCHEMA_VERSION = "bcrl-stage1-e013-runtime-v1"
RUNTIME_CONFIG_RELPATH = "configs/experiments/stage1_m5_e013_runtime.json"
RUNTIME_LOCK_RELPATH = "configs/experiments/stage1_m5_e013_runtime.lock.json"
CANONICAL_ENVELOPE_RELPATH = "configs/experiments/stage1_canonical_execution_envelope.json"
CANONICAL_ENVELOPE_LOCK_RELPATH = (
    "configs/experiments/stage1_canonical_execution_envelope.lock.json"
)
CANONICAL_SCHEMA_VERSION = "bcrl-stage1-canonical-execution-envelope-v1"
E011_RUNTIME_RELPATH = "configs/experiments/stage1_m5b_e011_runtime.json"
E012_RUNTIME_RELPATH = "configs/experiments/stage1_m5_e012_runtime.json"
EXPECTED_E012_OVERLAY_SHA256 = (
    "a71f4557e6ca752715f81b21a18d3a70d810950af47d99c9bed7c9bfdf30fcf1"
)
EXPECTED_E012_ENVELOPE = 18432
CHOSEN_ENVELOPE = 20480
REVISION_REASON = "sequence_capacity_headroom"
REQUIRED_TASK = "dask__dask-10042"
E011_FAILURE_MAX_SEQ = 16751
E012_REALIZED_MAX_SEQ = 17253
STRESS_UNIQUE_TASKS = 2
STRESS_STEPS = 1
STRESS_TRAIN_BATCH_SIZE = 2
CHECKPOINT_RELPATH = "checkpoints/stage1_e013_envelope_stress"
COMPUTE_HOST_HINT = "n30158"
GPU_HEALTHY_PEAK_MIB = 38000
E012_ENVELOPE = EXPECTED_E012_ENVELOPE

ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "parent",
        "e011_runtime",
        "e012_runtime",
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


def default_canonical_envelope_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / CANONICAL_ENVELOPE_LOCK_RELPATH


def default_e012_runtime_path(repo_root: Path) -> Path:
    return Path(repo_root) / E012_RUNTIME_RELPATH


def default_e013_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_e012_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), "E012")


def default_stress_checkpoint_dir(data_root: Path | None = None) -> Path:
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


def e012_untouched_errors(repo_root: Path) -> list[str]:
    errors: list[str] = []
    overlay = default_e012_runtime_path(repo_root)
    if not overlay.is_file():
        return [f"missing immutable E012 overlay {overlay}"]
    digest = sha256_file(overlay)
    if digest != EXPECTED_E012_OVERLAY_SHA256:
        errors.append(
            f"E012 overlay sha256 {digest} != frozen {EXPECTED_E012_OVERLAY_SHA256}"
        )
    payload = load_json(overlay)
    systems = ((payload.get("overrides") or {}).get("systems") or {})
    if int(systems.get("ppo_max_token_len_per_gpu") or 0) != EXPECTED_E012_ENVELOPE:
        errors.append("E012 overlay envelope is no longer 18432")
    if str(payload.get("experiment_id") or "") != "E012":
        errors.append("E012 overlay experiment_id mutated")
    return errors


def overlay_errors(
    overlay: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_OVERLAY_KEYS]
    if extra:
        errors.append(f"E013 overlay has disallowed keys: {extra}")
    if str(overlay.get("schema_version") or "") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if str(overlay.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("experiment_id must be E013")
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
        errors.append("launch.session_name must be E013")
    systems = overrides.get("systems") if isinstance(overrides.get("systems"), MappingABC) else {}
    if not systems:
        errors.append("overrides.systems is required")
    else:
        bad_sys = [key for key in systems.keys() if str(key) not in ALLOWED_SYSTEMS_KEYS]
        if bad_sys:
            errors.append(f"systems overrides disallowed: {bad_sys}")
        ppo_max = int(systems.get("ppo_max_token_len_per_gpu") or 0)
        if ppo_max != CHOSEN_ENVELOPE:
            errors.append(
                f"systems.ppo_max_token_len_per_gpu={ppo_max} must be {CHOSEN_ENVELOPE}"
            )
        if ppo_max not in ENVELOPE_CANDIDATES:
            errors.append(f"envelope {ppo_max} not in {ENVELOPE_CANDIDATES}")
        if ppo_max >= MAX_MODEL_LEN:
            errors.append("E013 envelope must stay below max_model_len=32768")
    e011 = overlay.get("e011_runtime") if isinstance(overlay.get("e011_runtime"), MappingABC) else {}
    if str(e011.get("sha256") or "") != EXPECTED_E011_OVERLAY_SHA256:
        errors.append(
            f"e011_runtime.sha256 {e011.get('sha256')} != {EXPECTED_E011_OVERLAY_SHA256}"
        )
    e012 = overlay.get("e012_runtime") if isinstance(overlay.get("e012_runtime"), MappingABC) else {}
    if str(e012.get("sha256") or "") != EXPECTED_E012_OVERLAY_SHA256:
        errors.append(
            f"e012_runtime.sha256 {e012.get('sha256')} != {EXPECTED_E012_OVERLAY_SHA256}"
        )
    if int(e012.get("ppo_max_token_len_per_gpu") or 0) != EXPECTED_E012_ENVELOPE:
        errors.append("e012_runtime.ppo_max_token_len_per_gpu must remain 18432")
    parent = overlay.get("parent") if isinstance(overlay.get("parent"), MappingABC) else {}
    if str(parent.get("sha256") or "") != EXPECTED_MAIN_SHA256:
        errors.append(f"parent.sha256 {parent.get('sha256')} != {EXPECTED_MAIN_SHA256}")
    if repo_root is not None:
        main_path = Path(repo_root) / "configs/experiments/stage1_m5_main.json"
        if main_path.is_file() and sha256_file(main_path) != EXPECTED_MAIN_SHA256:
            errors.append("stage1_m5_main.json was edited")
        e011_path = Path(repo_root) / E011_RUNTIME_RELPATH
        if e011_path.is_file() and sha256_file(e011_path) != EXPECTED_E011_OVERLAY_SHA256:
            errors.append("E011 overlay was edited; E013 must not mutate it")
        errors.extend(e012_untouched_errors(Path(repo_root)))
    return errors


def overlay_lock_errors(repo_root: Path) -> list[str]:
    overlay_path = default_runtime_config_path(repo_root)
    lock_path = default_runtime_lock_path(repo_root)
    errors: list[str] = []
    if not overlay_path.is_file():
        return [f"missing E013 overlay {overlay_path}"]
    if not lock_path.is_file():
        return [f"missing E013 lock {lock_path}"]
    actual = sha256_file(overlay_path)
    lock = load_json(lock_path)
    if str(lock.get("sha256") or "") != actual:
        errors.append(f"E013 overlay sha256 {actual} != lock {lock.get('sha256')}")
    if str(lock.get("parent_sha256") or "") != EXPECTED_MAIN_SHA256:
        errors.append("E013 lock parent_sha256 mismatch")
    if str(lock.get("e011_sha256") or "") != EXPECTED_E011_OVERLAY_SHA256:
        errors.append("E013 lock e011_sha256 mismatch")
    if str(lock.get("e012_sha256") or "") != EXPECTED_E012_OVERLAY_SHA256:
        errors.append("E013 lock e012_sha256 mismatch")
    if int(lock.get("ppo_max_token_len_per_gpu") or 0) != CHOSEN_ENVELOPE:
        errors.append("E013 lock envelope must be 20480")
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
        raise HardStopError("E013 runtime overlay contract failed", {"errors": errors})
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
        "e011_sha256": str((payload.get("e011_runtime") or {}).get("sha256") or ""),
        "e012_sha256": str((payload.get("e012_runtime") or {}).get("sha256") or ""),
        "overlay_path": str(path),
        "expected_placement": expected_hybrid_placement(
            n_gpus=int(gpu["n_gpus"]),
            tensor_model_parallel_size=int(gpu["tensor_model_parallel_size"]),
        ),
        "stress_train_batch_size": STRESS_TRAIN_BATCH_SIZE,
        "freeze_train_batch_size": TRAIN_BATCH_SIZE,
    }


def select_headroom_instance_ids(ranked: Sequence[Mapping[str, Any]]) -> list[str]:
    return select_long_prompt_instance_ids(
        ranked, n=STRESS_UNIQUE_TASKS, required_id=REQUIRED_TASK
    )


def ranked_from_e012_audit(repo_root: Path) -> list[dict[str, Any]]:
    audit_path = default_e012_output_dir(repo_root) / "capacity_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"E012 capacity_audit.json missing at {audit_path}; do not regenerate E012"
        )
    audit = load_json(audit_path)
    ranked = audit.get("ranked_top") or audit.get("ranked") or []
    if not ranked:
        decision = audit.get("decision") or {}
        ids = list(decision.get("selected_instance_ids") or [])
        ranked = [{"instance_id": iid, "prompt_token_count": 0} for iid in ids]
    return [dict(item) for item in ranked]


def packing_smokes(*, max_token_len: int) -> dict[str, Any]:
    known = packing_assert_covers(
        max_token_len=max_token_len, max_seq_len=E011_FAILURE_MAX_SEQ
    )
    realized = packing_assert_covers(
        max_token_len=max_token_len, max_seq_len=E012_REALIZED_MAX_SEQ
    )
    return {
        "e011_failure_16751": known,
        "e012_realized_17253": realized,
        "covers": bool(known.get("covers")) and bool(realized.get("covers")),
    }


def peak_gpu_memory_mib(sampler_path: Path) -> dict[str, Any]:
    peaks = {0: 0, 1: 0}
    n = 0
    if not Path(sampler_path).is_file():
        return {"gpu0_peak_mib": None, "gpu1_peak_mib": None, "peak_mib": None, "n_samples": 0}
    with Path(sampler_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            for gpu in rec.get("gpus") or []:
                idx = int(gpu.get("index") or 0)
                used = int(gpu.get("memory_used_mi") or 0)
                peaks[idx] = max(peaks.get(idx, 0), used)
    peak = max(peaks.values()) if peaks else 0
    return {
        "gpu0_peak_mib": peaks.get(0),
        "gpu1_peak_mib": peaks.get(1),
        "peak_mib": peak,
        "n_samples": n,
        "healthy_threshold_mib": GPU_HEALTHY_PEAK_MIB,
    }


def memory_healthy(*, oom: bool, peak_mib: float | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if oom:
        reasons.append("OOM")
    if peak_mib is None:
        reasons.append("gpu_peak_missing")
    elif float(peak_mib) >= GPU_HEALTHY_PEAK_MIB:
        reasons.append(f"gpu_peak_{int(peak_mib)} >= {GPU_HEALTHY_PEAK_MIB}")
    return (not reasons, reasons)


def build_canonical_envelope_payload(
    *,
    overlay_sha256: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "status": "frozen",
        "experiment_id": "canonical-execution-envelope",
        "source_experiment_id": EXPERIMENT_ID,
        "ppo_max_token_len_per_gpu": CHOSEN_ENVELOPE,
        "revision_reason": REVISION_REASON,
        "source_overlay": RUNTIME_CONFIG_RELPATH,
        "source_overlay_sha256": overlay_sha256,
        "e012_kept": {
            "path": E012_RUNTIME_RELPATH,
            "sha256": EXPECTED_E012_OVERLAY_SHA256,
            "ppo_max_token_len_per_gpu": EXPECTED_E012_ENVELOPE,
            "note": "Immutable historical 18432 stress; not the canonical rerun envelope.",
        },
        "parent_freeze": {
            "path": "configs/experiments/stage1_m5_main.json",
            "sha256": EXPECTED_MAIN_SHA256,
            "ppo_max_token_len_per_gpu_in_freeze_json": 16384,
            "note": "Research freeze file stays 16384; Hydra apply uses this execution overlay.",
        },
        "gpu": {
            "n_gpus": EXPECTED_N_GPUS,
            "tensor_model_parallel_size": EXPECTED_TP,
            "device": "2xA100-40GB",
        },
        "checkpoint_selection": {
            "rule": CHECKPOINT_SELECTION_RULE,
            "canonical_global_step": CANONICAL_CHECKPOINT_STEP,
        },
        "canonical_32_step_not_started": True,
        "not_m6_candidate": True,
        "evidence": {
            "realized_max_seq": evidence.get("realized_max_seq"),
            "margin": evidence.get("margin"),
            "gpu0_peak_mib": evidence.get("gpu0_peak_mib"),
            "gpu1_peak_mib": evidence.get("gpu1_peak_mib"),
            "oom": evidence.get("oom"),
            "n_steps_completed": evidence.get("n_steps_completed"),
        },
        "notes": [
            "Use this envelope for the next canonical 32-step rerun overlay.",
            "Do not treat E013 stress checkpoints as M6 candidates.",
            "Do not edit E012 artifacts or the 18432 overlay.",
        ],
    }


def write_canonical_envelope(
    repo_root: Path,
    *,
    overlay_sha256: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = build_canonical_envelope_payload(
        overlay_sha256=overlay_sha256, evidence=evidence
    )
    path = default_canonical_envelope_path(repo_root)
    lock_path = default_canonical_envelope_lock_path(repo_root)
    write_json(path, payload)
    digest = sha256_file(path)
    lock = {
        "path": CANONICAL_ENVELOPE_RELPATH,
        "sha256": digest,
        "ppo_max_token_len_per_gpu": CHOSEN_ENVELOPE,
        "source_experiment_id": EXPERIMENT_ID,
        "source_overlay_sha256": overlay_sha256,
        "e012_sha256": EXPECTED_E012_OVERLAY_SHA256,
        "parent_sha256": EXPECTED_MAIN_SHA256,
        "note": "Frozen execution envelope for the future canonical 32-step rerun.",
    }
    write_json(lock_path, lock)
    return {"payload": payload, "lock": lock, "sha256": digest}


def build_headroom_summary(*, evidence: Mapping[str, Any]) -> str:
    lines = [
        "# E013 systems-only 20480 envelope headroom",
        "",
        f"- READY_TO_FREEZE_CANONICAL: **{evidence.get('READY_TO_FREEZE_CANONICAL')}**",
        f"- status: **{evidence.get('status')}**",
        f"- stop_reason: {evidence.get('stop_reason')}",
        f"- chosen_envelope: `{evidence.get('chosen_envelope')}`",
        f"- e012_envelope_untouched: `{EXPECTED_E012_ENVELOPE}` sha `{EXPECTED_E012_OVERLAY_SHA256}`",
        f"- overlay_sha256: `{evidence.get('overlay_sha256')}`",
        f"- realized_max_seq: {evidence.get('realized_max_seq')}",
        f"- margin: {evidence.get('margin')}",
        f"- OOM: {evidence.get('oom')}",
        f"- gpu0_peak_mib: {evidence.get('gpu0_peak_mib')}",
        f"- gpu1_peak_mib: {evidence.get('gpu1_peak_mib')}",
        f"- memory_healthy: {evidence.get('memory_healthy')}",
        f"- dask_in_episodes: {evidence.get('dask_in_episodes')}",
        f"- n_steps_completed: {evidence.get('n_steps_completed')}",
        f"- research_freeze_unmodified: **{evidence.get('research_freeze_unmodified')}**",
        f"- wandb: {evidence.get('wandb_url')}",
        "",
        "E012 overlay/artifacts were not modified. 32-step canonical rerun was not started.",
        "",
    ]
    return "\n".join(lines) + "\n"


def ready_payload(
    *, ready: bool, reasons: Sequence[str], extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = {
        "READY_TO_FREEZE_CANONICAL": bool(ready),
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "not_m6_candidate": True,
        "canonical_32_step_not_started": True,
        "seed": SEED,
        "stress_train_batch_size": STRESS_TRAIN_BATCH_SIZE,
        "freeze_train_batch_size": TRAIN_BATCH_SIZE,
        "group_n": GROUP_N,
        "chosen_envelope": CHOSEN_ENVELOPE,
        "e012_envelope_kept": EXPECTED_E012_ENVELOPE,
    }
    if extra:
        payload.update(dict(extra))
    return payload
