"""E012 systems-only capacity overlay: raise actor packing envelope.

Does not edit stage1_m5_main.json, the E011 overlay, or E011 artifacts.
Does not retune reward / prompt / tools / parser / budget / sampling / G.
"""

from __future__ import annotations

import json
import math
import os
import socket
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.m4a import GROUP_N, OBS_TOKENS_LIMIT, load_json
from budget_coder_rl.eval.m4b import PROMPT_LENGTH, RESPONSE_LENGTH, write_json
from budget_coder_rl.eval.m5a import (
    MAX_MODEL_LEN,
    SEED,
    TOKEN_SLACK,
    TRAIN_BATCH_SIZE,
    default_output_dir,
)
from budget_coder_rl.eval.m5b import (
    CHECKPOINT_SELECTION_RULE,
    CANONICAL_CHECKPOINT_STEP,
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_NNODES,
    EXPECTED_TP,
    HardStopError,
    expected_hybrid_placement,
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E012"
MILESTONE = "M5B-E012-capacity"
SESSION_NAME = "E012"
OUTPUT_ENV = "BCRL_M5_OUTPUT_DIR"
SCHEMA_VERSION = "bcrl-stage1-e012-runtime-v1"
RUNTIME_CONFIG_RELPATH = "configs/experiments/stage1_m5_e012_runtime.json"
RUNTIME_LOCK_RELPATH = "configs/experiments/stage1_m5_e012_runtime.lock.json"
E011_RUNTIME_RELPATH = "configs/experiments/stage1_m5b_e011_runtime.json"
EXPECTED_E011_OVERLAY_SHA256 = (
    "69f149b74e0a39276655bc0f832a16c0535724ef4a302f18a6d0a8aedc6a2804"
)
REVISION_REASON = "sequence_capacity_exceeded"
REQUIRED_TASK = "dask__dask-10042"
E011_FAILURE_MAX_SEQ = 16751
E011_FAILURE_PROMPT = 12811
STRESS_UNIQUE_TASKS = 8
STRESS_STEPS = 2
ENVELOPE_CANDIDATES = (18432, 20480)
LONG_PROMPT_MIN = 8192
CHECKPOINT_RELPATH = "checkpoints/stage1_e012_capacity_stress"
COMPUTE_HOST_HINT = "n30158"

ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "parent",
        "e011_runtime",
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

PINNED_CAPACITY_NOTES = [
    "Pinned veRL rearrange_micro_batches asserts per-sequence max_seq_len "
    "<= ppo_max_token_len_per_gpu * ulysses_sp (sp=1). Nested input_ids use "
    "max(per-sample length); E011 reported tensor(16751).",
    "BCRL training_proxy = prompt + policy + observation matches that length.",
    "E011 actor.ppo_max_token_len_per_gpu=16384 was the blocker. "
    "rollout.log_prob_max_token_len_per_gpu is already 32768 via apply_m5_train_config. "
    "ref.log_prob remains 16384 but unused (use_kl_loss=false, LoRA ref_in_actor).",
    "M5A observed max 15251 is not a stochastic hard bound: same dask task "
    "reached 16751 in E011 from extra inserted observation tokens.",
    "Collate upper bound prompt_length+response_length=32768 is excluded "
    "(M5A packing-OOM risk). Do not jump to max_model_len to 'never fail'.",
]


def default_runtime_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_CONFIG_RELPATH


def default_runtime_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_LOCK_RELPATH


def default_e011_runtime_path(repo_root: Path) -> Path:
    return Path(repo_root) / E011_RUNTIME_RELPATH


def default_e012_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


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


def choose_envelope(needed: int) -> int:
    """Smallest 18432/20480 candidate covering needed. Refuse 32768."""
    need = int(needed)
    if need <= 0:
        raise ValueError(f"invalid needed envelope {need}")
    for cap in ENVELOPE_CANDIDATES:
        if int(cap) >= need:
            return int(cap)
    raise ValueError(
        f"needed {need} exceeds max candidate {ENVELOPE_CANDIDATES[-1]}; "
        f"do not jump to max_model_len={MAX_MODEL_LEN}"
    )


def select_long_prompt_instance_ids(
    ranked: Sequence[Mapping[str, Any]],
    *,
    n: int = STRESS_UNIQUE_TASKS,
    required_id: str = REQUIRED_TASK,
) -> list[str]:
    """Top-n by prompt length, forcing ``required_id``. Systems selection only."""
    rows = [item for item in ranked if str(item.get("instance_id") or "").strip()]
    if not rows:
        raise ValueError("ranked prompt list is empty")
    counts = {
        str(item["instance_id"]): float(item.get("prompt_token_count") or 0)
        for item in rows
    }
    ordered = [
        str(item["instance_id"])
        for item in sorted(rows, key=lambda row: -float(row.get("prompt_token_count") or 0))
    ]
    seen: list[str] = []
    for instance_id in ordered:
        if instance_id not in seen:
            seen.append(instance_id)
    if required_id not in seen:
        raise ValueError(f"required task {required_id} missing from ranking")
    selected = seen[: int(n)]
    if required_id not in selected:
        selected = seen[: max(0, int(n) - 1)] + [required_id]
    selected = sorted(selected, key=lambda iid: -counts.get(iid, 0.0))
    if len(selected) != int(n):
        raise ValueError(f"expected {n} long-prompt tasks, got {len(selected)}")
    if required_id not in selected:
        raise ValueError(f"{required_id} not in selected {selected}")
    return selected


def repeat_ids_for_steps(
    instance_ids: Sequence[str],
    *,
    n_steps: int = STRESS_STEPS,
) -> list[str]:
    unique = [str(item) for item in instance_ids]
    if not unique:
        raise ValueError("instance_ids is empty")
    return unique * int(n_steps)


def overlay_errors(
    overlay: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_OVERLAY_KEYS]
    if extra:
        errors.append(f"E012 overlay has disallowed keys: {extra}")
    if str(overlay.get("schema_version") or "") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if str(overlay.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("experiment_id must be E012")
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
        errors.append("launch.session_name must be E012")
    systems = overrides.get("systems") if isinstance(overrides.get("systems"), MappingABC) else {}
    if not systems:
        errors.append("overrides.systems is required")
    else:
        bad_sys = [key for key in systems.keys() if str(key) not in ALLOWED_SYSTEMS_KEYS]
        if bad_sys:
            errors.append(f"systems overrides disallowed: {bad_sys}")
        ppo_max = int(systems.get("ppo_max_token_len_per_gpu") or 0)
        if ppo_max not in ENVELOPE_CANDIDATES:
            errors.append(
                f"systems.ppo_max_token_len_per_gpu={ppo_max} not in {ENVELOPE_CANDIDATES}"
            )
        if ppo_max <= 16384:
            errors.append("E012 envelope must exceed frozen 16384")
        if ppo_max >= MAX_MODEL_LEN:
            errors.append("E012 envelope must stay below max_model_len=32768")
    e011 = overlay.get("e011_runtime") if isinstance(overlay.get("e011_runtime"), MappingABC) else {}
    if str(e011.get("sha256") or "") != EXPECTED_E011_OVERLAY_SHA256:
        errors.append(
            f"e011_runtime.sha256 {e011.get('sha256')} != {EXPECTED_E011_OVERLAY_SHA256}"
        )
    parent = overlay.get("parent") if isinstance(overlay.get("parent"), MappingABC) else {}
    if str(parent.get("sha256") or "") != EXPECTED_MAIN_SHA256:
        errors.append(f"parent.sha256 {parent.get('sha256')} != {EXPECTED_MAIN_SHA256}")
    if repo_root is not None:
        main_path = Path(repo_root) / "configs/experiments/stage1_m5_main.json"
        if main_path.is_file() and sha256_file(main_path) != EXPECTED_MAIN_SHA256:
            errors.append("stage1_m5_main.json was edited")
        e011_path = default_e011_runtime_path(Path(repo_root))
        if e011_path.is_file() and sha256_file(e011_path) != EXPECTED_E011_OVERLAY_SHA256:
            errors.append("E011 overlay was edited; E012 must not mutate it")
    return errors


def overlay_lock_errors(repo_root: Path) -> list[str]:
    overlay_path = default_runtime_config_path(repo_root)
    lock_path = default_runtime_lock_path(repo_root)
    errors: list[str] = []
    if not overlay_path.is_file():
        return [f"missing E012 overlay {overlay_path}"]
    if not lock_path.is_file():
        return [f"missing E012 lock {lock_path}"]
    actual = sha256_file(overlay_path)
    lock = load_json(lock_path)
    if str(lock.get("sha256") or "") != actual:
        errors.append(f"E012 overlay sha256 {actual} != lock {lock.get('sha256')}")
    if str(lock.get("parent_sha256") or "") != EXPECTED_MAIN_SHA256:
        errors.append("E012 lock parent_sha256 mismatch")
    if str(lock.get("e011_sha256") or "") != EXPECTED_E011_OVERLAY_SHA256:
        errors.append("E012 lock e011_sha256 mismatch")
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
        raise HardStopError("E012 runtime overlay contract failed", {"errors": errors})
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
        "overlay_path": str(path),
        "expected_placement": expected_hybrid_placement(
            n_gpus=int(gpu["n_gpus"]),
            tensor_model_parallel_size=int(gpu["tensor_model_parallel_size"]),
        ),
    }


def load_episode_proxy_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tokens = record.get("tokens") if isinstance(record.get("tokens"), MappingABC) else {}
            identity = record.get("identity") if isinstance(record.get("identity"), MappingABC) else {}
            prompt = tokens.get("prompt_token_count")
            if prompt is None:
                continue
            prompt_n = int(prompt)
            policy_n = int(tokens.get("policy_token_count") or 0)
            obs_n = int(tokens.get("observation_token_count") or 0)
            rows.append(
                {
                    "instance_id": str(identity.get("instance_id") or ""),
                    "prompt_token_count": prompt_n,
                    "policy_token_count": policy_n,
                    "observation_token_count": obs_n,
                    "policy_plus_obs": policy_n + obs_n,
                    "training_seq_proxy": prompt_n + policy_n + obs_n,
                    "source": str(path),
                }
            )
    return rows


def decide_capacity(
    *,
    ranked_prompts: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    slack: int = TOKEN_SLACK,
    failure_max_seq: int = E011_FAILURE_MAX_SEQ,
) -> dict[str, Any]:
    prompt_values = [float(item.get("prompt_token_count") or 0) for item in ranked_prompts]
    prompt_max = max(prompt_values) if prompt_values else 0.0
    long_rows = [
        item
        for item in episode_rows
        if float(item.get("prompt_token_count") or 0) >= LONG_PROMPT_MIN
    ]
    pool = long_rows or list(episode_rows)
    policy_obs_values = [float(item.get("policy_plus_obs") or 0) for item in pool]
    policy_obs_max = max(policy_obs_values) if policy_obs_values else 0.0
    needed_failure = int(failure_max_seq) + int(slack)
    needed_tail = int(math.ceil(prompt_max + policy_obs_max + float(slack)))
    needed = max(needed_failure, needed_tail)
    collate_bound = int(PROMPT_LENGTH) + int(RESPONSE_LENGTH)
    agentloop_tight_max = int(prompt_max) + min(
        int(RESPONSE_LENGTH),
        6 * 2048 + int(OBS_TOKENS_LIMIT) + 6 * 128,
    )
    envelope = choose_envelope(needed)
    selected = select_long_prompt_instance_ids(ranked_prompts)
    return {
        "failure_max_seq": int(failure_max_seq),
        "token_slack": int(slack),
        "needed_from_failure": needed_failure,
        "prompt_max_256": prompt_max,
        "policy_plus_obs_max_on_long": policy_obs_max,
        "n_long_prompt_episodes": len(long_rows),
        "needed_from_prompt_tail": needed_tail,
        "needed": needed,
        "collate_bound_excluded": collate_bound,
        "agentloop_conservative_prompt_max": agentloop_tight_max,
        "max_model_len_excluded": MAX_MODEL_LEN,
        "candidates": list(ENVELOPE_CANDIDATES),
        "chosen_envelope": envelope,
        "margin_vs_failure": envelope - int(failure_max_seq),
        "selected_instance_ids": selected,
        "rationale": (
            f"needed=max(16751+{slack}={needed_failure}, "
            f"prompt_max+policy_obs_max+slack={needed_tail})={needed}; "
            f"smallest candidate in {list(ENVELOPE_CANDIDATES)} is {envelope}. "
            "Not collate 32768."
        ),
    }


def packing_assert_covers(
    *,
    max_token_len: int,
    max_seq_len: int = E011_FAILURE_MAX_SEQ,
) -> dict[str, Any]:
    """Call pinned veRL rearrange_micro_batches with a synthetic 16751 sequence."""
    import torch

    from verl.utils.seqlen_balancing import rearrange_micro_batches

    seq = int(max_seq_len)
    envelope = int(max_token_len)
    attention_mask = torch.ones(1, seq, dtype=torch.int64)
    input_ids = torch.zeros(1, seq, dtype=torch.int64)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    ok = False
    error = None
    try:
        rearrange_micro_batches(batch, max_token_len=envelope)
        ok = True
    except AssertionError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        ok = False
    return {
        "max_seq_len": seq,
        "max_token_len": envelope,
        "covers": bool(ok) and envelope >= seq,
        "error": error,
        "note": "per-sequence packing assert; synthetic non-nested attention_mask width",
    }


def build_overlay_payload(*, ppo_max_token_len_per_gpu: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "experiment_id": EXPERIMENT_ID,
        "parent": {
            "path": "configs/experiments/stage1_m5_main.json",
            "sha256": EXPECTED_MAIN_SHA256,
            "lock_path": "configs/experiments/stage1_m5_main.lock.json",
            "schema_version": "bcrl-stage1-m5-main-v1",
        },
        "e011_runtime": {
            "path": E011_RUNTIME_RELPATH,
            "sha256": EXPECTED_E011_OVERLAY_SHA256,
        },
        "revision_reason": REVISION_REASON,
        "allowed_override_scope": ["gpu", "launch", "systems"],
        "checkpoint_selection": {
            "rule": CHECKPOINT_SELECTION_RULE,
            "canonical_global_step": CANONICAL_CHECKPOINT_STEP,
            "forbidden": "post-hoc train/W&B curve picking",
        },
        "overrides": {
            "gpu": {
                "n_gpus": EXPECTED_N_GPUS,
                "nnodes": EXPECTED_NNODES,
                "tensor_model_parallel_size": EXPECTED_TP,
                "device": "2xA100-40GB",
                "cuda_visible_devices": "0,1",
            },
            "launch": {
                "session_name": SESSION_NAME,
                "preferred": "tmux",
                "fallback": "sbatch",
            },
            "systems": {
                "ppo_max_token_len_per_gpu": int(ppo_max_token_len_per_gpu),
            },
        },
        "notes": [
            "Does not edit stage1_m5_main.json or the E011 overlay.",
            "Only actor packing envelope changes. Research/training semantics are frozen.",
            "rollout.log_prob_max_token_len_per_gpu stays 32768 (already matching).",
            "Stress checkpoints are not M6 candidates. Canonical E012 main is not started here.",
            *PINNED_CAPACITY_NOTES,
        ],
    }


def write_overlay_and_lock(
    repo_root: Path,
    *,
    ppo_max_token_len_per_gpu: int,
) -> dict[str, Any]:
    payload = build_overlay_payload(ppo_max_token_len_per_gpu=ppo_max_token_len_per_gpu)
    errors = overlay_errors(payload, repo_root=repo_root)
    if errors:
        raise HardStopError("E012 overlay invalid", {"errors": errors})
    overlay_path = default_runtime_config_path(repo_root)
    lock_path = default_runtime_lock_path(repo_root)
    write_json(overlay_path, payload)
    digest = sha256_file(overlay_path)
    lock = {
        "path": RUNTIME_CONFIG_RELPATH,
        "sha256": digest,
        "parent_path": "configs/experiments/stage1_m5_main.json",
        "parent_sha256": EXPECTED_MAIN_SHA256,
        "e011_path": E011_RUNTIME_RELPATH,
        "e011_sha256": EXPECTED_E011_OVERLAY_SHA256,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "revision_reason": REVISION_REASON,
        "ppo_max_token_len_per_gpu": int(ppo_max_token_len_per_gpu),
        "scope": "systems-capacity only",
        "note": "Do not edit stage1_m5_main.json. Do not overwrite E011 artifacts.",
    }
    write_json(lock_path, lock)
    return {"overlay": payload, "lock": lock, "sha256": digest}


def tokenize_prompt_rows(
    *,
    parquet_path: Path,
    ordered_ids: Sequence[str],
    tokenizer: Any,
    episode_prompt_by_id: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    import pandas as pd

    from budget_coder_rl.budget.state import BudgetState
    from budget_coder_rl.eval.m4a import BUDGET_VISIBLE
    from budget_coder_rl.protocol.prompt import (
        build_stage1_messages,
        extract_issue_text,
        policy_safe_repo,
    )

    frame = pd.read_parquet(parquet_path)
    extra_by_id: dict[str, dict[str, Any]] = {}
    prompt_by_id: dict[str, Any] = {}
    for extra, prompt in zip(frame["extra_info"].tolist(), frame["prompt"].tolist()):
        payload = dict(extra) if isinstance(extra, MappingABC) else {}
        instance_id = str(payload.get("instance_id") or "").strip()
        if instance_id and instance_id not in extra_by_id:
            extra_by_id[instance_id] = payload
            prompt_by_id[instance_id] = prompt
    measured = dict(episode_prompt_by_id or {})
    rows: list[dict[str, Any]] = []
    budget = BudgetState(
        obs_tokens_used=0,
        obs_tokens_limit=OBS_TOKENS_LIMIT,
        turns_used=0,
        turns_limit=6,
    )
    for instance_id in ordered_ids:
        iid = str(instance_id)
        extra = extra_by_id.get(iid) or {}
        if iid in measured:
            rows.append(
                {
                    "instance_id": iid,
                    "prompt_token_count": int(measured[iid]),
                    "source": "episode",
                }
            )
            continue
        issue = extract_issue_text(prompt_by_id.get(iid))
        messages = build_stage1_messages(
            issue,
            repo=policy_safe_repo(extra),
            budget_state=budget,
            budget_visible=BUDGET_VISIBLE,
        )
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "instance_id": iid,
                "prompt_token_count": int(len(list(ids))),
                "source": "tokenizer",
            }
        )
    return rows


def build_capacity_summary(*, evidence: Mapping[str, Any]) -> str:
    lines = [
        "# E012 systems-only capacity check",
        "",
        f"- READY_FOR_E012_MAIN: **{evidence.get('READY_FOR_E012_MAIN')}**",
        f"- status: **{evidence.get('status')}**",
        f"- stop_reason: {evidence.get('stop_reason')}",
        f"- chosen_envelope: `{evidence.get('chosen_envelope')}`",
        f"- overlay_sha256: `{evidence.get('overlay_sha256')}`",
        f"- parent_sha256: `{evidence.get('parent_sha256')}`",
        f"- e011_sha256: `{evidence.get('e011_sha256')}`",
        f"- realized_max_seq: {evidence.get('realized_max_seq')}",
        f"- margin: {evidence.get('margin')}",
        f"- OOM: {evidence.get('oom')}",
        f"- packing_assert_covers_16751: {evidence.get('packing_covers')}",
        f"- dask_in_episodes: {evidence.get('dask_in_episodes')}",
        f"- n_steps_completed: {evidence.get('n_steps_completed')}",
        f"- research_freeze_unmodified: **{evidence.get('research_freeze_unmodified')}**",
        f"- wandb: {evidence.get('wandb_url')}",
        "",
        "Stress checkpoints are not M6 candidates. 32-step E012 main was not started.",
        "",
    ]
    return "\n".join(lines) + "\n"


def ready_payload(*, ready: bool, reasons: Sequence[str], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "READY_FOR_E012_MAIN": bool(ready),
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "not_m6_candidate": True,
        "canonical_32_step_not_started": True,
        "seed": SEED,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "group_n": GROUP_N,
    }
    if extra:
        payload.update(dict(extra))
    return payload
