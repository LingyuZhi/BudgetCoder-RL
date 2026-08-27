"""E018 scaled-M6 frozen held-out eval namespace.

Consumes the immutable E015 freeze (stage1_m6_eval.json). Does not edit
E015/E014/E017 artifacts, parser, reward, or AgentLoop YAML. Candidate is
only E017 global_step_275.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.episode import summarize_episodes
from budget_coder_rl.eval.m3b import compact_review_case
from budget_coder_rl.eval.m4b import write_json
from budget_coder_rl.eval.m4c import (
    VLLM_LORA_INT_ID,
    adapter_payload_summary,
)
from budget_coder_rl.eval.m5a import default_output_dir
from budget_coder_rl.eval.m6 import (
    AGENT_LOOP_CONFIG_RELPATH,
    BUDGETS,
    CASE_RULES,
    EVAL_NAME,
    EXPECTED_EVAL_SHA256,
    EXPECTED_M3C_FREEZE_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    N_DEV_GROUPS,
    N_DEV_TASKS,
    PAIRED_SEED_BASE,
    PRIMARY_TRAINING_BUDGET,
    SMALL_REPO_N,
    SMOKE_N_TASKS,
    compare_pair,
    load_json,
    lock_errors as e015_lock_errors,
    obs_limit_from_row,
    paired_summary_stats,
    repo_from_row,
    instance_id_from_row,
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E018"
MILESTONE = "E018-SCALED-M6-EVAL"
SCHEMA_VERSION = "bcrl-stage1-m6-e018-v1"
SESSION_NAME = "E018"
WANDB_EXPERIMENT_NAME = "E018-scaled-m6-eval"
WANDB_PROJECT = "budget-coder-rl"
OVERLAY_RELPATH = "configs/experiments/stage1_m6_e018.json"
OVERLAY_LOCK_RELPATH = "configs/experiments/stage1_m6_e018.lock.json"
PARENT_RELPATH = "configs/experiments/stage1_m6_eval.json"
CHECKPOINT_RELPATH = "checkpoints/stage1_m5_scaled_e017"
TRAJECTORY_RELPATH = "trajectories/m6/E018"
E015_TRAJECTORY_RELPATH = "trajectories/m6/E015"
CANONICAL_RL_STEP = 275
FORBIDDEN_RL_STEPS = (8, 16, 24, 32, 256)
N_GPUS = 2
TENSOR_MODEL_PARALLEL_SIZE = 1
LORA_RANK = 16
LORA_ALPHA = 16
OUTPUT_ENV = "BCRL_E018_OUTPUT_DIR"

EXPECTED_OVERLAY_SHA256 = (
    "8749e58fbe88ce2560e37f4a32861e4a1c8ffc739136c2389dc582c055bb15f9"
)

FORBIDDEN_OUTPUT_IDS = (
    "E001",
    "E006",
    "E011",
    "E012",
    "E013",
    "E014",
    "E015",
    "E016",
    "E017",
)
FORBIDDEN_CHECKPOINT_MARKERS = (
    "stage1_m5_e014",
    "stage1_m5_main",
    "stage1_m5_scaled_e016",
    "global_step_256",
    "global_step_32",
)

ALLOWED_OVERLAY_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "eval_only",
        "not_training",
        "do_not_enter_m7",
        "inherits",
        "parent",
        "allowed_override_scope",
        "frozen_from_parent",
        "overrides",
        "prototype_overlay",
        "primary_comparison",
        "auxiliary_comparison",
        "scale_correction_comparison",
        "notes",
    }
)
ALLOWED_OVERRIDE_SECTIONS = frozenset(
    {"namespace", "checkpoint", "conditions", "wandb", "launch"}
)
FORBIDDEN_RESEARCH_SECTIONS = frozenset(
    {"sampling", "budgets", "evaluator", "reward", "parser", "agent_loop"}
)

KNOWN_CONDITION_IDS = frozenset({"B0", "B1", "M_scaled", "M1", "M1_proto"})

CONDITIONS = (
    {
        "id": "B0",
        "policy": "base",
        "budget_visible": False,
        "label": "Base Hidden",
    },
    {
        "id": "B1",
        "policy": "base",
        "budget_visible": True,
        "label": "Base Visible",
    },
    {
        "id": "M_scaled",
        "policy": "rl",
        "budget_visible": True,
        "label": "Scaled RL Visible",
        "checkpoint_step": CANONICAL_RL_STEP,
    },
)

TABLE_CONDITIONS = CONDITIONS + (
    {
        "id": "M1_proto",
        "policy": "rl",
        "budget_visible": True,
        "label": "Prototype RL",
        "source_experiment": "E015",
    },
)

EVAL_CRITICAL_RELPATHS = (
    "configs/agent_loop/repo_exploration_m3c.yaml",
    "configs/experiments/stage1_m6_eval.json",
    "src/budget_coder_rl/agent_loop/repo_exploration.py",
    "src/budget_coder_rl/agent_loop/tokenization.py",
    "src/budget_coder_rl/protocol/parser.py",
    "src/budget_coder_rl/budget/state.py",
    "src/budget_coder_rl/eval/localization.py",
    "scripts/eval/score_episodes.py",
)


def default_overlay_path(repo_root: Path) -> Path:
    return Path(repo_root) / OVERLAY_RELPATH


def default_overlay_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / OVERLAY_LOCK_RELPATH


def default_e018_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_trace_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / TRAJECTORY_RELPATH


def default_e015_trace_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / E015_TRAJECTORY_RELPATH


def default_rl_checkpoint_dir(data_root: Path | None = None) -> Path:
    return (
        Path(data_root or bcrl_data_root())
        / CHECKPOINT_RELPATH
        / f"global_step_{CANONICAL_RL_STEP}"
    )


def default_rl_actor_dir(data_root: Path | None = None) -> Path:
    return default_rl_checkpoint_dir(data_root) / "actor"


def condition_spec(condition_id: str) -> dict[str, Any]:
    for item in TABLE_CONDITIONS:
        if item["id"] == condition_id:
            return dict(item)
    raise KeyError(condition_id)


def condition_id_from_row(row: Mapping[str, Any]) -> str | None:
    """Prefer explicit condition_id. Do not map every RL row to M1."""
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    explicit = str(condition.get("condition_id") or row.get("condition_id") or "")
    if explicit in KNOWN_CONDITION_IDS:
        return explicit
    policy = str(condition.get("policy") or row.get("policy") or "").strip().lower()
    visible = condition.get("budget_visible")
    if visible is None:
        budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
        visible = budget.get("budget_visible")
    source = str(row.get("source_experiment") or "")
    if policy == "rl" and source == "E015":
        return "M1_proto"
    if policy == "rl":
        return "M_scaled"
    if visible is True:
        return "B1"
    if visible is False:
        return "B0"
    return None


def checkpoint_path_errors(path: Path | str) -> list[str]:
    text = str(path)
    errors: list[str] = []
    if "global_step_" not in text:
        errors.append("checkpoint path must contain global_step_")
    if f"global_step_{CANONICAL_RL_STEP}" not in text:
        errors.append(f"E018 candidate must be global_step_{CANONICAL_RL_STEP}")
    for step in FORBIDDEN_RL_STEPS:
        token = f"global_step_{step}"
        if token in text and f"global_step_{CANONICAL_RL_STEP}" not in text:
            errors.append(f"refusing forbidden RL checkpoint {text}")
        if token in text and step != CANONICAL_RL_STEP:
            if f"global_step_{CANONICAL_RL_STEP}" not in text:
                errors.append(f"refusing forbidden step {step} in {text}")
    if "global_step_256" in text and "global_step_275" not in text:
        errors.append(f"refusing E017 intermediate checkpoint {text}")
    if "stage1_m5_e014" in text:
        errors.append("refusing E014 prototype checkpoint as E018 candidate")
    if "stage1_m5_scaled_e017" not in text:
        errors.append("E018 candidate must live under stage1_m5_scaled_e017")
    return errors


def actor_dir_errors(path: Path, *, data_root: Path | None = None) -> list[str]:
    errors = checkpoint_path_errors(path)
    expected = default_rl_actor_dir(data_root)
    resolved = Path(path).resolve() if Path(path).exists() else Path(path)
    if "actor" not in Path(path).name and not str(path).rstrip("/").endswith("/actor"):
        errors.append("checkpoint path must point at the actor/ subdirectory")
    if Path(path).exists() and resolved != expected.resolve():
        errors.append(f"actor path {resolved} != expected {expected}")
    return errors


def latest_iteration_errors(checkpoint_path: Path) -> list[str]:
    path = Path(checkpoint_path)
    candidates = [
        path / "latest_checkpointed_iteration.txt",
        path.parent / "latest_checkpointed_iteration.txt",
        path.parent.parent / "latest_checkpointed_iteration.txt",
    ]
    marker = next((item for item in candidates if item.is_file()), None)
    if marker is None:
        return [f"missing latest_checkpointed_iteration.txt near {checkpoint_path}"]
    text = marker.read_text(encoding="utf-8").strip()
    if text != str(CANONICAL_RL_STEP):
        return [f"latest_checkpointed_iteration.txt={text!r} != {CANONICAL_RL_STEP}"]
    return []


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            return [f"refusing to write into {experiment_id} artifact directory {forbidden}"]
    return []


def overlay_errors(
    overlay: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_OVERLAY_KEYS]
    if extra:
        errors.append(f"E018 overlay has disallowed keys: {extra}")
    if str(overlay.get("schema_version") or "") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if str(overlay.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("experiment_id must be E018")
    if str(overlay.get("milestone") or "") != MILESTONE:
        errors.append(f"milestone must be {MILESTONE}")
    if overlay.get("eval_only") is not True:
        errors.append("eval_only must be true")
    if overlay.get("not_training") is not True:
        errors.append("not_training must be true")
    if overlay.get("do_not_enter_m7") is not True:
        errors.append("do_not_enter_m7 must be true")
    if str(overlay.get("inherits") or "") != PARENT_RELPATH:
        errors.append(f"inherits must be {PARENT_RELPATH}")
    parent = overlay.get("parent") if isinstance(overlay.get("parent"), MappingABC) else {}
    if str(parent.get("sha256") or "") != EXPECTED_EVAL_SHA256:
        errors.append(f"parent.sha256 {parent.get('sha256')} != {EXPECTED_EVAL_SHA256}")
    overrides = overlay.get("overrides") if isinstance(overlay.get("overrides"), MappingABC) else {}
    extra_sections = [key for key in overrides.keys() if str(key) not in ALLOWED_OVERRIDE_SECTIONS]
    if extra_sections:
        errors.append(f"overrides has disallowed sections: {extra_sections}")
    research = [key for key in overrides.keys() if str(key) in FORBIDDEN_RESEARCH_SECTIONS]
    if research:
        errors.append(f"must not change research sections: {research}")
    frozen = overlay.get("frozen_from_parent") if isinstance(overlay.get("frozen_from_parent"), MappingABC) else {}
    budgets = tuple(int(item) for item in (frozen.get("budgets") or []))
    if budgets != BUDGETS:
        errors.append(f"frozen budgets must stay {list(BUDGETS)}")
    if int(frozen.get("n_tasks") or 0) != N_DEV_TASKS:
        errors.append("frozen n_tasks must be 244")
    if int(frozen.get("paired_seed_base") or 0) != PAIRED_SEED_BASE:
        errors.append("frozen paired_seed_base drifted")
    sampling = frozen.get("sampling") if isinstance(frozen.get("sampling"), MappingABC) else {}
    if float(sampling.get("temperature") or 0) != 0.7:
        errors.append("frozen sampling.temperature must stay 0.7")
    if frozen.get("historical_runs_in_main_table") is True:
        errors.append("E001/E006 must not enter the main table")
    ckpt = overrides.get("checkpoint") if isinstance(overrides.get("checkpoint"), MappingABC) else {}
    if int(ckpt.get("global_step") or 0) != CANONICAL_RL_STEP:
        errors.append("checkpoint.global_step must be 275")
    if "global_step_275" not in str(ckpt.get("path_template") or ""):
        errors.append("checkpoint.path_template must be global_step_275")
    forbidden = {int(item) for item in (ckpt.get("forbidden_steps") or [])}
    if not set(FORBIDDEN_RL_STEPS).issubset(forbidden):
        errors.append("checkpoint.forbidden_steps must include 8/16/24/32/256")
    ids = [str(item.get("id")) for item in (overrides.get("conditions") or [])]
    if ids != ["B0", "B1", "M_scaled"]:
        errors.append("conditions must be B0, B1, M_scaled")
    wandb = overrides.get("wandb") if isinstance(overrides.get("wandb"), MappingABC) else {}
    if str(wandb.get("experiment_name") or "") != WANDB_EXPERIMENT_NAME:
        errors.append(f"wandb.experiment_name must be {WANDB_EXPERIMENT_NAME}")
    launch = overrides.get("launch") if isinstance(overrides.get("launch"), MappingABC) else {}
    if str(launch.get("session_name") or "") != SESSION_NAME:
        errors.append("launch.session_name must be E018")
    if repo_root is not None:
        errors.extend(e015_lock_errors(Path(repo_root)))
        parent_path = Path(repo_root) / PARENT_RELPATH
        if parent_path.is_file() and sha256_file(parent_path) != EXPECTED_EVAL_SHA256:
            errors.append("E015 freeze sha256 drifted")
        freeze = Path(repo_root) / "configs/experiments/stage1_m3c_freeze.json"
        if freeze.is_file() and sha256_file(freeze) != EXPECTED_M3C_FREEZE_SHA256:
            errors.append("M3C freeze sha256 drifted")
    return errors


def overlay_lock_errors(repo_root: Path) -> list[str]:
    overlay_path = default_overlay_path(repo_root)
    lock_path = default_overlay_lock_path(repo_root)
    if not overlay_path.is_file():
        return [f"missing E018 overlay {overlay_path}"]
    if not lock_path.is_file():
        return [f"missing E018 lock {lock_path}"]
    actual = sha256_file(overlay_path)
    lock = load_json(lock_path)
    errors: list[str] = []
    if str(lock.get("sha256") or "") != actual:
        errors.append(f"E018 overlay sha256 {actual} != lock {lock.get('sha256')}")
    if EXPECTED_OVERLAY_SHA256 != "0" * 64 and actual != EXPECTED_OVERLAY_SHA256:
        errors.append(f"E018 overlay sha256 {actual} != pinned {EXPECTED_OVERLAY_SHA256}")
    if str(lock.get("parent_sha256") or "") != EXPECTED_EVAL_SHA256:
        errors.append("E018 lock parent_sha256 mismatch")
    if str(lock.get("experiment_id") or "") != EXPERIMENT_ID:
        errors.append("E018 lock experiment_id must be E018")
    if int(lock.get("canonical_rl_step") or 0) != CANONICAL_RL_STEP:
        errors.append("E018 lock canonical_rl_step must be 275")
    if lock.get("eval_only") is not True:
        errors.append("E018 lock eval_only must be true")
    return errors


def write_overlay_lock(repo_root: Path) -> dict[str, Any]:
    overlay_path = default_overlay_path(repo_root)
    digest = sha256_file(overlay_path)
    lock = {
        "path": OVERLAY_RELPATH,
        "sha256": digest,
        "parent_path": PARENT_RELPATH,
        "parent_sha256": EXPECTED_EVAL_SHA256,
        "ordered_ids_sha256": EXPECTED_ORDERED_IDS_SHA256,
        "m3c_freeze_sha256": EXPECTED_M3C_FREEZE_SHA256,
        "experiment_id": EXPERIMENT_ID,
        "canonical_rl_step": CANONICAL_RL_STEP,
        "forbidden_rl_steps": list(FORBIDDEN_RL_STEPS),
        "eval_only": True,
        "not_training": True,
        "do_not_enter_m7": True,
        "note": "Namespace overlay only. Do not edit stage1_m6_eval.json. Candidate is E017 global_step_275.",
    }
    write_json(default_overlay_lock_path(repo_root), lock)
    return lock


def consume_e018_overlay(*, repo_root: Path, overlay: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = default_overlay_path(repo_root)
    payload = overlay if overlay is not None else load_json(path)
    errors = overlay_errors(payload, repo_root=repo_root)
    if overlay is None:
        errors.extend(overlay_lock_errors(repo_root))
    if errors:
        raise ValueError("E018 overlay contract failed: " + "; ".join(errors))
    return {
        "experiment_id": EXPERIMENT_ID,
        "overlay_sha256": sha256_file(path) if path.is_file() else None,
        "parent_sha256": EXPECTED_EVAL_SHA256,
        "canonical_rl_step": CANONICAL_RL_STEP,
        "wandb_experiment_name": WANDB_EXPERIMENT_NAME,
        "session_name": SESSION_NAME,
        "output_dir": str(default_e018_output_dir(repo_root)),
        "trajectory_dir": str(default_trace_dir()),
        "checkpoint_actor_dir": str(default_rl_actor_dir()),
    }


def iter_eval_jobs(
    tasks: Sequence[Mapping[str, Any]],
    *,
    condition_ids: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    wanted_conditions = list(condition_ids or [item["id"] for item in CONDITIONS])
    wanted_budgets = [int(item) for item in (budgets or BUDGETS)]
    jobs: list[dict[str, Any]] = []
    for task in tasks:
        for condition_id in wanted_conditions:
            spec = condition_spec(condition_id)
            for budget in wanted_budgets:
                jobs.append(
                    {
                        "task_index": int(task["task_index"]),
                        "instance_id": task["instance_id"],
                        "repo": task["repo"],
                        "condition_id": condition_id,
                        "policy": spec["policy"],
                        "budget_visible": bool(spec["budget_visible"]),
                        "obs_tokens_limit": int(budget),
                        "sampling_seed": int(task["sampling_seed"]),
                    }
                )
    return jobs


def jobs_for_phase(
    tasks: Sequence[Mapping[str, Any]],
    phase: str,
    *,
    reuse_base: bool = False,
    smoke_n_tasks: int = SMOKE_N_TASKS,
) -> list[dict[str, Any]]:
    name = str(phase)
    subset = list(tasks)[: int(smoke_n_tasks)] if name == "smoke" else list(tasks)
    if name == "smoke":
        condition_ids = ("M_scaled",) if reuse_base else ("B0", "B1", "M_scaled")
        return iter_eval_jobs(
            subset,
            condition_ids=condition_ids,
            budgets=(PRIMARY_TRAINING_BUDGET,),
        )
    if name == "base":
        if reuse_base:
            return []
        return iter_eval_jobs(tasks, condition_ids=("B0", "B1"))
    if name == "rl":
        return iter_eval_jobs(tasks, condition_ids=("M_scaled",))
    if name in {"all", "full"}:
        if reuse_base:
            return iter_eval_jobs(tasks, condition_ids=("M_scaled",))
        return iter_eval_jobs(tasks)
    raise ValueError(f"unknown E018 phase {phase!r}")


def split_jobs_by_policy(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = [dict(job) for job in jobs if str(job.get("policy")) == "base"]
    rl = [dict(job) for job in jobs if str(job.get("policy")) == "rl"]
    return base, rl


def resume_key(job: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(job["instance_id"]),
        str(job["condition_id"]),
        int(job["obs_tokens_limit"]),
    )


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not Path(path).is_file():
        return done
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = instance_id_from_row(row)
            condition_id = condition_id_from_row(row)
            limit = obs_limit_from_row(row)
            if instance_id and condition_id and limit is not None:
                done.add((instance_id, condition_id, int(limit)))
    return done


def group_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("termination") == "operational_error":
            continue
        instance_id = instance_id_from_row(row)
        condition_id = condition_id_from_row(row)
        limit = obs_limit_from_row(row)
        if not instance_id or condition_id is None or limit is None:
            continue
        slot = grouped.setdefault(
            (instance_id, int(limit)),
            {"instance_id": instance_id, "obs_tokens_limit": int(limit), "repo": repo_from_row(row)},
        )
        slot[condition_id] = row
    return grouped


def paired_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_id: str,
    right_id: str,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    matrix = group_matrix(rows)
    out: list[dict[str, Any]] = []
    for (instance_id, limit), slot in sorted(matrix.items()):
        if budget is not None and int(limit) != int(budget):
            continue
        if left_id not in slot or right_id not in slot:
            continue
        comparison = compare_pair(
            slot[left_id],
            slot[right_id],
            left_name=left_id,
            right_name=right_id,
        )
        comparison["left"] = slot[left_id]
        comparison["right"] = slot[right_id]
        comparison["instance_id"] = instance_id
        out.append(comparison)
    return out


def condition_rows(rows: Sequence[Mapping[str, Any]], condition_id: str, budget: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if condition_id_from_row(row) != condition_id:
            continue
        if obs_limit_from_row(row) != int(budget):
            continue
        if row.get("termination") == "operational_error":
            continue
        out.append(dict(row))
    return out


def cell_aggregate(rows: Sequence[Mapping[str, Any]], condition_id: str, budget: int) -> dict[str, Any]:
    subset = condition_rows(rows, condition_id, budget)
    summary = summarize_episodes(subset)
    summary["condition_id"] = condition_id
    summary["obs_tokens_limit"] = int(budget)
    summary["n_scored"] = len(subset)
    return summary


def main_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = list(condition_ids or [item["id"] for item in CONDITIONS])
    table: list[dict[str, Any]] = []
    for condition_id in wanted:
        spec = condition_spec(condition_id)
        for budget in BUDGETS:
            summary = cell_aggregate(rows, str(condition_id), int(budget))
            table.append(
                {
                    "condition_id": condition_id,
                    "label": spec.get("label"),
                    "policy": spec.get("policy"),
                    "budget_visible": spec.get("budget_visible"),
                    "source_experiment": spec.get("source_experiment") or "E018",
                    "obs_tokens_limit": budget,
                    "n": summary.get("n_episodes"),
                    "mean_localization_score": summary.get("mean_localization_score"),
                    "median_localization_score": summary.get("median_localization_score"),
                    "mean_file_precision_parse_ok": summary.get("mean_file_precision_parse_ok"),
                    "mean_file_recall_parse_ok": summary.get("mean_file_recall_parse_ok"),
                    "mean_file_f1_parse_ok": summary.get("mean_file_f1_parse_ok"),
                    "mean_symbol_precision_scored": summary.get("mean_symbol_precision_scored"),
                    "mean_symbol_recall_scored": summary.get("mean_symbol_recall_scored"),
                    "mean_symbol_f1_scored": summary.get("mean_symbol_f1_scored"),
                    "n_symbol_scored": summary.get("n_symbol_scored"),
                    "n_symbol_unavailable": summary.get("n_symbol_unavailable"),
                    "parse_ok_rate": summary.get("parse_ok_rate"),
                    "invalid_tool_rate": summary.get("invalid_tool_rate"),
                    "empty_submission_rate": summary.get("empty_submission_rate"),
                    "budget_exhaustion_rate": summary.get("budget_exhaustion_rate"),
                    "mean_repo_observation_tokens": summary.get("mean_repo_observation_tokens"),
                    "mean_budget_utilization": summary.get("mean_budget_utilization"),
                    "mean_n_events": summary.get("mean_n_events"),
                    "mean_n_tool_ok": summary.get("mean_n_tool_ok"),
                    "mean_policy_token_count": summary.get("mean_policy_token_count"),
                    "mean_n_search": summary.get("mean_n_search"),
                    "mean_n_read": summary.get("mean_n_read"),
                    "finish_rate": summary.get("finish_rate"),
                }
            )
    return table


def quality_budget_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_ids: Sequence[str] = ("B1", "M_scaled"),
) -> dict[str, Any]:
    series: dict[str, list[dict[str, Any]]] = {}
    for condition_id in condition_ids:
        points = []
        for budget in BUDGETS:
            summary = cell_aggregate(rows, condition_id, budget)
            points.append(
                {
                    "obs_tokens_limit": budget,
                    "mean_localization_score": summary.get("mean_localization_score"),
                    "median_localization_score": summary.get("median_localization_score"),
                    "mean_repo_observation_tokens": summary.get("mean_repo_observation_tokens"),
                    "n": summary.get("n_episodes"),
                }
            )
        series[str(condition_id)] = points
    return {"x": "obs_tokens_limit", "y": "mean_localization_score", "series": series}


def _score_field(item: Mapping[str, Any], condition_id: str) -> float | None:
    value = item.get(f"{condition_id}_score")
    if value is None:
        return None
    return float(value)


def _parse_field(item: Mapping[str, Any], condition_id: str) -> bool:
    return bool(item.get(f"{condition_id}_parse_ok"))


def _invalid_field(item: Mapping[str, Any], condition_id: str) -> bool:
    return bool(item.get(f"{condition_id}_invalid"))


def case_candidates(
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_id: str = "B1",
    right_id: str = "M_scaled",
    rules: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    spec = dict(CASE_RULES)
    spec.update(dict(rules or {}))
    pools: dict[str, list[dict[str, Any]]] = {
        "rl_win": [],
        "base_win": [],
        "both_fail": [],
        "protocol_regression": [],
    }
    for item in pairs:
        delta = item.get("delta_localization_score")
        left = _score_field(item, left_id)
        right = _score_field(item, right_id)
        if delta is None or left is None or right is None:
            continue
        packet = {
            "instance_id": item.get("instance_id"),
            "repo": item.get("repo"),
            "obs_tokens_limit": item.get("obs_tokens_limit"),
            "delta": float(delta),
            f"{left_id}_score": float(left),
            f"{right_id}_score": float(right),
            f"{left_id}_parse_ok": _parse_field(item, left_id),
            f"{right_id}_parse_ok": _parse_field(item, right_id),
            f"{left_id}_invalid": _invalid_field(item, left_id),
            f"{right_id}_invalid": _invalid_field(item, right_id),
            f"{left_id}_termination": item.get(f"{left_id}_termination"),
            f"{right_id}_termination": item.get(f"{right_id}_termination"),
        }
        if (
            float(delta) >= float(spec["delta_rl_win_min"])
            and float(left) <= float(spec["base_score_max_for_rl_win"])
            and float(right) >= float(spec["rl_score_min_for_rl_win"])
        ):
            pools["rl_win"].append(packet)
        if float(delta) <= float(spec["delta_base_win_max"]):
            pools["base_win"].append(packet)
        if float(left) <= float(spec["both_fail_max"]) and float(right) <= float(spec["both_fail_max"]):
            pools["both_fail"].append(packet)
        if _parse_field(item, left_id) and not _parse_field(item, right_id):
            pools["protocol_regression"].append(packet)
        elif _invalid_field(item, right_id) and not _invalid_field(item, left_id) and float(delta) < 0:
            pools["protocol_regression"].append(packet)
    for key in pools:
        reverse = key != "both_fail"
        pools[key].sort(
            key=lambda row: (
                -abs(float(row["delta"])) if reverse else abs(float(row["delta"])),
                0 if int(row.get("obs_tokens_limit") or 0) == int(spec["prefer_budget"]) else 1,
                str(row.get("instance_id") or ""),
            )
        )
    return pools


def select_case_studies(
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_id: str = "B1",
    right_id: str = "M_scaled",
    rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = dict(CASE_RULES)
    spec.update(dict(rules or {}))
    pools = case_candidates(pairs, left_id=left_id, right_id=right_id, rules=spec)
    selected: dict[str, dict[str, Any] | None] = {
        "rl_win": None,
        "base_win": None,
        "both_fail": None,
    }
    used_repos: set[str] = set()

    def take(category: str, pool_name: str) -> None:
        pool = list(pools.get(pool_name) or [])
        if spec.get("prefer_parse_ok") and category != "both_fail":
            preferred = [
                item
                for item in pool
                if item.get(f"{left_id}_parse_ok") or item.get(f"{right_id}_parse_ok")
            ]
            pool = preferred or pool
        chosen = None
        for item in pool:
            repo = str(item.get("repo") or "")
            if spec.get("prefer_cross_repo") and repo in used_repos and any(
                str(other.get("repo") or "") not in used_repos for other in pool
            ):
                continue
            chosen = item
            break
        if chosen is None and pool:
            chosen = pool[0]
        selected[category] = chosen
        if chosen is not None:
            used_repos.add(str(chosen.get("repo") or ""))

    take("rl_win", "rl_win")
    if spec.get("include_protocol_regression_in_base_win") and pools.get("protocol_regression"):
        proto = [
            item
            for item in pools["protocol_regression"]
            if item.get("delta") is not None and float(item["delta"]) <= 0
        ]
        if proto:
            pools["base_win"] = proto + [
                item for item in pools["base_win"] if item not in proto
            ]
    take("base_win", "base_win")
    take("both_fail", "both_fail")
    return {
        "rules": spec,
        "left_id": left_id,
        "right_id": right_id,
        "n_candidates": {key: len(value) for key, value in pools.items()},
        "selected": selected,
        "pools": {key: value[:12] for key, value in pools.items()},
    }


def attach_case_trajectories(
    selection: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_id: str = "B1",
    right_id: str = "M_scaled",
) -> dict[str, Any]:
    payload = json.loads(json.dumps(selection, default=str))
    by_key = {
        (str(item.get("instance_id")), int(item.get("obs_tokens_limit") or 0)): item
        for item in pairs
    }
    cases = []
    for category, chosen in (payload.get("selected") or {}).items():
        if not chosen:
            cases.append({"category": category, "missing": True})
            continue
        key = (str(chosen.get("instance_id")), int(chosen.get("obs_tokens_limit") or 0))
        pair = by_key.get(key)
        if pair is None:
            cases.append({"category": category, "missing": True, **chosen})
            continue
        left = pair.get("left")
        right = pair.get("right")
        cases.append(
            {
                "category": category,
                "instance_id": chosen.get("instance_id"),
                "repo": chosen.get("repo"),
                "obs_tokens_limit": chosen.get("obs_tokens_limit"),
                "delta": chosen.get("delta"),
                left_id: compact_review_case(left, partner=right, reason=f"e018_{category}"),
                right_id: compact_review_case(right, partner=left, reason=f"e018_{category}"),
                "behavior_notes": _behavior_notes(left, right, category, left_id=left_id, right_id=right_id),
                f"{left_id}_events": list((left or {}).get("events") or []),
                f"{right_id}_events": list((right or {}).get("events") or []),
                f"{left_id}_final_submission": (left or {}).get("final_submission"),
                f"{right_id}_final_submission": (right or {}).get("final_submission"),
            }
        )
    payload["cases"] = cases
    return payload


def _behavior_notes(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    category: str,
    *,
    left_id: str,
    right_id: str,
) -> dict[str, Any]:
    def queries(row: Mapping[str, Any] | None) -> list[str]:
        if not row:
            return []
        out = []
        for event in row.get("events") or []:
            if not isinstance(event, MappingABC):
                continue
            if event.get("action_name") == "search":
                args = event.get("action_arguments") or {}
                query = args.get("query")
                if query:
                    out.append(str(query))
        return out

    def first_read_turn(row: Mapping[str, Any] | None) -> int | None:
        if not row:
            return None
        for event in row.get("events") or []:
            if isinstance(event, MappingABC) and event.get("action_name") == "read":
                return int(event.get("turn") or 0)
        return None

    left_q = queries(left)
    right_q = queries(right)
    return {
        f"{left_id}_first_query": left_q[0] if left_q else None,
        f"{right_id}_first_query": right_q[0] if right_q else None,
        f"{left_id}_n_queries": len(left_q),
        f"{right_id}_n_queries": len(right_q),
        "same_first_query": bool(left_q and right_q and left_q[0] == right_q[0]),
        f"{left_id}_first_read_turn": first_read_turn(left),
        f"{right_id}_first_read_turn": first_read_turn(right),
        "category": category,
        "protocol_degradation": bool(
            right
            and left
            and not bool((right.get("localization") or {}).get("parse_ok"))
            and bool((left.get("localization") or {}).get("parse_ok"))
        ),
    }


def per_repo_breakdown(
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_id: str = "B1",
    right_id: str = "M_scaled",
) -> list[dict[str, Any]]:
    by_repo: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in pairs:
        repo = str(item.get("repo") or "unknown")
        by_repo[repo].append(item)
    out: list[dict[str, Any]] = []
    for repo in sorted(by_repo):
        group = by_repo[repo]
        stats = paired_summary_stats(group, left_id=left_id, right_id=right_id, group_ids={})
        stats["repo"] = repo
        stats["n"] = len(group)
        stats["significance_ok"] = len(group) >= SMALL_REPO_N
        out.append(stats)
    return out


def _ci_excludes_zero(stats: Mapping[str, Any]) -> bool:
    boot = stats.get("bootstrap") or {}
    low = boot.get("low")
    high = boot.get("high")
    if low is None or high is None:
        return False
    return float(low) > 0 or float(high) < 0


def _mean_delta(stats: Mapping[str, Any]) -> float | None:
    value = stats.get("mean_delta")
    if value is None:
        return None
    return float(value)


def scientific_conclusion(paired_rl: Mapping[str, Any]) -> str:
    """POSITIVE / WEAK POSITIVE / NULL / NEGATIVE from Scaled vs B1 CIs."""
    deltas = []
    positives = 0
    negatives = 0
    excludes = 0
    for budget in BUDGETS:
        stats = paired_rl.get(str(budget)) or {}
        mean = _mean_delta(stats)
        if mean is None:
            continue
        deltas.append(mean)
        if _ci_excludes_zero(stats):
            excludes += 1
            if mean > 0:
                positives += 1
            elif mean < 0:
                negatives += 1
    if not deltas:
        return "NULL"
    if positives == len(deltas) and excludes == len(deltas):
        return "POSITIVE"
    if negatives == len(deltas) and excludes == len(deltas):
        return "NEGATIVE"
    if all(value > 0 for value in deltas):
        return "WEAK POSITIVE"
    if all(value < 0 for value in deltas) and negatives >= 1:
        return "NEGATIVE"
    return "NULL"


def reliability_class(
    *,
    conclusion: str,
    b1_cells: Sequence[Mapping[str, Any]],
    scaled_cells: Sequence[Mapping[str, Any]],
) -> str:
    """A-E from localization conclusion plus parse_ok / invalid / empty."""

    def by_budget(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
        return {int(item["obs_tokens_limit"]): item for item in rows}

    b1 = by_budget(b1_cells)
    scaled = by_budget(scaled_cells)
    parse_deltas = []
    invalid_deltas = []
    for budget in BUDGETS:
        left = b1.get(int(budget)) or {}
        right = scaled.get(int(budget)) or {}
        lp = left.get("parse_ok_rate")
        rp = right.get("parse_ok_rate")
        li = left.get("invalid_tool_rate")
        ri = right.get("invalid_tool_rate")
        if lp is not None and rp is not None:
            parse_deltas.append(float(rp) - float(lp))
        if li is not None and ri is not None:
            invalid_deltas.append(float(ri) - float(li))
    parse_up = bool(parse_deltas) and all(delta > 0.02 for delta in parse_deltas)
    parse_down = bool(parse_deltas) and all(delta < -0.02 for delta in parse_deltas)
    invalid_down = bool(invalid_deltas) and all(delta < -0.02 for delta in invalid_deltas)
    invalid_up = bool(invalid_deltas) and all(delta > 0.02 for delta in invalid_deltas)
    protocol_improved = parse_up or invalid_down
    protocol_regressed = parse_down or invalid_up
    loc_up = conclusion in {"POSITIVE", "WEAK POSITIVE"}
    loc_down = conclusion == "NEGATIVE"
    loc_null = conclusion == "NULL"
    if loc_up and protocol_improved:
        return "A"
    if loc_up:
        return "B"
    if loc_null and protocol_improved:
        return "C"
    if loc_down or protocol_regressed:
        return "E"
    return "D"


def tag_source(row: Mapping[str, Any], source_experiment: str, *, condition_id: str | None = None) -> dict[str, Any]:
    payload = json.loads(json.dumps(row, default=str))
    payload["source_experiment"] = source_experiment
    if condition_id:
        condition = payload.get("condition") if isinstance(payload.get("condition"), dict) else {}
        condition["condition_id"] = condition_id
        payload["condition"] = condition
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def import_e015_condition(
    path: Path,
    *,
    condition_id: str,
    remap_to: str | None = None,
) -> list[dict[str, Any]]:
    from budget_coder_rl.eval import m6 as m6_mod

    out: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        if row.get("termination") == "operational_error":
            continue
        cid = m6_mod.condition_id_from_row(row)
        if cid != condition_id:
            continue
        out.append(tag_source(row, "E015", condition_id=remap_to or cid))
    return out


def count_condition_cells(rows: Sequence[Mapping[str, Any]], condition_id: str) -> int:
    n = 0
    for row in rows:
        if row.get("termination") == "operational_error":
            continue
        if condition_id_from_row(row) == condition_id:
            n += 1
    return n


def git_changed_files(repo_root: Path, old_commit: str) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", old_commit, "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return list(EVAL_CRITICAL_RELPATHS)
    changed = {line.strip() for line in output.splitlines() if line.strip()}
    return [path for path in EVAL_CRITICAL_RELPATHS if path in changed]


def reuse_base_audit(
    repo_root: Path,
    *,
    data_root: Path | None = None,
    e015_provenance: Mapping[str, Any] | None = None,
    e015_scored_path: Path | None = None,
) -> dict[str, Any]:
    """Allow E015 B0/B1 reuse only under strict provenance. Default is re-run."""
    reasons: list[str] = []
    parent_path = Path(repo_root) / PARENT_RELPATH
    if not parent_path.is_file() or sha256_file(parent_path) != EXPECTED_EVAL_SHA256:
        reasons.append("E015 freeze sha256 mismatch")
    yaml_path = Path(repo_root) / AGENT_LOOP_CONFIG_RELPATH
    yaml_sha = sha256_file(yaml_path) if yaml_path.is_file() else None
    provenance = dict(e015_provenance or {})
    if not provenance:
        prov_path = default_output_dir(Path(repo_root), "E015") / "provenance.json"
        if prov_path.is_file():
            provenance = load_json(prov_path)
    recorded_yaml = ((provenance.get("agent_loop_config") or {}).get("sha256"))
    if recorded_yaml and yaml_sha and recorded_yaml != yaml_sha:
        reasons.append("AgentLoop YAML sha256 != E015 provenance")
    project = provenance.get("budget_coder_rl") or {}
    if project.get("dirty") is True:
        reasons.append("E015 provenance worktree was dirty")
    e015_commit = str(project.get("commit") or "")
    if e015_commit:
        drifted = git_changed_files(Path(repo_root), e015_commit)
        if drifted:
            reasons.append("eval-critical files changed since E015 commit: " + ",".join(drifted[:8]))
    scored = Path(e015_scored_path) if e015_scored_path is not None else (
        default_e015_trace_dir(data_root) / "episodes_scored.jsonl"
    )
    n_b0 = n_b1 = n_err = 0
    if scored.is_file():
        from budget_coder_rl.eval import m6 as m6_mod

        for row in load_jsonl(scored):
            if row.get("termination") == "operational_error":
                n_err += 1
                continue
            cid = m6_mod.condition_id_from_row(row)
            if cid == "B0":
                n_b0 += 1
            elif cid == "B1":
                n_b1 += 1
        if n_b0 != N_DEV_TASKS * len(BUDGETS) or n_b1 != N_DEV_TASKS * len(BUDGETS):
            reasons.append(f"E015 B0/B1 counts {n_b0}/{n_b1} != 732/732")
        if n_err:
            reasons.append(f"E015 scored operational_error={n_err}")
    else:
        reasons.append(f"missing E015 scored episodes {scored}")
    allow = not reasons
    return {
        "allow_reuse": allow,
        "decision": "reuse_e015_b0_b1" if allow else "rerun_b0_b1",
        "reasons": reasons,
        "e015_commit": e015_commit or None,
        "e015_dirty": project.get("dirty"),
        "n_b0": n_b0,
        "n_b1": n_b1,
        "yaml_sha256": yaml_sha,
        "recorded_yaml_sha256": recorded_yaml,
        "parent_sha256": EXPECTED_EVAL_SHA256,
        "n_dev_groups": N_DEV_GROUPS,
    }


def shard_file_fingerprints(actor_dir: Path) -> list[dict[str, Any]]:
    files = []
    root = Path(actor_dir)
    if not root.is_dir():
        return files
    for path in sorted(root.glob("model_world_size_*.pt")):
        files.append(
            {
                "relpath": path.name,
                "nbytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return files


def treatment_integrity_errors(
    payload: Mapping[str, Any],
    *,
    require_generate: bool = True,
) -> list[str]:
    """CPU taxonomy. HTTP /v1/models and 'RL!=Base' are not sufficient."""
    errors: list[str] = []
    actor = str(payload.get("checkpoint_actor_dir") or "")
    errors.extend(checkpoint_path_errors(actor))
    if payload.get("load_ok") is not True:
        errors.append("FSDP load_checkpoint did not complete")
    if payload.get("update_weights_ok") is not True:
        errors.append("official update_weights did not complete")
    listed = [int(item) for item in list(payload.get("listed_lora_ids") or [])]
    if int(VLLM_LORA_INT_ID) not in listed:
        errors.append(f"list_loras missing adapter id {VLLM_LORA_INT_ID}")
    if payload.get("lora_as_adapter") is False:
        errors.append("lora_as_adapter is false")
    if require_generate and payload.get("lora_request_attached") is not True:
        errors.append("generate did not attach LoRARequest")
    attached_id = payload.get("lora_int_id")
    if payload.get("lora_request_attached") and attached_id is not None:
        if int(attached_id) != int(VLLM_LORA_INT_ID):
            errors.append(f"generate attached lora_int_id={attached_id} != {VLLM_LORA_INT_ID}")
    load_fp = payload.get("load_fingerprint") if isinstance(payload.get("load_fingerprint"), MappingABC) else {}
    sync_fp = payload.get("sync_payload") if isinstance(payload.get("sync_payload"), MappingABC) else {}
    summary = adapter_payload_summary(sync_fp) if sync_fp else {}
    if sync_fp:
        if not summary.get("peft_config_present"):
            errors.append("adapter sync peft_config missing")
        if int(summary.get("n_adapter_tensors") or 0) <= 0:
            errors.append("adapter sync payload has no LoRA tensors")
        if not summary.get("adapter_nonzero"):
            errors.append("adapter payload looks like an empty/zero LoRA")
    load_digest = str(load_fp.get("digest") or "")
    sync_digest = str(summary.get("digest") or sync_fp.get("digest") or "")
    if load_digest and sync_digest and load_digest != sync_digest:
        errors.append("load-time LoRA digest != update_weights payload digest")
    if not load_digest and payload.get("require_load_digest") is True:
        errors.append("missing load-time LoRA fingerprint digest")
    if payload.get("http_saw_adapter") is False and not listed:
        errors.append("HTTP /v1/models missing adapter is not a pass; engine list_loras also empty")
    if payload.get("used_output_difference_as_proof") is True:
        errors.append("must not use RL vs Base output difference as LoRA-active proof")
    return errors


def evidence_dir(output_dir: Path | None = None) -> Path:
    raw = os.environ.get(OUTPUT_ENV)
    path = Path(raw) if raw else Path(output_dir or ".")
    path.mkdir(parents=True, exist_ok=True)
    return path
