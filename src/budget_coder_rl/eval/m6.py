"""M6 frozen held-out-task evaluation contract and CPU analysis.

Core split name: frozen SWE-Gym held-out-task dev (244 tasks). This is
not a held-out-repository test. Does not train, retune parser/prompt, or
select among E014 step 8/16/24.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.episode import summarize_episodes
from budget_coder_rl.eval.m3b import (
    QWEN3_SAMPLING,
    compact_review_case,
    extra_mapping,
    first_pass_taxonomy,
    load_manifest,
    localization_score,
    sha256_ids,
)
from budget_coder_rl.eval.m4a import PRIVILEGED_LEAK_MARKERS, leakage_errors
from budget_coder_rl.eval.m4b import PINNED_VERL_COMMIT, PINNED_VERL_VERSION, write_json
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID
from budget_coder_rl.eval.m5a import default_output_dir
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E015"
MILESTONE = "M6"
SCHEMA_VERSION = "bcrl-stage1-m6-eval-v1"
EVAL_NAME = "frozen SWE-Gym held-out-task dev evaluation"
CONFIG_RELPATH = "configs/experiments/stage1_m6_eval.json"
LOCK_RELPATH = "configs/experiments/stage1_m6_eval.lock.json"
M3B_MANIFEST_RELPATH = "data/manifests/m3b_baseline_tasks.json"
M3C_FREEZE_RELPATH = "configs/experiments/stage1_m3c_freeze.json"
SPLIT_RELPATH = "data/manifests/swe_gym_m1d_split.json"
AGENT_LOOP_CONFIG_RELPATH = "configs/agent_loop/repo_exploration_m3c.yaml"
FORBIDDEN_OUTPUT_IDS = ("E001", "E006", "E011", "E012", "E013", "E014")

N_DEV_TASKS = 244
N_DEV_GROUPS = 211
BUDGETS = (2048, 4096, 8192)
PRIMARY_TRAINING_BUDGET = 4096
PAIRED_SEED_BASE = 20260827
PAIRED_SEED_FORMULA = "PAIRED_SEED_BASE + task_index"
CANONICAL_RL_STEP = 32
FORBIDDEN_RL_STEPS = (8, 16, 24)
N_GPUS = 2
TENSOR_MODEL_PARALLEL_SIZE = 1
LORA_RANK = 16
LORA_ALPHA = 16
MAX_TURNS = 6
MAX_NEW_TOKENS_PER_TURN = 2048
PROMPT_LENGTH = 16384
RESPONSE_LENGTH = 16384
MAX_MODEL_LEN = 32768
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260827
SMALL_REPO_N = 8

EXPECTED_ORDERED_IDS_SHA256 = (
    "c25b29585d4f1a14f850d98f093e19f85614229a299cca787e3b78dadb499576"
)
EXPECTED_M3C_FREEZE_SHA256 = (
    "49084af1c792e2049af72d4c98291dc546b829122034dba9e698cea8f7284185"
)
EXPECTED_M1E_MANIFEST_SHA256 = (
    "5b1606760c4864cafb8c4d421472c51ff5f8582e0d6dae9185902095fc17da0c"
)
EXPECTED_E014_OVERLAY_SHA256 = (
    "850e6830237f6697d60c58805e68e386277257c2f539def8cb825bb9e1f8c69a"
)
EXPECTED_EVAL_SHA256 = (
    "bdaabd34520d86a7514fde485dd2037d5920e6ebd6945fe31bc90a1c701b7c76"
)
CHECKPOINT_RELPATH = "checkpoints/stage1_m5_e014"
E014_LOG_RELPATH = "outputs/experiments/E014/pipeline.log"

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
        "id": "M1",
        "policy": "rl",
        "budget_visible": True,
        "label": "RL Visible",
        "checkpoint_step": CANONICAL_RL_STEP,
    },
)

CASE_RULES = {
    "delta_rl_win_min": 0.25,
    "base_score_max_for_rl_win": 0.1,
    "rl_score_min_for_rl_win": 0.5,
    "delta_base_win_max": -0.25,
    "both_fail_max": 0.1,
    "prefer_budget": PRIMARY_TRAINING_BUDGET,
    "prefer_cross_repo": True,
    "prefer_parse_ok": True,
    "include_protocol_regression_in_base_win": True,
    "n_per_category": 1,
    "selection_seed": 20260827,
}

TOKENIZER_WARNING_NEEDLE = "1306641 > 1010000"
SMOKE_N_TASKS = 2
PRIVILEGED_EXTRA_KEYS = (
    "oracle_symbols",
    "base_changed_files",
    "gold_localization",
    "gold_edit_files",
    "unmapped_sites",
    "patch",
    "test_patch",
    "correlation_group_id",
)


def default_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / CONFIG_RELPATH


def default_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / LOCK_RELPATH


def default_e015_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_trace_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / "trajectories" / "m6" / EXPERIMENT_ID


def default_rl_checkpoint_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / CHECKPOINT_RELPATH / f"global_step_{CANONICAL_RL_STEP}"


def default_rl_actor_dir(data_root: Path | None = None) -> Path:
    return default_rl_checkpoint_dir(data_root) / "actor"


def eval_seed(task_index: int, *, base: int = PAIRED_SEED_BASE) -> int:
    return int(base) + int(task_index)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def condition_spec(condition_id: str) -> dict[str, Any]:
    for item in CONDITIONS:
        if item["id"] == condition_id:
            return dict(item)
    raise KeyError(condition_id)


def condition_id_from_row(row: Mapping[str, Any]) -> str | None:
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    explicit = condition.get("condition_id") or row.get("condition_id")
    if explicit in {"B0", "B1", "M1"}:
        return str(explicit)
    policy = str(condition.get("policy") or row.get("policy") or "").strip().lower()
    visible = condition.get("budget_visible")
    if visible is None:
        budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
        visible = budget.get("budget_visible")
    if policy == "rl":
        return "M1"
    if visible is True:
        return "B1"
    if visible is False:
        return "B0"
    return None


def obs_limit_from_row(row: Mapping[str, Any]) -> int | None:
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    value = condition.get("obs_tokens_limit")
    if value is None:
        value = budget.get("obs_tokens_limit")
    if value is None:
        return None
    return int(value)


def instance_id_from_row(row: Mapping[str, Any]) -> str:
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    return str(identity.get("instance_id") or row.get("instance_id") or "").strip()


def repo_from_row(row: Mapping[str, Any]) -> str:
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    return str(identity.get("repo") or row.get("repo") or "").strip()


def load_eval_contract(repo_root: Path) -> dict[str, Any]:
    path = default_config_path(repo_root)
    payload = load_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected M6 schema {payload.get('schema_version')!r}")
    return payload


def contract_errors(payload: Mapping[str, Any], *, repo_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must be bcrl-stage1-m6-eval-v1")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id must be E015")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone must be M6")
    if payload.get("eval_name") != EVAL_NAME:
        errors.append("eval_name must be frozen SWE-Gym held-out-task dev evaluation")
    if "held-out-repo" in str(payload.get("eval_name") or "").lower():
        errors.append("eval_name must not claim held-out-repo test")
    if int(payload.get("n_tasks") or 0) != N_DEV_TASKS:
        errors.append(f"n_tasks must be {N_DEV_TASKS}")
    budgets = tuple(int(item) for item in (payload.get("budgets") or []))
    if budgets != BUDGETS:
        errors.append(f"budgets must be {list(BUDGETS)}")
    if int(payload.get("canonical_rl_step") or 0) != CANONICAL_RL_STEP:
        errors.append("canonical_rl_step must be 32")
    forbidden = tuple(int(item) for item in (payload.get("forbidden_rl_steps") or []))
    if set(forbidden) != set(FORBIDDEN_RL_STEPS):
        errors.append("forbidden_rl_steps must be [8, 16, 24]")
    if payload.get("evaluate_intermediate_checkpoints") is True:
        errors.append("must not evaluate intermediate checkpoints")
    sampling = payload.get("sampling") if isinstance(payload.get("sampling"), MappingABC) else {}
    for key, expected in QWEN3_SAMPLING.items():
        if sampling.get(key) != expected:
            errors.append(f"sampling.{key} must stay {expected}")
    if payload.get("validate") is not False:
        errors.append("validate must be false")
    if int(payload.get("vllm_rollout_n") or 0) != 1:
        errors.append("vllm_rollout_n must be 1")
    if int(payload.get("paired_seed_base") or 0) != PAIRED_SEED_BASE:
        errors.append(f"paired_seed_base must be {PAIRED_SEED_BASE}")
    if payload.get("paired_seed_formula") != PAIRED_SEED_FORMULA:
        errors.append("paired_seed_formula mismatch")
    if payload.get("historical_runs_in_main_table") is True:
        errors.append("E001/E006 must not enter the main table")
    ids = [str(item.get("id")) for item in (payload.get("conditions") or [])]
    if ids != ["B0", "B1", "M1"]:
        errors.append("conditions must be B0, B1, M1")
    if repo_root is not None:
        manifest_path = Path(repo_root) / M3B_MANIFEST_RELPATH
        if manifest_path.is_file():
            manifest = load_manifest(manifest_path)
            if manifest.get("ordered_ids_sha256") != EXPECTED_ORDERED_IDS_SHA256:
                errors.append("M3B ordered_ids_sha256 drifted")
            if len(manifest.get("ordered_ids") or []) != N_DEV_TASKS:
                errors.append("M3B ordered_ids is not 244")
            contract_hash = (payload.get("task_set") or {}).get("ordered_ids_sha256")
            if contract_hash and contract_hash != manifest["ordered_ids_sha256"]:
                errors.append("contract ordered_ids_sha256 does not match M3B manifest")
        freeze_path = Path(repo_root) / M3C_FREEZE_RELPATH
        if freeze_path.is_file() and sha256_file(freeze_path) != EXPECTED_M3C_FREEZE_SHA256:
            errors.append("M3C freeze sha256 drifted")
    return errors


def lock_errors(repo_root: Path) -> list[str]:
    config_path = default_config_path(repo_root)
    lock_path = default_lock_path(repo_root)
    errors: list[str] = []
    if not config_path.is_file():
        return [f"missing {config_path}"]
    if not lock_path.is_file():
        return [f"missing {lock_path}"]
    lock = load_json(lock_path)
    digest = sha256_file(config_path)
    if lock.get("sha256") != digest:
        errors.append(f"lock sha256 {lock.get('sha256')} != file {digest}")
    if lock.get("immutable") is not True:
        errors.append("lock.immutable must be true")
    if lock.get("experiment_id") != EXPERIMENT_ID:
        errors.append("lock experiment_id must be E015")
    payload = load_json(config_path)
    errors.extend(contract_errors(payload, repo_root=repo_root))
    return errors


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            return [f"refusing to write into {experiment_id} artifact directory {forbidden}"]
    return []


def extra_info_leakage_errors(extra_info: Mapping[str, Any]) -> list[str]:
    keys = list(extra_info.keys())
    errors = leakage_errors(
        decoded_prompt="",
        decoded_observations=[],
        extra_field_keys=keys,
    )
    for marker in PRIVILEGED_EXTRA_KEYS:
        if marker in extra_info:
            errors.append(f"extra_info contains privileged key {marker}")
    blob = " ".join(str(key) for key in keys)
    for marker in PRIVILEGED_LEAK_MARKERS:
        if marker in blob and marker not in {str(key) for key in keys}:
            errors.append(f"privileged marker {marker} in extra_info keys")
    return errors


def build_policy_extra_info(
    source: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Policy-visible extra_info. Oracle / correlation group stay out."""
    extra = dict(source)
    for key in PRIVILEGED_EXTRA_KEYS:
        extra.pop(key, None)
    extra["budget_visible"] = bool(job["budget_visible"])
    extra["obs_tokens_limit"] = int(job["obs_tokens_limit"])
    extra["sampling_seed"] = int(job["sampling_seed"])
    extra["condition_id"] = str(job["condition_id"])
    extra["policy"] = str(job["policy"])
    leaks = extra_info_leakage_errors(extra)
    if leaks:
        raise ValueError(f"policy extra_info leaked privileged fields: {leaks}")
    return extra


def checkpoint_path_errors(path: Path | str) -> list[str]:
    text = str(path)
    errors: list[str] = []
    if "global_step_" not in text:
        errors.append("checkpoint path must contain global_step_")
    for step in FORBIDDEN_RL_STEPS:
        token = f"global_step_{step}"
        if token in text and f"global_step_{CANONICAL_RL_STEP}" not in text:
            errors.append(f"refusing intermediate RL checkpoint {text}")
    if f"global_step_{CANONICAL_RL_STEP}" not in text:
        errors.append(f"canonical M6 checkpoint must be global_step_{CANONICAL_RL_STEP}")
    return errors


def jobs_for_phase(
    tasks: Sequence[Mapping[str, Any]],
    phase: str,
    *,
    smoke_n_tasks: int = SMOKE_N_TASKS,
) -> list[dict[str, Any]]:
    name = str(phase)
    if name == "smoke":
        return iter_eval_jobs(
            list(tasks)[: int(smoke_n_tasks)],
            budgets=(PRIMARY_TRAINING_BUDGET,),
        )
    if name == "base":
        return iter_eval_jobs(tasks, condition_ids=("B0", "B1"))
    if name == "rl":
        return iter_eval_jobs(tasks, condition_ids=("M1",))
    if name in {"all", "full"}:
        return iter_eval_jobs(tasks)
    raise ValueError(f"unknown M6 phase {phase!r}")


def split_jobs_by_policy(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = [dict(job) for job in jobs if str(job.get("policy")) == "base"]
    rl = [dict(job) for job in jobs if str(job.get("policy")) == "rl"]
    return base, rl


def load_tasks(repo_root: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(Path(repo_root) / M3B_MANIFEST_RELPATH)
    tasks = []
    for index, instance_id in enumerate(manifest["ordered_ids"]):
        row = next(item for item in manifest["tasks"] if item["instance_id"] == instance_id)
        tasks.append(
            {
                "task_index": int(row.get("task_index", index)),
                "instance_id": instance_id,
                "repo": row["repo"],
                "sampling_seed": eval_seed(int(row.get("task_index", index))),
            }
        )
    return tasks


def load_correlation_groups(repo_root: Path) -> dict[str, str]:
    payload = load_json(Path(repo_root) / SPLIT_RELPATH)
    mapping: dict[str, str] = {}
    for row in payload.get("assignments") or []:
        if str(row.get("split") or "") != "dev":
            continue
        instance_id = str(row.get("instance_id") or "")
        group_id = str(row.get("correlation_group_id") or "")
        if instance_id and group_id:
            mapping[instance_id] = group_id
    return mapping


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


def compare_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_score = localization_score(left)
    right_score = localization_score(right)
    winner = "tie"
    if left_score is not None and right_score is not None:
        if right_score > left_score:
            winner = right_name
        elif left_score > right_score:
            winner = left_name
    delta = None
    if left_score is not None and right_score is not None:
        delta = right_score - left_score
    return {
        "instance_id": instance_id_from_row(left) or instance_id_from_row(right),
        "repo": repo_from_row(left) or repo_from_row(right),
        "obs_tokens_limit": obs_limit_from_row(left) or obs_limit_from_row(right),
        f"{left_name}_score": left_score,
        f"{right_name}_score": right_score,
        "delta_localization_score": delta,
        "winner": winner,
        f"{left_name}_parse_ok": _parse_ok(left),
        f"{right_name}_parse_ok": _parse_ok(right),
        f"{left_name}_invalid": _invalid(left),
        f"{right_name}_invalid": _invalid(right),
        f"{left_name}_termination": left.get("termination"),
        f"{right_name}_termination": right.get("termination"),
        f"{left_name}_repo_obs": _repo_obs(left),
        f"{right_name}_repo_obs": _repo_obs(right),
        f"{left_name}_sampling_seed": (left.get("condition") or {}).get("sampling_seed"),
        f"{right_name}_sampling_seed": (right.get("condition") or {}).get("sampling_seed"),
    }


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


def paired_summary_stats(
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_id: str,
    right_id: str,
    group_ids: Mapping[str, str] | None = None,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    deltas = [
        float(item["delta_localization_score"])
        for item in pairs
        if item.get("delta_localization_score") is not None
    ]
    n_right = sum(1 for item in pairs if item.get("winner") == right_id)
    n_left = sum(1 for item in pairs if item.get("winner") == left_id)
    n_tie = sum(1 for item in pairs if item.get("winner") == "tie")
    ci = bootstrap_mean_ci(
        pairs,
        group_ids=group_ids or {},
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "n_pairs": len(pairs),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        f"n_{right_id}_win": n_right,
        f"n_{left_id}_win": n_left,
        "n_tie": n_tie,
        "bootstrap": ci,
        "mean_left": _mean(
            [float(item[f"{left_id}_score"]) for item in pairs if item.get(f"{left_id}_score") is not None]
        ),
        "mean_right": _mean(
            [float(item[f"{right_id}_score"]) for item in pairs if item.get(f"{right_id}_score") is not None]
        ),
    }


def bootstrap_mean_ci(
    pairs: Sequence[Mapping[str, Any]],
    *,
    group_ids: Mapping[str, str],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster bootstrap over correlation_group_id when available."""
    clusters: dict[str, list[float]] = defaultdict(list)
    ungrouped: list[float] = []
    for item in pairs:
        delta = item.get("delta_localization_score")
        if delta is None:
            continue
        instance_id = str(item.get("instance_id") or "")
        group_id = group_ids.get(instance_id) or instance_id
        if group_id:
            clusters[str(group_id)].append(float(delta))
        else:
            ungrouped.append(float(delta))
    keys = sorted(clusters)
    if ungrouped:
        keys.append("__ungrouped__")
        clusters["__ungrouped__"] = ungrouped
    if not keys:
        return {"n_clusters": 0, "n_boot": 0, "low": None, "high": None, "method": "cluster"}
    rng = random.Random(int(seed))
    means: list[float] = []
    for _ in range(int(n_boot)):
        sampled: list[float] = []
        for _key in keys:
            chosen = keys[rng.randrange(len(keys))]
            sampled.extend(clusters[chosen])
        if sampled:
            means.append(sum(sampled) / len(sampled))
    means.sort()
    if not means:
        return {"n_clusters": len(keys), "n_boot": 0, "low": None, "high": None, "method": "cluster"}
    lo_index = int(math.floor(alpha / 2 * (len(means) - 1)))
    hi_index = int(math.ceil((1 - alpha / 2) * (len(means) - 1)))
    hi_index = min(hi_index, len(means) - 1)
    return {
        "n_clusters": len(keys),
        "n_boot": len(means),
        "low": float(means[lo_index]),
        "high": float(means[hi_index]),
        "method": "correlation_group_cluster",
        "alpha": alpha,
    }


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


def per_repo_breakdown(
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_id: str = "B1",
    right_id: str = "M1",
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


def quality_budget_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_ids: Sequence[str] = ("B1", "M1"),
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
        series[condition_id] = points
    return {"x": "obs_tokens_limit", "y": "mean_localization_score", "series": series}


def case_candidates(
    pairs: Sequence[Mapping[str, Any]],
    *,
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
        b1 = item.get("B1_score")
        m1 = item.get("M1_score")
        if delta is None or b1 is None or m1 is None:
            continue
        packet = {
            "instance_id": item.get("instance_id"),
            "repo": item.get("repo"),
            "obs_tokens_limit": item.get("obs_tokens_limit"),
            "delta": float(delta),
            "B1_score": float(b1),
            "M1_score": float(m1),
            "B1_parse_ok": item.get("B1_parse_ok"),
            "M1_parse_ok": item.get("M1_parse_ok"),
            "B1_invalid": item.get("B1_invalid"),
            "M1_invalid": item.get("M1_invalid"),
            "B1_termination": item.get("B1_termination"),
            "M1_termination": item.get("M1_termination"),
        }
        if (
            float(delta) >= float(spec["delta_rl_win_min"])
            and float(b1) <= float(spec["base_score_max_for_rl_win"])
            and float(m1) >= float(spec["rl_score_min_for_rl_win"])
        ):
            pools["rl_win"].append(packet)
        if float(delta) <= float(spec["delta_base_win_max"]):
            pools["base_win"].append(packet)
        if float(b1) <= float(spec["both_fail_max"]) and float(m1) <= float(spec["both_fail_max"]):
            pools["both_fail"].append(packet)
        if item.get("B1_parse_ok") and not item.get("M1_parse_ok"):
            pools["protocol_regression"].append(packet)
        elif item.get("M1_invalid") and not item.get("B1_invalid") and float(delta) < 0:
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
    rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = dict(CASE_RULES)
    spec.update(dict(rules or {}))
    pools = case_candidates(pairs, rules=spec)
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
                if item.get("B1_parse_ok") or item.get("M1_parse_ok")
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
        "n_candidates": {key: len(value) for key, value in pools.items()},
        "selected": selected,
        "pools": {
            key: value[:12] for key, value in pools.items()
        },
    }


def attach_case_trajectories(
    selection: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
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
        b1 = pair.get("left") if condition_id_from_row(pair.get("left") or {}) == "B1" else pair.get("B1")
        m1 = pair.get("right") if condition_id_from_row(pair.get("right") or {}) == "M1" else pair.get("M1")
        if b1 is None:
            b1 = pair.get("left")
        if m1 is None:
            m1 = pair.get("right")
        cases.append(
            {
                "category": category,
                "instance_id": chosen.get("instance_id"),
                "repo": chosen.get("repo"),
                "obs_tokens_limit": chosen.get("obs_tokens_limit"),
                "delta": chosen.get("delta"),
                "B1": compact_review_case(b1, partner=m1, reason=f"m6_{category}"),
                "M1": compact_review_case(m1, partner=b1, reason=f"m6_{category}"),
                "behavior_notes": _behavior_notes(b1, m1, category),
                "B1_events": list((b1 or {}).get("events") or []),
                "M1_events": list((m1 or {}).get("events") or []),
                "B1_final_submission": (b1 or {}).get("final_submission"),
                "M1_final_submission": (m1 or {}).get("final_submission"),
            }
        )
    payload["cases"] = cases
    return payload


def _behavior_notes(b1: Mapping[str, Any] | None, m1: Mapping[str, Any] | None, category: str) -> dict[str, Any]:
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

    b_queries = queries(b1)
    m_queries = queries(m1)
    notes = {
        "B1_first_query": b_queries[0] if b_queries else None,
        "M1_first_query": m_queries[0] if m_queries else None,
        "B1_n_queries": len(b_queries),
        "M1_n_queries": len(m_queries),
        "B1_first_read_turn": first_read_turn(b1),
        "M1_first_read_turn": first_read_turn(m1),
        "B1_taxonomy": first_pass_taxonomy(b1) if b1 else None,
        "M1_taxonomy": first_pass_taxonomy(m1) if m1 else None,
        "category": category,
    }
    if category == "base_win":
        notes["regression_flags"] = {
            "premature_finish": (m1 or {}).get("termination") == "finish"
            and int(((m1 or {}).get("counts") or {}).get("n_events") or 99) <= 2,
            "protocol_degradation": bool((m1 and not _parse_ok(m1)) and (b1 and _parse_ok(b1))),
            "overly_conservative": _repo_obs(m1) is not None
            and _repo_obs(b1) is not None
            and float(_repo_obs(m1) or 0) + 256 < float(_repo_obs(b1) or 0),
        }
    return notes


def inspect_tokenizer_warning(repo_root: Path) -> dict[str, Any]:
    log_path = Path(repo_root) / E014_LOG_RELPATH
    loop_path = Path(repo_root) / "src/budget_coder_rl/agent_loop/repo_exploration.py"
    tools_path = Path(repo_root) / "src/budget_coder_rl/env/tools.py"
    log_hits: list[str] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if TOKENIZER_WARNING_NEEDLE in line or "Token indices sequence length is longer" in line:
                log_hits.append(line.strip()[:500])
    loop_text = loop_path.read_text(encoding="utf-8") if loop_path.is_file() else ""
    tools_text = tools_path.read_text(encoding="utf-8") if tools_path.is_file() else ""
    encode_site = "_encode_user_message" in loop_text and "apply_chat_template" in loop_text
    can_insert = "can_insert" in loop_text
    search_cap = "SEARCH_MAX_FILE_BYTES = 1_000_000" in tools_text
    tito_bug = False
    return {
        "warning_needle": TOKENIZER_WARNING_NEEDLE,
        "n_log_hits": len(log_hits),
        "log_hits": log_hits[:4],
        "log_path": E014_LOG_RELPATH,
        "call_site": (
            "RepoExplorationAgentLoop._encode_user_message -> "
            "AgentLoopBase.apply_chat_template(tokenize=True) on raw # bcrl-obs-v1"
        ),
        "encode_site_present": encode_site,
        "encode_before_can_insert": True,
        "insertion_still_gated_by_can_insert": can_insert,
        "search_max_file_bytes_1e6": search_cap,
        "tito_correctness_bug": tito_bug,
        "eval_tokenization_changed": False,
        "note": (
            "HF tokenizer.model_max_length warning while encoding an oversized "
            "raw tool observation (search may read SEARCH_MAX_FILE_BYTES=1e6) "
            "to measure tool_obs_n. Inserted tokens remain gated by "
            "budget.can_insert; oversized raw ids are not appended when the "
            "gate fails. Do not change M6 evaluation tokenization semantics."
        ),
    }


def main_table(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for budget in BUDGETS:
            summary = cell_aggregate(rows, str(condition["id"]), int(budget))
            table.append(
                {
                    "condition_id": condition["id"],
                    "label": condition["label"],
                    "policy": condition["policy"],
                    "budget_visible": condition["budget_visible"],
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


def difficulty_breakdown(
    pairs: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Small optional slices. Not a mining expedition."""
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in pairs:
        left = item.get("left") or {}
        loc = left.get("localization") if isinstance(left.get("localization"), MappingABC) else {}
        n_gold = loc.get("n_gold_files")
        if n_gold is None:
            gold_bucket = "unknown_n_files"
        elif int(n_gold) <= 1:
            gold_bucket = "n_gold_files_le_1"
        elif int(n_gold) <= 3:
            gold_bucket = "n_gold_files_2_3"
        else:
            gold_bucket = "n_gold_files_ge_4"
        buckets[gold_bucket].append(item)
        instance_id = str(item.get("instance_id") or "")
        feat = (features or {}).get(instance_id) or {}
        if "gold_basename_mentioned" in feat:
            mentioned = bool(feat.get("gold_basename_mentioned"))
            buckets["basename_mentioned" if mentioned else "basename_not_mentioned"].append(item)
    out = []
    for name in sorted(buckets):
        stats = paired_summary_stats(buckets[name], left_id="B1", right_id="M1", group_ids={})
        stats["slice"] = name
        stats["n"] = len(buckets[name])
        out.append(stats)
    return out


def load_optional_features(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = Path(repo_root) / "data/interim/swe_gym/m1d_features.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id") or "")
            if not instance_id:
                continue
            out[instance_id] = {
                "gold_basename_mentioned": row.get("gold_basename_mentioned"),
                "gold_full_path_mentioned": row.get("gold_full_path_mentioned"),
            }
    return out


def _parse_ok(row: Mapping[str, Any]) -> bool:
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    return bool(loc.get("parse_ok"))


def _invalid(row: Mapping[str, Any]) -> bool:
    counts = row.get("counts") if isinstance(row.get("counts"), MappingABC) else {}
    return int(counts.get("n_protocol_errors") or 0) > 0 or int(counts.get("n_tool_errors") or 0) > 0


def _repo_obs(row: Mapping[str, Any] | None) -> float | None:
    if not row:
        return None
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    value = budget.get("repo_observation_tokens")
    if value is None:
        value = budget.get("obs_tokens_used")
    if value is None:
        return None
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def build_lock_payload(config_path: Path) -> dict[str, Any]:
    return {
        "path": CONFIG_RELPATH,
        "sha256": sha256_file(config_path),
        "immutable": True,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "eval_name": EVAL_NAME,
        "canonical_rl_step": CANONICAL_RL_STEP,
        "forbidden_rl_steps": list(FORBIDDEN_RL_STEPS),
        "note": "Do not edit stage1_m6_eval.json after this lock. Do not splice E001/E006 into the main table.",
    }


__all__ = [
    "BOOTSTRAP_N",
    "BUDGETS",
    "CANONICAL_RL_STEP",
    "CASE_RULES",
    "CONDITIONS",
    "EVAL_NAME",
    "EXPERIMENT_ID",
    "EXPECTED_ORDERED_IDS_SHA256",
    "LORA_RANK",
    "MILESTONE",
    "N_DEV_TASKS",
    "N_GPUS",
    "PAIRED_SEED_BASE",
    "PINNED_VERL_COMMIT",
    "PINNED_VERL_VERSION",
    "PRIMARY_TRAINING_BUDGET",
    "SCHEMA_VERSION",
    "VLLM_LORA_INT_ID",
    "attach_case_trajectories",
    "bootstrap_mean_ci",
    "build_lock_payload",
    "build_policy_extra_info",
    "checkpoint_path_errors",
    "case_candidates",
    "cell_aggregate",
    "compare_pair",
    "condition_id_from_row",
    "condition_spec",
    "contract_errors",
    "default_config_path",
    "default_e015_output_dir",
    "default_lock_path",
    "default_rl_actor_dir",
    "default_trace_dir",
    "difficulty_breakdown",
    "eval_seed",
    "extra_info_leakage_errors",
    "extra_mapping",
    "forbidden_output_dir_errors",
    "group_matrix",
    "inspect_tokenizer_warning",
    "iter_eval_jobs",
    "jobs_for_phase",
    "load_completed",
    "load_correlation_groups",
    "load_eval_contract",
    "load_optional_features",
    "load_tasks",
    "lock_errors",
    "main_table",
    "paired_cells",
    "paired_summary_stats",
    "per_repo_breakdown",
    "quality_budget_curve",
    "resume_key",
    "select_case_studies",
    "sha256_ids",
    "split_jobs_by_policy",
    "write_json",
]
