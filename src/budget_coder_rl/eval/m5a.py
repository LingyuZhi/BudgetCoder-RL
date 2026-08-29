"""M5A CPU helpers: isolated veRL, seqlen stats, main-run contract, gates.

Does not run GPU, the optimizer, or vLLM. Does not edit the M3C freeze JSON.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.eval.m3c import (
    QUANTILE_LEVELS,
    group_reward_stats,
    quantile,
)
from budget_coder_rl.eval.m4a import (
    ADVANTAGE_ABS_EPS,
    BUDGET_VISIBLE,
    GROUP_N,
    OBS_TOKENS_LIMIT,
    artifact_hashes,
    freeze_contract_errors,
    load_candidate_ordered_ids,
    load_json,
)
from budget_coder_rl.eval.m4b import (
    LORA_ALPHA,
    LORA_RANK,
    PINNED_VERL_COMMIT,
    PINNED_VERL_VERSION,
    PROMPT_LENGTH,
    RESPONSE_LENGTH,
    write_json,
)
from budget_coder_rl.eval.episode import action_counts, build_episode_record
from budget_coder_rl.eval.provenance import git_info, sha256_file
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root

EXPERIMENT_ID = "E010"
M5B_EXPERIMENT_ID = "E011"
MILESTONE = "M5A"
OUTPUT_ENV = "BCRL_M5_OUTPUT_DIR"
REWARD_FN_RELPATH = "src/budget_coder_rl/reward/localization_score.py"
MAIN_CONFIG_RELPATH = "configs/historical/stage1_m5_main.json"
PILOT_CONFIG_RELPATH = "configs/historical/stage1_m5a_pilot.json"
E007_EPISODES_RELPATH = "trajectories/m3c/E007/episodes.jsonl"
E003_MASK_RELPATH = "outputs/experiments/E003/loss_mask_evidence.json"
E008_EPISODES_RELPATH = "outputs/experiments/E008/episodes.jsonl"

SHARED_VERL_ROOT = Path(os.environ.get("BCRL_VERL_ROOT") or Path.home() / "verl")
PREFERRED_SIBLING_VERL = Path(
    os.environ.get("BCRL_VERL_SIBLING")
    or (Path(__file__).resolve().parents[3].parent / "deps" / "verl")
)
ISOLATED_VERL_RELPATH = "runtimes/verl-8481f9f"


def default_verl_source(repo_root: Path | None = None) -> Path:
    env = os.environ.get("BCRL_VERL_ROOT")
    if env:
        return Path(env)
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root).resolve().parent / "deps" / "verl")
    sibling = Path(__file__).resolve().parents[3].parent / "deps" / "verl"
    candidates.append(sibling)
    try:
        import verl

        candidates.append(Path(verl.__file__).resolve().parents[1])
    except Exception:
        pass
    for path in candidates:
        if path.is_dir():
            return path
    return sibling

N_CANDIDATES = 256
TRAIN_BATCH_SIZE = 8
PILOT_STEPS = 4
MAIN_STEPS = 32
SEED = 20260826
ACTOR_LR = 1.0e-6
TOKEN_SLACK = 256
PPO_MAX_TOKEN_CANDIDATES = (12288, 16384)
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.5
N_GPUS = 1
TENSOR_MODEL_PARALLEL_SIZE = 1
WANDB_PROJECT = "budget-coder-rl"

ALLOWED_PILOT_OVERRIDE_KEYS = frozenset(
    {
        "schema_version",
        "milestone",
        "experiment_id",
        "disposable",
        "not_m6_candidate",
        "inherits",
        "overrides",
        "trainer",
        "checkpoint",
        "n_pilot_tasks",
        "n_pilot_steps",
        "pilot_instance_policy",
        "notes",
    }
)

PILOT_TRAINER_OVERRIDE_KEYS = frozenset(
    {
        "total_training_steps",
        "experiment_name",
        "default_local_dir",
        "save_freq",
        "max_actor_ckpt_to_keep",
        "resume_mode",
    }
)

VERL_CHECKOUT_TEXT = """# M5 isolated veRL checkout

Pinned: `{commit}` (`{version}`, fork LingyuZhi/rtrl-verl).

## Isolation rule

M4 imported a shared dirty tree (`~/verl` with realtimegym files).
M5 prepends this isolated checkout on `PYTHONPATH` and HARD FAILs if
`import verl` resolves to the shared tree.

- isolated_root: `{isolated}`
- source_git: `{source}`
- dirty: `{dirty}`
- matches_pin: `{matches}`

Do not `pip install -e` over the shared environment. Do not vendor veRL.
"""


def default_output_dir(repo_root: Path, experiment_id: str = EXPERIMENT_ID) -> Path:
    return Path(repo_root) / "outputs" / "experiments" / experiment_id


def default_main_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / MAIN_CONFIG_RELPATH


def default_pilot_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / PILOT_CONFIG_RELPATH


def default_isolated_verl_root(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / ISOLATED_VERL_RELPATH


def default_e007_episodes_path(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / E007_EPISODES_RELPATH


def expected_candidate_sha256(freeze: Mapping[str, Any]) -> str:
    manifest = freeze.get("train_candidate_manifest")
    if not isinstance(manifest, MappingABC):
        return ""
    return str(manifest.get("file_sha256") or "")


def select_prefix_instance_ids(
    ordered_ids: Sequence[str],
    *,
    n_tasks: int = TRAIN_BATCH_SIZE,
    n_steps: int = PILOT_STEPS,
) -> list[str]:
    """First ``n_tasks * n_steps`` frozen candidate ids. No mixed-reward filter."""
    need = int(n_tasks) * int(n_steps)
    selected = [str(item) for item in list(ordered_ids)[:need]]
    if len(selected) < need:
        raise ValueError(
            f"need {need} prefix candidate ids, found {len(selected)}"
        )
    return selected


def length_stats(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(item) for item in values]
    if not numbers:
        return {
            "n": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "quantiles": {},
        }
    ordered = sorted(numbers)
    mean = sum(ordered) / len(ordered)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "mean": mean,
        "median": quantile(ordered, 0.50),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": ordered[-1],
        "quantiles": {
            f"p{int(level * 100)}": quantile(ordered, level) for level in QUANTILE_LEVELS
        },
    }


def propose_ppo_max_token_len_per_gpu(
    measured_max: int | float | None,
    *,
    slack: int = TOKEN_SLACK,
) -> int:
    """Pick a stable cap: >= measured_max+slack, not the 32768 packing envelope."""
    if measured_max is None:
        raise ValueError("measured_max is required")
    needed = int(math.ceil(float(measured_max))) + int(slack)
    if needed <= 0:
        raise ValueError(f"invalid needed token cap {needed}")
    if needed > MAX_MODEL_LEN:
        raise ValueError(
            f"measured_max+slack={needed} exceeds max_model_len={MAX_MODEL_LEN}"
        )
    for cap in PPO_MAX_TOKEN_CANDIDATES:
        if int(cap) >= needed:
            return int(cap)
    rounded = int(math.ceil(needed / 256.0) * 256)
    return min(MAX_MODEL_LEN, max(rounded, PPO_MAX_TOKEN_CANDIDATES[-1]))


def load_episode_token_rows(path: Path) -> list[dict[str, Any]]:
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
            condition = record.get("condition") if isinstance(record.get("condition"), MappingABC) else {}
            prompt = tokens.get("prompt_token_count")
            policy = tokens.get("policy_token_count")
            observation = tokens.get("observation_token_count")
            if prompt is None:
                continue
            prompt_n = int(prompt)
            policy_n = int(policy or 0)
            obs_n = int(observation or 0)
            rows.append(
                {
                    "instance_id": str(identity.get("instance_id") or ""),
                    "prompt_token_count": prompt_n,
                    "policy_token_count": policy_n,
                    "observation_token_count": obs_n,
                    "training_seq_proxy": prompt_n + policy_n + obs_n,
                    "obs_tokens_limit": condition.get("obs_tokens_limit"),
                    "budget_visible": condition.get("budget_visible"),
                    "source": str(path),
                }
            )
    return rows


def load_mask_unpadded_response_rows(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    payload = load_json(path)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    out: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, MappingABC):
            continue
        n_policy = int(item.get("n_policy") or 0)
        n_obs = int(item.get("n_observation") or 0)
        out.append(
            {
                "instance_id": str(item.get("instance_id") or ""),
                "n_policy": n_policy,
                "n_observation": n_obs,
                "unpadded_response": n_policy + n_obs,
                "prompt_width": item.get("prompt_width"),
                "response_width": item.get("response_width"),
                "source": str(path),
            }
        )
    return out


def characterize_training_seq_lengths(
    *,
    episode_paths: Sequence[Path],
    mask_path: Path | None = None,
    candidate_ids: Sequence[str] | None = None,
    require_obs_limit: int = OBS_TOKENS_LIMIT,
) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    for path in episode_paths:
        episode_rows.extend(load_episode_token_rows(path))
    budget_rows = [
        row
        for row in episode_rows
        if row.get("obs_tokens_limit") in {None, require_obs_limit}
        or int(row.get("obs_tokens_limit") or require_obs_limit) == int(require_obs_limit)
    ]
    candidate_set = {str(item) for item in (candidate_ids or [])}
    candidate_rows = [
        row for row in budget_rows if row.get("instance_id") in candidate_set
    ] if candidate_set else []
    seq_values = [float(row["training_seq_proxy"]) for row in budget_rows]
    prompt_values = [float(row["prompt_token_count"]) for row in budget_rows]
    policy_values = [float(row["policy_token_count"]) for row in budget_rows]
    obs_values = [float(row["observation_token_count"]) for row in budget_rows]
    mask_rows = load_mask_unpadded_response_rows(mask_path) if mask_path else []
    mask_response = [float(row["unpadded_response"]) for row in mask_rows]
    measured_max = None
    if seq_values:
        measured_max = max(seq_values)
    proposed = propose_ppo_max_token_len_per_gpu(measured_max) if measured_max is not None else None
    return {
        "n_episode_rows": len(episode_rows),
        "n_budget_rows": len(budget_rows),
        "n_candidate_overlap_rows": len(candidate_rows),
        "obs_tokens_limit": int(require_obs_limit),
        "prompt": length_stats(prompt_values),
        "policy": length_stats(policy_values),
        "observation": length_stats(obs_values),
        "training_seq_proxy": length_stats(seq_values),
        "e003_unpadded_response": length_stats(mask_response),
        "candidate_overlap_training_seq_proxy": length_stats(
            [float(row["training_seq_proxy"]) for row in candidate_rows]
        ),
        "measured_max": measured_max,
        "proposed_ppo_max_token_len_per_gpu": proposed,
        "assert_risk_if_8192": bool(measured_max is not None and measured_max > 8192),
        "packing_oom_risk_if_32768": True,
        "sources": [str(path) for path in episode_paths] + ([str(mask_path)] if mask_path else []),
        "note": (
            "training_seq_proxy = prompt+policy+observation from research JSONL; "
            "not a substitute for training token IDs"
        ),
    }


def inherited_m3c_knobs(freeze: Mapping[str, Any]) -> dict[str, Any]:
    sampling = freeze.get("sampling") if isinstance(freeze.get("sampling"), MappingABC) else {}
    envelope = freeze.get("envelope") if isinstance(freeze.get("envelope"), MappingABC) else {}
    return {
        "model": freeze.get("model"),
        "primary_training_B_obs": freeze.get("primary_training_B_obs"),
        "budget_visible": freeze.get("budget_visible"),
        "sampling": dict(sampling),
        "validate": freeze.get("validate"),
        "vllm_rollout_n": freeze.get("vllm_rollout_n"),
        "proposed_grpo_rollout_n": freeze.get("proposed_grpo_rollout_n"),
        "max_turns": freeze.get("max_turns"),
        "max_new_tokens_per_turn": freeze.get("max_new_tokens_per_turn"),
        "envelope": dict(envelope),
        "localization_reward": freeze.get("localization_reward"),
        "verl": freeze.get("verl"),
    }


def m4_validated_knobs() -> dict[str, Any]:
    return {
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": "all-linear",
        "actor_strategy": "fsdp",
        "fsdp_param_offload": True,
        "fsdp_optimizer_offload": True,
        "model_dtype": "bf16",
        "enable_gradient_checkpointing": True,
        "actor_lr": ACTOR_LR,
        "entropy_coeff": 0.0,
        "use_kl_loss": False,
        "use_kl_in_reward": False,
        "load_format": "safetensors",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "checkpoint_path": "RayPPOTrainer._save_checkpoint / FSDPCheckpointManager",
        "no_m4c_lora_fingerprint_hooks": True,
        "no_custom_vllm_server": True,
    }


def build_main_contract(
    *,
    freeze: Mapping[str, Any],
    freeze_path: Path,
    candidate_path: Path,
    seqlen: Mapping[str, Any],
    verl_isolated: Mapping[str, Any],
    project_commit: str | None,
    ppo_max_token_len_per_gpu: int,
) -> dict[str, Any]:
    freeze_sha = sha256_file(freeze_path) if Path(freeze_path).is_file() else None
    candidate_sha = sha256_file(candidate_path) if Path(candidate_path).is_file() else None
    expected = expected_candidate_sha256(freeze)
    train_batch = TRAIN_BATCH_SIZE
    group_n = GROUP_N
    n_pool = int(freeze.get("train_candidate_manifest", {}).get("n_selected") or N_CANDIDATES)
    if n_pool % train_batch != 0:
        raise ValueError(f"n_pool={n_pool} is not divisible by train_batch_size={train_batch}")
    steps = n_pool // train_batch
    if steps != MAIN_STEPS:
        raise ValueError(f"derived steps {steps} != frozen MAIN_STEPS {MAIN_STEPS}")
    return {
        "schema_version": "bcrl-stage1-m5-main-v1",
        "milestone": "M5B",
        "experiment_id": M5B_EXPERIMENT_ID,
        "inherits_m3c_freeze": {
            "path": str(freeze_path),
            "sha256": freeze_sha,
            "schema_version": freeze.get("schema_version"),
        },
        "train_candidate_manifest": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
            "expected_sha256": expected,
            "n_selected": n_pool,
            "ordered_ids_sha256": freeze.get("train_candidate_manifest", {}).get(
                "ordered_ids_sha256"
            ),
        },
        "inherited_m3c": inherited_m3c_knobs(freeze),
        "inherited_m4_runtime": m4_validated_knobs(),
        "newly_frozen": {
            "gpu": {
                "n_gpus": N_GPUS,
                "nnodes": 1,
                "tensor_model_parallel_size": TENSOR_MODEL_PARALLEL_SIZE,
                "device": "1xA100-40GB",
            },
            "data": {
                "train_batch_size": train_batch,
                "shuffle": False,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "max_prompt_length": PROMPT_LENGTH,
                "max_response_length": RESPONSE_LENGTH,
            },
            "actor": {
                "ppo_mini_batch_size": train_batch,
                "ppo_epochs": 1,
                "use_dynamic_bsz": True,
                "ppo_max_token_len_per_gpu": int(ppo_max_token_len_per_gpu),
                "log_prob_max_token_len_per_gpu": MAX_MODEL_LEN,
                "calculate_entropy": True,
                "entropy_coeff": 0.0,
                "use_kl_loss": False,
                "optim_lr": ACTOR_LR,
                "lr_warmup_steps": 0,
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "use_kl_in_reward": False,
                "rollout_n": group_n,
                "vllm_n": 1,
            },
            "trainer": {
                "total_epochs": 1,
                "total_training_steps": steps,
                "val_before_train": False,
                "test_freq": -1,
                "save_freq": 8,
                "max_actor_ckpt_to_keep": 2,
                "resume_mode": "auto",
                "logger": ["console", "wandb"],
                "project_name": WANDB_PROJECT,
                "experiment_name": f"{M5B_EXPERIMENT_ID}-m5-main",
                "seed": SEED,
                "critic_enable": False,
            },
            "checkpoint": {
                "backend": "FSDPCheckpointManager",
                "contents": ["model", "optimizer", "extra"],
                "approx_size_gib": 9.2,
                "retention_n": 2,
                "directory_template": "$BCRL_DATA_ROOT/checkpoints/stage1_m5_main",
                "no_lora_only_migration": True,
            },
            "hard_stop": {
                "one_pass_over_frozen_pool": True,
                "silent_second_epoch": False,
                "continuation_requires_new_version": True,
                "abort_on": [
                    "OOM",
                    "veRL pin mismatch",
                    "M3C freeze hash mismatch",
                    "shared dirty veRL import",
                ],
            },
        },
        "token_length": dict(seqlen),
        "runtime_provenance": {
            "budget_coder_rl_commit": project_commit,
            "verl": dict(verl_isolated),
        },
        "notes": [
            "Do not silently edit this file after READY_FOR_M5B=true.",
            "Second pass over the 256-task pool is a versioned continuation.",
            "Pilot weights are not M6 candidates.",
        ],
        "gate": {
            "READY_FOR_M5B": False,
            "immutable": False,
            "pilot_experiment_id": EXPERIMENT_ID,
        },
    }


def build_pilot_overlay(
    *,
    output_dir: Path,
    n_steps: int = PILOT_STEPS,
    n_tasks: int = TRAIN_BATCH_SIZE,
) -> dict[str, Any]:
    return {
        "schema_version": "bcrl-stage1-m5a-pilot-v1",
        "milestone": MILESTONE,
        "experiment_id": EXPERIMENT_ID,
        "disposable": True,
        "not_m6_candidate": True,
        "inherits": MAIN_CONFIG_RELPATH,
        "n_pilot_tasks": int(n_tasks) * int(n_steps),
        "n_pilot_steps": int(n_steps),
        "pilot_instance_policy": "frozen ordered_ids prefix; no mixed-reward cherry-pick",
        "overrides": {
            "experiment_id": EXPERIMENT_ID,
            "trainer": {
                "total_training_steps": int(n_steps),
                "experiment_name": f"{EXPERIMENT_ID}-m5a-pilot",
                "default_local_dir": str(Path(output_dir) / "checkpoints"),
                "save_freq": 1,
                "max_actor_ckpt_to_keep": 1,
                "resume_mode": "disable",
            },
            "checkpoint": {
                "directory_template": str(Path(output_dir) / "checkpoints"),
                "retention_n": 1,
            },
        },
        "notes": [
            "Pilot uses the same LoRA/LR/batch/token/GPU knobs as stage1_m5_main.json.",
            "Only duration, experiment id, and checkpoint retention differ.",
        ],
    }


def merge_main_and_pilot(
    main: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    merged = json.loads(json.dumps(main))
    overrides = overlay.get("overrides") if isinstance(overlay.get("overrides"), MappingABC) else {}
    trainer_over = overrides.get("trainer") if isinstance(overrides.get("trainer"), MappingABC) else {}
    ckpt_over = overrides.get("checkpoint") if isinstance(overrides.get("checkpoint"), MappingABC) else {}
    newly = merged.setdefault("newly_frozen", {})
    trainer = newly.setdefault("trainer", {})
    checkpoint = newly.setdefault("checkpoint", {})
    for key, value in trainer_over.items():
        trainer[str(key)] = value
    for key, value in ckpt_over.items():
        checkpoint[str(key)] = value
    if overrides.get("experiment_id"):
        merged["experiment_id"] = overrides["experiment_id"]
    merged["milestone"] = overlay.get("milestone", merged.get("milestone"))
    merged["disposable"] = overlay.get("disposable", False)
    return merged


def pilot_override_errors(
    overlay: Mapping[str, Any],
    *,
    main: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    extra = [key for key in overlay.keys() if str(key) not in ALLOWED_PILOT_OVERRIDE_KEYS]
    if extra:
        errors.append(f"pilot overlay has disallowed keys: {extra}")
    overrides = overlay.get("overrides") if isinstance(overlay.get("overrides"), MappingABC) else {}
    allowed_override_sections = {"experiment_id", "trainer", "checkpoint"}
    extra_sections = [key for key in overrides.keys() if str(key) not in allowed_override_sections]
    if extra_sections:
        errors.append(f"pilot overrides has disallowed keys: {extra_sections}")
    trainer = overrides.get("trainer") if isinstance(overrides.get("trainer"), MappingABC) else {}
    bad_trainer = [key for key in trainer.keys() if str(key) not in PILOT_TRAINER_OVERRIDE_KEYS]
    if bad_trainer:
        errors.append(f"pilot trainer overrides disallowed: {bad_trainer}")
    if main is not None:
        inherited = (main.get("inherited_m3c") or {}).get("primary_training_B_obs")
        if inherited != OBS_TOKENS_LIMIT:
            errors.append("merged main lost B_obs=4096")
        newly = main.get("newly_frozen") if isinstance(main.get("newly_frozen"), MappingABC) else {}
        actor = newly.get("actor") if isinstance(newly.get("actor"), MappingABC) else {}
        data = newly.get("data") if isinstance(newly.get("data"), MappingABC) else {}
        if int(data.get("train_batch_size") or 0) != TRAIN_BATCH_SIZE:
            errors.append("main train_batch_size != 8")
        if actor.get("use_kl_loss") is True:
            errors.append("main use_kl_loss must stay false")
    return errors


def _is_clean_pin(info: Mapping[str, Any], pinned: str) -> bool:
    return str(info.get("commit") or "") == str(pinned) and not bool(info.get("dirty"))


def ensure_isolated_verl_checkout(
    *,
    isolated_root: Path,
    source_git: Path,
    pinned_commit: str = PINNED_VERL_COMMIT,
    create: bool = True,
) -> dict[str, Any]:
    isolated = Path(isolated_root).resolve()
    source = Path(source_git).resolve()
    shared = SHARED_VERL_ROOT.resolve()
    if isolated == shared:
        raise SystemExit(
            "HARD FAIL: isolated veRL root must not be the shared dirty ~/verl tree"
        )
    if isolated.exists():
        info = git_info(isolated)
        info["isolated_root"] = str(isolated)
        info["source_git"] = str(source)
        info["created"] = False
        info["matches_pin"] = str(info.get("commit") or "") == pinned_commit
        if not _is_clean_pin(info, pinned_commit):
            raise SystemExit(
                "HARD FAIL: existing isolated veRL is not a clean pin "
                f"commit={info.get('commit')!r} dirty={info.get('dirty')} "
                f"path={isolated}"
            )
        return info
    if not create:
        raise FileNotFoundError(f"isolated veRL checkout missing: {isolated}")
    if not source.is_dir():
        raise SystemExit(f"HARD FAIL: source veRL git missing: {source}")
    isolated.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "git",
            "-C",
            str(source),
            "worktree",
            "add",
            "--detach",
            str(isolated),
            pinned_commit,
        ]
    )
    info = git_info(isolated)
    info["isolated_root"] = str(isolated)
    info["source_git"] = str(source)
    info["created"] = True
    info["matches_pin"] = str(info.get("commit") or "") == pinned_commit
    if not _is_clean_pin(info, pinned_commit):
        raise SystemExit(
            "HARD FAIL: newly created isolated veRL is not clean/pinned: "
            f"{info}"
        )
    return info


def isolated_pythonpath(isolated_root: Path, repo_root: Path) -> str:
    isolated = str(Path(isolated_root).resolve())
    src = str(Path(repo_root).resolve() / "src")
    shared = str(SHARED_VERL_ROOT.resolve())
    sibling = str(PREFERRED_SIBLING_VERL.resolve())
    parts = [isolated, src]
    existing = os.environ.get("PYTHONPATH") or ""
    for item in existing.split(os.pathsep):
        if not item:
            continue
        resolved = str(Path(item).resolve()) if item else item
        if resolved in {shared, sibling, isolated, src}:
            continue
        if resolved.endswith(str(Path("verl"))) and resolved != isolated:
            continue
        parts.append(item)
    return os.pathsep.join(parts)


def prepend_isolated_verl(isolated_root: Path, repo_root: Path) -> str:
    isolated = str(Path(isolated_root).resolve())
    merged = isolated_pythonpath(isolated_root, repo_root)
    os.environ["PYTHONPATH"] = merged
    import sys

    if isolated in sys.path:
        sys.path.remove(isolated)
    sys.path.insert(0, isolated)
    src = str(Path(repo_root).resolve() / "src")
    if src not in sys.path:
        sys.path.insert(1, src)
    return merged


def imported_verl_errors(
    *,
    isolated_root: Path,
    pinned_commit: str = PINNED_VERL_COMMIT,
) -> tuple[list[str], dict[str, Any]]:
    import verl

    isolated = Path(isolated_root).resolve()
    source = Path(verl.__file__).resolve().parents[1]
    info = git_info(source)
    info["version"] = getattr(verl, "__version__", None)
    info["import_path"] = str(Path(verl.__file__).resolve())
    info["source_root"] = str(source)
    info["isolated_root"] = str(isolated)
    errors: list[str] = []
    if source == SHARED_VERL_ROOT.resolve():
        errors.append("imported veRL is the shared dirty ~/verl tree")
    if source != isolated:
        errors.append(
            f"imported veRL source {source} != isolated checkout {isolated}"
        )
    if str(info.get("commit") or "") != pinned_commit:
        errors.append(
            f"imported veRL commit {info.get('commit')!r} != pin {pinned_commit}"
        )
    if info.get("dirty"):
        errors.append(f"isolated veRL is dirty: {info.get('dirty_files')}")
    return errors, info


def write_verl_checkout_md(path: Path, info: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        VERL_CHECKOUT_TEXT.format(
            commit=info.get("commit") or PINNED_VERL_COMMIT,
            version=info.get("version") or PINNED_VERL_VERSION,
            isolated=info.get("isolated_root") or info.get("path"),
            source=info.get("source_git") or "",
            dirty=info.get("dirty"),
            matches=info.get("matches_pin"),
        ),
        encoding="utf-8",
    )


def coerce_sequence(value: Any) -> list[Any]:
    """Convert non-tensor batch fields to a list without truth-testing arrays."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, str)) and str(value).lower() in {"true", "false"}:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray)):
        return False
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    if hasattr(value, "shape") and getattr(value, "shape", None) not in {(), None}:
        try:
            return bool(value.any())
        except Exception:
            return False
    try:
        return bool(value)
    except (ValueError, TypeError):
        return False


def compute_bcrl_step_metrics(
    *,
    uids: Sequence[str],
    rewards: Sequence[float],
    extra_rows: Sequence[Mapping[str, Any]],
    group_n: int = GROUP_N,
) -> dict[str, Any]:
    if len(uids) != len(rewards):
        raise ValueError("uids/rewards length mismatch")
    grouped: dict[str, list[float]] = defaultdict(list)
    order: list[str] = []
    for uid, reward in zip(uids, rewards):
        key = str(uid)
        if key not in grouped:
            order.append(key)
        grouped[key].append(float(reward))
    group_stats = [group_reward_stats(grouped[uid]) for uid in order]
    n_groups = len(group_stats)
    n_mixed = sum(1 for item in group_stats if item.get("mixed"))
    n_zero = sum(1 for item in group_stats if item.get("zero_variance"))
    n_nonzero_adv = 0
    for item in group_stats:
        values = list(item.get("rewards") or [])
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        if any(abs(value - mean) > ADVANTAGE_ABS_EPS for value in values):
            n_nonzero_adv += 1
    file_f1 = [_as_float(row.get("file_f1")) for row in extra_rows]
    symbol_f1 = [_as_float(row.get("symbol_f1")) for row in extra_rows]
    parse_ok = [_truthy(row.get("parse_ok")) for row in extra_rows]
    exhausted = [
        _truthy(row.get("budget_exhausted"))
        or str(row.get("termination") or "") == "budget_exhausted"
        for row in extra_rows
    ]
    obs_used = [_as_float(row.get("obs_tokens_used")) for row in extra_rows]
    policy_tokens = [_as_float(row.get("policy_token_count")) for row in extra_rows]
    prompt_tokens = [_as_float(row.get("prompt_token_count")) for row in extra_rows]
    obs_tokens = [_as_float(row.get("observation_token_count")) for row in extra_rows]
    invalid = 0
    n_protocol = 0
    for row in extra_rows:
        events_raw = row.get("events")
        if isinstance(events_raw, list):
            events = events_raw
        else:
            events = [
                item
                for item in coerce_sequence(events_raw)
                if isinstance(item, MappingABC)
            ]
        counts = action_counts(events, termination=row.get("termination"))
        n_protocol += int(counts.get("n_protocol_errors") or 0)
        if int(counts.get("n_protocol_errors") or 0) or int(counts.get("n_tool_errors") or 0):
            invalid += 1
    n = len(rewards)
    seq = []
    for prompt, policy, observation in zip(prompt_tokens, policy_tokens, obs_tokens):
        if prompt is None:
            continue
        seq.append(float(prompt) + float(policy or 0) + float(observation or 0))

    def _mean(values: Sequence[float | None]) -> float | None:
        present = [float(item) for item in values if item is not None]
        if not present:
            return None
        return sum(present) / len(present)

    def _std(values: Sequence[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
        return math.sqrt(var)

    reward_list = [float(item) for item in rewards]
    return {
        "bcrl/reward/mean": _mean(reward_list),
        "bcrl/reward/std": _std(reward_list) if reward_list else 0.0,
        "bcrl/group/n": n_groups,
        "bcrl/group/mixed_fraction": (n_mixed / n_groups) if n_groups else None,
        "bcrl/group/zero_var_fraction": (n_zero / n_groups) if n_groups else None,
        "bcrl/group/nonzero_adv_fraction": (n_nonzero_adv / n_groups) if n_groups else None,
        "bcrl/group/reward_std_mean": _mean([float(item.get("std") or 0.0) for item in group_stats]),
        "bcrl/file_f1/mean": _mean(file_f1),
        "bcrl/symbol_f1/mean": _mean(symbol_f1),
        "bcrl/parse_ok_rate": (sum(1 for item in parse_ok if item) / n) if n else None,
        "bcrl/budget_exhaustion_rate": (sum(1 for item in exhausted if item) / n) if n else None,
        "bcrl/invalid_action_rate": (invalid / n) if n else None,
        "bcrl/protocol_error_count": n_protocol,
        "bcrl/obs_tokens_used/mean": _mean(obs_used),
        "bcrl/policy_tokens/mean": _mean(policy_tokens),
        "bcrl/prompt_tokens/mean": _mean(prompt_tokens),
        "bcrl/seq/training_proxy_mean": _mean(seq),
        "bcrl/seq/training_proxy_max": max(seq) if seq else None,
        "bcrl/n_trajectories": n,
        "bcrl/group_n_expected": int(group_n),
        "bcrl/any_nonzero_advantage": bool(n_nonzero_adv),
        "bcrl/any_mixed_group": bool(n_mixed),
    }


def compact_episode_from_extra(extra: Mapping[str, Any]) -> dict[str, Any]:
    localization = {
        key: extra.get(key)
        for key in (
            "parse_ok",
            "file_f1",
            "symbol_f1",
            "localization_score",
            "symbol_status",
            "submission_missing",
        )
        if key in extra
    }
    record = build_episode_record(extra, localization=localization or None)
    record.pop("unpadded_prompt_ids", None)
    identity_tokens = record.get("tokens") if isinstance(record.get("tokens"), dict) else {}
    # Drop raw token-id segments; keep counts only.
    record["segments"] = [
        {"kind": item.get("kind"), "n_tokens": item.get("n_tokens")}
        for item in (record.get("segments") or [])
    ]
    record["trace_role"] = "research_debug_not_training_tokens"
    record["tokens"] = identity_tokens
    return record


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True, default=str) + "\n")


def m5_freeze_consume_errors(
    freeze: Mapping[str, Any],
    *,
    freeze_path: Path,
    candidate_path: Path,
) -> list[str]:
    errors = list(freeze_contract_errors(freeze))
    if freeze.get("not_trained") is not True:
        errors.append("M3C freeze not_trained is not true (file was edited)")
    expected = expected_candidate_sha256(freeze)
    actual = sha256_file(candidate_path) if Path(candidate_path).is_file() else ""
    if expected and actual and expected != actual:
        errors.append(
            f"train candidate sha256 {actual} != freeze.train_candidate_manifest.file_sha256 {expected}"
        )
    if not Path(freeze_path).is_file():
        errors.append(f"missing freeze {freeze_path}")
    return errors


def ready_for_m5b_errors(evidence: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not evidence.get("verl_isolated_clean"):
        reasons.append("veRL checkout is not isolated/clean/pinned")
    if not evidence.get("m3c_freeze_ok"):
        reasons.append("M3C freeze consume check failed")
    if not evidence.get("seqlen_characterized"):
        reasons.append("token-length characterization missing")
    if not evidence.get("ppo_max_token_len_frozen"):
        reasons.append("ppo_max_token_len_per_gpu not frozen")
    if evidence.get("assert_risk_if_8192") and int(
        evidence.get("ppo_max_token_len_per_gpu") or 0
    ) <= 8192:
        reasons.append("frozen token cap still 8192 despite seqs > 8192")
    if not evidence.get("pilot_completed"):
        reasons.append("short multi-step rehearsal did not complete")
    if evidence.get("pilot_oom"):
        reasons.append("pilot hit OOM")
    if int(evidence.get("n_steps_completed") or 0) < 2:
        reasons.append("pilot completed fewer than 2 optimizer steps")
    if int(evidence.get("n_steps_nonzero_advantage") or 0) < 1:
        reasons.append("no pilot step had non-zero GRPO advantage")
    if not evidence.get("metrics_jsonl_present"):
        reasons.append("metrics.jsonl missing")
    if not evidence.get("wandb_logged"):
        reasons.append("W&B run was not logged")
    if not evidence.get("main_config_written"):
        reasons.append("stage1_m5_main.json missing")
    if not evidence.get("checkpoint_policy_frozen"):
        reasons.append("checkpoint/resume policy not frozen")
    return reasons


def m5a_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    reasons = ready_for_m5b_errors(evidence)
    return {
        "pass": not reasons,
        "READY_FOR_M5B": not reasons,
        "reasons": reasons,
    }


def main_contract_immutable(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("gate") if isinstance(payload.get("gate"), MappingABC) else {}
    return bool(gate.get("immutable") or gate.get("READY_FOR_M5B"))


def stamp_main_contract_gate(
    payload: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    stamped_at: str,
) -> dict[str, Any]:
    stamped = json.loads(json.dumps(payload, default=str))
    stamped["gate"] = {
        "READY_FOR_M5B": True,
        "immutable": True,
        "pilot_experiment_id": EXPERIMENT_ID,
        "m5b_experiment_id": M5B_EXPERIMENT_ID,
        "evidence_dir": "outputs/experiments/E010",
        "ready_for_m5b_path": "outputs/experiments/E010/READY_FOR_M5B.json",
        "n_pilot_steps": int(evidence.get("n_steps_completed") or 0),
        "n_steps_nonzero_advantage": int(evidence.get("n_steps_nonzero_advantage") or 0),
        "ppo_max_token_len_per_gpu": int(evidence.get("ppo_max_token_len_per_gpu") or 0),
        "wandb_run_id": ((evidence.get("wandb_run") or {}) if isinstance(evidence.get("wandb_run"), MappingABC) else {}).get("id"),
        "stamped_at": stamped_at,
    }
    return stamped


def checkpoint_shard_manifest(path: Path) -> dict[str, Any]:
    root = Path(path)
    summary = checkpoint_dir_manifest(root)
    shards: list[dict[str, Any]] = []
    if root.exists():
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            size = int(file_path.stat().st_size)
            shards.append(
                {
                    "relpath": str(file_path.relative_to(root)),
                    "size_bytes": size,
                    "size_mib": round(size / (1024**2), 2),
                }
            )
    summary["shards"] = shards
    return summary


def checkpoint_dir_manifest(path: Path) -> dict[str, Any]:
    root = Path(path)
    if not root.exists():
        return {"path": str(root), "exists": False, "n_files": 0, "size_bytes": 0}
    n_files = 0
    size = 0
    steps: list[str] = []
    for current, _dirs, files in os.walk(root):
        if Path(current).name.startswith("global_step_"):
            steps.append(Path(current).name)
        for name in files:
            file_path = Path(current) / name
            try:
                n_files += 1
                size += int(file_path.stat().st_size)
            except OSError:
                continue
    return {
        "path": str(root),
        "exists": True,
        "n_files": n_files,
        "size_bytes": size,
        "size_gib": round(size / (1024**3), 3),
        "global_steps": sorted(set(steps)),
    }


def summarize_metrics_jsonl(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {"path": str(path), "n_rows": 0, "steps": []}
    steps: list[int] = []
    n_nonzero = 0
    n_mixed = 0
    rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows += 1
            step = payload.get("step")
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), MappingABC) else payload
            if step is not None:
                steps.append(int(step))
            if metrics.get("bcrl/any_nonzero_advantage"):
                n_nonzero += 1
            if metrics.get("bcrl/any_mixed_group"):
                n_mixed += 1
    unique_steps = sorted(set(steps))
    return {
        "path": str(path),
        "n_rows": rows,
        "steps": unique_steps,
        "n_steps": len(unique_steps),
        "n_steps_nonzero_advantage": n_nonzero,
        "n_steps_mixed": n_mixed,
    }
