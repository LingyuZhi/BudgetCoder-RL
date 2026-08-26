"""Scaled M5 contract: 2193 unique + hash padding, not prototype 256.

Does not edit M3C/M5-main/E014/E015 freeze files. Does not start the
275-step main run. Oracle is used only for symbol_applicable class
counts, never as a drop or reward-cherry-pick rule.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_materialize import EXPECTED_TRAIN_ROWS
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.e013 import CHOSEN_ENVELOPE
from budget_coder_rl.eval.e014 import EXPECTED_CANONICAL_SHA256
from budget_coder_rl.eval.m3b import repo_round_robin_ids, sha256_ids
from budget_coder_rl.eval.m3c import OVERLONG_INSTANCE_IDS
from budget_coder_rl.eval.m4a import GROUP_N, OBS_TOKENS_LIMIT, load_json
from budget_coder_rl.eval.m4b import LORA_ALPHA, LORA_RANK, PROMPT_LENGTH, RESPONSE_LENGTH
from budget_coder_rl.eval.m5a import (
    ACTOR_LR,
    MAX_MODEL_LEN,
    SEED,
    TRAIN_BATCH_SIZE,
    inherited_m3c_knobs,
    m4_validated_knobs,
)
from budget_coder_rl.eval.m5b import (
    EXPECTED_M3C_SHA256,
    EXPECTED_MAIN_SHA256,
    EXPECTED_N_GPUS,
    EXPECTED_NNODES,
    EXPECTED_TP,
    expected_hybrid_placement,
)
from budget_coder_rl.eval.m5_scale_audit import (
    EXPECTED_CANDIDATE_FILE_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    M1D_SPLIT_RELPATH,
    load_m1d_train_rows,
    load_train_identity_rows,
)
from budget_coder_rl.eval.provenance import sha256_file

CANDIDATE_SCHEMA = "bcrl-m5-scaled-train-candidates-v1"
CONTRACT_SCHEMA = "bcrl-stage1-m5-scaled-v1"
PREFLIGHT_SCHEMA = "bcrl-stage1-m5-scaled-e016-preflight-v1"
CANDIDATE_RELPATH = "data/manifests/m5_scaled_train_candidates.json"
CONTRACT_RELPATH = "configs/experiments/stage1_m5_scaled.json"
CONTRACT_LOCK_RELPATH = "configs/experiments/stage1_m5_scaled.lock.json"
PREFLIGHT_RELPATH = "configs/experiments/stage1_m5_scaled_e016_preflight.json"
PREFLIGHT_LOCK_RELPATH = "configs/experiments/stage1_m5_scaled_e016_preflight.lock.json"
CANONICAL_ENVELOPE_RELPATH = "configs/experiments/stage1_canonical_execution_envelope.json"
M3C_FREEZE_RELPATH = "configs/experiments/stage1_m3c_freeze.json"
M3C_CANDIDATE_RELPATH = "data/manifests/m3c_train_candidates.json"
M5_MAIN_RELPATH = "configs/experiments/stage1_m5_main.json"
E014_RUNTIME_RELPATH = "configs/experiments/stage1_m5_e014_runtime.json"
MILESTONE = "M5-SCALED"
EXPERIMENT_ID = "scaled-m5"
PREFLIGHT_EXPERIMENT_ID = "E016"
PREFLIGHT_SESSION_NAME = "E016"
WANDB_EXPERIMENT_NAME = "E016-scaled-preflight"
CHECKPOINT_RELPATH = "checkpoints/stage1_m5_scaled"
PREFLIGHT_CHECKPOINT_RELPATH = "checkpoints/stage1_m5_scaled_e016"

PADDING_SALT = "stage1-m5-scaled-padding-v1"
PADDING_POLICY_NAME = "sha256_salt_pipe_instance_id_lowest_7"
PADDING_RULE = (
    'salt = "stage1-m5-scaled-padding-v1"; '
    'key(id) = SHA256(utf-8("{salt}|{instance_id}")).hexdigest(); '
    "pad_ids = first 7 of unique_ids sorted by (key(id), id); "
    "padded_ids = unique_ordered_ids + pad_ids. "
    "Padding is dataloader-only (drop_last remainder). "
    "It does not change scientific eligibility."
)
N_UNIQUE = 2193
N_PAD = 7
N_ROWS = 2200
MAIN_STEPS = 275
PREFLIGHT_STEPS = 2
PREFLIGHT_MAX_STEPS = 2
N_TRAJECTORIES = N_ROWS * GROUP_N
PPO_MAX_TOKEN_LEN = CHOSEN_ENVELOPE
SAVE_FREQ = 32
MAX_ACTOR_CKPT_TO_KEEP = 2
CKPT_SHARD_GIB = 9.2
DISK_MIN_GIB = 100.0
N_SYMBOL_UNAVAILABLE = 145
N_SYMBOL_APPLICABLE = 2048
EXPECTED_UNIQUE_IDS_SHA256 = (
    "a32795cd5515a9465068fce0fe9deef334fa919b4fa79e5a2735b855905c0e32"
)
EXPECTED_PADDED_IDS_SHA256 = (
    "24c27349d4476c7df4dedd00a33ba000a8f9e8677509bc7d0cd05269b4316b7a"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "df785b9e35c3b6403e6f5cc0819ea18d94bf2cb43bac7de9b38473cbaaea5855"
)
EXPECTED_CONTRACT_SHA256 = (
    "672f064399a1d42062dd4360b4bd22b30f101988f3325e29338781e934e9ae8a"
)
EXPECTED_PAD_IDS = (
    "pydantic__pydantic-8650",
    "Project-MONAI__MONAI-2837",
    "python__mypy-11140",
    "Project-MONAI__MONAI-5117",
    "Project-MONAI__MONAI-2321",
    "pandas-dev__pandas-50719",
    "python__mypy-10415",
)
PREFIX_REPEAT_PAD_IDS = (
    "Project-MONAI__MONAI-1010",
    "bokeh__bokeh-12779",
    "conan-io__conan-10213",
    "dask__dask-10009",
    "facebookresearch__hydra-1006",
    "getmoto__moto-4787",
    "iterative__dvc-10213",
)
EXPECTED_E014_RUNTIME_SHA256 = (
    "850e6830237f6697d60c58805e68e386277257c2f539def8cb825bb9e1f8c69a"
)
SYMBOL_UNAVAILABLE_IN_FIRST_BATCH = "conan-io__conan-10213"
E014_ELAPSED_S = 5851.6
E014_STEPS = 32

FORBIDDEN_OUTPUT_IDS = ("E011", "E012", "E013", "E014", "E015")


def default_candidate_path(repo_root: Path) -> Path:
    return Path(repo_root) / CANDIDATE_RELPATH


def default_contract_path(repo_root: Path) -> Path:
    return Path(repo_root) / CONTRACT_RELPATH


def default_contract_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / CONTRACT_LOCK_RELPATH


def default_preflight_path(repo_root: Path) -> Path:
    return Path(repo_root) / PREFLIGHT_RELPATH


def default_preflight_lock_path(repo_root: Path) -> Path:
    return Path(repo_root) / PREFLIGHT_LOCK_RELPATH


def default_preflight_output_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "outputs" / "experiments" / PREFLIGHT_EXPERIMENT_ID


def default_scaled_checkpoint_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / CHECKPOINT_RELPATH


def default_preflight_checkpoint_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / PREFLIGHT_CHECKPOINT_RELPATH


def padding_key(instance_id: str, *, salt: str = PADDING_SALT) -> str:
    blob = f"{salt}|{instance_id}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def select_hash_padding_ids(
    unique_ids: Sequence[str],
    *,
    n_pad: int = N_PAD,
    salt: str = PADDING_SALT,
) -> list[str]:
    unique = [str(item) for item in unique_ids]
    ranked = sorted(unique, key=lambda item: (padding_key(item, salt=salt), item))
    selected = ranked[: int(n_pad)]
    if len(selected) != int(n_pad):
        raise ValueError(f"need {n_pad} padding ids, got {len(selected)}")
    if len(set(selected)) != len(selected):
        raise ValueError("padding ids are not unique")
    return selected


def pad_unique_ids(
    unique_ids: Sequence[str],
    *,
    batch_size: int = TRAIN_BATCH_SIZE,
    salt: str = PADDING_SALT,
) -> dict[str, Any]:
    unique = [str(item) for item in unique_ids]
    n_unique = len(unique)
    remainder = n_unique % int(batch_size) if unique else 0
    pad_n = 0 if remainder == 0 else int(batch_size) - remainder
    pad_ids = select_hash_padding_ids(unique, n_pad=pad_n, salt=salt) if pad_n else []
    padded = unique + pad_ids
    n_rows = len(padded)
    if pad_n and n_rows % int(batch_size) != 0:
        raise ValueError(f"padded rows {n_rows} not divisible by batch {batch_size}")
    return {
        "policy_name": PADDING_POLICY_NAME,
        "salt": salt,
        "rule": PADDING_RULE,
        "batch_size": int(batch_size),
        "n_unique": n_unique,
        "remainder_if_unpadded": remainder,
        "n_pad": pad_n,
        "pad_ids": pad_ids,
        "n_rows_padded": n_rows,
        "padded_ids": padded,
        "unique_ids_sha256": sha256_ids(unique),
        "padded_ids_sha256": sha256_ids(padded),
        "optimizer_steps": (n_rows // int(batch_size)) if padded else 0,
        "silent_drop_if_unpadded_drop_last": remainder,
        "scientific_eligibility_n": n_unique,
    }


def save_event_steps(total_steps: int, save_freq: int) -> list[int]:
    freq = int(save_freq)
    total = int(total_steps)
    if freq <= 0 or total <= 0:
        return []
    steps = list(range(freq, total + 1, freq))
    if total not in steps:
        steps.append(total)
    return steps


def checkpoint_audit() -> dict[str, Any]:
    events = save_event_steps(MAIN_STEPS, SAVE_FREQ)
    unpruned = round(len(events) * CKPT_SHARD_GIB, 1)
    if_pruned = round(MAX_ACTOR_CKPT_TO_KEEP * CKPT_SHARD_GIB, 1)
    return {
        "pinned_paths": {
            "save": "verl/trainer/ppo/ray_trainer.py RayPPOTrainer._save_checkpoint",
            "actor_retention": (
                "verl/utils/checkpoint/fsdp_checkpoint_manager.py "
                "FSDPCheckpointManager.register_checkpoint"
            ),
            "remove": "BaseCheckpointManager.remove_previous_save_local_path",
        },
        "tracked_path": "global_step_N/actor",
        "does_not_delete_parent_global_step_dir": True,
        "data_pt_location": "global_step_N/data.pt",
        "terminal_save_if_save_freq_positive": True,
        "resume_mode_auto": "latest_checkpointed_iteration.txt then actor/ + data.pt",
        "e014_empirical": (
            "max_actor_ckpt_to_keep=2 left global_step_{8,16,24,32} on disk; "
            "do not assume prune deletes global_step_*"
        ),
        "save_freq": SAVE_FREQ,
        "max_actor_ckpt_to_keep": MAX_ACTOR_CKPT_TO_KEEP,
        "save_events": events,
        "n_save_events": len(events),
        "terminal_global_step": MAIN_STEPS,
        "shard_gib": CKPT_SHARD_GIB,
        "storage_bound_unpruned_gib": unpruned,
        "storage_bound_if_actor_prune_works_gib": if_pruned,
        "disk_min_gib": DISK_MIN_GIB,
        "worst_resume_gap_steps": SAVE_FREQ - 1,
        "canonical_later_candidate": f"only global_step_{MAIN_STEPS}",
        "custom_gc_this_round": False,
    }


def predicted_main_compute() -> dict[str, Any]:
    seconds_per_step = E014_ELAPSED_S / float(E014_STEPS)
    elapsed_s = seconds_per_step * float(MAIN_STEPS)
    hours = elapsed_s / 3600.0
    return {
        "source": "linear from E014 5851.6s / 32 steps",
        "seconds_per_step": round(seconds_per_step, 3),
        "estimated_wall_clock_s": round(elapsed_s, 1),
        "estimated_wall_clock_h": round(hours, 2),
        "estimated_gpu_hours_2xA100": round(hours * 2.0, 1),
        "n_steps": MAIN_STEPS,
        "n_trajectories": N_TRAJECTORIES,
        "long_prompt_tail_may_be_slower": True,
    }


def load_old_256(repo_root: Path) -> list[str]:
    payload = load_json(Path(repo_root) / M3C_CANDIDATE_RELPATH)
    ids = [str(item) for item in payload.get("ordered_ids") or []]
    if len(ids) != 256:
        raise ValueError(f"M3C ordered_ids n={len(ids)} != 256")
    if sha256_ids(ids) != EXPECTED_ORDERED_IDS_SHA256:
        raise ValueError("M3C ordered_ids_sha256 drifted")
    return ids


def symbol_status_by_id(
    unique_ids: Sequence[str],
    *,
    oracle: Any | None,
    m3c_skipped_unavailable: Sequence[str] | None = None,
) -> dict[str, Any]:
    unique = [str(item) for item in unique_ids]
    true_ids: list[str] = []
    false_ids: list[str] = []
    unknown_ids: list[str] = []
    if oracle is not None:
        for instance_id in unique:
            if instance_id not in oracle:
                unknown_ids.append(instance_id)
                continue
            if oracle.get(instance_id).symbol_applicable:
                true_ids.append(instance_id)
            else:
                false_ids.append(instance_id)
    else:
        skipped = {str(item) for item in (m3c_skipped_unavailable or [])}
        for instance_id in unique:
            if instance_id in skipped:
                false_ids.append(instance_id)
            else:
                true_ids.append(instance_id)
    return {
        "true_ids": true_ids,
        "false_ids": false_ids,
        "unknown_ids": unknown_ids,
        "n_true": len(true_ids),
        "n_false": len(false_ids),
        "n_unknown": len(unknown_ids),
        "oracle_replayed": oracle is not None,
    }


def build_scaled_train_manifest(
    rows: Sequence[Mapping[str, str]],
    *,
    repo_root: Path,
    oracle: Any | None = None,
    identity_source: str = "m1e_train_parquet",
) -> dict[str, Any]:
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"train identities {len(rows)} != {EXPECTED_TRAIN_ROWS}")
    blocked = frozenset(str(item) for item in OVERLONG_INSTANCE_IDS)
    by_id_all = {str(row["instance_id"]): dict(row) for row in rows}
    excluded = sorted(blocked)
    # Same order as the M5 scale audit: round-robin the full train universe,
    # then drop hard-unusable. Filtering first would change repo interleaving.
    ordered_all = repo_round_robin_ids(rows)
    ordered = [iid for iid in ordered_all if iid not in blocked]
    kept_rows = [by_id_all[iid] for iid in ordered]
    if len(kept_rows) != N_UNIQUE:
        raise ValueError(f"unique pool {len(kept_rows)} != {N_UNIQUE}")
    if len(ordered) != N_UNIQUE:
        raise ValueError(f"ordered unique {len(ordered)} != {N_UNIQUE}")
    unique_hash = sha256_ids(ordered)
    if unique_hash != EXPECTED_UNIQUE_IDS_SHA256:
        raise ValueError(
            f"unique ordered_ids_sha256 {unique_hash} != {EXPECTED_UNIQUE_IDS_SHA256}"
        )
    old_256 = load_old_256(repo_root)
    old_set = set(old_256)
    ordered_set = set(ordered)
    missing_old = [item for item in old_256 if item not in ordered_set]
    if missing_old:
        raise ValueError(f"old M3C 256 not subset of scaled pool: {missing_old[:8]}")
    if any(item in ordered_set for item in excluded):
        raise ValueError("overlong id leaked into scaled unique pool")

    m3c = load_json(Path(repo_root) / M3C_CANDIDATE_RELPATH)
    skipped_symbol = [
        str(item) for item in (m3c.get("skipped") or {}).get("symbol_unavailable") or []
    ]
    symbol = symbol_status_by_id(
        ordered, oracle=oracle, m3c_skipped_unavailable=skipped_symbol
    )
    if symbol["n_false"] != N_SYMBOL_UNAVAILABLE:
        raise ValueError(
            f"symbol_applicable=false n={symbol['n_false']} != {N_SYMBOL_UNAVAILABLE}"
        )
    if symbol["n_unknown"]:
        raise ValueError(f"symbol_applicable unknown: {symbol['unknown_ids'][:8]}")
    if symbol["n_true"] != N_SYMBOL_APPLICABLE:
        raise ValueError(
            f"symbol_applicable=true n={symbol['n_true']} != {N_SYMBOL_APPLICABLE}"
        )

    pad = pad_unique_ids(ordered)
    if pad["n_pad"] != N_PAD or pad["n_rows_padded"] != N_ROWS:
        raise ValueError(f"pad arithmetic drifted: {pad['n_pad']} {pad['n_rows_padded']}")
    if pad["optimizer_steps"] != MAIN_STEPS:
        raise ValueError(f"optimizer_steps {pad['optimizer_steps']} != {MAIN_STEPS}")
    if tuple(pad["pad_ids"]) != EXPECTED_PAD_IDS:
        raise ValueError(f"pad_ids {pad['pad_ids']} != frozen EXPECTED_PAD_IDS")
    if pad["padded_ids_sha256"] != EXPECTED_PADDED_IDS_SHA256:
        raise ValueError("padded_ids_sha256 drifted")
    if pad["pad_ids"] == list(PREFIX_REPEAT_PAD_IDS):
        raise ValueError("padding accidentally reused prefix-repeat IDs")

    by_id = {str(row["instance_id"]): row for row in kept_rows}
    repo_counts = Counter(str(by_id[iid]["repo"]) for iid in ordered)
    groups = {
        str(by_id[iid].get("correlation_group_id") or "")
        for iid in ordered
    }
    groups.discard("")
    false_set = set(symbol["false_ids"])
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "split": "train",
        "milestone": MILESTONE,
        "identity_source": identity_source,
        "selection_algorithm": (
            "repo_name_sort + instance_id_lexicographic + repo_round_robin "
            "over full M1E train, then drop hard-unusable "
            "(Project-MONAI__MONAI-6344); matches M5_SCALE_AUDIT unique order"
        ),
        "oracle_used_for_class_filter": True,
        "oracle_used_as_drop": False,
        "gold_used_for_cherry_pick": False,
        "reward_used_for_selection": False,
        "zero_variance_used_as_drop": False,
        "rule_text": (
            "Start from frozen M1E train (2194). Exclude only "
            "Project-MONAI__MONAI-6344. Keep all remaining identities, including "
            "symbol_applicable=false (file-only localization_score). Order with "
            "repo_name_sort + instance_id_lexicographic + repo_round_robin. "
            "Append 7 hash-selected padding rows so 2193 unique -> 2200 "
            "dataloader rows under drop_last=True. Do not cherry-pick by "
            "E014/E015 reward or performance."
        ),
        "n_universe": EXPECTED_TRAIN_ROWS,
        "n_unique": N_UNIQUE,
        "n_rows": N_ROWS,
        "n_pad": N_PAD,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "optimizer_steps": MAIN_STEPS,
        "group_n": GROUP_N,
        "expected_trajectories": N_TRAJECTORIES,
        "excluded_from_m1e_train": excluded,
        "skipped": {
            "overlong_prompt": excluded,
            "symbol_unavailable_kept": symbol["false_ids"],
            "missing_oracle": [],
        },
        "symbol_applicable_true": symbol["n_true"],
        "symbol_applicable_false": symbol["n_false"],
        "symbol_applicable_unknown": symbol["n_unknown"],
        "oracle_replayed": symbol["oracle_replayed"],
        "old_256_coverage_n": len(old_256),
        "old_256_coverage_ratio": 1.0,
        "old_256_is_prefix": ordered[:256] == old_256,
        "n_repos": len(repo_counts),
        "n_correlation_groups": len(groups),
        "per_repo_counts": dict(sorted(repo_counts.items())),
        "padding": {
            "policy_name": pad["policy_name"],
            "salt": pad["salt"],
            "rule": pad["rule"],
            "n_pad": pad["n_pad"],
            "pad_ids": pad["pad_ids"],
            "remainder_if_unpadded": pad["remainder_if_unpadded"],
            "silent_drop_if_unpadded_drop_last": pad["silent_drop_if_unpadded_drop_last"],
            "scientific_eligibility_n": pad["scientific_eligibility_n"],
        },
        "ordered_ids": ordered,
        "padded_ids": pad["padded_ids"],
        "unique_ids_sha256": unique_hash,
        "padded_ids_sha256": pad["padded_ids_sha256"],
        "ordered_ids_sha256": unique_hash,
        "tasks": [
            {
                "task_index": index,
                "instance_id": instance_id,
                "repo": by_id[instance_id]["repo"],
                "symbol_applicable": instance_id not in false_set,
                "padding_row": False,
            }
            for index, instance_id in enumerate(ordered)
        ],
        "padding_rows": [
            {
                "row_index": N_UNIQUE + index,
                "instance_id": instance_id,
                "repo": by_id[instance_id]["repo"],
                "padding_row": True,
            }
            for index, instance_id in enumerate(pad["pad_ids"])
        ],
        "notes": [
            "unique N=2193 is scientific eligibility.",
            "2200 rows exist only so StatefulDataLoader(drop_last=True) does not silent-drop.",
            "symbol_unavailable uses frozen file-only localization_score; reward formula unchanged.",
        ],
    }


def load_padded_ids(path: Path) -> list[str]:
    payload = load_json(path)
    ids = [str(item) for item in payload.get("padded_ids") or []]
    if len(ids) != N_ROWS:
        raise ValueError(f"{path} padded_ids n={len(ids)} != {N_ROWS}")
    return ids


def load_unique_ids(path: Path) -> list[str]:
    payload = load_json(path)
    ids = [str(item) for item in payload.get("ordered_ids") or []]
    if len(ids) != N_UNIQUE:
        raise ValueError(f"{path} ordered_ids n={len(ids)} != {N_UNIQUE}")
    return ids


def consume_scaled_errors(repo_root: Path) -> list[str]:
    errors = historical_untouched_errors(repo_root)
    candidate_path = default_candidate_path(repo_root)
    contract_path = default_contract_path(repo_root)
    if not candidate_path.is_file():
        errors.append(f"missing scaled manifest {candidate_path}")
        return errors
    if not contract_path.is_file():
        errors.append(f"missing scaled contract {contract_path}")
        return errors
    candidate_sha = sha256_file(candidate_path)
    if candidate_sha != EXPECTED_MANIFEST_FILE_SHA256:
        errors.append(
            f"scaled manifest sha256 {candidate_sha} != {EXPECTED_MANIFEST_FILE_SHA256}"
        )
    contract_sha = sha256_file(contract_path)
    if contract_sha != EXPECTED_CONTRACT_SHA256:
        errors.append(
            f"scaled contract sha256 {contract_sha} != {EXPECTED_CONTRACT_SHA256}"
        )
    payload = load_json(candidate_path)
    errors.extend(manifest_errors(payload))
    contract = load_json(contract_path)
    errors.extend(scaled_contract_errors(contract))
    return errors


def historical_untouched_errors(repo_root: Path) -> list[str]:
    errors: list[str] = []
    checks = (
        (Path(repo_root) / M3C_FREEZE_RELPATH, EXPECTED_M3C_SHA256, "stage1_m3c_freeze.json"),
        (Path(repo_root) / M5_MAIN_RELPATH, EXPECTED_MAIN_SHA256, "stage1_m5_main.json"),
        (
            Path(repo_root) / M3C_CANDIDATE_RELPATH,
            EXPECTED_CANDIDATE_FILE_SHA256,
            "m3c_train_candidates.json",
        ),
        (
            Path(repo_root) / E014_RUNTIME_RELPATH,
            EXPECTED_E014_RUNTIME_SHA256,
            "stage1_m5_e014_runtime.json",
        ),
        (
            Path(repo_root) / CANONICAL_ENVELOPE_RELPATH,
            EXPECTED_CANONICAL_SHA256,
            "canonical envelope",
        ),
    )
    for path, expected, label in checks:
        if not path.is_file():
            errors.append(f"missing {label} {path}")
            continue
        digest = sha256_file(path)
        if digest != expected:
            errors.append(f"{label} sha256 {digest} != frozen {expected}")
    return errors


def manifest_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        errors.append(f"schema_version {payload.get('schema_version')!r}")
    if int(payload.get("n_unique") or 0) != N_UNIQUE:
        errors.append("n_unique != 2193")
    if int(payload.get("n_rows") or 0) != N_ROWS:
        errors.append("n_rows != 2200")
    if int(payload.get("optimizer_steps") or 0) != MAIN_STEPS:
        errors.append("optimizer_steps != 275")
    if int(payload.get("expected_trajectories") or 0) != N_TRAJECTORIES:
        errors.append("expected_trajectories != 8800")
    if payload.get("unique_ids_sha256") != EXPECTED_UNIQUE_IDS_SHA256:
        errors.append("unique_ids_sha256 drifted")
    if payload.get("padded_ids_sha256") != EXPECTED_PADDED_IDS_SHA256:
        errors.append("padded_ids_sha256 drifted")
    pad_ids = [str(item) for item in (payload.get("padding") or {}).get("pad_ids") or []]
    if tuple(pad_ids) != EXPECTED_PAD_IDS:
        errors.append("pad_ids drifted")
    if pad_ids == list(PREFIX_REPEAT_PAD_IDS):
        errors.append("pad_ids are prefix-repeat")
    ordered = [str(item) for item in payload.get("ordered_ids") or []]
    padded = [str(item) for item in payload.get("padded_ids") or []]
    if len(ordered) != N_UNIQUE:
        errors.append(f"ordered_ids n={len(ordered)}")
    if len(padded) != N_ROWS:
        errors.append(f"padded_ids n={len(padded)}")
    if padded != ordered + pad_ids:
        errors.append("padded_ids != ordered_ids + pad_ids")
    if int(payload.get("symbol_applicable_false") or 0) != N_SYMBOL_UNAVAILABLE:
        errors.append("symbol_applicable_false != 145")
    if int(payload.get("old_256_coverage_n") or 0) != 256:
        errors.append("old_256_coverage_n != 256")
    excluded = [str(item) for item in payload.get("excluded_from_m1e_train") or []]
    if excluded != ["Project-MONAI__MONAI-6344"]:
        errors.append(f"excluded {excluded}")
    if payload.get("gold_used_for_cherry_pick") or payload.get("reward_used_for_selection"):
        errors.append("manifest used gold/reward selection")
    if payload.get("oracle_used_as_drop"):
        errors.append("oracle used as drop")
    remainder = int((payload.get("padding") or {}).get("remainder_if_unpadded") or 0)
    if remainder != 1:
        errors.append("remainder_if_unpadded != 1")
    if SYMBOL_UNAVAILABLE_IN_FIRST_BATCH not in set(ordered[:8]):
        errors.append("first batch missing known symbol-unavailable id")
    return errors


def build_scaled_contract(
    *,
    freeze: Mapping[str, Any],
    freeze_path: Path,
    candidate: Mapping[str, Any],
    candidate_path: Path,
    envelope_path: Path,
    project_commit: str | None,
) -> dict[str, Any]:
    errors = manifest_errors(candidate)
    if errors:
        raise ValueError(f"scaled manifest invalid: {errors}")
    freeze_sha = sha256_file(freeze_path)
    if freeze_sha != EXPECTED_M3C_SHA256:
        raise ValueError(f"M3C freeze hash {freeze_sha} != {EXPECTED_M3C_SHA256}")
    candidate_sha = sha256_file(candidate_path)
    envelope_sha = sha256_file(envelope_path)
    if envelope_sha != EXPECTED_CANONICAL_SHA256:
        raise ValueError(f"canonical envelope hash {envelope_sha} != {EXPECTED_CANONICAL_SHA256}")
    placement = expected_hybrid_placement(
        n_gpus=EXPECTED_N_GPUS, tensor_model_parallel_size=EXPECTED_TP
    )
    audit = checkpoint_audit()
    compute = predicted_main_compute()
    inherited = inherited_m3c_knobs(freeze)
    if int(inherited.get("proposed_grpo_rollout_n") or 0) != GROUP_N:
        raise ValueError("M3C proposed_grpo_rollout_n drifted from 4")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "milestone": MILESTONE,
        "experiment_id": EXPERIMENT_ID,
        "inherits_m3c_freeze": {
            "path": str(freeze_path),
            "sha256": freeze_sha,
            "schema_version": freeze.get("schema_version"),
        },
        "canonical_envelope": {
            "path": str(envelope_path),
            "sha256": envelope_sha,
            "ppo_max_token_len_per_gpu": PPO_MAX_TOKEN_LEN,
        },
        "train_candidate_manifest": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
            "schema_version": candidate.get("schema_version"),
            "n_unique": N_UNIQUE,
            "n_rows": N_ROWS,
            "unique_ids_sha256": candidate.get("unique_ids_sha256"),
            "padded_ids_sha256": candidate.get("padded_ids_sha256"),
            "pad_ids": list((candidate.get("padding") or {}).get("pad_ids") or []),
        },
        "scale_context": {
            "codescout_4b_public": {
                "instances": 1600,
                "steps": 200,
                "batch": 8,
                "G": 8,
                "trajectories": 12800,
            },
            "this_run": {
                "unique": N_UNIQUE,
                "steps": MAIN_STEPS,
                "batch": TRAIN_BATCH_SIZE,
                "G": GROUP_N,
                "trajectories": N_TRAJECTORIES,
            },
            "note": (
                "Deliberate compute-constrained scale match, not hyperparameter "
                "reproduction. Do not raise G to 8 to chase CodeScout."
            ),
        },
        "inherited_m3c": inherited,
        "inherited_m4_runtime": m4_validated_knobs(),
        "newly_frozen": {
            "gpu": {
                "n_gpus": EXPECTED_N_GPUS,
                "nnodes": EXPECTED_NNODES,
                "tensor_model_parallel_size": EXPECTED_TP,
                "device": "2xA100-40GB",
                "cuda_visible_devices": "0,1",
                "fsdp_world_size": placement["fsdp_world_size"],
                "n_vllm_replicas": placement["n_vllm_replicas"],
            },
            "data": {
                "train_batch_size": TRAIN_BATCH_SIZE,
                "shuffle": False,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "max_prompt_length": PROMPT_LENGTH,
                "max_response_length": RESPONSE_LENGTH,
                "n_unique": N_UNIQUE,
                "n_rows": N_ROWS,
                "drop_last": True,
                "padding_policy": PADDING_POLICY_NAME,
            },
            "actor": {
                "ppo_mini_batch_size": TRAIN_BATCH_SIZE,
                "ppo_epochs": 1,
                "use_dynamic_bsz": True,
                "ppo_max_token_len_per_gpu": PPO_MAX_TOKEN_LEN,
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
                "rollout_n": GROUP_N,
                "vllm_n": 1,
            },
            "trainer": {
                "total_epochs": 1,
                "total_training_steps": MAIN_STEPS,
                "val_before_train": False,
                "test_freq": -1,
                "save_freq": SAVE_FREQ,
                "max_actor_ckpt_to_keep": MAX_ACTOR_CKPT_TO_KEEP,
                "resume_mode": "auto",
                "logger": ["console", "wandb"],
                "project_name": "budget-coder-rl",
                "experiment_name": "scaled-m5-main",
                "seed": SEED,
                "critic_enable": False,
            },
            "checkpoint": {
                "backend": "FSDPCheckpointManager",
                "contents": ["model", "optimizer", "extra"],
                "approx_size_gib": CKPT_SHARD_GIB,
                "retention_n": MAX_ACTOR_CKPT_TO_KEEP,
                "save_freq": SAVE_FREQ,
                "save_events": audit["save_events"],
                "storage_bound_unpruned_gib": audit["storage_bound_unpruned_gib"],
                "directory_template": f"$BCRL_DATA_ROOT/{CHECKPOINT_RELPATH}",
                "terminal_global_step": MAIN_STEPS,
                "no_lora_only_migration": True,
                "audit": audit,
            },
            "hard_stop": {
                "one_pass_over_padded_rows": True,
                "silent_second_epoch": False,
                "do_not_start_from_this_file_in_e016": True,
                "continuation_requires_new_version": True,
                "abort_on": [
                    "OOM",
                    "veRL pin mismatch",
                    "M3C freeze hash mismatch",
                    "scaled manifest hash mismatch",
                    "dataloader silent drop",
                    "shared dirty veRL import",
                    "disk below 100 GiB",
                ],
            },
        },
        "compute_estimate": compute,
        "runtime_provenance": {
            "budget_coder_rl_commit": project_commit,
        },
        "notes": [
            "Do not silently edit this file after freeze. Do not start the 275-step run from E016.",
            "G stays 4. This experiment only corrects training-data / training-step scale.",
            "Do not edit stage1_m3c_freeze.json, stage1_m5_main.json, E014, or E015.",
        ],
        "gate": {
            "READY_FOR_SCALED_M5_MAIN": False,
            "immutable": False,
            "preflight_experiment_id": PREFLIGHT_EXPERIMENT_ID,
        },
    }


def scaled_contract_errors(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        errors.append("scaled contract schema drifted")
    newly = contract.get("newly_frozen") if isinstance(contract.get("newly_frozen"), MappingABC) else {}
    data = newly.get("data") if isinstance(newly.get("data"), MappingABC) else {}
    actor = newly.get("actor") if isinstance(newly.get("actor"), MappingABC) else {}
    algo = newly.get("algorithm") if isinstance(newly.get("algorithm"), MappingABC) else {}
    trainer = newly.get("trainer") if isinstance(newly.get("trainer"), MappingABC) else {}
    gpu = newly.get("gpu") if isinstance(newly.get("gpu"), MappingABC) else {}
    if int(data.get("n_unique") or 0) != N_UNIQUE:
        errors.append("contract n_unique != 2193")
    if int(data.get("n_rows") or 0) != N_ROWS:
        errors.append("contract n_rows != 2200")
    if int(data.get("train_batch_size") or 0) != TRAIN_BATCH_SIZE:
        errors.append("train_batch_size != 8")
    if data.get("shuffle") is not False:
        errors.append("shuffle must stay false")
    if int(algo.get("rollout_n") or 0) != GROUP_N:
        errors.append("G != 4")
    if int(algo.get("vllm_n") or 0) != 1:
        errors.append("vllm_n != 1")
    if int(trainer.get("total_training_steps") or 0) != MAIN_STEPS:
        errors.append("total_training_steps != 275")
    if int(trainer.get("total_epochs") or 0) != 1:
        errors.append("total_epochs != 1")
    if int(trainer.get("save_freq") or 0) != SAVE_FREQ:
        errors.append("save_freq != 32")
    if int(actor.get("ppo_max_token_len_per_gpu") or 0) != PPO_MAX_TOKEN_LEN:
        errors.append("ppo_max != 20480")
    if int(gpu.get("n_gpus") or 0) != EXPECTED_N_GPUS:
        errors.append("n_gpus != 2")
    if int(gpu.get("tensor_model_parallel_size") or 0) != EXPECTED_TP:
        errors.append("TP != 1")
    inherited = contract.get("inherited_m3c") if isinstance(contract.get("inherited_m3c"), MappingABC) else {}
    if inherited.get("primary_training_B_obs") != OBS_TOKENS_LIMIT:
        errors.append("B_obs != 4096")
    if inherited.get("budget_visible") is not True:
        errors.append("budget_visible is not true")
    if int(inherited.get("proposed_grpo_rollout_n") or 0) != GROUP_N:
        errors.append("inherited G != 4")
    n_rows = int(data.get("n_rows") or 0)
    batch = int(data.get("train_batch_size") or 0)
    steps = int(trainer.get("total_training_steps") or 0)
    if batch and n_rows / batch != steps:
        errors.append(f"one-pass broken: {n_rows}/{batch} != {steps}")
    return errors


def build_preflight_overlay(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    n_steps: int = PREFLIGHT_STEPS,
) -> dict[str, Any]:
    if int(n_steps) < 1 or int(n_steps) > PREFLIGHT_MAX_STEPS:
        raise ValueError(f"preflight steps {n_steps} not in 1..{PREFLIGHT_MAX_STEPS}")
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "milestone": "E016-SCALED-PREFLIGHT",
        "experiment_id": PREFLIGHT_EXPERIMENT_ID,
        "disposable": True,
        "not_main_run": True,
        "do_not_start_275": True,
        "inherits": CONTRACT_RELPATH,
        "n_preflight_steps": int(n_steps),
        "n_preflight_tasks": TRAIN_BATCH_SIZE * int(n_steps),
        "n_preflight_trajectories": TRAIN_BATCH_SIZE * int(n_steps) * GROUP_N,
        "instance_policy": (
            "full 2200-row padded parquet in frozen padded_ids order; "
            "trainer.total_training_steps stops after 2 SequentialSampler batches. "
            "No reward cherry-pick. No long-risk reorder."
        ),
        "overrides": {
            "experiment_id": PREFLIGHT_EXPERIMENT_ID,
            "trainer": {
                "total_training_steps": int(n_steps),
                "experiment_name": WANDB_EXPERIMENT_NAME,
                "default_local_dir": str(checkpoint_dir),
                "save_freq": 1,
                "max_actor_ckpt_to_keep": 1,
                "resume_mode": "disable",
            },
            "checkpoint": {
                "directory_template": str(checkpoint_dir),
                "retention_n": 1,
                "save_freq": 1,
            },
        },
        "output_dir": str(output_dir),
        "notes": [
            "Preflight validates dataset/loader/scale contract on the official 2xA100 topology.",
            "HARD FAIL if total_training_steps > 2. Do not launch scaled main from this overlay.",
        ],
    }


def preflight_overlay_errors(
    overlay: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if overlay.get("schema_version") != PREFLIGHT_SCHEMA:
        errors.append("preflight schema drifted")
    if overlay.get("experiment_id") != PREFLIGHT_EXPERIMENT_ID:
        errors.append("preflight experiment_id != E016")
    if overlay.get("do_not_start_275") is not True:
        errors.append("preflight must set do_not_start_275")
    n_steps = int(overlay.get("n_preflight_steps") or 0)
    if n_steps < 1 or n_steps > PREFLIGHT_MAX_STEPS:
        errors.append(f"preflight steps {n_steps} exceed cap {PREFLIGHT_MAX_STEPS}")
    trainer = ((overlay.get("overrides") or {}).get("trainer") or {})
    if int(trainer.get("total_training_steps") or 0) > PREFLIGHT_MAX_STEPS:
        errors.append("preflight trainer.total_training_steps > 2")
    if contract is not None:
        errors.extend(scaled_contract_errors(contract))
        newly = contract.get("newly_frozen") or {}
        algo = newly.get("algorithm") or {}
        if int(algo.get("rollout_n") or 0) != GROUP_N:
            errors.append("preflight inherit G != 4")
    return errors


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            return [f"refusing to write into {experiment_id} artifact directory {forbidden}"]
    return []


def observe_checkpoint_retention(checkpoint_root: Path) -> dict[str, Any]:
    root = Path(checkpoint_root)
    if not root.exists():
        return {"path": str(root), "exists": False, "global_steps": []}
    steps: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("global_step_"):
            continue
        actor = child / "actor"
        actor_files = list(actor.glob("*")) if actor.is_dir() else []
        steps.append(
            {
                "name": child.name,
                "actor_exists": actor.is_dir(),
                "actor_n_files": len(actor_files),
                "data_pt_exists": (child / "data.pt").is_file(),
            }
        )
    return {
        "path": str(root),
        "exists": True,
        "global_steps": steps,
        "n_step_dirs": len(steps),
        "n_actor_dirs": sum(1 for item in steps if item["actor_exists"]),
        "note": (
            "max_keep tracks actor/ only. A missing actor/ with leftover data.pt "
            "means parent global_step_* was not deleted."
        ),
    }


def ready_payload(
    *,
    ready: bool,
    reasons: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "READY_FOR_SCALED_M5_MAIN": bool(ready),
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unique_n": N_UNIQUE,
        "padded_n": N_ROWS,
        "optimizer_steps_main": MAIN_STEPS,
        "group_n": GROUP_N,
        "preflight_experiment_id": PREFLIGHT_EXPERIMENT_ID,
        "do_not_start_main": True,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def build_preflight_summary(*, evidence: Mapping[str, Any]) -> str:
    pad_ids = evidence.get("pad_ids") or []
    pad_block = "\n".join(f"- `{item}`" for item in pad_ids)
    per_repo = evidence.get("per_repo_counts") or {}
    repo_block = "\n".join(f"- `{repo}`: {count}" for repo, count in sorted(per_repo.items()))
    audit = evidence.get("checkpoint_audit") or {}
    compute = evidence.get("compute_estimate") or {}
    lines = [
        "# E016 scaled-M5 preflight",
        "",
        f"- READY_FOR_SCALED_M5_MAIN: **{evidence.get('READY_FOR_SCALED_M5_MAIN')}**",
        f"- status: **{evidence.get('status')}**",
        f"- stop_reason: `{evidence.get('stop_reason')}`",
        f"- do_not_start_main: **true**",
        "",
        "## Scaled manifest",
        "",
        f"- unique: **{evidence.get('n_unique')}** hash `{evidence.get('unique_ids_sha256')}`",
        f"- padded rows: **{evidence.get('n_rows')}** hash `{evidence.get('padded_ids_sha256')}`",
        f"- padding rule: `{PADDING_POLICY_NAME}` salt `{PADDING_SALT}`",
        f"- one-pass: {evidence.get('n_rows')}/{evidence.get('train_batch_size')} = **{evidence.get('main_steps')}**",
        f"- expected trajectories: **{evidence.get('n_trajectories')}**",
        f"- symbol_applicable true/false: {evidence.get('symbol_applicable_true')} / {evidence.get('symbol_applicable_false')}",
        f"- old M3C 256 coverage: {evidence.get('old_256_coverage_n')}/256",
        "",
        "### Exact 7 padding IDs",
        "",
        pad_block or "- (missing)",
        "",
        "### Per-repo unique counts",
        "",
        repo_block or "- (missing)",
        "",
        "## Frozen scaled config",
        "",
        f"- contract: `{CONTRACT_RELPATH}` sha256 `{evidence.get('contract_sha256')}`",
        f"- manifest: `{CANDIDATE_RELPATH}` sha256 `{evidence.get('manifest_sha256')}`",
        f"- G: **{GROUP_N}** (not 8)",
        f"- ppo_max: **{PPO_MAX_TOKEN_LEN}**",
        f"- B_obs: {OBS_TOKENS_LIMIT}, budget_visible=true",
        "",
        "## Checkpoint retention audit",
        "",
        f"- save_freq: **{audit.get('save_freq')}**",
        f"- max_actor_ckpt_to_keep: {audit.get('max_actor_ckpt_to_keep')} (does not imply global_step_* delete)",
        f"- save events: {audit.get('save_events')}",
        f"- terminal: global_step_{audit.get('terminal_global_step')}",
        f"- storage bound unpruned: **{audit.get('storage_bound_unpruned_gib')} GiB**",
        f"- storage bound if actor prune works: {audit.get('storage_bound_if_actor_prune_works_gib')} GiB",
        f"- disk min: {audit.get('disk_min_gib')} GiB",
        f"- E014 empirical: {audit.get('e014_empirical')}",
        f"- preflight retention observation: `{evidence.get('ckpt_retention_observation')}`",
        "",
        "## Preflight runtime",
        "",
        f"- steps: **{evidence.get('n_steps_completed')}** / {evidence.get('preflight_steps')}",
        f"- trajectories: **{evidence.get('n_trajectories_preflight')}**",
        f"- realized max seq: {evidence.get('realized_max_seq')}",
        f"- GPU0 peak MiB: {evidence.get('gpu0_peak_mib')}",
        f"- GPU1 peak MiB: {evidence.get('gpu1_peak_mib')}",
        f"- pg_loss finite: {evidence.get('pg_loss_finite')}",
        f"- grad_norm finite: {evidence.get('grad_norm_finite')}",
        f"- advantage health: nonzero_adv_steps={evidence.get('n_steps_nonzero_advantage')} mixed={evidence.get('n_steps_mixed')}",
        f"- OOM: {evidence.get('oom')}",
        f"- packing covers: {evidence.get('packing_covers')}",
        f"- protocol / invalid_action (observational): {evidence.get('invalid_action_rate')}",
        f"- symbol_unavailable seen: {evidence.get('symbol_unavailable_seen')}",
        f"- placement: FSDP world={evidence.get('fsdp_world_size')} vLLM replicas={evidence.get('n_vllm_replicas')} TP={evidence.get('tensor_model_parallel_size')}",
        f"- elapsed_s: {evidence.get('elapsed_s')}",
        "",
        "## Estimated main wall-clock",
        "",
        f"- ~**{compute.get('estimated_wall_clock_h')} h** / **{compute.get('estimated_gpu_hours_2xA100')} GPU-h** ({compute.get('source')})",
        "",
        "## Gate",
        "",
        f"`READY_FOR_SCALED_M5_MAIN={evidence.get('READY_FOR_SCALED_M5_MAIN')}`",
        "",
        "Do **not** start the 275-step scaled main run from this preflight.",
        "",
        f"Gate reasons: {evidence.get('gate_reasons')}",
    ]
    return "\n".join(lines) + "\n"


def load_identity_rows_for_build(
    repo_root: Path,
    *,
    require_parquet: bool = True,
) -> dict[str, Any]:
    info = load_train_identity_rows(repo_root)
    if info.get("errors"):
        raise ValueError(f"identity load errors: {info['errors']}")
    if require_parquet and info.get("source") != "m1e_train_parquet":
        raise ValueError("scaled freeze requires M1E train parquet identities")
    if info.get("source") != "m1e_train_parquet":
        m1d = load_m1d_train_rows(Path(repo_root) / M1D_SPLIT_RELPATH)
        return {
            "rows": m1d,
            "source": "m1d_split",
            "n": len(m1d),
        }
    return info
