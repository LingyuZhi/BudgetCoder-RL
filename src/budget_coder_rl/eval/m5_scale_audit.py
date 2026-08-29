"""M5 scale-correction audit: reconstruct 2194 → 2048 → 256 without training.

Does not start GPU, Ray, vLLM, or rewrite frozen M3C/M5/E014/E015 artifacts.
Oracle sidecar is used only as a class-filter replay, never for reward
cherry-picking.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym import length_stats, sha256_file
from budget_coder_rl.data.swe_gym_materialize import (
    EXPECTED_TRAIN_ROWS,
    dataset_manifest_path,
    oracle_parquet_path,
    train_parquet_path,
)
from budget_coder_rl.eval.m3b import repo_round_robin_ids, sha256_ids
from budget_coder_rl.eval.m3c import (
    CANDIDATE_RELPATH,
    FREEZE_RELPATH,
    OVERLONG_INSTANCE_IDS,
    SELECTION_ALGORITHM,
    TRAIN_CANDIDATE_TARGET_N,
    build_train_candidate_manifest,
    default_candidate_path,
    load_jsonl,
    load_split_identities,
)
from budget_coder_rl.eval.m4a import load_json
from budget_coder_rl.eval.oracle import EvaluatorOracleIndex, load_evaluator_oracle

AUDIT_RELDIR = "outputs/experiments/M5_SCALE_AUDIT"
AUDIT_SCHEMA = "bcrl-m5-scale-audit-v1"
PROPOSED_SCHEMA = "bcrl-m5-scaled-train-candidates-proposal-v1"
M1D_SPLIT_RELPATH = "data/manifests/swe_gym_m1d_split.json"
M1D_POLICY_RELPATH = "data/manifests/swe_gym_m1d_policy.json"
M1E_MANIFEST_RELPATH = "data/manifests/swe_gym_m1e_dataset_manifest.json"
M5_MAIN_RELPATH = "configs/historical/stage1_m5_main.json"
ENVELOPE_RELPATH = "configs/historical/stage1_canonical_execution_envelope.json"
M2C_STATS_RELPATH = "data/stats/swe_gym_m2c_prompt_length.json"
M2C_OVERLONG_RELPATH = "data/interim/swe_gym/m2c_prompt_overlong.jsonl"

PROMPT_LENGTH = 16384
PPO_MAX_TOKEN_LEN = 20480
TRAIN_BATCH_SIZE = 8
GROUP_N = 4
MAX_TURNS = 6
MAX_NEW_TOKENS_PER_TURN = 2048
OBS_TOKENS_LIMIT = 4096
SELECTION_CODE = (
    "src/budget_coder_rl/eval/m3c.py:build_train_candidate_manifest "
    "+ scripts/data/build_m3c_train_candidates.py"
)

EXPECTED_ORDERED_IDS_SHA256 = (
    "8c3c25db97daae005646352284959353b7473cdf487f364f07bf2b33bfca3710"
)
EXPECTED_ELIGIBLE_IDS_SHA256 = (
    "f919cf157988b7b26c1767b6c30a8f7518170e6333929f575cd6f31e6c19125a"
)
EXPECTED_CANDIDATE_FILE_SHA256 = (
    "3ece05681486bf28dc99637f98674723ddf4797024e0af856f5725fb71c7e81b"
)
EXPECTED_M3C_FREEZE_SHA256 = (
    "49084af1c792e2049af72d4c98291dc546b829122034dba9e698cea8f7284185"
)
EXPECTED_M5_MAIN_SHA256 = (
    "fac90e49b1c3d6bc42beff57cdc73a407b2d9f88cb83f748dc4670d6dfc9837b"
)
EXPECTED_M1E_MANIFEST_SHA256 = (
    "5b1606760c4864cafb8c4d421472c51ff5f8582e0d6dae9185902095fc17da0c"
)
EXPECTED_M1D_SPLIT_SHA256 = (
    "0141d75066ef488bd501ffbf5c73703d6888f6f08a6cbb186d600179dc3b8a89"
)

E014_STEPS = 32
E014_TRAJECTORIES = 1024
E014_ELAPSED_S = 5851.6
E014_CKPT_GIB = 9.2
E014_SAVE_FREQ = 8

PAD_POLICY_NAME = "prefix_repeat_to_batch_multiple"
PAD_POLICY_RULE = (
    "If unique N % train_batch_size != 0, append the first "
    "(train_batch_size - N % train_batch_size) ordered unique ids. "
    "Repeats are explicit, hashed, and versioned. Do not rely on pinned "
    "veRL StatefulDataLoader(drop_last=True) to drop the remainder."
)


def audit_dir(repo_root: Path) -> Path:
    return Path(repo_root) / AUDIT_RELDIR


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def pad_ordered_ids(
    ordered: Sequence[str],
    *,
    batch_size: int = TRAIN_BATCH_SIZE,
) -> dict[str, Any]:
    unique = [str(item) for item in ordered]
    n_unique = len(unique)
    remainder = n_unique % int(batch_size) if unique else 0
    pad_n = 0 if remainder == 0 else int(batch_size) - remainder
    pad_ids = unique[:pad_n]
    padded = unique + pad_ids
    return {
        "policy_name": PAD_POLICY_NAME,
        "rule": PAD_POLICY_RULE,
        "batch_size": int(batch_size),
        "n_unique": n_unique,
        "remainder_if_unpadded": remainder,
        "n_pad": pad_n,
        "pad_ids": pad_ids,
        "n_rows_padded": len(padded),
        "padded_ids": padded,
        "unique_ids_sha256": sha256_ids(unique),
        "padded_ids_sha256": sha256_ids(padded),
        "optimizer_steps": (len(padded) // int(batch_size)) if padded else 0,
        "silent_drop_if_unpadded_drop_last": remainder,
    }


def load_m1d_train_rows(split_path: Path) -> list[dict[str, str]]:
    payload = load_json(split_path)
    assignments = payload.get("assignments") or []
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in assignments:
        if not isinstance(item, MappingABC):
            continue
        if str(item.get("split") or "") != "train":
            continue
        instance_id = str(item.get("instance_id") or "").strip()
        repo = str(item.get("repo") or "").strip()
        group = str(item.get("correlation_group_id") or "").strip()
        if not instance_id or not repo:
            raise ValueError("M1D train assignment missing instance_id or repo")
        if instance_id in seen:
            raise ValueError(f"duplicate M1D train instance_id: {instance_id}")
        seen.add(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "split": "train",
                "correlation_group_id": group,
            }
        )
    if len(rows) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"M1D train rows {len(rows)} != {EXPECTED_TRAIN_ROWS}")
    return rows


def load_train_identity_rows(
    repo_root: Path,
    *,
    train_parquet: Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    m1d_path = root / M1D_SPLIT_RELPATH
    m1d_rows = load_m1d_train_rows(m1d_path)
    m1d_by_id = {row["instance_id"]: row for row in m1d_rows}
    parquet = Path(train_parquet) if train_parquet is not None else train_parquet_path(root)
    source = "m1d_split"
    parquet_sha = None
    rows = m1d_rows
    errors: list[str] = []
    if parquet.is_file():
        parquet_sha = sha256_file(parquet)
        expected = (
            load_json(dataset_manifest_path(root))
            .get("artifacts", {})
            .get("train", {})
            .get("sha256")
        )
        if expected and parquet_sha != expected:
            errors.append(
                f"train parquet sha256 {parquet_sha} != M1E manifest {expected}"
            )
        parquet_rows = load_split_identities(parquet, expected_split="train")
        parquet_ids = [row["instance_id"] for row in parquet_rows]
        m1d_ids = [row["instance_id"] for row in m1d_rows]
        if set(parquet_ids) != set(m1d_ids):
            errors.append("train parquet instance_id set != M1D train set")
        merged: list[dict[str, str]] = []
        for row in parquet_rows:
            instance_id = row["instance_id"]
            extra = m1d_by_id.get(instance_id) or {}
            merged.append(
                {
                    "instance_id": instance_id,
                    "repo": row["repo"],
                    "split": "train",
                    "correlation_group_id": str(extra.get("correlation_group_id") or ""),
                }
            )
        rows = merged
        source = "m1e_train_parquet"
    return {
        "rows": rows,
        "source": source,
        "parquet_path": str(parquet) if parquet.is_file() else None,
        "parquet_sha256": parquet_sha,
        "m1d_split_sha256": sha256_file(m1d_path),
        "errors": errors,
        "n": len(rows),
    }


def replay_m3c_filters(
    rows: Sequence[Mapping[str, str]],
    oracle: EvaluatorOracleIndex | None,
    *,
    overlong_ids: Sequence[str] | frozenset[str] = OVERLONG_INSTANCE_IDS,
) -> dict[str, Any]:
    blocked = frozenset(str(item) for item in overlong_ids)
    skipped_overlong: list[str] = []
    skipped_symbol: list[str] = []
    skipped_missing: list[str] = []
    eligible: list[str] = []
    for row in rows:
        instance_id = str(row["instance_id"])
        if instance_id in blocked:
            skipped_overlong.append(instance_id)
            continue
        if oracle is None:
            continue
        if instance_id not in oracle:
            skipped_missing.append(instance_id)
            continue
        if not oracle.get(instance_id).symbol_applicable:
            skipped_symbol.append(instance_id)
            continue
        eligible.append(instance_id)
    return {
        "skipped_overlong": skipped_overlong,
        "skipped_symbol": skipped_symbol,
        "skipped_missing": skipped_missing,
        "eligible_ids_encounter_order": eligible,
        "oracle_replayed": oracle is not None,
    }


def reconstruct_eligible_from_skipped(
    rows: Sequence[Mapping[str, str]],
    skipped: Mapping[str, Sequence[str]],
) -> list[str]:
    excluded = set()
    for key in ("overlong_prompt", "symbol_unavailable", "missing_oracle"):
        excluded.update(str(item) for item in skipped.get(key) or [])
    return [str(row["instance_id"]) for row in rows if str(row["instance_id"]) not in excluded]


def classify_historical_exclusions(
    *,
    train_ids: Sequence[str],
    skipped: Mapping[str, Sequence[str]],
    selected: Sequence[str],
    eligible: Sequence[str],
    hard_unusable: Sequence[str],
) -> dict[str, Any]:
    train_set = [str(item) for item in train_ids]
    selected_set = set(str(item) for item in selected)
    eligible_set = set(str(item) for item in eligible)
    overlong = [str(item) for item in skipped.get("overlong_prompt") or []]
    symbol = [str(item) for item in skipped.get("symbol_unavailable") or []]
    missing = [str(item) for item in skipped.get("missing_oracle") or []]
    hard = [str(item) for item in hard_unusable]
    class_a = sorted(set(overlong) | set(missing) | set(hard))
    class_c = sorted(set(symbol) - set(class_a))
    class_d = sorted(iid for iid in eligible_set if iid not in selected_set)
    class_b: list[str] = []
    classified = set(class_a) | set(class_b) | set(class_c) | set(class_d)
    historical_excluded = [iid for iid in train_set if iid not in selected_set]
    leftover = sorted(set(historical_excluded) - classified)
    return {
        "A_genuine_hard_unusable": {
            "n": len(class_a),
            "instance_ids": class_a,
            "reason": (
                "Initial prompt exceeds frozen AgentLoop prompt_length=16384 "
                "and/or missing oracle / reward-undefined. Do not raise ppo_max=20480 "
                "to include these."
            ),
        },
        "B_old_systems_limitation": {
            "n": len(class_b),
            "instance_ids": class_b,
            "reason": (
                "No extra train rows were dropped solely for 1xA100 / old "
                "ppo_max=16384 beyond the hard-overlong singleton. Long prompts "
                "that still fit prompt_length=16384 stay in the scaled pool."
            ),
        },
        "C_valid_symbol_unavailable": {
            "n": len(class_c),
            "instance_ids": class_c,
            "reason": (
                "M1D symbol_oracle is auxiliary, not eligibility. M4A/M3C freeze "
                "already scores file-only localization_score when "
                "symbol_applicable=false. M3C required symbol_applicable only as "
                "a first-RL class filter."
            ),
        },
        "D_prototype_compute_exclusion": {
            "n": len(class_d),
            "instance_ids": class_d,
            "reason": (
                "target_n=256 repo-round-robin prefix of the 2048 eligible pool. "
                "Compute-budget first-RL subset, not scientific ineligibility."
            ),
        },
        "unclassified_leftover": leftover,
        "n_historical_excluded_from_256": len(historical_excluded),
        "n_classified": len(classified),
    }


def pool_stats(
    ordered_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, str]],
    *,
    symbol_applicable_by_id: Mapping[str, bool] | None = None,
    prompt_tokens_by_id: Mapping[str, int] | None = None,
    old_256: Sequence[str] | None = None,
) -> dict[str, Any]:
    ordered = [str(item) for item in ordered_ids]
    repos = Counter(
        str((rows_by_id.get(iid) or {}).get("repo") or "") for iid in ordered
    )
    groups = {
        str((rows_by_id.get(iid) or {}).get("correlation_group_id") or "")
        for iid in ordered
    }
    groups.discard("")
    symbol_true = 0
    symbol_false = 0
    symbol_unknown = 0
    if symbol_applicable_by_id is not None:
        for iid in ordered:
            flag = symbol_applicable_by_id.get(iid)
            if flag is True:
                symbol_true += 1
            elif flag is False:
                symbol_false += 1
            else:
                symbol_unknown += 1
    old = [str(item) for item in (old_256 or [])]
    old_set = set(old)
    coverage_n = sum(1 for iid in old if iid in set(ordered))
    prompt_values = []
    if prompt_tokens_by_id is not None:
        prompt_values = [
            int(prompt_tokens_by_id[iid])
            for iid in ordered
            if iid in prompt_tokens_by_id
        ]
    return {
        "n_tasks": len(ordered),
        "n_repos": len([key for key in repos if key]),
        "per_repo_counts": dict(sorted(repos.items())),
        "n_correlation_groups": len(groups),
        "symbol_applicable_true": symbol_true,
        "symbol_applicable_false": symbol_false,
        "symbol_applicable_unknown": symbol_unknown,
        "old_256_coverage_n": coverage_n,
        "old_256_coverage_ratio": (coverage_n / len(old)) if old else None,
        "prompt_length_stats": length_stats(prompt_values) if prompt_values else None,
        "ordered_ids_sha256": sha256_ids(ordered),
    }


def dataloader_semantics() -> dict[str, Any]:
    return {
        "pinned_verl": {
            "version": "0.8.0.dev0",
            "fork_commit": "8481f9f9880d0f46a75b3db0329d3de8abad3d81",
            "paths": {
                "train_dataloader": (
                    "verl/trainer/ppo/ray_trainer.py "
                    "RayPPOTrainer._create_dataloader"
                ),
                "sampler": "verl/trainer/main_ppo.py create_rl_sampler",
            },
        },
        "train_dataloader": {
            "class": "torchdata.stateful_dataloader.StatefulDataLoader",
            "drop_last": True,
            "note": (
                "Hardcoded True in pinned RayPPOTrainer. A unique N not "
                "divisible by train_batch_size silently drops N % batch_size "
                "rows per epoch. Partial final batch is not available without "
                "patching veRL trainer core."
            ),
        },
        "sampler": {
            "shuffle_false": "torch.utils.data.SequentialSampler",
            "shuffle_true": "torchdata.stateful_dataloader.sampler.RandomSampler",
            "stage1_m5": "shuffle=false, SequentialSampler, parquet row order",
        },
        "epoch_boundary": {
            "total_training_steps_default": (
                "len(train_dataloader) * trainer.total_epochs"
            ),
            "override": (
                "if trainer.total_training_steps is not None, that value wins"
            ),
            "stage1_m5_main": (
                "total_epochs=1 and total_training_steps=32 were both set; "
                "32 == 256/8 so one pass. A scaled run must set "
                "total_training_steps = n_rows_padded / 8, not reuse 32."
            ),
        },
        "resume": {
            "StatefulDataLoader": "sampler state is checkpointed",
            "stage1_m5_main_resume_mode": "auto",
            "e014_resume_mode": "disable (empty ckpt dir)",
            "note": (
                "Resume is valid for a crashed scaled run. Do not resume E014 "
                "into a new pool."
            ),
        },
        "recommendation": {
            "do_not_patch_verl": True,
            "do_not_silent_drop": True,
            "policy": PAD_POLICY_NAME,
            "rule": PAD_POLICY_RULE,
            "partial_final_batch": "rejected; requires veRL trainer core change",
        },
    }


def compute_scale_estimate(
    *,
    n_unique: int,
    n_padded: int,
    optimizer_steps: int,
    group_n: int = GROUP_N,
) -> dict[str, Any]:
    seconds_per_step = E014_ELAPSED_S / float(E014_STEPS)
    elapsed_s = optimizer_steps * seconds_per_step
    trajectories = int(n_padded) * int(group_n)
    unique_trajectories = int(n_unique) * int(group_n)
    n_saves = (
        optimizer_steps // E014_SAVE_FREQ
        if optimizer_steps >= E014_SAVE_FREQ
        else 0
    )
    return {
        "e014_baseline": {
            "n_tasks": 256,
            "optimizer_steps": E014_STEPS,
            "trajectories": E014_TRAJECTORIES,
            "elapsed_s": E014_ELAPSED_S,
            "wall_clock_h": round(E014_ELAPSED_S / 3600.0, 3),
            "gpu_hours_2xA100": round(E014_ELAPSED_S / 3600.0 * 2.0, 2),
            "ckpt_gib_per_shard": E014_CKPT_GIB,
        },
        "proposed_one_pass": {
            "n_unique": n_unique,
            "n_rows_padded": n_padded,
            "optimizer_steps": optimizer_steps,
            "rollout_trajectories_padded": trajectories,
            "rollout_trajectories_unique": unique_trajectories,
            "group_n": group_n,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "seconds_per_step_from_e014": round(seconds_per_step, 1),
            "elapsed_s_linear": round(elapsed_s, 1),
            "wall_clock_h_linear": round(elapsed_s / 3600.0, 2),
            "gpu_hours_2xA100_linear": round(elapsed_s / 3600.0 * 2.0, 1),
            "step_multiplier_vs_e014": round(optimizer_steps / float(E014_STEPS), 3),
            "traj_multiplier_vs_e014": round(trajectories / float(E014_TRAJECTORIES), 3),
            "checkpoint_saves_if_save_freq_8": n_saves,
            "checkpoint_gib_if_unpruned": round(n_saves * E014_CKPT_GIB, 1),
            "checkpoint_gib_if_max_keep_2": round(min(n_saves, 2) * E014_CKPT_GIB, 1),
        },
        "codescout_context_only": {
            "note": (
                "CodeScout-4B public config is about 1.6K instances / 200 steps "
                "/ batch=8 / G=8. Cited only as a neighboring workload scale, "
                "not a hyperparameter recipe to copy."
            ),
            "instances": 1600,
            "steps": 200,
            "batch": 8,
            "G": 8,
        },
        "later_freeze_storage_note": (
            "E014 did not prune intermediate shards. Scaled freeze should raise "
            "save_freq and/or enforce max_actor_ckpt_to_keep. Not implemented here."
        ),
    }


def _symbol_map_from_oracle(
    oracle: EvaluatorOracleIndex | None,
    instance_ids: Sequence[str],
) -> dict[str, bool] | None:
    if oracle is None:
        return None
    out: dict[str, bool] = {}
    for iid in instance_ids:
        if iid in oracle:
            out[str(iid)] = bool(oracle.get(iid).symbol_applicable)
    return out


def _symbol_map_from_skipped(
    train_ids: Sequence[str],
    skipped: Mapping[str, Sequence[str]],
    selected: Sequence[str],
) -> dict[str, bool]:
    unavailable = set(str(item) for item in skipped.get("symbol_unavailable") or [])
    selected_set = set(str(item) for item in selected)
    out: dict[str, bool] = {}
    for iid in train_ids:
        key = str(iid)
        if key in unavailable:
            out[key] = False
        elif key in selected_set:
            out[key] = True
        else:
            # eligible non-selected under M3C also required symbol_applicable
            if key not in set(skipped.get("overlong_prompt") or []) and key not in set(
                skipped.get("missing_oracle") or []
            ):
                out[key] = True
    return out


def load_overlong_token_cache(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    for row in load_jsonl(path):
        instance_id = str(row.get("instance_id") or "").strip()
        if instance_id and "n_tokens" in row:
            out[instance_id] = int(row["n_tokens"])
    return out


def inspect_prompt_lengths(
    prompt_tokens_by_id: Mapping[str, int] | None,
    *,
    m2c_stats: Mapping[str, Any] | None = None,
    overlong_cache: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    over_16384: list[dict[str, Any]] = []
    over_8192: list[dict[str, Any]] = []
    over_20480: list[dict[str, Any]] = []
    source = "unavailable"
    stats = None
    records_source: Mapping[str, int] | None = None
    full_coverage = False
    if prompt_tokens_by_id:
        source = "live_tokenizer"
        records_source = prompt_tokens_by_id
        full_coverage = True
        stats = length_stats(list(prompt_tokens_by_id.values()))
    elif overlong_cache:
        source = "m2c_overlong_jsonl"
        records_source = overlong_cache
        if m2c_stats:
            stats = (m2c_stats.get("train") or {}).get("stats")
    elif m2c_stats:
        source = "frozen_m2c_stats"
        train = (m2c_stats.get("train") or {}).get("stats") or {}
        over = (m2c_stats.get("train") or {}).get("over_limits") or {}
        stats = train
        n_over_16384 = int((over.get("16384") or {}).get("n_over") or 0)
        n_over_8192 = int((over.get("8192") or {}).get("n_over") or 0)
        if n_over_16384 == 1:
            over_16384.append(
                {
                    "instance_id": next(iter(OVERLONG_INSTANCE_IDS)),
                    "n_tokens": train.get("max"),
                    "attribution": (
                        "M2C train max with n>16384=1 and hardcoded "
                        "OVERLONG_INSTANCE_IDS singleton"
                    ),
                }
            )
        over_8192 = list(over_16384)
        if n_over_8192 > len(over_16384):
            over_8192.append(
                {
                    "instance_id": None,
                    "n_unresolved": n_over_8192 - len(over_16384),
                    "note": (
                        "Tokenizer unavailable; remaining >8192 ids not resolved"
                    ),
                }
            )
    if records_source:
        for iid, n_tokens in sorted(
            records_source.items(), key=lambda item: (-int(item[1]), item[0])
        ):
            record = {"instance_id": iid, "n_tokens": int(n_tokens)}
            if int(n_tokens) > PPO_MAX_TOKEN_LEN:
                over_20480.append(record)
            if int(n_tokens) > PROMPT_LENGTH:
                over_16384.append(record)
            if int(n_tokens) > 8192:
                over_8192.append(record)
        if not full_coverage and m2c_stats:
            expected_8192 = int(
                ((m2c_stats.get("train") or {}).get("over_limits") or {})
                .get("8192", {})
                .get("n_over")
                or 0
            )
            if expected_8192 and len(over_8192) != expected_8192:
                over_8192.append(
                    {
                        "instance_id": None,
                        "n_unresolved": expected_8192 - len(
                            [item for item in over_8192 if item.get("instance_id")]
                        ),
                        "note": "overlong jsonl did not resolve every >8192 train id",
                    }
                )
    packing_certain = [
        item
        for item in over_16384
        if int(item.get("n_tokens") or 0) >= PPO_MAX_TOKEN_LEN
    ]
    packing_likely = []
    for item in over_16384:
        n_tokens = int(item.get("n_tokens") or 0)
        if n_tokens and n_tokens < PPO_MAX_TOKEN_LEN:
            packing_likely.append(
                {
                    **item,
                    "margin_vs_20480_prompt_only": PPO_MAX_TOKEN_LEN - n_tokens,
                    "note": (
                        "Any non-trivial response+obs likely exceeds actor "
                        f"ppo_max={PPO_MAX_TOKEN_LEN}"
                    ),
                }
            )
    return {
        "source": source,
        "prompt_length_ceiling": PROMPT_LENGTH,
        "ppo_max_token_len_per_gpu": PPO_MAX_TOKEN_LEN,
        "stats": stats,
        "full_train_coverage": full_coverage,
        "n_over_8192": len([item for item in over_8192 if item.get("instance_id")]),
        "n_over_16384": len(over_16384),
        "n_over_20480": len(over_20480),
        "over_8192": over_8192,
        "over_16384": over_16384,
        "over_20480": over_20480,
        "hard_initial_prompt_blocker": over_16384,
        "packing_certain_prompt_ge_20480": packing_certain,
        "packing_likely_even_with_short_response": packing_likely,
        "stochastic_packing_note": (
            "Tasks with prompt in (8192, 16384] can start (prompt_length=16384) "
            "and may still abort if AgentLoop response+obs pushes the packed "
            "sequence over 20480. That is a runtime abort risk, not a "
            "dataset eligibility drop."
        ),
    }


def build_ledger_rows(
    *,
    n_train: int,
    skipped: Mapping[str, Sequence[str]],
    n_eligible: int,
    n_selected: int,
    hashes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    n_overlong = len(skipped.get("overlong_prompt") or [])
    n_symbol = len(skipped.get("symbol_unavailable") or [])
    n_missing = len(skipped.get("missing_oracle") or [])
    n_tail = n_eligible - n_selected
    return [
        {
            "stage": "M1D eligibility",
            "input_n": 2438,
            "predicate": "swe-gym-stage1-eligible-all-v1 keep-all; no zero-symbol/difficulty drop",
            "code": "data/manifests/swe_gym_m1d_policy.json",
            "deterministic": True,
            "oracle_used": False,
            "reward_used": False,
            "excluded_n": 0,
            "output_n": 2438,
            "output_hash": hashes.get("m1d_policy_sha256"),
        },
        {
            "stage": "M1D split train",
            "input_n": 2438,
            "predicate": "group-repo v1, seed=42, largest-remainder repo quota, subset-sum groups",
            "code": "scripts/data/prepare_swe_gym.py verify-split + data/manifests/swe_gym_m1d_split.json",
            "deterministic": True,
            "oracle_used": False,
            "reward_used": False,
            "excluded_n": 244,
            "output_n": n_train,
            "output_hash": hashes.get("m1d_split_sha256"),
        },
        {
            "stage": "M1E train parquet",
            "input_n": n_train,
            "predicate": "materialize identities; no extra drop; row order instance_id lexicographic",
            "code": "scripts/data/prepare_swe_gym.py materialize",
            "deterministic": True,
            "oracle_used": False,
            "reward_used": False,
            "excluded_n": 0,
            "output_n": n_train,
            "output_hash": hashes.get("train_parquet_sha256"),
        },
        {
            "stage": "M3C exclude overlong prompt",
            "input_n": n_train,
            "predicate": "instance_id in OVERLONG_INSTANCE_IDS (tokenizer length > 16384)",
            "code": "src/budget_coder_rl/eval/m3c.py:OVERLONG_INSTANCE_IDS; scripts/data/build_m3c_train_candidates.py",
            "deterministic": True,
            "oracle_used": False,
            "reward_used": False,
            "excluded_n": n_overlong,
            "excluded_instance_ids": list(skipped.get("overlong_prompt") or []),
            "output_n": n_train - n_overlong,
            "output_hash": None,
        },
        {
            "stage": "M3C exclude missing oracle",
            "input_n": n_train - n_overlong,
            "predicate": "instance_id not in evaluator_oracle.parquet",
            "code": "scripts/data/build_m3c_train_candidates.py",
            "deterministic": True,
            "oracle_used": True,
            "reward_used": False,
            "excluded_n": n_missing,
            "excluded_instance_ids": list(skipped.get("missing_oracle") or []),
            "output_n": n_train - n_overlong - n_missing,
            "output_hash": hashes.get("oracle_parquet_sha256"),
        },
        {
            "stage": "M3C exclude symbol_applicable=false",
            "input_n": n_train - n_overlong - n_missing,
            "predicate": "not evaluator sidecar symbol_applicable (class filter, not reward pick)",
            "code": "scripts/data/build_m3c_train_candidates.py",
            "deterministic": True,
            "oracle_used": True,
            "reward_used": False,
            "excluded_n": n_symbol,
            "output_n": n_eligible,
            "output_hash": hashes.get("eligible_ids_sha256"),
        },
        {
            "stage": "M3C first-RL subset",
            "input_n": n_eligible,
            "predicate": (
                f"{SELECTION_ALGORITHM}; take first target_n={TRAIN_CANDIDATE_TARGET_N}"
            ),
            "code": SELECTION_CODE,
            "deterministic": True,
            "oracle_used": False,
            "reward_used": False,
            "excluded_n": n_tail,
            "output_n": n_selected,
            "output_hash": hashes.get("ordered_ids_sha256"),
        },
    ]


def run_audit(
    repo_root: Path,
    *,
    prompt_tokens_by_id: Mapping[str, int] | None = None,
    require_live_oracle: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    candidate_path = default_candidate_path(root)
    freeze_path = root / FREEZE_RELPATH
    m5_main_path = root / M5_MAIN_RELPATH
    m1e_manifest_path = root / M1E_MANIFEST_RELPATH
    m1d_policy_path = root / M1D_POLICY_RELPATH
    envelope_path = root / ENVELOPE_RELPATH
    m2c_path = root / M2C_STATS_RELPATH

    candidate = load_json(candidate_path)
    freeze = load_json(freeze_path)
    m5_main = load_json(m5_main_path)
    m1e_manifest = load_json(m1e_manifest_path)
    m1d_policy = load_json(m1d_policy_path)
    envelope = load_json(envelope_path) if envelope_path.is_file() else {}
    m2c_stats = load_json(m2c_path) if m2c_path.is_file() else None

    hashes = {
        "candidate_file_sha256": sha256_file(candidate_path),
        "m3c_freeze_sha256": sha256_file(freeze_path),
        "m5_main_sha256": sha256_file(m5_main_path),
        "m1e_manifest_sha256": sha256_file(m1e_manifest_path),
        "m1d_policy_sha256": sha256_file(m1d_policy_path),
        "m1d_split_sha256": sha256_file(root / M1D_SPLIT_RELPATH),
        "canonical_envelope_sha256": (
            sha256_file(envelope_path) if envelope_path.is_file() else None
        ),
        "ordered_ids_sha256": candidate.get("ordered_ids_sha256"),
        "eligible_ids_sha256": candidate.get("eligible_ids_sha256"),
        "rule_text_sha256": candidate.get("rule_text_sha256"),
        "train_parquet_sha256": None,
        "oracle_parquet_sha256": None,
    }
    for key, expected in (
        ("candidate_file_sha256", EXPECTED_CANDIDATE_FILE_SHA256),
        ("m3c_freeze_sha256", EXPECTED_M3C_FREEZE_SHA256),
        ("m5_main_sha256", EXPECTED_M5_MAIN_SHA256),
        ("m1e_manifest_sha256", EXPECTED_M1E_MANIFEST_SHA256),
        ("m1d_split_sha256", EXPECTED_M1D_SPLIT_SHA256),
        ("ordered_ids_sha256", EXPECTED_ORDERED_IDS_SHA256),
        ("eligible_ids_sha256", EXPECTED_ELIGIBLE_IDS_SHA256),
    ):
        if hashes.get(key) != expected:
            errors.append(f"{key} {hashes.get(key)} != frozen {expected}")

    identities = load_train_identity_rows(root)
    errors.extend(identities["errors"])
    hashes["train_parquet_sha256"] = identities.get("parquet_sha256")
    hashes["m1d_split_sha256"] = identities["m1d_split_sha256"]
    rows = identities["rows"]
    rows_by_id = {row["instance_id"]: row for row in rows}
    train_ids = [row["instance_id"] for row in rows]

    oracle_path = oracle_parquet_path(root)
    oracle: EvaluatorOracleIndex | None = None
    if oracle_path.is_file():
        hashes["oracle_parquet_sha256"] = sha256_file(oracle_path)
        expected_oracle = (
            m1e_manifest.get("artifacts", {}).get("evaluator_oracle", {}).get("sha256")
        )
        if expected_oracle and hashes["oracle_parquet_sha256"] != expected_oracle:
            errors.append(
                "oracle parquet sha256 "
                f"{hashes['oracle_parquet_sha256']} != M1E manifest {expected_oracle}"
            )
        oracle = load_evaluator_oracle(oracle_path)
    elif require_live_oracle:
        errors.append(f"evaluator oracle parquet missing: {oracle_path}")
    else:
        warnings.append(
            f"evaluator oracle parquet missing at {oracle_path}; "
            "replaying eligibility from frozen skipped lists"
        )

    frozen_skipped = {
        "overlong_prompt": [str(item) for item in (candidate.get("skipped") or {}).get("overlong_prompt") or []],
        "symbol_unavailable": [
            str(item)
            for item in (candidate.get("skipped") or {}).get("symbol_unavailable") or []
        ],
        "missing_oracle": [
            str(item) for item in (candidate.get("skipped") or {}).get("missing_oracle") or []
        ],
    }
    selected = [str(item) for item in candidate.get("ordered_ids") or []]
    replay = replay_m3c_filters(rows, oracle)
    if replay["oracle_replayed"]:
        live_skipped = {
            "overlong_prompt": replay["skipped_overlong"],
            "symbol_unavailable": replay["skipped_symbol"],
            "missing_oracle": replay["skipped_missing"],
        }
        for key in frozen_skipped:
            if set(live_skipped[key]) != set(frozen_skipped[key]):
                errors.append(f"live skipped.{key} != frozen candidate skipped.{key}")
        rebuilt = build_train_candidate_manifest(
            [{"instance_id": row["instance_id"], "repo": row["repo"]} for row in rows],
            eligible_ids=replay["eligible_ids_encounter_order"],
            skipped=live_skipped,
            rule_text=str(candidate.get("rule_text") or ""),
            target_n=int(candidate.get("target_n") or TRAIN_CANDIDATE_TARGET_N),
        )
        if rebuilt["ordered_ids_sha256"] != candidate["ordered_ids_sha256"]:
            errors.append("replayed ordered_ids_sha256 != frozen candidate")
        if rebuilt["eligible_ids_sha256"] != candidate["eligible_ids_sha256"]:
            errors.append("replayed eligible_ids_sha256 != frozen candidate")
        if rebuilt["n_eligible"] != candidate["n_eligible"]:
            errors.append("replayed n_eligible != frozen candidate")
        eligible_ids = [
            iid
            for iid in repo_round_robin_ids(rows)
            if iid in set(replay["eligible_ids_encounter_order"])
        ]
    else:
        eligible_ids = [
            iid
            for iid in repo_round_robin_ids(rows)
            if iid in set(reconstruct_eligible_from_skipped(rows, frozen_skipped))
        ]
        if sha256_ids(eligible_ids) != candidate.get("eligible_ids_sha256"):
            errors.append(
                "eligible reconstruction from skipped lists != frozen eligible_ids_sha256"
            )

    if sha256_ids(selected) != candidate.get("ordered_ids_sha256"):
        errors.append("sha256(ordered_ids) != candidate.ordered_ids_sha256")
    if set(selected) - set(eligible_ids):
        errors.append("selected ids not subset of eligible")

    n_overlong = len(frozen_skipped["overlong_prompt"])
    n_symbol = len(frozen_skipped["symbol_unavailable"])
    n_missing = len(frozen_skipped["missing_oracle"])
    n_eligible = len(eligible_ids)
    expected_eligible = EXPECTED_TRAIN_ROWS - n_overlong - n_symbol - n_missing
    if n_eligible != expected_eligible:
        errors.append(f"eligible {n_eligible} != 2194-{n_overlong}-{n_symbol}-{n_missing}")
    if n_eligible != int(candidate.get("n_eligible") or 0):
        errors.append("eligible count != candidate.n_eligible")

    overlong_cache = load_overlong_token_cache(root / M2C_OVERLONG_RELPATH)
    train_id_set = set(train_ids)
    if overlong_cache:
        overlong_cache = {
            key: value for key, value in overlong_cache.items() if key in train_id_set
        }
    prompt_audit = inspect_prompt_lengths(
        prompt_tokens_by_id,
        m2c_stats=m2c_stats,
        overlong_cache=overlong_cache if prompt_tokens_by_id is None else None,
    )
    hard_from_prompt = [
        str(item["instance_id"])
        for item in prompt_audit["over_16384"]
        if item.get("instance_id")
    ]
    hard_unusable = sorted(set(frozen_skipped["overlong_prompt"]) | set(hard_from_prompt) | set(frozen_skipped["missing_oracle"]))

    historical = classify_historical_exclusions(
        train_ids=train_ids,
        skipped=frozen_skipped,
        selected=selected,
        eligible=eligible_ids,
        hard_unusable=hard_unusable,
    )
    if historical["unclassified_leftover"]:
        errors.append(
            "unclassified historical exclusions: "
            + ",".join(historical["unclassified_leftover"][:10])
        )

    symbol_from_oracle = _symbol_map_from_oracle(oracle, train_ids)
    symbol_from_skipped = _symbol_map_from_skipped(train_ids, frozen_skipped, selected)
    symbol_map = symbol_from_oracle if symbol_from_oracle is not None else symbol_from_skipped

    n_selected_applicable = sum(1 for iid in selected if symbol_map.get(iid) is True)
    n_selected_unavailable = sum(1 for iid in selected if symbol_map.get(iid) is False)
    if n_selected_unavailable:
        errors.append(
            f"M3C ordered_ids contains symbol_applicable=false: {n_selected_unavailable}"
        )
    if n_selected_applicable != len(selected) and symbol_from_oracle is not None:
        errors.append(
            f"selected symbol_applicable true={n_selected_applicable} != {len(selected)}"
        )

    train_symbol_true = sum(1 for iid in train_ids if symbol_map.get(iid) is True)
    train_symbol_false = sum(1 for iid in train_ids if symbol_map.get(iid) is False)
    train_symbol_unknown = len(train_ids) - train_symbol_true - train_symbol_false

    primary_ids = [
        iid
        for iid in repo_round_robin_ids(rows)
        if iid not in set(hard_unusable)
    ]
    fallback_ids = list(eligible_ids)
    primary_pad = pad_ordered_ids(primary_ids)
    fallback_pad = pad_ordered_ids(fallback_ids)

    primary_stats = pool_stats(
        primary_ids,
        rows_by_id,
        symbol_applicable_by_id=symbol_map,
        prompt_tokens_by_id=prompt_tokens_by_id,
        old_256=selected,
    )
    fallback_stats = pool_stats(
        fallback_ids,
        rows_by_id,
        symbol_applicable_by_id=symbol_map,
        prompt_tokens_by_id=prompt_tokens_by_id,
        old_256=selected,
    )

    proposed_exclusions = {
        "A_genuine_hard_unusable": historical["A_genuine_hard_unusable"],
        "B_old_systems_limitation": {
            "n": 0,
            "instance_ids": [],
            "reason": (
                "Not used as a scaled-pool drop. Long-but-legal prompts remain "
                "in the primary pool; see prompt_audit.over_8192."
            ),
        },
        "C_valid_symbol_unavailable": {
            "n": 0,
            "instance_ids": [],
            "kept_in_primary_n": historical["C_valid_symbol_unavailable"]["n"],
            "reason": (
                "Included in the primary scaled pool. File-only reward fallback "
                "is the frozen M3A/M4A semantics."
            ),
        },
        "D_prototype_compute_exclusion": {
            "n": 0,
            "instance_ids": [],
            "kept_in_primary_n": historical["D_prototype_compute_exclusion"]["n"],
            "reason": "Included in the primary scaled pool. 256 was first-RL only.",
        },
    }

    scale = compute_scale_estimate(
        n_unique=primary_pad["n_unique"],
        n_padded=primary_pad["n_rows_padded"],
        optimizer_steps=primary_pad["optimizer_steps"],
    )
    dloader = dataloader_semantics()

    n_train_m1e = int(m1e_manifest.get("split_checks", {}).get("train_rows") or EXPECTED_TRAIN_ROWS)
    if n_train_m1e != EXPECTED_TRAIN_ROWS:
        errors.append(f"M1E train_rows {n_train_m1e} != {EXPECTED_TRAIN_ROWS}")
    if int(m1d_policy.get("excluded_count", 0)) != 0:
        errors.append("M1D policy excluded_count != 0")
    if int(envelope.get("ppo_max_token_len_per_gpu") or 0) != PPO_MAX_TOKEN_LEN:
        errors.append("canonical envelope ppo_max is not 20480")

    m5_uses_256 = int(
        (m5_main.get("train_candidate_manifest") or {}).get("n_selected") or 0
    ) == 256
    if not m5_uses_256:
        errors.append("stage1_m5_main.json is not the 256-task freeze")

    ready = not errors and not historical["unclassified_leftover"] and bool(primary_ids)
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "READY_FOR_SCALED_M5_DESIGN": ready,
        "errors": errors,
        "warnings": warnings,
        "identity_source": identities["source"],
        "hashes": hashes,
        "ledger": build_ledger_rows(
            n_train=len(train_ids),
            skipped=frozen_skipped,
            n_eligible=n_eligible,
            n_selected=len(selected),
            hashes=hashes,
        ),
        "m3c_replay": {
            "oracle_replayed": replay["oracle_replayed"],
            "n_universe": len(train_ids),
            "n_overlong": n_overlong,
            "n_missing_oracle": n_missing,
            "n_symbol_unavailable": n_symbol,
            "n_eligible": n_eligible,
            "n_selected": len(selected),
            "arithmetic": (
                f"{len(train_ids)} - {n_overlong} overlong - {n_symbol} symbol "
                f"- {n_missing} missing_oracle = {n_eligible} eligible; "
                f"{n_eligible} - {len(selected)} = {n_eligible - len(selected)} "
                "prototype_compute_exclusion"
            ),
            "skipped": frozen_skipped,
            "selected_ordered_ids": selected,
            "eligible_ids_sha256": sha256_ids(eligible_ids),
            "ordered_ids_sha256": sha256_ids(selected),
        },
        "symbol_applicable_resolution": {
            "m3c_construction_requires_symbol_applicable": True,
            "n_selected": len(selected),
            "n_selected_symbol_applicable_true": n_selected_applicable,
            "n_selected_symbol_applicable_false": n_selected_unavailable,
            "expected_if_filter_applied": f"{len(selected)}/{len(selected)}",
            "train_symbol_applicable_true": train_symbol_true,
            "train_symbol_applicable_false": train_symbol_false,
            "train_symbol_applicable_unknown": train_symbol_unknown,
            "skipped_symbol_n": n_symbol,
            "doc_confusion": (
                "current_progress M3C line 'skipped overlong 1, symbol 145' "
                "next to '256 / eligible 2048' was misread as 145/256 "
                "symbol-oracle candidates. 145 is the train skip count; the "
                "256 selected tasks are all symbol_applicable=true."
            ),
            "reward_frozen_semantics": (
                "symbol_unavailable -> file-only localization_score, "
                "symbol_status=unavailable, symbol_f1=null. Implemented in "
                "budget_coder_rl.eval.localization and consumed by M4A "
                "compute_score. No reward change in this audit."
            ),
            "stale_docs": [
                "M3C candidate rule_text still says require symbol_applicable; that is true for the 256 first-RL subset, not for Stage-1 eligibility.",
                "my_docs/03_training_and_eval.md allows filtering symbol-unavailable from the initial RL set if prevalence is high; that was used for M3C-256, not a reason to keep dropping them at scaled M5.",
                "Do not edit those frozen files this round; SUMMARY is the corrected conclusion.",
            ],
        },
        "prompt_audit": prompt_audit,
        "historical_exclusions": {
            key: (
                {k: v for k, v in value.items() if k != "instance_ids"}
                | {"n": value["n"], "instance_ids_n": len(value.get("instance_ids") or [])}
                if isinstance(value, dict) and "instance_ids" in value
                else value
            )
            for key, value in historical.items()
        },
        "historical_exclusion_ids": {
            "A": historical["A_genuine_hard_unusable"]["instance_ids"],
            "B": historical["B_old_systems_limitation"]["instance_ids"],
            "C": historical["C_valid_symbol_unavailable"]["instance_ids"],
            "D": historical["D_prototype_compute_exclusion"]["instance_ids"],
        },
        "primary_pool": {
            "name": "m5_scaled_train_primary",
            "recommendation": "primary",
            "n_unique": len(primary_ids),
            "excluded_from_m1e_train": hard_unusable,
            "ordering": SELECTION_ALGORITHM,
            "ordering_note": (
                "Round-robin over the new eligible set (train minus hard-unusable). "
                "Old 256 is covered as a set, not as a prefix."
            ),
            "stats": primary_stats,
            "pad": {
                k: v
                for k, v in primary_pad.items()
                if k not in {"padded_ids"}
            },
            "ordered_ids": primary_ids,
        },
        "fallback_pool": {
            "name": "m5_scaled_train_fallback_m3c_eligible",
            "recommendation": "fallback_only",
            "n_unique": len(fallback_ids),
            "note": (
                "Keep M3C symbol_applicable class filter (2048). Inferior to the "
                "M1 keep-all + file-only fallback contract. Use only if a later "
                "review rejects mixed file-only / file+symbol rewards in one run."
            ),
            "stats": fallback_stats,
            "pad": {k: v for k, v in fallback_pad.items() if k not in {"padded_ids"}},
            "ordered_ids": fallback_ids,
        },
        "proposed_pool_exclusions": proposed_exclusions,
        "dataloader_semantics": dloader,
        "compute_scale": scale,
        "preserved_lineage": {
            "do_not_modify": [
                str(CANDIDATE_RELPATH),
                str(FREEZE_RELPATH),
                str(M5_MAIN_RELPATH),
                "outputs/experiments/E014",
                "outputs/experiments/E015",
                "$BCRL_DATA_ROOT/checkpoints/stage1_m5_e014",
            ],
            "e014": "valid 256-task proof-of-concept main run",
            "e015": "valid evaluation of E014; null result is not a bug",
            "next_contract_after_review": [
                "configs/historical/stage1_m5_scaled.json",
                "data/manifests/m5_scaled_train_candidates.json",
            ],
            "next_experiment_id": "E016+",
        },
        "m1d_policy_keep_all": {
            "policy_version": m1d_policy.get("policy_version"),
            "eligible_count": m1d_policy.get("eligible_count"),
            "excluded_count": m1d_policy.get("excluded_count"),
            "symbol_oracle_not_eligibility": (m1d_policy.get("symbol_oracle") or {}).get(
                "not_an_eligibility_criterion"
            ),
        },
    }
    payload["primary_pool"]["padded_ids"] = primary_pad["padded_ids"]
    payload["fallback_pool"]["padded_ids"] = fallback_pad["padded_ids"]
    return payload


def render_summary(payload: Mapping[str, Any]) -> str:
    ready = bool(payload.get("READY_FOR_SCALED_M5_DESIGN"))
    ledger = payload.get("ledger") or []
    replay = payload.get("m3c_replay") or {}
    symbol = payload.get("symbol_applicable_resolution") or {}
    prompt = payload.get("prompt_audit") or {}
    primary = payload.get("primary_pool") or {}
    fallback = payload.get("fallback_pool") or {}
    hist_ids = payload.get("historical_exclusion_ids") or {}
    scale = (payload.get("compute_scale") or {}).get("proposed_one_pass") or {}
    e014 = (payload.get("compute_scale") or {}).get("e014_baseline") or {}
    pad = primary.get("pad") or {}
    stats = primary.get("stats") or {}
    lines: list[str] = [
        "# M5 Scale Correction Audit",
        "",
        f"- READY_FOR_SCALED_M5_DESIGN: **{str(ready).lower()}**",
        f"- identity_source: `{payload.get('identity_source')}`",
        f"- oracle_replayed: **{replay.get('oracle_replayed')}**",
        "- E014 / E015 / M3C freeze / M5 main freeze: **not modified**",
        "",
        "## 1. Exact `2194 → 2048 → 256` ledger",
        "",
        str(replay.get("arithmetic") or ""),
        "",
        "| Stage | Input N | Predicate / operation | Excluded N | Output N | Hash |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in ledger:
        hist = row.get("output_hash") or "—"
        if isinstance(hist, str) and len(hist) > 12:
            hist = hist[:12] + "…"
        pred = str(row.get("predicate") or "").replace("|", "/")
        lines.append(
            f"| {row.get('stage')} | {row.get('input_n')} | {pred} | "
            f"{row.get('excluded_n')} | {row.get('output_n')} | `{hist}` |"
        )
    lines.extend(
        [
            "",
            "2194 → 2048 is **not** a mysterious 146-row scientific filter. It is:",
            "",
            f"- **A / hard overlong:** {len(hist_ids.get('A') or [])} (`Project-MONAI__MONAI-6344`)",
            f"- **C / symbol_applicable=false class filter:** {len(hist_ids.get('C') or [])}",
            f"- **missing oracle:** {int(replay.get('n_missing_oracle') or 0)}",
            "",
            f"Then 2048 → 256 drops **{len(hist_ids.get('D') or [])}** by "
            "`target_n=256` repo-round-robin (`D / prototype_compute_exclusion`).",
            "",
            "Code: `scripts/data/build_m3c_train_candidates.py` + "
            "`src/budget_coder_rl/eval/m3c.py:build_train_candidate_manifest`.",
            "Deterministic. Oracle used as a **class filter** only. Gold patch / "
            "per-task reward / n=4 zero-variance were **not** drop rules.",
            "",
            "## 2. Exclusion classes",
            "",
            "| Class | Historical vs 256 | vs proposed primary | Meaning |",
            "| --- | ---: | ---: | --- |",
            f"| A genuine hard-unusable | {len(hist_ids.get('A') or [])} | {len(hist_ids.get('A') or [])} | prompt>16384 and/or missing oracle |",
            f"| B old systems limitation | {len(hist_ids.get('B') or [])} | 0 | none remain dropped; long-legal prompts stay |",
            f"| C valid symbol-unavailable | {len(hist_ids.get('C') or [])} | 0 | keep; file-only reward is frozen |",
            f"| D prototype_compute_exclusion | {len(hist_ids.get('D') or [])} | 0 | keep; 256 was first-RL only |",
            "",
            "Full instance_id sets: `exclusions.json`.",
            "",
            "## 3. Symbol-applicable inconsistency",
            "",
            "M3C construction **does** require `symbol_applicable=true` for the 256-subset.",
            f"Measured selected: **{symbol.get('n_selected_symbol_applicable_true')} / "
            f"{symbol.get('n_selected')}** true, "
            f"**{symbol.get('n_selected_symbol_applicable_false')}** false.",
            "",
            f"Train `symbol_applicable=false`: **{symbol.get('train_symbol_applicable_false')}** "
            f"(frozen skip list n={symbol.get('skipped_symbol_n')}).",
            "",
            str(symbol.get("doc_confusion") or ""),
            "",
            "Reward (unchanged this round): "
            + str(symbol.get("reward_frozen_semantics") or ""),
            "",
            "## 4. Capacity exclusions under prompt_length=16384 and ppo_max=20480",
            "",
            f"- prompt_length (AgentLoop `PromptTooLongError`): **{PROMPT_LENGTH}**",
            f"- actor packing envelope: **{PPO_MAX_TOKEN_LEN}** (do not raise)",
            f"- prompt audit source: `{prompt.get('source')}`",
            f"- n prompt >8192 with resolved id: {prompt.get('n_over_8192')}",
            f"- n prompt >16384: {prompt.get('n_over_16384')}",
            f"- n prompt >20480: {prompt.get('n_over_20480')}",
            "",
            "`Project-MONAI__MONAI-6344` remains a **true hard blocker** at current "
            "prompt_length=16384. The 20480 overlay did not raise `rollout.prompt_length`. "
            "Its ~19.5k initial prompt also leaves too little room under 20480 packing "
            "for a real multi-turn response. **Do not include it** by changing 20480.",
            "",
            "Other prompts in (8192, 16384] can start. Stochastic AgentLoop growth may "
            "still abort a rare long trajectory; that is runtime risk, not eligibility.",
            "",
            "Resolved train prompts >8192:",
            "",
        ]
    )
    for item in prompt.get("over_8192") or []:
        iid = item.get("instance_id")
        n_tokens = item.get("n_tokens")
        if iid:
            lines.append(f"- `{iid}`: {n_tokens} tokens")
        elif item.get("note"):
            lines.append(f"- unresolved: {item}")
    excluded = primary.get("excluded_from_m1e_train") or []
    excluded_txt = ", ".join(f"`{item}`" for item in excluded) or "none"
    lines.extend(
        [
            "",
            "## 5. Recommended scaled train pool N",
            "",
            f"**Primary: unique N={primary.get('n_unique')}** "
            f"(M1E train minus {excluded_txt}).",
            "",
            f"Fallback (not recommended): unique N={fallback.get('n_unique')} "
            "(keep M3C `symbol_applicable` class filter).",
            "",
            "Do **not** reuse 256 as the scaled main pool.",
            "",
            "## 6. Per-repo / group / symbol stats (primary)",
            "",
            f"- unique tasks: **{stats.get('n_tasks')}**",
            f"- correlation groups: **{stats.get('n_correlation_groups')}**",
            f"- repos: **{stats.get('n_repos')}**",
            f"- symbol_applicable true/false/unknown: "
            f"{stats.get('symbol_applicable_true')} / "
            f"{stats.get('symbol_applicable_false')} / "
            f"{stats.get('symbol_applicable_unknown')}",
            f"- old-256 coverage: **{stats.get('old_256_coverage_n')} / 256** "
            f"(ratio={stats.get('old_256_coverage_ratio')})",
            f"- unique ordered_ids_sha256: `{stats.get('ordered_ids_sha256')}`",
            f"- padded_ids_sha256: `{pad.get('padded_ids_sha256')}`",
            "",
            "Per-repo unique counts:",
            "",
        ]
    )
    for repo, count in (stats.get("per_repo_counts") or {}).items():
        lines.append(f"- `{repo}`: {count}")
    prompt_stats = stats.get("prompt_length_stats") or prompt.get("stats") or {}
    if prompt_stats:
        lines.extend(
            [
                "",
                "Prompt-length quantiles (primary, if available):",
                "",
                f"- n={prompt_stats.get('n')} min={prompt_stats.get('min')} "
                f"mean={prompt_stats.get('mean')} p50={prompt_stats.get('p50')} "
                f"p90={prompt_stats.get('p90')} p95={prompt_stats.get('p95')} "
                f"p99={prompt_stats.get('p99')} max={prompt_stats.get('max')}",
            ]
        )
    lines.extend(
        [
            "",
            "Ordering: `repo_name_sort + instance_id_lexicographic + repo_round_robin` "
            "over train minus hard-unusable. Adding the 145 symbol-unavailable rows "
            "**changes** round-robin, so old 256 is not a prefix.",
            "",
            "## 7. One-pass steps, trajectories, remainder",
            "",
            f"- unique N={pad.get('n_unique')}; N % 8 = {pad.get('remainder_if_unpadded')}",
            f"- pad policy: `{pad.get('policy_name')}` n_pad={pad.get('n_pad')} "
            f"→ padded rows={pad.get('n_rows_padded')}",
            f"- optimizer steps / one dataset pass: **{pad.get('optimizer_steps')}**",
            f"- trajectories (padded × G=4): **{scale.get('rollout_trajectories_padded')}**",
            f"- unique task exposures: {pad.get('n_unique')}; extra pad exposures: {pad.get('n_pad')}",
            "",
            "Pinned veRL `StatefulDataLoader(drop_last=True)` would silent-drop "
            f"**{pad.get('silent_drop_if_unpadded_drop_last')}** unique task(s) if we "
            "trained on unpadded N. Partial last batch would require a veRL core patch. "
            "**Recommendation:** dataset-side prefix repeat; do not patch veRL.",
            "",
            "`shuffle=false` → SequentialSampler. `total_training_steps` must equal "
            "padded_rows/8. Do not reuse M5 `total_training_steps=32`. "
            "`m5a.build_main_contract` currently requires n_pool%8==0 and steps==32; "
            "a later scaled contract must not inherit that 32-step freeze.",
            "",
            "## 8. Estimated compute / storage vs E014",
            "",
            "| | E014 | scaled primary (padded) |",
            "| --- | ---: | ---: |",
            f"| optimizer steps | {e014.get('optimizer_steps')} | {scale.get('optimizer_steps')} |",
            f"| trajectories | {e014.get('trajectories')} | {scale.get('rollout_trajectories_padded')} |",
            f"| wall-clock (linear from E014 ~{scale.get('seconds_per_step_from_e014')} s/step) | "
            f"{e014.get('wall_clock_h')} h | ~{scale.get('wall_clock_h_linear')} h |",
            f"| GPU-hours (2×A100) | {e014.get('gpu_hours_2xA100')} | ~{scale.get('gpu_hours_2xA100_linear')} |",
            f"| multiplier | 1× | {scale.get('step_multiplier_vs_e014')}× |",
            f"| ckpt if save_freq=8 unpruned | ~4 shards | "
            f"~{scale.get('checkpoint_gib_if_unpruned')} GiB "
            f"({scale.get('checkpoint_saves_if_save_freq_8')} saves) |",
            f"| ckpt if max_keep=2 | | ~{scale.get('checkpoint_gib_if_max_keep_2')} GiB |",
            "",
            "CodeScout-4B (~1.6K / 200 steps / G=8) is neighboring scale only, not a recipe.",
            "",
            "## 9. Suggested new manifest / config lineage",
            "",
            "After user review (not this round):",
            "",
            "- `configs/historical/stage1_m5_scaled.json`",
            "- `data/manifests/m5_scaled_train_candidates.json`",
            "- experiment id **E016+**",
            "",
            "Keep immutable: `stage1_m3c_freeze.json`, `stage1_m5_main.json`, E014, E015.",
            "This audit writes the proposal only under `outputs/experiments/M5_SCALE_AUDIT/`.",
            "",
            "## 10. Gate",
            "",
            f"`READY_FOR_SCALED_M5_DESIGN={str(ready).lower()}`",
            "",
            "Pass means: every M1 train row is labeled A/B/C/D or kept; the proposed "
            "pool is deterministic and hashed; remainder is an explicit pad, not "
            "`drop_last`. Do **not** start scaled M5 training until the user freezes "
            "the new contract.",
            "",
        ]
    )
    errors = payload.get("errors") or []
    warnings = payload.get("warnings") or []
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in errors)
        lines.append("")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in warnings)
        lines.append("")
    return "\n".join(lines) + "\n"


def write_audit_artifacts(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema_version": AUDIT_SCHEMA,
        "READY_FOR_SCALED_M5_DESIGN": payload.get("READY_FOR_SCALED_M5_DESIGN"),
        "identity_source": payload.get("identity_source"),
        "hashes": payload.get("hashes"),
        "ledger": payload.get("ledger"),
        "m3c_replay": {
            key: value
            for key, value in (payload.get("m3c_replay") or {}).items()
            if key != "selected_ordered_ids"
        },
        "errors": payload.get("errors"),
        "warnings": payload.get("warnings"),
    }
    exclusions = {
        "schema_version": AUDIT_SCHEMA,
        "historical_exclusions": payload.get("historical_exclusions"),
        "historical_exclusion_ids": payload.get("historical_exclusion_ids"),
        "proposed_pool_exclusions": payload.get("proposed_pool_exclusions"),
        "symbol_applicable_resolution": payload.get("symbol_applicable_resolution"),
    }
    primary = payload.get("primary_pool") or {}
    fallback = payload.get("fallback_pool") or {}
    proposed = {
        "schema_version": PROPOSED_SCHEMA,
        "primary": {
            key: value
            for key, value in primary.items()
            if key != "padded_ids"
        },
        "fallback": {
            key: value
            for key, value in fallback.items()
            if key != "padded_ids"
        },
        "primary_padded_ids": primary.get("padded_ids"),
        "fallback_padded_ids": fallback.get("padded_ids"),
        "preserved_lineage": payload.get("preserved_lineage"),
        "compute_scale": payload.get("compute_scale"),
    }
    prompt = dict(payload.get("prompt_audit") or {})
    dloader = payload.get("dataloader_semantics") or {}
    ready = {
        "READY_FOR_SCALED_M5_DESIGN": payload.get("READY_FOR_SCALED_M5_DESIGN"),
        "n_primary_unique": (payload.get("primary_pool") or {}).get("n_unique"),
        "primary_ordered_ids_sha256": ((payload.get("primary_pool") or {}).get("stats") or {}).get(
            "ordered_ids_sha256"
        ),
        "primary_padded_ids_sha256": ((payload.get("primary_pool") or {}).get("pad") or {}).get(
            "padded_ids_sha256"
        ),
        "errors": payload.get("errors"),
        "warnings": payload.get("warnings"),
    }
    write_json(out / "ledger.json", ledger)
    write_json(out / "exclusions.json", exclusions)
    write_json(out / "proposed_pool.json", proposed)
    write_json(out / "prompt_lengths.json", prompt)
    write_json(out / "dataloader_semantics.json", dloader)
    write_json(out / "READY_FOR_SCALED_M5_DESIGN.json", ready)
    summary_path = out / "SUMMARY.md"
    summary_path.write_text(render_summary(payload), encoding="utf-8")
    return {
        "SUMMARY.md": str(summary_path),
        "ledger.json": str(out / "ledger.json"),
        "exclusions.json": str(out / "exclusions.json"),
        "proposed_pool.json": str(out / "proposed_pool.json"),
        "prompt_lengths.json": str(out / "prompt_lengths.json"),
        "dataloader_semantics.json": str(out / "dataloader_semantics.json"),
        "READY_FOR_SCALED_M5_DESIGN.json": str(out / "READY_FOR_SCALED_M5_DESIGN.json"),
    }
