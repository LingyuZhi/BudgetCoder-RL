"""M4A CPU helpers: smoke task selection and GRPO group evidence.

Does not run GPU, RewardLoop, or the optimizer. Gold lists stay in the
evaluator sidecar; this module only reads instance_id identities and
already-computed scalar rewards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.eval.m3c import (
    CANDIDATE_RELPATH,
    FREEZE_RELPATH,
    group_reward_stats,
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E008"
MILESTONE = "M4A"
GROUP_N = 4
OBS_TOKENS_LIMIT = 4096
BUDGET_VISIBLE = True
REWARD_NUM_WORKERS = 2
E007_GROUPS_RELPATH = "outputs/experiments/E007/m3c_groups.json"
PRIVILEGED_LEAK_MARKERS = (
    "oracle_symbols",
    "base_changed_files",
    "gold_edit_files",
    "unmapped_sites",
)
ADVANTAGE_ABS_EPS = 1e-8
SCORE_MATCH_EPS = 1e-6


def default_candidate_path(repo_root: Path) -> Path:
    return Path(repo_root) / CANDIDATE_RELPATH


def default_freeze_path(repo_root: Path) -> Path:
    return Path(repo_root) / FREEZE_RELPATH


def default_e007_groups_path(repo_root: Path) -> Path:
    return Path(repo_root) / E007_GROUPS_RELPATH


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_candidate_ordered_ids(path: Path) -> list[str]:
    payload = load_json(path)
    ids = [str(item) for item in payload.get("ordered_ids") or []]
    if not ids:
        raise ValueError(f"{path} has empty ordered_ids")
    return ids


def load_e007_groups(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{path} missing groups[]")
    return [item for item in groups if isinstance(item, MappingABC)]


def mixed_instance_ids(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        stats = group.get("stats") if isinstance(group.get("stats"), MappingABC) else {}
        if not stats.get("mixed"):
            continue
        instance_id = str(group.get("instance_id") or "").strip()
        if not instance_id or instance_id in seen:
            continue
        seen.add(instance_id)
        ordered.append(instance_id)
    return ordered


def select_smoke_instance_ids(
    ordered_ids: Sequence[str],
    groups: Sequence[Mapping[str, Any]],
    *,
    n: int = GROUP_N,
) -> list[str]:
    """First ``n`` train-candidate ids that were mixed in E007.

    Uses candidate order (repo-round-robin), not gold or per-task reward.
    """
    mixed = set(mixed_instance_ids(groups))
    selected = [str(item) for item in ordered_ids if str(item) in mixed][: int(n)]
    if len(selected) < int(n):
        raise ValueError(
            f"need {n} E007-mixed train-candidate ids, found {len(selected)}"
        )
    return selected


def scalar_advantage(
    token_advantages: Sequence[float],
    response_mask: Sequence[int],
) -> float:
    """Recover the GRPO outcome scalar from mask-broadcast token advantages."""
    weighted = 0.0
    denom = 0.0
    for value, bit in zip(token_advantages, response_mask):
        flag = float(bit)
        weighted += float(value) * flag
        denom += flag
    if denom <= 0:
        return 0.0
    return weighted / denom


def leakage_errors(
    *,
    decoded_prompt: str,
    decoded_observations: Sequence[str],
    extra_field_keys: Sequence[str],
) -> list[str]:
    """Flag oracle field names in prompt/obs text or extra_fields keys.

    Gold path strings in a repo ``read`` observation are not leakage.
    """
    errors: list[str] = []
    keys = {str(item) for item in extra_field_keys}
    for marker in PRIVILEGED_LEAK_MARKERS:
        if marker in keys:
            errors.append(f"extra_fields contains privileged key {marker}")
    blob = decoded_prompt + "\n" + "\n".join(str(item) for item in decoded_observations)
    for marker in PRIVILEGED_LEAK_MARKERS:
        if marker in blob:
            errors.append(f"{marker} appeared in decoded prompt/observations")
    return errors


def scores_match(rm_score: float, localization_score: float) -> bool:
    return abs(float(rm_score) - float(localization_score)) < SCORE_MATCH_EPS


def assemble_group_evidence(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in members]
    instance_ids = [str(item.get("instance_id") or "") for item in rows]
    uids = [str(item.get("uid") or "") for item in rows]
    rewards = [float(item["rm_score"]) for item in rows]
    advantages = [float(item["advantage_scalar"]) for item in rows]
    stats = group_reward_stats(rewards)
    same_task = len(set(instance_ids)) == 1 and bool(instance_ids[0])
    same_uid = len(set(uids)) == 1 and bool(uids[0])
    n_ok = len(rows) == GROUP_N
    match = all(
        scores_match(float(item["rm_score"]), float(item["localization_score"]))
        for item in rows
    )
    nonzero_advantage = any(abs(value) > ADVANTAGE_ABS_EPS for value in advantages)
    gate = bool(
        same_task
        and n_ok
        and same_uid
        and stats["mixed"]
        and nonzero_advantage
        and match
    )
    return {
        "instance_id": instance_ids[0] if instance_ids else None,
        "uid": uids[0] if uids else None,
        "n_members": len(rows),
        "group_n": GROUP_N,
        "same_task": same_task,
        "same_uid": same_uid,
        "rewards": rewards,
        "advantages": advantages,
        "stats": stats,
        "mixed": stats["mixed"],
        "nonzero_advantage": nonzero_advantage,
        "rm_matches_localization": match,
        "members": rows,
        "gate_pass": gate,
    }


def freeze_contract_errors(freeze: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(freeze.get("primary_training_B_obs") or 0) != OBS_TOKENS_LIMIT:
        errors.append("freeze primary_training_B_obs != 4096")
    if freeze.get("budget_visible") is not True:
        errors.append("freeze budget_visible is not true")
    if int(freeze.get("proposed_grpo_rollout_n") or 0) != GROUP_N:
        errors.append("freeze proposed_grpo_rollout_n != 4")
    if int(freeze.get("vllm_rollout_n") or -1) != 1:
        errors.append("freeze vllm_rollout_n != 1")
    sampling = freeze.get("sampling") if isinstance(freeze.get("sampling"), MappingABC) else {}
    if int(sampling.get("n") or -1) != 1:
        errors.append("freeze sampling.n != 1")
    if freeze.get("validate") is not False:
        errors.append("freeze validate is not false")
    return errors


def artifact_hashes(paths: Mapping[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = Path(path)
        out[name] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    return out
