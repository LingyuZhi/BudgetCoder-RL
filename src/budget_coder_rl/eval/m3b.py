"""M3B baseline task order, paired comparison, and failure taxonomy.

Does not read gold patches or evaluator oracles when building the task
manifest. Scoring happens after rollout via the frozen sidecar.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_materialize import (
    EXPECTED_DEV_ROWS,
    dev_parquet_path,
)
from budget_coder_rl.eval.episode import summarize_episodes

MANIFEST_SCHEMA = "bcrl-m3b-tasks-v1"
MANIFEST_RELPATH = "data/manifests/m3b_baseline_tasks.json"
SELECTION_ALGORITHM = "repo_name_sort + instance_id_lexicographic + repo_round_robin"
PRIMARY_N = 80
PAIRED_SEED_BASE = 20260825
PROVISIONAL_OBS_TOKENS_LIMIT = 8192
REVIEW_SEED = 20260825
QWEN3_SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "n": 1,
}
TAXONOMY_LABELS = (
    "wrong_search_query",
    "search_too_broad",
    "irrelevant_read",
    "premature_finish",
    "budget_waste",
    "repeated_search",
    "correct_file_wrong_symbol",
    "invalid_action",
    "budget_exhausted",
)


def extra_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, MappingABC):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    return {}


def load_dev_identities(parquet_path: Path) -> list[dict[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(parquet_path, columns=["extra_info"])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in frame.to_dict(orient="records"):
        extra = extra_mapping(record.get("extra_info"))
        instance_id = str(extra.get("instance_id") or "").strip()
        repo = str(extra.get("repo") or "").strip()
        split = str(extra.get("split") or "").strip()
        if not instance_id or not repo:
            raise ValueError("dev parquet row missing instance_id or repo")
        if instance_id in seen:
            raise ValueError(f"duplicate instance_id in dev parquet: {instance_id}")
        if split and split != "dev":
            raise ValueError(f"{instance_id}: extra_info.split={split!r} expected dev")
        seen.add(instance_id)
        rows.append({"instance_id": instance_id, "repo": repo, "split": "dev"})
    return rows


def repo_round_robin_ids(rows: Sequence[Mapping[str, str]]) -> list[str]:
    by_repo: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo"])].append(str(row["instance_id"]))
    for repo in by_repo:
        by_repo[repo].sort()
    queues = [by_repo[repo][:] for repo in sorted(by_repo)]
    ordered: list[str] = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def sha256_ids(instance_ids: Sequence[str]) -> str:
    blob = "\n".join(instance_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def paired_seed(task_index: int, *, base: int = PAIRED_SEED_BASE) -> int:
    return int(base) + int(task_index)


def build_manifest(
    rows: Sequence[Mapping[str, str]],
    *,
    primary_n: int = PRIMARY_N,
    expected_n: int = EXPECTED_DEV_ROWS,
) -> dict[str, Any]:
    if len(rows) != expected_n:
        raise ValueError(f"dev identities {len(rows)} != expected {expected_n}")
    ordered = repo_round_robin_ids(rows)
    if len(ordered) != expected_n:
        raise ValueError(f"ordered ids {len(ordered)} != expected {expected_n}")
    if primary_n > len(ordered):
        raise ValueError(f"primary_n {primary_n} > universe {len(ordered)}")
    by_id = {row["instance_id"]: row for row in rows}
    repo_counts: dict[str, int] = defaultdict(int)
    primary_repo_counts: dict[str, int] = defaultdict(int)
    for instance_id in ordered:
        repo_counts[by_id[instance_id]["repo"]] += 1
    for instance_id in ordered[:primary_n]:
        primary_repo_counts[by_id[instance_id]["repo"]] += 1
    tasks = [
        {
            "task_index": index,
            "instance_id": instance_id,
            "repo": by_id[instance_id]["repo"],
            "set": "primary" if index < primary_n else "remainder",
            "sampling_seed": paired_seed(index),
        }
        for index, instance_id in enumerate(ordered)
    ]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "split": "dev",
        "selection_algorithm": SELECTION_ALGORITHM,
        "oracle_used": False,
        "gold_used": False,
        "paired_seed_base": PAIRED_SEED_BASE,
        "paired_seed_formula": "PAIRED_SEED_BASE + task_index",
        "provisional_obs_tokens_limit": PROVISIONAL_OBS_TOKENS_LIMIT,
        "obs_tokens_limit_note": (
            "provisional M3B budget; not a frozen Stage-1 training budget"
        ),
        "n_universe": len(ordered),
        "n_primary": primary_n,
        "n_remainder": len(ordered) - primary_n,
        "ordered_ids": ordered,
        "primary_ids": ordered[:primary_n],
        "remainder_ids": ordered[primary_n:],
        "ordered_ids_sha256": sha256_ids(ordered),
        "primary_ids_sha256": sha256_ids(ordered[:primary_n]),
        "repo_counts_universe": dict(sorted(repo_counts.items())),
        "repo_counts_primary": dict(sorted(primary_repo_counts.items())),
        "n_repos_universe": len(repo_counts),
        "n_repos_primary": len(primary_repo_counts),
        "tasks": tasks,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema {payload.get('schema_version')!r}")
    if payload.get("oracle_used") or payload.get("gold_used"):
        raise ValueError("M3B manifest must not be built from oracle/gold")
    return payload


def default_manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / MANIFEST_RELPATH


def build_manifest_from_dev_parquet(
    repo_root: Path,
    *,
    parquet_path: Path | None = None,
    primary_n: int = PRIMARY_N,
) -> dict[str, Any]:
    path = parquet_path or dev_parquet_path(repo_root)
    rows = load_dev_identities(path)
    return build_manifest(rows, primary_n=primary_n)


def localization_score(row: Mapping[str, Any]) -> float | None:
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    value = loc.get("localization_score")
    if value is None:
        return None
    return float(value)


def pair_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
        instance_id = str(identity.get("instance_id") or row.get("instance_id") or "")
        if not instance_id:
            continue
        visible = None
        condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
        budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
        if condition.get("budget_visible") is not None:
            visible = bool(condition.get("budget_visible"))
        elif budget.get("budget_visible") is not None:
            visible = bool(budget.get("budget_visible"))
        slot = grouped.setdefault(instance_id, {"instance_id": instance_id})
        if visible is True:
            slot["visible"] = row
        elif visible is False:
            slot["hidden"] = row
    return grouped


def paired_rows(grouped: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for instance_id in sorted(grouped):
        pair = grouped[instance_id]
        if "hidden" in pair and "visible" in pair:
            out.append(
                {
                    "instance_id": instance_id,
                    "hidden": pair["hidden"],
                    "visible": pair["visible"],
                }
            )
    return out


def _num(row: Mapping[str, Any], *path: str) -> float | None:
    current: Any = row
    for key in path:
        if not isinstance(current, MappingABC):
            return None
        current = current.get(key)
    if current is None:
        return None
    return float(current)


def compare_pair(hidden: Mapping[str, Any], visible: Mapping[str, Any]) -> dict[str, Any]:
    h_score = localization_score(hidden)
    v_score = localization_score(visible)
    winner = "tie"
    if h_score is not None and v_score is not None:
        if v_score > h_score:
            winner = "visible"
        elif h_score > v_score:
            winner = "hidden"
    h_actions = _action_sequence(hidden)
    v_actions = _action_sequence(visible)
    return {
        "instance_id": (hidden.get("identity") or {}).get("instance_id"),
        "hidden_score": h_score,
        "visible_score": v_score,
        "delta_localization_score": (
            None if h_score is None or v_score is None else v_score - h_score
        ),
        "winner": winner,
        "hidden_repo_obs": _num(hidden, "budget", "repo_observation_tokens")
        or _num(hidden, "budget", "obs_tokens_used"),
        "visible_repo_obs": _num(visible, "budget", "repo_observation_tokens")
        or _num(visible, "budget", "obs_tokens_used"),
        "hidden_turns": _num(hidden, "counts", "n_events"),
        "visible_turns": _num(visible, "counts", "n_events"),
        "hidden_termination": hidden.get("termination"),
        "visible_termination": visible.get("termination"),
        "action_sequence_equal": h_actions == v_actions,
        "hidden_actions": h_actions,
        "visible_actions": v_actions,
        "hidden_sampling_seed": (hidden.get("condition") or {}).get("sampling_seed"),
        "visible_sampling_seed": (visible.get("condition") or {}).get("sampling_seed"),
    }


def paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = pair_episodes(rows)
    pairs = [compare_pair(item["hidden"], item["visible"]) for item in paired_rows(grouped)]
    n_visible_win = sum(1 for item in pairs if item["winner"] == "visible")
    n_hidden_win = sum(1 for item in pairs if item["winner"] == "hidden")
    n_tie = sum(1 for item in pairs if item["winner"] == "tie")
    deltas = [
        item["delta_localization_score"]
        for item in pairs
        if item["delta_localization_score"] is not None
    ]
    delta_obs = []
    for item in pairs:
        if item["hidden_repo_obs"] is not None and item["visible_repo_obs"] is not None:
            delta_obs.append(item["visible_repo_obs"] - item["hidden_repo_obs"])
    hidden_rows = [item["hidden"] for item in paired_rows(grouped)]
    visible_rows = [item["visible"] for item in paired_rows(grouped)]
    return {
        "n_completed_pairs": len(pairs),
        "n_visible_win": n_visible_win,
        "n_hidden_win": n_hidden_win,
        "n_tie": n_tie,
        "mean_delta_localization_score": _mean(deltas),
        "median_delta_localization_score": _median(deltas),
        "mean_delta_repo_observation_tokens": _mean(delta_obs),
        "median_delta_repo_observation_tokens": _median(delta_obs),
        "n_action_sequence_equal": sum(1 for item in pairs if item["action_sequence_equal"]),
        "hidden": summarize_episodes(hidden_rows),
        "visible": summarize_episodes(visible_rows),
        "pairs": pairs,
    }


def _action_sequence(row: Mapping[str, Any]) -> list[str]:
    events = row.get("events") or []
    names: list[str] = []
    for event in events:
        if not isinstance(event, MappingABC):
            continue
        name = event.get("action_name") or event.get("action_type")
        if name:
            names.append(str(name))
    return names


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


def score_band(score: float | None) -> str:
    if score is None or score <= 0:
        return "zero"
    if score >= 0.66:
        return "high"
    return "medium"


def first_pass_taxonomy(row: Mapping[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    behavior = row.get("behavior") if isinstance(row.get("behavior"), MappingABC) else {}
    counts = row.get("counts") if isinstance(row.get("counts"), MappingABC) else {}
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    termination = row.get("termination")
    events = list(row.get("events") or [])
    n_search = int(behavior.get("n_search") or 0)
    n_read = int(behavior.get("n_read") or 0)
    n_empty = int(behavior.get("n_empty_search_hits") or 0)
    n_repeat_search = int(behavior.get("n_repeated_search_queries") or 0)
    n_repeat_read = int(behavior.get("n_repeated_reads") or 0)
    file_f1 = loc.get("file_f1")
    symbol_f1 = loc.get("symbol_f1")
    assistant_turns = sum(
        1
        for event in events
        if isinstance(event, MappingABC)
        and (event.get("action_name") or event.get("action_type"))
    )
    if termination == "budget_exhausted" or (
        row.get("budget") or {}
    ).get("budget_exhausted"):
        labels.append("budget_exhausted")
    if int(counts.get("n_protocol_errors") or 0) or int(counts.get("n_tool_errors") or 0):
        labels.append("invalid_action")
    if n_empty and n_search and n_empty >= max(1, n_search // 2):
        labels.append("wrong_search_query")
    if n_search >= 3 and n_read == 0:
        labels.append("search_too_broad")
    if n_read >= 2 and (file_f1 is not None and float(file_f1) == 0):
        labels.append("irrelevant_read")
    if termination == "finish" and assistant_turns <= 2 and (
        file_f1 is not None and float(file_f1) == 0
    ):
        labels.append("premature_finish")
    util = None
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    used = budget.get("repo_observation_tokens")
    if used is None:
        used = budget.get("obs_tokens_used")
    limit = budget.get("obs_tokens_limit")
    if used is not None and limit:
        util = float(used) / float(limit)
        if util >= 0.7 and (file_f1 is not None and float(file_f1) == 0):
            labels.append("budget_waste")
    if n_repeat_search > 0 or n_repeat_read > 0:
        labels.append("repeated_search")
    if (
        file_f1 is not None
        and float(file_f1) >= 0.99
        and symbol_f1 is not None
        and float(symbol_f1) == 0
        and loc.get("symbol_status") == "scored"
    ):
        labels.append("correct_file_wrong_symbol")
    unique = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    if not unique:
        unique.append("unclassified")
    knowledge_like = "correct_file_wrong_symbol" in unique and len(unique) == 1
    failure_class = "coding_knowledge" if knowledge_like else "exploration_policy"
    if unique == ["unclassified"] and loc.get("parse_ok") and float(loc.get("localization_score") or 0) >= 0.66:
        failure_class = "success_or_partial_success"
    return {
        "labels": unique,
        "primary_label": unique[0],
        "failure_class": failure_class,
        "n_search": n_search,
        "n_read": n_read,
        "n_empty_search_hits": n_empty,
        "utilization": util,
    }


def compact_review_case(
    row: Mapping[str, Any],
    *,
    partner: Mapping[str, Any] | None = None,
    reason: str,
) -> dict[str, Any]:
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    taxonomy = first_pass_taxonomy(row)
    packet = {
        "reason": reason,
        "instance_id": identity.get("instance_id"),
        "repo": identity.get("repo"),
        "budget_visible": (row.get("condition") or {}).get("budget_visible"),
        "termination": row.get("termination"),
        "localization_score": loc.get("localization_score"),
        "file_f1": loc.get("file_f1"),
        "symbol_f1": loc.get("symbol_f1"),
        "symbol_status": loc.get("symbol_status"),
        "final_submission": row.get("final_submission"),
        "actions": _action_sequence(row),
        "behavior": row.get("behavior"),
        "budget": {
            "repo_observation_tokens": (row.get("budget") or {}).get(
                "repo_observation_tokens"
            ),
            "budget_metadata_tokens": (row.get("budget") or {}).get(
                "budget_metadata_tokens"
            ),
            "total_env_tokens": (row.get("budget") or {}).get("total_env_tokens"),
            "obs_tokens_used": (row.get("budget") or {}).get("obs_tokens_used"),
            "obs_tokens_limit": (row.get("budget") or {}).get("obs_tokens_limit"),
            "budget_exhausted": (row.get("budget") or {}).get("budget_exhausted"),
        },
        "search_queries": [
            (event.get("action_arguments") or {}).get("query")
            for event in (row.get("events") or [])
            if isinstance(event, MappingABC) and event.get("action_name") == "search"
        ],
        "read_paths": (row.get("behavior") or {}).get("read_paths"),
        "taxonomy": taxonomy,
    }
    if partner is not None:
        packet["partner_budget_visible"] = (partner.get("condition") or {}).get(
            "budget_visible"
        )
        packet["partner_score"] = localization_score(partner)
        packet["partner_actions"] = _action_sequence(partner)
        packet["partner_termination"] = partner.get("termination")
    return packet


def _stable_rng(seed: int) -> Any:
    import random

    rng = random.Random(int(seed))
    return rng


def select_review_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_target: int = 24,
    seed: int = REVIEW_SEED,
) -> list[dict[str, Any]]:
    grouped = pair_episodes(rows)
    pairs = paired_rows(grouped)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        hidden = pair["hidden"]
        visible = pair["visible"]
        comparison = compare_pair(hidden, visible)
        h_score = comparison["hidden_score"]
        v_score = comparison["visible_score"]
        for label, row, partner in (
            ("hidden", hidden, visible),
            ("visible", visible, hidden),
        ):
            score = h_score if label == "hidden" else v_score
            buckets[f"score_{score_band(score)}_{label}"].append(
                compact_review_case(row, partner=partner, reason=f"score_{score_band(score)}")
            )
            if row.get("termination") == "budget_exhausted":
                buckets["budget_exhausted"].append(
                    compact_review_case(
                        row, partner=partner, reason="budget_exhausted"
                    )
                )
            counts = row.get("counts") or {}
            if int(counts.get("n_protocol_errors") or 0) or int(
                counts.get("n_tool_errors") or 0
            ):
                buckets["invalid_action"].append(
                    compact_review_case(row, partner=partner, reason="invalid_action")
                )
            if row.get("termination") == "max_turns":
                buckets["max_turns"].append(
                    compact_review_case(row, partner=partner, reason="max_turns")
                )
            submission = row.get("final_submission")
            locations = (
                submission.get("locations") if isinstance(submission, MappingABC) else None
            )
            if not isinstance(locations, list) or not locations:
                buckets["empty_submission"].append(
                    compact_review_case(row, partner=partner, reason="empty_submission")
                )
        if not comparison["action_sequence_equal"] or abs(
            float(comparison["delta_localization_score"] or 0)
        ) >= 0.25:
            buckets["hidden_visible_divergent"].append(
                compact_review_case(
                    visible, partner=hidden, reason="hidden_visible_divergent"
                )
            )
    rng = _stable_rng(seed)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, str]] = set()
    priority = [
        "budget_exhausted",
        "invalid_action",
        "empty_submission",
        "max_turns",
        "hidden_visible_divergent",
        "score_high_hidden",
        "score_high_visible",
        "score_medium_hidden",
        "score_medium_visible",
        "score_zero_hidden",
        "score_zero_visible",
    ]
    for key in priority:
        pool = list(buckets.get(key) or [])
        rng.shuffle(pool)
        take = 2 if n_target >= 20 else 1
        for item in pool[:take]:
            marker = (item.get("instance_id"), item.get("budget_visible"), item.get("reason"))
            if marker in seen:
                continue
            seen.add(marker)
            selected.append(item)
            if len(selected) >= n_target:
                return selected
    leftover = []
    for key, pool in buckets.items():
        for item in pool:
            marker = (item.get("instance_id"), item.get("budget_visible"), item.get("reason"))
            if marker not in seen:
                leftover.append(item)
    rng.shuffle(leftover)
    for item in leftover:
        marker = (item.get("instance_id"), item.get("budget_visible"), item.get("reason"))
        seen.add(marker)
        selected.append(item)
        if len(selected) >= n_target:
            break
    return selected
