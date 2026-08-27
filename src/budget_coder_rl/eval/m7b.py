"""M7B offline train–eval invalid-action discrepancy audit.

Compares frozen E017 training research JSONL with E018 held-out eval.
Does not modify parse_action, prompt, reward, AgentLoop, or E017/E018 artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from budget_coder_rl.eval.m7a import (
    PRIMARY_EVAL_BUDGET,
    classify_event,
    episode_events,
    episode_is_invalid,
    episode_parse_ok,
    event_is_invalid,
    filter_e018_cell,
    localization_score,
)

SCHEMA_VERSION = "bcrl-m7b-v1"
MILESTONE = "M7B"
EXPERIMENT_ID = "M7B"
TRAJ_PER_STEP = 32
N_TRAIN_STEPS = 275
GROUP_N = 4
N_UNIQUE_TRAIN = 2193
BIN_SIZE = 25
EARLY16 = (1, 16)
LATE16 = (260, 275)
PHASE_EARLY = (1, 91)
PHASE_MID = (92, 183)
PHASE_LATE = (184, 275)
EXPECTED_TEMPERATURE = 0.7
EXPECTED_TOP_P = 0.8
EXPECTED_TOP_K = 20
QWEN3_OBS_LIMIT = 4096
TRACKED_TAXONOMY = (
    "multiple_actions",
    "framing_unbalanced_tags",
    "tool_semantic_misuse",
    "other_protocol",
)
VERDICTS = (
    "supported",
    "weakly_supported",
    "mixed",
    "rejected",
    "insufficient",
)
PROTOCOL_LEARNING_DELTA = 0.05
PROMPT_MIX_REL = 0.15
B1_MSCALED_CLOSE = 0.03
EVENT_GAP_TRAIN_EVAL = 0.20


def global_step_from_index(
    index: int,
    *,
    traj_per_step: int = TRAJ_PER_STEP,
) -> int:
    """Reconstruct optimizer step from JSONL write order. 1-based."""
    if index < 0:
        raise ValueError(f"jsonl index must be >= 0, got {index}")
    if traj_per_step <= 0:
        raise ValueError("traj_per_step must be positive")
    return int(index) // int(traj_per_step) + 1


def coarsen_taxonomy(bucket: str | None) -> str | None:
    """Map M7A labels onto the four M7B tracked groups."""
    if bucket is None:
        return None
    if bucket in {"multiple_actions", "framing_unbalanced_tags", "tool_semantic_misuse"}:
        return bucket
    if bucket == "runtime_infra":
        return "runtime_infra"
    return "other_protocol"


def phase_of_step(step: int) -> str:
    if PHASE_EARLY[0] <= step <= PHASE_EARLY[1]:
        return "early"
    if PHASE_MID[0] <= step <= PHASE_MID[1]:
        return "mid"
    if PHASE_LATE[0] <= step <= PHASE_LATE[1]:
        return "late"
    return "other"


def bin_bounds(step: int, *, bin_size: int = BIN_SIZE) -> tuple[int, int]:
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    start = ((step - 1) // bin_size) * bin_size + 1
    end = start + bin_size - 1
    return start, end


def is_padding_index(
    index: int,
    *,
    n_unique: int = N_UNIQUE_TRAIN,
    group_n: int = GROUP_N,
) -> bool:
    """Padding rows are appended after the unique pool, then repeated G times."""
    return int(index) >= int(n_unique) * int(group_n)


def iter_jsonl_indexed(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_index, row) for every non-empty JSON object.

    Unlike M7A ``iter_jsonl``, error rows without events still increment the
    index so ``global_step = index // 32 + 1`` stays aligned with write order.
    """
    index = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                index += 1
                continue
            yield index, row
            index += 1


def first_turn_event(row: Mapping[str, Any]) -> dict[str, Any] | None:
    events = episode_events(row)
    if not events:
        return None
    for event in events:
        turn = event.get("turn")
        try:
            if int(turn) == 1:
                return dict(event)
        except (TypeError, ValueError):
            continue
    return dict(events[0])


def compact_episode_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Per-episode counters. Does not keep raw_action strings."""
    events = episode_events(row)
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    tokens = row.get("tokens") if isinstance(row.get("tokens"), MappingABC) else {}
    knobs = episode_knobs(row)
    n_invalid = 0
    n_protocol = 0
    n_tool = 0
    taxonomy: Counter[str] = Counter()
    for event in events:
        if not event_is_invalid(event):
            continue
        n_invalid += 1
        kind = event.get("error_kind")
        if kind == "protocol":
            n_protocol += 1
        elif kind == "tool":
            n_tool += 1
        bucket = coarsen_taxonomy(classify_event(event) or "other_protocol")
        if bucket:
            taxonomy[bucket] += 1
    first = first_turn_event(row)
    first_invalid = bool(first is not None and event_is_invalid(first))
    first_protocol = bool(first is not None and first.get("error_kind") == "protocol")
    first_taxonomy = None
    if first is not None and first_invalid:
        first_taxonomy = coarsen_taxonomy(classify_event(first) or "other_protocol")
    prompt = tokens.get("prompt_token_count")
    try:
        prompt_n = float(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_n = None
    loc = localization_score(row)
    instance_id = identity.get("instance_id") or row.get("instance_id")
    repo = identity.get("repo") or row.get("repo")
    if not repo and isinstance(instance_id, str) and "__" in instance_id:
        repo = instance_id.split("__", 1)[0].replace("_", "/", 1)
    return {
        "instance_id": instance_id,
        "repo": repo,
        "parse_ok": episode_parse_ok(row),
        "invalid": episode_is_invalid(row),
        "localization_score": loc,
        "n_events": len(events),
        "n_invalid_events": n_invalid,
        "n_protocol_events": n_protocol,
        "n_tool_error_events": n_tool,
        "taxonomy": dict(taxonomy),
        "first_turn_present": first is not None,
        "first_turn_invalid": first_invalid,
        "first_turn_protocol": first_protocol,
        "first_turn_taxonomy": first_taxonomy,
        "prompt_token_count": prompt_n,
        "knobs": knobs,
        "error_row": bool(row.get("error") and "events" not in row),
    }


def episode_knobs(row: Mapping[str, Any]) -> dict[str, Any]:
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    sampling = condition.get("sampling") if isinstance(condition.get("sampling"), MappingABC) else {}
    obs = condition.get("obs_tokens_limit")
    if obs is None:
        obs = budget.get("obs_tokens_limit")
    visible = condition.get("budget_visible")
    if visible is None:
        visible = budget.get("budget_visible")
    return {
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
        "do_sample": sampling.get("do_sample"),
        "obs_tokens_limit": obs,
        "budget_visible": visible,
        "max_turns": condition.get("max_turns"),
        "max_new_tokens_per_turn": condition.get("max_new_tokens_per_turn"),
        "sampling_seed": condition.get("sampling_seed"),
        "n_sampling_keys": len(sampling),
    }


def temperature_is_greedy(value: Any) -> bool:
    if value is None:
        return False
    if value in {0, 0.0}:
        return True
    try:
        return abs(float(value)) < 1e-12
    except (TypeError, ValueError):
        return False


def _empty_step() -> dict[str, Any]:
    return {
        "n_episodes": 0,
        "n_error_rows": 0,
        "n_invalid_episodes": 0,
        "n_parse_ok": 0,
        "n_events": 0,
        "n_invalid_events": 0,
        "n_protocol_events": 0,
        "n_tool_error_events": 0,
        "n_first_turn": 0,
        "n_first_turn_invalid": 0,
        "n_first_turn_protocol": 0,
        "loc_sum": 0.0,
        "loc_n": 0,
        "prompt_sum": 0.0,
        "prompt_n": 0,
        "n_pad": 0,
        "gold_sum": 0.0,
        "gold_n": 0,
        "taxonomy_event": Counter(),
        "taxonomy_first_protocol": Counter(),
        "unique_ids": set(),
        "repos": defaultdict(set),
        "gold_by_unique": {},
        "prompt_by_unique": {},
    }


def _accumulate_episode(
    acc: dict[str, Any],
    metrics: Mapping[str, Any],
    *,
    padding: bool,
    n_gold: int | None,
) -> None:
    if metrics.get("error_row"):
        acc["n_error_rows"] += 1
        return
    acc["n_episodes"] += 1
    acc["n_events"] += int(metrics["n_events"])
    acc["n_invalid_events"] += int(metrics["n_invalid_events"])
    acc["n_protocol_events"] += int(metrics["n_protocol_events"])
    acc["n_tool_error_events"] += int(metrics["n_tool_error_events"])
    if metrics["invalid"]:
        acc["n_invalid_episodes"] += 1
    if metrics["parse_ok"]:
        acc["n_parse_ok"] += 1
    if metrics["first_turn_present"]:
        acc["n_first_turn"] += 1
        if metrics["first_turn_invalid"]:
            acc["n_first_turn_invalid"] += 1
        if metrics["first_turn_protocol"]:
            acc["n_first_turn_protocol"] += 1
            bucket = metrics.get("first_turn_taxonomy") or "other_protocol"
            acc["taxonomy_first_protocol"][bucket] += 1
    loc = metrics.get("localization_score")
    if loc is not None:
        acc["loc_sum"] += float(loc)
        acc["loc_n"] += 1
    prompt = metrics.get("prompt_token_count")
    if prompt is not None:
        acc["prompt_sum"] += float(prompt)
        acc["prompt_n"] += 1
    if padding:
        acc["n_pad"] += 1
    for key, count in (metrics.get("taxonomy") or {}).items():
        acc["taxonomy_event"][key] += int(count)
    instance_id = metrics.get("instance_id")
    repo = metrics.get("repo") or "unknown"
    if instance_id:
        acc["unique_ids"].add(str(instance_id))
        acc["repos"][str(repo)].add(str(instance_id))
        if prompt is not None and str(instance_id) not in acc["prompt_by_unique"]:
            acc["prompt_by_unique"][str(instance_id)] = float(prompt)
        if n_gold is not None and str(instance_id) not in acc["gold_by_unique"]:
            acc["gold_by_unique"][str(instance_id)] = int(n_gold)
    if n_gold is not None:
        acc["gold_sum"] += float(n_gold)
        acc["gold_n"] += 1


def _finalize_acc(acc: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    n_ep = int(acc["n_episodes"])
    n_ev = int(acc["n_events"])
    n_inv = int(acc["n_invalid_events"])
    n_ft = int(acc["n_first_turn"])
    tax = dict(acc["taxonomy_event"])
    tax_first = dict(acc["taxonomy_first_protocol"])
    unique_prompts = list(acc["prompt_by_unique"].values())
    unique_gold = list(acc["gold_by_unique"].values())
    out = {
        "n_episodes": n_ep,
        "n_error_rows": int(acc["n_error_rows"]),
        "n_unique_ids": len(acc["unique_ids"]),
        "n_invalid_episodes": int(acc["n_invalid_episodes"]),
        "n_parse_ok": int(acc["n_parse_ok"]),
        "n_events": n_ev,
        "n_invalid_events": n_inv,
        "n_protocol_events": int(acc["n_protocol_events"]),
        "n_tool_error_events": int(acc["n_tool_error_events"]),
        "n_first_turn": n_ft,
        "n_first_turn_invalid": int(acc["n_first_turn_invalid"]),
        "n_first_turn_protocol": int(acc["n_first_turn_protocol"]),
        "n_pad": int(acc["n_pad"]),
        "episode_invalid_rate": _rate(acc["n_invalid_episodes"], n_ep),
        "event_invalid_rate": _rate(n_inv, n_ev),
        "parse_ok_rate": _rate(acc["n_parse_ok"], n_ep),
        "first_turn_invalid_rate": _rate(acc["n_first_turn_invalid"], n_ft),
        "first_turn_protocol_rate": _rate(acc["n_first_turn_protocol"], n_ft),
        "mean_localization_score": _mean_sum(acc["loc_sum"], acc["loc_n"]),
        "mean_prompt_token_count": _mean_sum(acc["prompt_sum"], acc["prompt_n"]),
        "mean_prompt_token_count_unique": _mean_list(unique_prompts),
        "mean_n_gold_files_unique": _mean_list(unique_gold) if unique_gold else None,
        "taxonomy_event_counts": {key: int(tax.get(key, 0)) for key in TRACKED_TAXONOMY},
        "taxonomy_event_shares": _shares(tax, n_inv, TRACKED_TAXONOMY),
        "taxonomy_first_protocol_counts": {
            key: int(tax_first.get(key, 0)) for key in TRACKED_TAXONOMY
        },
        "taxonomy_first_protocol_shares": _shares(
            tax_first, int(acc["n_first_turn_protocol"]), TRACKED_TAXONOMY
        ),
        "repo_unique_counts": {
            repo: len(ids) for repo, ids in sorted(acc["repos"].items())
        },
    }
    if extra:
        out.update(dict(extra))
    return out


def _rate(num: Any, den: Any) -> float | None:
    den_i = int(den)
    if den_i <= 0:
        return None
    return float(num) / float(den_i)


def _mean_sum(total: Any, n: Any) -> float | None:
    n_i = int(n)
    if n_i <= 0:
        return None
    return float(total) / float(n_i)


def _mean_list(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(item) for item in values) / float(len(values))


def _shares(
    counts: Mapping[str, Any],
    denom: int,
    keys: Sequence[str],
) -> dict[str, float | None]:
    return {key: _rate(counts.get(key, 0), denom) for key in keys}


def _hist_add(hist: dict[str, Counter], field: str, value: Any) -> None:
    if value is None:
        hist[field]["<missing>"] += 1
        return
    if isinstance(value, bool):
        key = str(value)
    elif isinstance(value, float):
        key = f"{value:.4g}"
    else:
        key = str(value)
    hist[field][key] += 1


def analyze_training_stream(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    *,
    gold_by_id: Mapping[str, int] | None = None,
    n_unique: int = N_UNIQUE_TRAIN,
    group_n: int = GROUP_N,
    traj_per_step: int = TRAJ_PER_STEP,
) -> dict[str, Any]:
    """One-pass E017 analysis. ``rows`` is (jsonl_index, episode)."""
    steps: dict[int, dict[str, Any]] = defaultdict(_empty_step)
    knobs_hist: dict[str, Counter] = defaultdict(Counter)
    n_temp_zero = 0
    n_lines = 0
    n_usable = 0
    gold = gold_by_id or {}
    gold_available = bool(gold)
    for index, row in rows:
        n_lines += 1
        step = global_step_from_index(index, traj_per_step=traj_per_step)
        metrics = compact_episode_metrics(row)
        padding = is_padding_index(index, n_unique=n_unique, group_n=group_n)
        instance_id = str(metrics.get("instance_id") or "")
        n_gold = gold.get(instance_id) if instance_id and instance_id in gold else None
        _accumulate_episode(steps[step], metrics, padding=padding, n_gold=n_gold)
        if metrics.get("error_row"):
            continue
        n_usable += 1
        knobs = metrics.get("knobs") or {}
        for field in (
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "obs_tokens_limit",
            "budget_visible",
            "max_turns",
            "max_new_tokens_per_turn",
        ):
            _hist_add(knobs_hist, field, knobs.get(field))
        seed = knobs.get("sampling_seed")
        _hist_add(knobs_hist, "sampling_seed_present", seed is not None)
        if temperature_is_greedy(knobs.get("temperature")):
            n_temp_zero += 1
    step_rows = []
    for step in sorted(steps):
        finalized = _finalize_acc(steps[step], extra={"global_step": step})
        step_rows.append(finalized)
    pooled = _empty_step()
    for acc in steps.values():
        _merge_acc(pooled, acc)
    return {
        "n_jsonl_lines": n_lines,
        "n_usable_episodes": n_usable,
        "n_temp_zero": n_temp_zero,
        "gold_available": gold_available,
        "knob_histograms": {key: dict(counter) for key, counter in knobs_hist.items()},
        "pooled": _finalize_acc(pooled),
        "step_rows": step_rows,
    }


def _merge_acc(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for key in (
        "n_episodes",
        "n_error_rows",
        "n_invalid_episodes",
        "n_parse_ok",
        "n_events",
        "n_invalid_events",
        "n_protocol_events",
        "n_tool_error_events",
        "n_first_turn",
        "n_first_turn_invalid",
        "n_first_turn_protocol",
        "n_pad",
        "gold_n",
    ):
        dst[key] += int(src[key])
    dst["loc_sum"] += float(src["loc_sum"])
    dst["loc_n"] += int(src["loc_n"])
    dst["prompt_sum"] += float(src["prompt_sum"])
    dst["prompt_n"] += int(src["prompt_n"])
    dst["gold_sum"] += float(src["gold_sum"])
    dst["taxonomy_event"].update(src["taxonomy_event"])
    dst["taxonomy_first_protocol"].update(src["taxonomy_first_protocol"])
    dst["unique_ids"].update(src["unique_ids"])
    for repo, ids in src["repos"].items():
        dst["repos"][repo].update(ids)
    for iid, prompt in src["prompt_by_unique"].items():
        dst["prompt_by_unique"].setdefault(iid, prompt)
    for iid, gold in src["gold_by_unique"].items():
        dst["gold_by_unique"].setdefault(iid, gold)


def analyze_eval_cell(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    acc = _empty_step()
    knobs_hist: dict[str, Counter] = defaultdict(Counter)
    n_temp_zero = 0
    n = 0
    for row in rows:
        n += 1
        metrics = compact_episode_metrics(row)
        _accumulate_episode(acc, metrics, padding=False, n_gold=None)
        knobs = metrics.get("knobs") or {}
        for field in (
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "obs_tokens_limit",
            "budget_visible",
            "max_turns",
            "max_new_tokens_per_turn",
        ):
            _hist_add(knobs_hist, field, knobs.get(field))
        _hist_add(knobs_hist, "sampling_seed_present", knobs.get("sampling_seed") is not None)
        if temperature_is_greedy(knobs.get("temperature")):
            n_temp_zero += 1
    out = _finalize_acc(acc)
    out["n_source_rows"] = n
    out["n_temp_zero"] = n_temp_zero
    out["knob_histograms"] = {key: dict(counter) for key, counter in knobs_hist.items()}
    return out


def slice_steps(
    step_rows: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> dict[str, Any]:
    acc = _empty_step()
    n_steps = 0
    for row in step_rows:
        step = int(row["global_step"])
        if start <= step <= end:
            n_steps += 1
            _merge_acc(acc, _acc_from_finalized(row))
    out = _finalize_acc(acc, extra={"step_start": start, "step_end": end, "n_steps": n_steps})
    return out


def _acc_from_finalized(row: Mapping[str, Any]) -> dict[str, Any]:
    acc = _empty_step()
    acc["n_episodes"] = int(row.get("n_episodes") or 0)
    acc["n_error_rows"] = int(row.get("n_error_rows") or 0)
    acc["n_invalid_episodes"] = int(row.get("n_invalid_episodes") or 0)
    acc["n_parse_ok"] = int(row.get("n_parse_ok") or 0)
    acc["n_events"] = int(row.get("n_events") or 0)
    acc["n_invalid_events"] = int(row.get("n_invalid_events") or 0)
    acc["n_protocol_events"] = int(row.get("n_protocol_events") or 0)
    acc["n_tool_error_events"] = int(row.get("n_tool_error_events") or 0)
    acc["n_first_turn"] = int(row.get("n_first_turn") or 0)
    acc["n_first_turn_invalid"] = int(row.get("n_first_turn_invalid") or 0)
    acc["n_first_turn_protocol"] = int(row.get("n_first_turn_protocol") or 0)
    acc["n_pad"] = int(row.get("n_pad") or 0)
    mean_loc = row.get("mean_localization_score")
    if mean_loc is not None and acc["n_episodes"]:
        loc_n = int(row.get("n_episodes") or 0)
        acc["loc_sum"] = float(mean_loc) * loc_n
        acc["loc_n"] = loc_n
    mean_prompt = row.get("mean_prompt_token_count")
    if mean_prompt is not None:
        acc["prompt_sum"] = float(mean_prompt) * int(row.get("n_episodes") or 0)
        acc["prompt_n"] = int(row.get("n_episodes") or 0)
    acc["taxonomy_event"].update(row.get("taxonomy_event_counts") or {})
    acc["taxonomy_first_protocol"].update(row.get("taxonomy_first_protocol_counts") or {})
    step_token = row.get("global_step") or row.get("bin_start") or id(row)
    for repo, count in (row.get("repo_unique_counts") or {}).items():
        for i in range(int(count)):
            token = f"{step_token}::{repo}::{i}"
            acc["repos"][str(repo)].add(token)
            acc["unique_ids"].add(token)
    mean_gold = row.get("mean_n_gold_files_unique")
    n_unique = int(row.get("n_unique_ids") or 0)
    if mean_gold is not None and n_unique:
        acc["gold_sum"] = float(mean_gold) * n_unique
        acc["gold_n"] = n_unique
        for i in range(n_unique):
            acc["gold_by_unique"][f"{step_token}::gold::{i}"] = float(mean_gold)
    return acc


def bin_step_rows(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    bin_size: int = BIN_SIZE,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        start, end = bin_bounds(int(row["global_step"]), bin_size=bin_size)
        grouped[(start, end)].append(row)
    out = []
    for start, end in sorted(grouped):
        sliced = slice_steps(grouped[(start, end)], start, end)
        sliced["bin_start"] = start
        sliced["bin_end"] = end
        sliced["bin_label"] = f"{start}-{end}"
        out.append(sliced)
    return out


def load_step_bcrl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
            rows.append(
                {
                    "global_steps": int(payload.get("global_steps") or 0),
                    "n_trajectories": metrics.get("bcrl/n_trajectories"),
                    "invalid_action_rate": metrics.get("bcrl/invalid_action_rate"),
                    "parse_ok_rate": metrics.get("bcrl/parse_ok_rate"),
                    "protocol_error_count": metrics.get("bcrl/protocol_error_count"),
                    "reward_mean": metrics.get("bcrl/reward/mean"),
                }
            )
    return rows


def cross_check_step_bcrl(
    step_rows: Sequence[Mapping[str, Any]],
    bcrl_rows: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    by_step = {int(row["global_step"]): row for row in step_rows}
    n_checked = 0
    n_rate_mismatch = 0
    n_count_mismatch = 0
    examples: list[dict[str, Any]] = []
    for rec in bcrl_rows:
        step = int(rec["global_steps"])
        series = by_step.get(step)
        if series is None:
            continue
        n_checked += 1
        left = series.get("episode_invalid_rate")
        right = rec.get("invalid_action_rate")
        if left is not None and right is not None and abs(float(left) - float(right)) > 1e-9:
            n_rate_mismatch += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "global_step": step,
                        "kind": "invalid_action_rate",
                        "reconstructed": left,
                        "step_bcrl": right,
                    }
                )
        proto_left = series.get("n_protocol_events")
        proto_right = rec.get("protocol_error_count")
        if proto_left is not None and proto_right is not None:
            if int(proto_left) != int(proto_right):
                n_count_mismatch += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "global_step": step,
                            "kind": "protocol_error_count",
                            "reconstructed": proto_left,
                            "step_bcrl": proto_right,
                        }
                    )
        traj = rec.get("n_trajectories")
        if traj is not None and int(traj) != int(series.get("n_episodes") or 0):
            n_count_mismatch += 1
    return {
        "n_checked": n_checked,
        "n_rate_mismatch": n_rate_mismatch,
        "n_count_mismatch": n_count_mismatch,
        "pass": n_rate_mismatch == 0 and n_count_mismatch == 0,
        "examples": examples,
        "atol": atol,
    }


def load_pad_ids(train_candidates: Mapping[str, Any] | None) -> set[str]:
    if not train_candidates:
        return set()
    padding = train_candidates.get("padding") if isinstance(train_candidates, MappingABC) else {}
    ids = padding.get("pad_ids") if isinstance(padding, MappingABC) else None
    if not ids:
        ids = train_candidates.get("pad_ids")
    return {str(item) for item in (ids or [])}


def load_optional_gold(repo_root: Path) -> dict[str, int]:
    """Evaluator-only gold file counts. Empty if sidecars are absent."""
    jsonl = Path(repo_root) / "data" / "interim" / "swe_gym" / "m1c_oracle.jsonl"
    if jsonl.is_file():
        out: dict[str, int] = {}
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                instance_id = str(row.get("instance_id") or "")
                if not instance_id:
                    continue
                n_gold = row.get("n_gold_edit_files")
                if n_gold is None:
                    files = row.get("base_changed_files") or []
                    n_gold = len(files) if isinstance(files, list) else None
                if n_gold is None:
                    continue
                out[instance_id] = int(n_gold)
        return out
    try:
        from budget_coder_rl.eval.oracle import default_oracle_path, load_evaluator_oracle
    except Exception:
        return {}
    parquet = default_oracle_path(repo_root)
    if not Path(parquet).is_file():
        return {}
    try:
        index = load_evaluator_oracle(parquet)
    except Exception:
        return {}
    out = {}
    for instance_id in getattr(index, "_rows", {}):
        row = index.get(instance_id)
        out[instance_id] = len(row.base_changed_files)
    return out


def _nested_get(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, MappingABC):
            return None
        cur = cur.get(key)
    return cur


def audit_execution_contract(
    *,
    e017_provenance: Mapping[str, Any] | None,
    e017_config: Mapping[str, Any] | None,
    e018_provenance: Mapping[str, Any] | None,
    e018_overlay: Mapping[str, Any] | None,
    e018_integrity: Mapping[str, Any] | None,
    e017_empirical: Mapping[str, Any] | None,
    e018_cells: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    e017_rollout = _nested_get(e017_config, "actor_rollout_ref", "rollout") or {}
    e017_val = e017_rollout.get("val_kwargs") if isinstance(e017_rollout, MappingABC) else {}
    e017_trainer = _nested_get(e017_config, "trainer") or {}
    e017_sampling = (e017_provenance or {}).get("sampling_rollout") or {}
    e017_intended = (e017_provenance or {}).get("sampling_intended") or {}
    e018_frozen = (e018_overlay or {}).get("frozen_from_parent") or {}
    e018_sampling = e018_frozen.get("sampling") if isinstance(e018_frozen, MappingABC) else {}
    e018_intended = (e018_provenance or {}).get("sampling_intended") or e018_sampling
    agent_e017 = (e017_provenance or {}).get("agent_loop_config") or {}
    integrity = e018_integrity or {}
    e017_temp_zero = int((e017_empirical or {}).get("n_temp_zero") or 0)
    e018_temp_zero = 0
    for cell in (e018_cells or {}).values():
        e018_temp_zero += int(cell.get("n_temp_zero") or 0)
    greedy_override = e017_temp_zero > 0 or e018_temp_zero > 0
    n_train = e017_sampling.get("n") if e017_sampling.get("n") is not None else e017_rollout.get("n")
    n_eval = (e018_provenance or {}).get("vllm_rollout_n")
    if n_eval is None:
        n_eval = e018_sampling.get("n")
    validate_train = e017_trainer.get("val_before_train")
    test_freq = e017_trainer.get("test_freq")
    validate_eval = (e018_provenance or {}).get("validate")
    if validate_eval is None:
        validate_eval = e018_frozen.get("validate")
    unmatched = []
    if n_train != n_eval:
        unmatched.append({"knob": "n", "e017": n_train, "e018": n_eval})
    unmatched.append(
        {
            "knob": "sampling_seed",
            "e017": "stripped_unseeded",
            "e018": "paired_seed",
        }
    )
    unmatched.append(
        {
            "knob": "task_pool",
            "e017": "train_2193_one_pass",
            "e018": "held_out_task_dev_244",
        }
    )
    unmatched.append(
        {
            "knob": "lora",
            "e017": "fresh_then_evolving",
            "e018_B1": "none",
            "e018_M_scaled": "adapter_123_frozen_step_275",
        }
    )
    sampling_matched = (
        _close_num(e017_sampling.get("temperature"), EXPECTED_TEMPERATURE)
        and _close_num(
            (e018_intended or {}).get("temperature") or e018_sampling.get("temperature"),
            EXPECTED_TEMPERATURE,
        )
        and not greedy_override
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "verl_validate_override": {
            "path": "verl/experimental/agent_loop/agent_loop.py AgentLoopWorker.generate_sequences",
            "behavior": (
                "If batch.meta_info['validate'] is True, sampling temperature/top_p/top_k "
                "are replaced by rollout.val_kwargs (E017 val_kwargs is greedy T=0)."
            ),
            "e017_val_kwargs": dict(e017_val) if isinstance(e017_val, MappingABC) else e017_val,
            "e017_val_before_train": validate_train,
            "e017_test_freq": test_freq,
            "e018_validate": validate_eval,
            "episode_temperature_zero": {
                "e017": e017_temp_zero,
                "e018": e018_temp_zero,
            },
            "greedy_override_detected": greedy_override,
        },
        "e017": {
            "sampling_rollout": dict(e017_sampling) if isinstance(e017_sampling, MappingABC) else e017_sampling,
            "sampling_intended": dict(e017_intended) if isinstance(e017_intended, MappingABC) else e017_intended,
            "rollout_temperature": e017_rollout.get("temperature") if isinstance(e017_rollout, MappingABC) else None,
            "rollout_top_p": e017_rollout.get("top_p") if isinstance(e017_rollout, MappingABC) else None,
            "rollout_top_k": e017_rollout.get("top_k") if isinstance(e017_rollout, MappingABC) else None,
            "rollout_do_sample": e017_rollout.get("do_sample") if isinstance(e017_rollout, MappingABC) else None,
            "rollout_n": n_train,
            "lora_rank": _nested_get(e017_config, "actor_rollout_ref", "model", "lora_rank"),
            "agent_loop_config": agent_e017,
            "empirical_knobs": (e017_empirical or {}).get("knob_histograms"),
            "n_temp_zero": e017_temp_zero,
        },
        "e018": {
            "sampling_intended": dict(e018_intended) if isinstance(e018_intended, MappingABC) else e018_intended,
            "frozen_sampling": dict(e018_sampling) if isinstance(e018_sampling, MappingABC) else e018_sampling,
            "validate": validate_eval,
            "vllm_rollout_n": n_eval,
            "max_turns": e018_frozen.get("max_turns") if isinstance(e018_frozen, MappingABC) else None,
            "primary_training_B_obs": (
                e018_frozen.get("primary_training_B_obs")
                if isinstance(e018_frozen, MappingABC)
                else None
            ),
            "agent_loop_config": (
                e018_frozen.get("agent_loop_config") if isinstance(e018_frozen, MappingABC) else None
            ),
            "treatment_integrity_pass": integrity.get("pass")
            if "pass" in integrity
            else (integrity.get("listed_lora_ids") == [123] or integrity.get("load_ok")),
            "listed_lora_ids": integrity.get("listed_lora_ids"),
            "lora_request_attached": integrity.get("lora_request_attached"),
            "cell_empirical": {
                key: cell.get("knob_histograms") for key, cell in (e018_cells or {}).items()
            },
            "n_temp_zero": e018_temp_zero,
        },
        "matched_sampling_temperature": bool(sampling_matched),
        "execution_matched": False,
        "unmatched": unmatched,
        "note": (
            "execution_matched is false because n, sampling_seed policy, task pool, "
            "and LoRA lifecycle differ even when temperature/top_p/top_k/validate/B_obs match. "
            "Do not attribute the 46.6% vs ~11% gap to RL unless H2 survives these mismatches."
        ),
    }
    return contract


def _close_num(value: Any, expected: float, atol: float = 1e-6) -> bool:
    if value is None:
        return False
    try:
        return abs(float(value) - float(expected)) <= atol
    except (TypeError, ValueError):
        return False


def stratify_phases(
    step_rows: Sequence[Mapping[str, Any]],
    *,
    gold_available: bool,
) -> dict[str, Any]:
    phases = {
        "early": slice_steps(step_rows, *PHASE_EARLY),
        "mid": slice_steps(step_rows, *PHASE_MID),
        "late": slice_steps(step_rows, *PHASE_LATE),
    }
    early_prompt = phases["early"].get("mean_prompt_token_count")
    late_prompt = phases["late"].get("mean_prompt_token_count")
    rel = None
    if early_prompt and late_prompt and float(early_prompt) != 0:
        rel = abs(float(late_prompt) - float(early_prompt)) / abs(float(early_prompt))
    mix_stable = rel is not None and rel < PROMPT_MIX_REL
    return {
        "gold_available": gold_available,
        "phases": phases,
        "prompt_length_rel_change_early_to_late": rel,
        "prompt_mix_stable": mix_stable,
        "note": (
            "Train is one-pass shuffle=false repo-round-robin. Repo mix is less "
            "confounded than sequential-by-repo, but instance_id order within repo "
            "can still drift difficulty. First-turn protocol-only is the behavioral probe."
        ),
    }


def matched_comparison(
    *,
    late16: Mapping[str, Any],
    last_bin: Mapping[str, Any] | None,
    e018_cells: Mapping[str, Mapping[str, Any]],
    execution_matched: bool,
) -> dict[str, Any]:
    b1 = e018_cells.get("B1@4096") or {}
    scaled = e018_cells.get("M_scaled@4096") or {}
    return {
        "execution_matched": execution_matched,
        "forbid_rl_attribution": not execution_matched,
        "units": {
            "event_invalid_rate": "assistant turns with error_kind in {protocol, tool}",
            "first_turn_protocol_rate": "episodes whose first turn has error_kind=protocol",
            "parse_ok": "episode-level final submission parse via _truthy",
        },
        "e017_late16": _compact_cell(late16, label="E017 steps 260-275"),
        "e017_last_bin": _compact_cell(last_bin, label="E017 last 25-step bin") if last_bin else None,
        "e018_B1_4096": _compact_cell(b1, label="E018 B1@4096"),
        "e018_M_scaled_4096": _compact_cell(scaled, label="E018 M_scaled@4096"),
        "deltas": {
            "late16_minus_M_scaled_event_invalid": _sub(
                late16.get("event_invalid_rate"), scaled.get("event_invalid_rate")
            ),
            "late16_minus_B1_event_invalid": _sub(
                late16.get("event_invalid_rate"), b1.get("event_invalid_rate")
            ),
            "M_scaled_minus_B1_event_invalid": _sub(
                scaled.get("event_invalid_rate"), b1.get("event_invalid_rate")
            ),
            "late16_minus_M_scaled_first_turn_protocol": _sub(
                late16.get("first_turn_protocol_rate"),
                scaled.get("first_turn_protocol_rate"),
            ),
            "M_scaled_minus_B1_first_turn_protocol": _sub(
                scaled.get("first_turn_protocol_rate"),
                b1.get("first_turn_protocol_rate"),
            ),
        },
    }


def _compact_cell(cell: Mapping[str, Any] | None, *, label: str) -> dict[str, Any] | None:
    if not cell:
        return None
    return {
        "label": label,
        "n_episodes": cell.get("n_episodes"),
        "n_events": cell.get("n_events"),
        "episode_invalid_rate": cell.get("episode_invalid_rate"),
        "event_invalid_rate": cell.get("event_invalid_rate"),
        "first_turn_invalid_rate": cell.get("first_turn_invalid_rate"),
        "first_turn_protocol_rate": cell.get("first_turn_protocol_rate"),
        "parse_ok_rate": cell.get("parse_ok_rate"),
        "mean_localization_score": cell.get("mean_localization_score"),
        "taxonomy_event_shares": cell.get("taxonomy_event_shares"),
        "taxonomy_first_protocol_shares": cell.get("taxonomy_first_protocol_shares"),
    }


def _sub(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def hypothesis_verdicts(
    *,
    contract: Mapping[str, Any],
    pooled: Mapping[str, Any],
    early16: Mapping[str, Any],
    late16: Mapping[str, Any],
    stratification: Mapping[str, Any],
    matched: Mapping[str, Any],
    step_check: Mapping[str, Any],
) -> dict[str, Any]:
    greedy = bool(
        _nested_get(contract, "verl_validate_override", "greedy_override_detected")
    )
    sampling_t_ok = bool(contract.get("matched_sampling_temperature"))
    b1 = matched.get("e018_B1_4096") or {}
    scaled = matched.get("e018_M_scaled_4096") or {}
    deltas = matched.get("deltas") or {}
    ftp_early = early16.get("first_turn_protocol_rate")
    ftp_late = late16.get("first_turn_protocol_rate")
    ftp_drop = _sub(ftp_early, ftp_late)
    mix_stable = bool(stratification.get("prompt_mix_stable"))
    b1_vs_scaled_event = deltas.get("M_scaled_minus_B1_event_invalid")
    late_vs_eval = deltas.get("late16_minus_M_scaled_event_invalid")
    train_event = late16.get("event_invalid_rate") or pooled.get("event_invalid_rate")
    eval_event = scaled.get("event_invalid_rate")

    if greedy:
        h1_greedy = "supported"
        h1 = "supported"
    elif sampling_t_ok:
        h1_greedy = "rejected"
        h1 = "weakly_supported"
    else:
        h1_greedy = "insufficient"
        h1 = "mixed"
    h1_evidence = [
        f"greedy_validate_override={h1_greedy} (episode T=0 count E017="
        f"{_nested_get(contract, 'verl_validate_override', 'episode_temperature_zero', 'e017')}, "
        f"E018={_nested_get(contract, 'verl_validate_override', 'episode_temperature_zero', 'e018')})",
        "val_kwargs stay greedy by design; E017 val_before_train=false test_freq=-1; "
        "E018 build_batch(validate=False)",
        f"secondary unmatched: {contract.get('unmatched')}",
        "n=4 vs n=1 and seed vs unseeded are real mismatches but are not by themselves "
        "a 4x event-invalid gap without task-pool evidence",
    ]

    ftp_eval = b1.get("first_turn_protocol_rate")
    ftp_late_vs_eval = _sub(ftp_late, ftp_eval)
    no_heldout_protocol_gain = (
        b1_vs_scaled_event is not None
        and abs(float(b1_vs_scaled_event)) <= B1_MSCALED_CLOSE
        and late_vs_eval is not None
        and float(late_vs_eval) >= EVENT_GAP_TRAIN_EVAL
    )
    if no_heldout_protocol_gain:
        h2 = "rejected"
    elif ftp_drop is None:
        h2 = "insufficient"
    elif ftp_drop >= PROTOCOL_LEARNING_DELTA and mix_stable:
        h2 = "weakly_supported"
    elif ftp_drop >= PROTOCOL_LEARNING_DELTA and not mix_stable:
        h2 = "mixed"
    else:
        h2 = "rejected"
    h2_evidence = [
        f"first_turn_protocol early16={_fmt(ftp_early)} late16={_fmt(ftp_late)} drop={_fmt(ftp_drop)}",
        f"prompt mix stable early→late: {mix_stable} "
        f"(rel={_fmt(stratification.get('prompt_length_rel_change_early_to_late'))})",
        f"E018 B1 vs M_scaled event invalid Δ={_fmt(b1_vs_scaled_event)} "
        f"(first-turn protocol both {_fmt(ftp_eval)})",
        f"E017 late16 vs E018 M_scaled event invalid Δ={_fmt(late_vs_eval)}; "
        f"late16 first-turn protocol minus B1={_fmt(ftp_late_vs_eval)}",
        "Transferable protocol-compliance learning requires a first-turn protocol drop "
        "after mix control AND a held-out gap vs B1. B1≈M_scaled with late-train still "
        "dirty rejects a learned-compliance explanation of the 46.6% vs ~11% gap.",
    ]

    if late_vs_eval is not None and float(late_vs_eval) >= EVENT_GAP_TRAIN_EVAL:
        h3 = "supported"
    elif train_event is not None and eval_event is not None:
        h3 = "supported" if float(train_event) - float(eval_event) >= EVENT_GAP_TRAIN_EVAL else "mixed"
    else:
        h3 = "insufficient"
    h3_evidence = [
        "Task pools unmatched by construction: E017 train 2193 one-pass vs E018 held-out-task dev 244",
        f"early16 event invalid={_fmt(early16.get('event_invalid_rate'))} "
        f"(base+fresh LoRA on train) vs E018 B1={_fmt(b1.get('event_invalid_rate'))} (base on dev)",
        f"late16 event invalid={_fmt(late16.get('event_invalid_rate'))} vs M_scaled="
        f"{_fmt(scaled.get('event_invalid_rate'))}",
        f"repo unique early={phases_repo_preview(stratification, 'early')} "
        f"late={phases_repo_preview(stratification, 'late')}",
    ]

    check_ok = bool(step_check.get("pass"))
    if check_ok:
        h4 = "rejected"
    else:
        h4 = "mixed"
    h4_evidence = [
        "M7A already showed episode any-error 0.952 vs event 0.466; that is not the remaining 0.466 vs ~0.11 gap",
        "M7B uses the same event_is_invalid / first-turn predicates on train and eval JSONL",
        f"step reconstruction vs step_bcrl pass={check_ok} "
        f"rate_mismatch={step_check.get('n_rate_mismatch')} "
        f"count_mismatch={step_check.get('n_count_mismatch')}",
        "parse_ok uses _truthy so the JSONL string trap cannot inflate rates",
    ]

    items = [
        _verdict_item("H1", "sampling/execution mismatch", h1, h1_evidence, extra={
            "greedy_validate_override": h1_greedy,
            "secondary_mismatches": ["n=4 vs n=1", "unseeded vs paired seed"],
        }),
        _verdict_item("H2", "RL learned protocol compliance", h2, h2_evidence),
        _verdict_item("H3", "train/dev task-distribution effect", h3, h3_evidence),
        _verdict_item("H4", "logging/metric mismatch", h4, h4_evidence),
    ]
    rank = sorted(
        items,
        key=lambda item: _verdict_rank_score(item["id"], item["verdict"], late_vs_eval),
        reverse=True,
    )
    primary = rank[0]["id"] if rank else None
    return {
        "schema_version": SCHEMA_VERSION,
        "items": items,
        "primary_gap_contributor": primary,
        "ranking": [item["id"] for item in rank],
        "gate": {
            "execution_matched": bool(matched.get("execution_matched")),
            "protocol_compliance_learning_credible": h2 in {"supported", "weakly_supported"},
            "do_not_start_intervention": True,
        },
    }


def phases_repo_preview(stratification: Mapping[str, Any], phase: str) -> str:
    phases = stratification.get("phases") or {}
    cell = phases.get(phase) or {}
    counts = cell.get("repo_unique_counts") or {}
    top = sorted(counts.items(), key=lambda kv: -int(kv[1]))[:3]
    return ", ".join(f"{repo}:{n}" for repo, n in top) or "n/a"


def _verdict_item(
    hid: str,
    title: str,
    verdict: str,
    evidence: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict}")
    item = {"id": hid, "title": title, "verdict": verdict, "evidence": list(evidence)}
    if extra:
        item.update(dict(extra))
    return item


def _verdict_rank_score(hid: str, verdict: str, late_vs_eval: Any) -> float:
    weights = {
        "supported": 4.0,
        "weakly_supported": 2.0,
        "mixed": 1.0,
        "insufficient": 0.2,
        "rejected": 0.0,
    }
    score = weights[verdict]
    if hid == "H3":
        score += 0.5
    if hid == "H4":
        score -= 0.4
    if hid == "H1" and verdict == "weakly_supported":
        score -= 0.3
    return score


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return value


def step_series_csv_rows(step_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in step_rows:
        shares = item.get("taxonomy_event_shares") or {}
        rows.append(
            {
                "global_step": item.get("global_step"),
                "n_episodes": item.get("n_episodes"),
                "n_events": item.get("n_events"),
                "episode_invalid_rate": item.get("episode_invalid_rate"),
                "event_invalid_rate": item.get("event_invalid_rate"),
                "first_turn_invalid_rate": item.get("first_turn_invalid_rate"),
                "first_turn_protocol_rate": item.get("first_turn_protocol_rate"),
                "parse_ok_rate": item.get("parse_ok_rate"),
                "mean_localization_score": item.get("mean_localization_score"),
                "n_protocol_events": item.get("n_protocol_events"),
                "n_pad": item.get("n_pad"),
                "mean_prompt_token_count": item.get("mean_prompt_token_count"),
                "share_multiple_actions": shares.get("multiple_actions"),
                "share_framing_unbalanced_tags": shares.get("framing_unbalanced_tags"),
                "share_tool_semantic_misuse": shares.get("tool_semantic_misuse"),
                "share_other_protocol": shares.get("other_protocol"),
            }
        )
    return rows


def binned_csv_rows(binned: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in binned:
        shares = item.get("taxonomy_event_shares") or {}
        first = item.get("taxonomy_first_protocol_shares") or {}
        rows.append(
            {
                "bin_start": item.get("bin_start"),
                "bin_end": item.get("bin_end"),
                "n_steps": item.get("n_steps"),
                "n_episodes": item.get("n_episodes"),
                "n_unique_ids": item.get("n_unique_ids"),
                "episode_invalid_rate": item.get("episode_invalid_rate"),
                "event_invalid_rate": item.get("event_invalid_rate"),
                "first_turn_invalid_rate": item.get("first_turn_invalid_rate"),
                "first_turn_protocol_rate": item.get("first_turn_protocol_rate"),
                "parse_ok_rate": item.get("parse_ok_rate"),
                "mean_localization_score": item.get("mean_localization_score"),
                "mean_prompt_token_count": item.get("mean_prompt_token_count"),
                "mean_n_gold_files_unique": item.get("mean_n_gold_files_unique"),
                "share_multiple_actions": shares.get("multiple_actions"),
                "share_framing_unbalanced_tags": shares.get("framing_unbalanced_tags"),
                "share_tool_semantic_misuse": shares.get("tool_semantic_misuse"),
                "share_other_protocol": shares.get("other_protocol"),
                "first_share_multiple_actions": first.get("multiple_actions"),
                "first_share_framing_unbalanced_tags": first.get("framing_unbalanced_tags"),
                "first_share_other_protocol": first.get("other_protocol"),
            }
        )
    return rows


def taxonomy_over_time_rows(binned: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in binned:
        counts = item.get("taxonomy_event_counts") or {}
        shares = item.get("taxonomy_event_shares") or {}
        first_counts = item.get("taxonomy_first_protocol_counts") or {}
        first_shares = item.get("taxonomy_first_protocol_shares") or {}
        for key in TRACKED_TAXONOMY:
            rows.append(
                {
                    "bin_start": item.get("bin_start"),
                    "bin_end": item.get("bin_end"),
                    "category": key,
                    "n": counts.get(key),
                    "share": shares.get(key),
                    "first_turn_protocol_n": first_counts.get(key),
                    "first_turn_protocol_share": first_shares.get(key),
                }
            )
    return rows


STEP_CSV_FIELDS = (
    "global_step",
    "n_episodes",
    "n_events",
    "episode_invalid_rate",
    "event_invalid_rate",
    "first_turn_invalid_rate",
    "first_turn_protocol_rate",
    "parse_ok_rate",
    "mean_localization_score",
    "n_protocol_events",
    "n_pad",
    "mean_prompt_token_count",
    "share_multiple_actions",
    "share_framing_unbalanced_tags",
    "share_tool_semantic_misuse",
    "share_other_protocol",
)
BIN_CSV_FIELDS = (
    "bin_start",
    "bin_end",
    "n_steps",
    "n_episodes",
    "n_unique_ids",
    "episode_invalid_rate",
    "event_invalid_rate",
    "first_turn_invalid_rate",
    "first_turn_protocol_rate",
    "parse_ok_rate",
    "mean_localization_score",
    "mean_prompt_token_count",
    "mean_n_gold_files_unique",
    "share_multiple_actions",
    "share_framing_unbalanced_tags",
    "share_tool_semantic_misuse",
    "share_other_protocol",
    "first_share_multiple_actions",
    "first_share_framing_unbalanced_tags",
    "first_share_other_protocol",
)
TAXONOMY_CSV_FIELDS = (
    "bin_start",
    "bin_end",
    "category",
    "n",
    "share",
    "first_turn_protocol_n",
    "first_turn_protocol_share",
)


def write_curves_svg(binned: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Minimal SVG so a curve exists when matplotlib is absent."""
    series = [
        ("event_invalid_rate", "#1f77b4"),
        ("first_turn_protocol_rate", "#d62728"),
        ("episode_invalid_rate", "#ff7f0e"),
        ("parse_ok_rate", "#2ca02c"),
        ("mean_localization_score", "#9467bd"),
    ]
    width, height = 840, 420
    left, right, top, bottom = 56, 16, 24, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [0.5 * (float(row["bin_start"]) + float(row["bin_end"])) for row in binned]
    if not xs:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return
    x_min, x_max = min(xs), max(xs)
    if x_max <= x_min:
        x_max = x_min + 1.0

    def tx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def ty(y: float) -> float:
        return top + (1.0 - y) * plot_h

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='16' font-size='12' font-family='sans-serif'>"
        "E017 binned rates (bin means only; source JSONL unchanged)</text>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333'/>",
    ]
    for i in range(5):
        y = i / 4
        parts.append(
            f"<line x1='{left}' y1='{ty(y)}' x2='{left + plot_w}' y2='{ty(y)}' "
            f"stroke='#eee'/><text x='{left - 8}' y='{ty(y) + 4}' font-size='10' "
            f"text-anchor='end' font-family='sans-serif'>{y:.2f}</text>"
        )
    legend_x = left + 8
    legend_y = top + 12
    for name, color in series:
        pts = []
        for x, row in zip(xs, binned):
            raw = row.get(name)
            if raw is None:
                continue
            pts.append(f"{tx(x):.1f},{ty(float(raw)):.1f}")
        if pts:
            parts.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}'/>"
            )
        parts.append(
            f"<rect x='{legend_x}' y='{legend_y - 8}' width='10' height='10' fill='{color}'/>"
            f"<text x='{legend_x + 14}' y='{legend_y}' font-size='10' font-family='sans-serif'>{name}</text>"
        )
        legend_y += 14
    parts.append(
        f"<text x='{left + plot_w / 2}' y='{height - 8}' font-size='11' text-anchor='middle' "
        "font-family='sans-serif'>global_step (bin midpoint)</text>"
    )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_curves_png(binned: Sequence[Mapping[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    xs = [0.5 * (float(row["bin_start"]) + float(row["bin_end"])) for row in binned]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name in (
        "event_invalid_rate",
        "first_turn_protocol_rate",
        "episode_invalid_rate",
        "parse_ok_rate",
        "mean_localization_score",
    ):
        ys = [row.get(name) for row in binned]
        if any(item is not None for item in ys):
            ax.plot(xs, [float(item) if item is not None else float("nan") for item in ys], marker="o", label=name)
    ax.set_xlabel("global_step (bin midpoint)")
    ax.set_ylabel("rate / score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("E017 binned invalid-action series")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def render_summary(payload: Mapping[str, Any]) -> str:
    contract = payload.get("execution_contract") or {}
    pooled = payload.get("e017_pooled") or {}
    early16 = payload.get("early16") or {}
    late16 = payload.get("late16") or {}
    matched = payload.get("matched_comparison") or {}
    verdicts = payload.get("hypothesis_verdicts") or {}
    check = payload.get("step_bcrl_check") or {}
    strat = payload.get("stratification") or {}
    cells = payload.get("e018_cells") or {}
    override = contract.get("verl_validate_override") or {}
    lines = [
        "# M7B — Train–Eval Invalid-Action Discrepancy Audit",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        "- status: **diagnostic only** (no parser / prompt / reward / RL change)",
        "- frozen E017 / E018 artifacts: **not modified**",
        f"- primary gap: E017 event invalid {_fmt(pooled.get('event_invalid_rate'))} "
        f"vs E018 M_scaled@4096 {_fmt((matched.get('e018_M_scaled_4096') or {}).get('event_invalid_rate'))}",
        f"- execution_matched: **{matched.get('execution_matched')}** "
        f"(forbid RL attribution: {matched.get('forbid_rl_attribution')})",
        f"- primary_gap_contributor: **{(verdicts.get('primary_gap_contributor'))}**",
        f"- protocol-compliance learning credible: "
        f"**{(verdicts.get('gate') or {}).get('protocol_compliance_learning_credible')}**",
        f"- step reconstruction vs step_bcrl: **{check.get('pass')}**",
        "",
        "This note explains the remaining **turn-level** train/eval invalid gap. "
        "M7A already showed that training `invalid_action≈0.952` is episode any-error.",
        "",
        "## 0. Gate",
        "",
        "Do not start parser, prompt, reward, sampling, or training intervention from this audit.",
        "The 46.6% vs ~11% comparison is not an RL win unless H2 survives execution and task-mix controls.",
        "",
        "## 1. Execution contract",
        "",
        "veRL `validate=True` replaces temperature/top_p/top_k with `rollout.val_kwargs` "
        "(E017 val_kwargs is greedy T=0 by design, so a mistaken validate flag would be obvious).",
        "",
        _md_kv_table(
            [
                ("E017 val_before_train", override.get("e017_val_before_train")),
                ("E017 test_freq", override.get("e017_test_freq")),
                ("E018 validate", override.get("e018_validate")),
                ("greedy_override_detected", override.get("greedy_override_detected")),
                ("E017 episode T=0", (override.get("episode_temperature_zero") or {}).get("e017")),
                ("E018 episode T=0", (override.get("episode_temperature_zero") or {}).get("e018")),
                ("matched_sampling_temperature", contract.get("matched_sampling_temperature")),
                ("E017 rollout.n", (contract.get("e017") or {}).get("rollout_n")),
                ("E018 vllm_rollout_n", (contract.get("e018") or {}).get("vllm_rollout_n")),
            ]
        ),
        "",
        "Unmatched knobs (do not ignore when attributing rates):",
        "",
    ]
    for item in contract.get("unmatched") or []:
        knob = item.get("knob")
        extra = {key: value for key, value in item.items() if key != "knob"}
        pretty = ", ".join(f"{key}={value}" for key, value in extra.items())
        lines.append(f"- `{knob}`: {pretty}")
    lines.extend(
        [
            "",
            "## 2. Metric alignment",
            "",
            "- Episode invalid = any protocol or tool error in the trajectory (training `bcrl/invalid_action_rate`).",
            "- Event invalid = assistant turns with `error_kind in {protocol, tool}` (M7A diagnostic; this audit's primary rate).",
            "- First-turn protocol-only = first assistant turn has `error_kind=protocol` (H2 probe; ignores later tool path errors).",
            "- `parse_ok` uses `_truthy` (string `'False'` is false).",
            "",
            _md_kv_table(
                [
                    ("E017 n_episodes", pooled.get("n_episodes")),
                    ("E017 n_events", pooled.get("n_events")),
                    ("E017 episode invalid", pooled.get("episode_invalid_rate")),
                    ("E017 event invalid", pooled.get("event_invalid_rate")),
                    ("E017 first-turn invalid", pooled.get("first_turn_invalid_rate")),
                    ("E017 first-turn protocol", pooled.get("first_turn_protocol_rate")),
                    ("E017 parse_ok", pooled.get("parse_ok_rate")),
                ]
            ),
            "",
            "## 3. Temporal (E017)",
            "",
            "Smoothing is bin/window means only. Source JSONL is unchanged. "
            f"Bins are {BIN_SIZE} steps; early16=steps {EARLY16[0]}–{EARLY16[1]}; "
            f"late16=steps {LATE16[0]}–{LATE16[1]}.",
            "",
            _md_kv_table(
                [
                    ("early16 event invalid", early16.get("event_invalid_rate")),
                    ("early16 first-turn protocol", early16.get("first_turn_protocol_rate")),
                    ("early16 parse_ok", early16.get("parse_ok_rate")),
                    ("early16 loc reward", early16.get("mean_localization_score")),
                    ("late16 event invalid", late16.get("event_invalid_rate")),
                    ("late16 first-turn protocol", late16.get("first_turn_protocol_rate")),
                    ("late16 parse_ok", late16.get("parse_ok_rate")),
                    ("late16 loc reward", late16.get("mean_localization_score")),
                    (
                        "first-turn protocol drop (early−late)",
                        _sub(
                            early16.get("first_turn_protocol_rate"),
                            late16.get("first_turn_protocol_rate"),
                        ),
                    ),
                ]
            ),
            "",
            "### Taxonomy over time (invalid events, bin shares)",
            "",
            _taxonomy_bin_table(payload.get("binned") or []),
            "",
            "## 4. Task-order confound",
            "",
            strat.get("note") or "",
            "",
            _md_kv_table(
                [
                    ("gold_available", strat.get("gold_available")),
                    (
                        "prompt rel change early→late",
                        strat.get("prompt_length_rel_change_early_to_late"),
                    ),
                    ("prompt mix stable", strat.get("prompt_mix_stable")),
                    ("early unique n", (strat.get("phases") or {}).get("early", {}).get("n_unique_ids")),
                    ("late unique n", (strat.get("phases") or {}).get("late", {}).get("n_unique_ids")),
                    (
                        "early mean prompt",
                        (strat.get("phases") or {}).get("early", {}).get("mean_prompt_token_count"),
                    ),
                    (
                        "late mean prompt",
                        (strat.get("phases") or {}).get("late", {}).get("mean_prompt_token_count"),
                    ),
                    (
                        "early first-turn protocol",
                        (strat.get("phases") or {}).get("early", {}).get("first_turn_protocol_rate"),
                    ),
                    (
                        "late first-turn protocol",
                        (strat.get("phases") or {}).get("late", {}).get("first_turn_protocol_rate"),
                    ),
                    (
                        "early mean n_gold_files",
                        (strat.get("phases") or {}).get("early", {}).get("mean_n_gold_files_unique"),
                    ),
                    (
                        "late mean n_gold_files",
                        (strat.get("phases") or {}).get("late", {}).get("mean_n_gold_files_unique"),
                    ),
                ]
            ),
            "",
            "Round-robin exhausts smaller repos first, so the late window is pandas-heavy. "
            "That is a task-order confound: do not read a late first-turn protocol dip as "
            "protocol learning without the unique-id mix table.",
            "",
            _phase_repo_table(strat),
            "",
            "## 5. Matched comparison (event-level)",
            "",
            "Sampling temperature/top_p/top_k/validate/B_obs/max_turns are the intended match. "
            "`n`, seed, task pool, and LoRA lifecycle are **not** matched. "
            "Do not treat late-train vs E018 as an RL causal contrast.",
            "",
            _matched_table(matched),
            "",
            "E018 cells at B_obs=4096:",
            "",
        ]
    )
    for key in ("B1@4096", "M_scaled@4096"):
        cell = cells.get(key) or {}
        lines.append(
            f"- `{key}`: n={cell.get('n_episodes')} event_invalid={_fmt(cell.get('event_invalid_rate'))} "
            f"first_turn_protocol={_fmt(cell.get('first_turn_protocol_rate'))} "
            f"parse_ok={_fmt(cell.get('parse_ok_rate'))}"
        )
    lines.extend(
        [
            "",
            "Taxonomy contrast: E017 invalid mass stays `multiple_actions` + "
            "`framing_unbalanced_tags` across bins (tool-semantic share ~0.20–0.25, not the "
            "decline). E018 B1/M_scaled@4096 invalid events have **zero** `multiple_actions` "
            "and **zero** `framing_unbalanced_tags`; remaining invalid is other_protocol + "
            "tool_semantic_misuse. First-turn protocol is ~0.66 train vs 0.008 eval.",
            "",
            "## 6. Hypothesis verdicts",
            "",
        ]
    )
    for item in verdicts.get("items") or []:
        lines.append(f"### {item.get('id')} — {item.get('title')}")
        lines.append("")
        lines.append(f"- verdict: **{item.get('verdict')}**")
        if item.get("greedy_validate_override"):
            lines.append(f"- greedy_validate_override: **{item.get('greedy_validate_override')}**")
        for ev in item.get("evidence") or []:
            lines.append(f"- {ev}")
        lines.append("")
    lines.extend(
        [
            f"Primary gap contributor: **{verdicts.get('primary_gap_contributor')}** "
            f"(rank {verdicts.get('ranking')}).",
            "",
            "## Not verified",
            "",
            "- Token-level training trajectories (research JSONL only).",
            "- Causal effect of any future parser/prompt/reward change.",
            "- A same-task train-vs-eval replay of the frozen policy (not run).",
            "",
            "## Risks",
            "",
            "- `global_step` is reconstructed from JSONL write order (32 rows/step). "
            f"Cross-check vs `step_bcrl.jsonl` pass={check.get('pass')}.",
            "- G=4 copies inflate episode counts; mix tables use unique instance_id.",
            "- Gold-file join is evaluator-only (`m1c_oracle.jsonl` / sidecar parquet) and is omitted when absent.",
            "",
            "## Next minimal step",
            "",
            "Stop. Do not start intervention. If a later milestone continues, keep parser, "
            "prompt, and reward frozen until a matched train/eval execution exists or H3 is "
            "accepted as the main source of the invalid-action table gap.",
            "",
        ]
    )
    return "\n".join(lines)


def _taxonomy_bin_table(binned: Sequence[Mapping[str, Any]]) -> str:
    if not binned:
        return "_no bins_"
    header = (
        "| bin | multiple_actions | framing_unbalanced_tags | "
        "tool_semantic_misuse | other_protocol |"
    )
    lines = [header, "| --- | ---: | ---: | ---: | ---: |"]
    for item in binned:
        shares = item.get("taxonomy_event_shares") or {}
        lines.append(
            f"| {item.get('bin_label') or item.get('bin_start')} | "
            f"{_fmt(shares.get('multiple_actions'))} | "
            f"{_fmt(shares.get('framing_unbalanced_tags'))} | "
            f"{_fmt(shares.get('tool_semantic_misuse'))} | "
            f"{_fmt(shares.get('other_protocol'))} |"
        )
    return "\n".join(lines)


def _phase_repo_table(strat: Mapping[str, Any]) -> str:
    phases = strat.get("phases") or {}
    repos: set[str] = set()
    for cell in phases.values():
        repos.update((cell.get("repo_unique_counts") or {}).keys())
    if not repos:
        return "_no repo counts_"
    lines = ["| repo | early | mid | late |", "| --- | ---: | ---: | ---: |"]
    for repo in sorted(repos):
        vals = []
        for name in ("early", "mid", "late"):
            counts = (phases.get(name) or {}).get("repo_unique_counts") or {}
            vals.append(str(int(counts.get(repo) or 0)))
        lines.append(f"| {repo} | {vals[0]} | {vals[1]} | {vals[2]} |")
    return "\n".join(lines)


def _matched_table(matched: Mapping[str, Any]) -> str:
    rows = [
        matched.get("e017_late16"),
        matched.get("e017_last_bin"),
        matched.get("e018_B1_4096"),
        matched.get("e018_M_scaled_4096"),
    ]
    lines = [
        "| cell | n | event invalid | first-turn protocol | parse_ok | loc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not row:
            continue
        lines.append(
            f"| {row.get('label')} | {row.get('n_episodes')} | "
            f"{_fmt(row.get('event_invalid_rate'))} | "
            f"{_fmt(row.get('first_turn_protocol_rate'))} | "
            f"{_fmt(row.get('parse_ok_rate'))} | "
            f"{_fmt(row.get('mean_localization_score'))} |"
        )
    return "\n".join(lines)


def _md_kv_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| item | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def drop_unserializable_sets(row: Mapping[str, Any]) -> dict[str, Any]:
    """Finalized step rows are already JSON-safe; keep as a guard."""
    out = {}
    for key, value in row.items():
        if isinstance(value, set):
            continue
        if isinstance(value, defaultdict):
            out[key] = dict(value)
        else:
            out[key] = value
    return out
