"""Research episode records. Not a substitute for veRL token trajectories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping, Sequence

EPISODE_SCHEMA_VERSION = "bcrl-episode-v1"
TRACE_ROLE = "research_debug_not_training_tokens"


def build_episode_record(
    extra_fields: Mapping[str, Any],
    *,
    localization: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a machine-readable episode from AgentLoop extra_fields.

    Must not copy oracle gold lists. Localization metrics are joined by the
    caller after rollout.
    """
    events = list(extra_fields.get("events") or [])
    segments = list(extra_fields.get("segments") or [])
    obs_used = extra_fields.get("obs_tokens_used")
    if obs_used is None:
        obs_used = sum(
            len(item.get("token_ids") or [])
            for item in segments
            if item.get("kind") == "observation"
        )
    record: dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "trace_role": extra_fields.get("trace_role") or TRACE_ROLE,
        "identity": {
            "instance_id": extra_fields.get("instance_id"),
            "repo": extra_fields.get("repo"),
            "base_commit": extra_fields.get("base_commit"),
            "split": extra_fields.get("split"),
        },
        "condition": {
            "budget_visible": extra_fields.get("budget_visible"),
            "obs_tokens_limit": extra_fields.get("obs_tokens_limit"),
            "max_turns": extra_fields.get("max_turns"),
            "max_new_tokens_per_turn": extra_fields.get("max_new_tokens_per_turn"),
            "sampling": dict(sampling or extra_fields.get("sampling_params") or {}),
        },
        "termination": extra_fields.get("termination"),
        "final_submission": extra_fields.get("final_submission"),
        "budget": {
            "obs_tokens_used": obs_used,
            "obs_tokens_limit": extra_fields.get("obs_tokens_limit"),
            "obs_tokens_remaining": extra_fields.get("obs_tokens_remaining"),
            "budget_exhausted": extra_fields.get("budget_exhausted"),
            "budget_visible": extra_fields.get("budget_visible"),
        },
        "tokens": {
            "prompt_token_count": extra_fields.get("prompt_token_count"),
            "policy_token_count": extra_fields.get("policy_token_count"),
            "observation_token_count": extra_fields.get("observation_token_count", obs_used),
            "tool_observation_token_count": extra_fields.get("tool_observation_token_count"),
            "response_token_count": _response_token_count(segments, extra_fields),
        },
        "counts": action_counts(events, termination=extra_fields.get("termination")),
        "behavior": behavior_stats(events),
        "events": compact_events(events),
        "segments": [
            {
                "kind": item.get("kind"),
                "n_tokens": len(item.get("token_ids") or []),
            }
            for item in segments
        ],
        "model_name_or_path": extra_fields.get("model_name_or_path"),
    }
    if localization is not None:
        record["localization"] = dict(localization)
    if provenance is not None:
        record["provenance"] = dict(provenance)
    return record


def compact_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for event in events:
        row = {str(key): event[key] for key in event if key != "observation"}
        compact.append(row)
    return compact


def action_counts(
    events: Sequence[Mapping[str, Any]],
    *,
    termination: str | None = None,
) -> dict[str, Any]:
    n_tool_ok = 0
    n_protocol = 0
    n_tool_error = 0
    n_finish = 0
    parse_codes: dict[str, int] = {}
    tool_names: dict[str, int] = {}
    for event in events:
        kind = event.get("error_kind")
        name = event.get("action_name") or event.get("action_type")
        if kind == "protocol":
            n_protocol += 1
        elif kind == "tool":
            n_tool_error += 1
        elif name == "finish" or event.get("action_type") == "finish":
            n_finish += 1
        elif name in {"tree", "search", "read"}:
            n_tool_ok += 1
            tool_names[str(name)] = tool_names.get(str(name), 0) + 1
        code = event.get("parse_error_code") or (
            event.get("error_code") if kind == "protocol" else None
        )
        if code:
            parse_codes[str(code)] = parse_codes.get(str(code), 0) + 1
    return {
        "n_events": len(events),
        "n_tool_ok": n_tool_ok,
        "n_protocol_errors": n_protocol,
        "n_tool_errors": n_tool_error,
        "n_finish": n_finish,
        "termination_finish": termination == "finish",
        "parse_error_codes": parse_codes,
        "tool_ok_by_name": tool_names,
    }


def behavior_stats(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    search_queries: list[str] = []
    search_paths: set[str] = set()
    read_paths: set[str] = set()
    tree_paths: set[str] = set()
    n_empty_search = 0
    n_search = 0
    n_read = 0
    n_tree = 0
    for event in events:
        name = event.get("action_name")
        args = event.get("action_arguments") if isinstance(event.get("action_arguments"), MappingABC) else {}
        if name == "search":
            n_search += 1
            query = args.get("query")
            if query is not None:
                search_queries.append(str(query))
            path = args.get("path")
            if path:
                search_paths.add(str(path))
            headers = _headers_from_event(event)
            if headers.get("match_count") == "0":
                n_empty_search += 1
        elif name == "read":
            n_read += 1
            path = args.get("path")
            if path:
                read_paths.add(str(path))
        elif name == "tree":
            n_tree += 1
            path = args.get("path")
            if path:
                tree_paths.add(str(path))
    query_counts = Counter(search_queries)
    n_repeated_query = int(sum(count - 1 for count in query_counts.values() if count > 1))
    return {
        "n_search": n_search,
        "n_read": n_read,
        "n_tree": n_tree,
        "n_empty_search_hits": n_empty_search,
        "n_repeated_search_queries": n_repeated_query,
        "unique_search_queries": len(set(search_queries)),
        "unique_search_paths": len(search_paths),
        "unique_read_paths": len(read_paths),
        "unique_tree_paths": len(tree_paths),
        "read_paths": sorted(read_paths),
        "search_paths": sorted(search_paths),
    }


def summarize_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_finish = 0
    n_exhausted = 0
    n_parse_ok = 0
    n_symbol_scored = 0
    n_symbol_unavailable = 0
    file_f1: list[float] = []
    symbol_f1: list[float] = []
    scores: list[float] = []
    obs_used: list[float] = []
    policy_tokens: list[float] = []
    for row in rows:
        termination = row.get("termination")
        if termination == "finish":
            n_finish += 1
        budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
        if budget.get("budget_exhausted") or termination == "budget_exhausted":
            n_exhausted += 1
        loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
        if loc.get("parse_ok"):
            n_parse_ok += 1
        status = loc.get("symbol_status")
        if status == "scored":
            n_symbol_scored += 1
            if loc.get("symbol_f1") is not None:
                symbol_f1.append(float(loc["symbol_f1"]))
        elif status == "unavailable":
            n_symbol_unavailable += 1
        if loc.get("file_f1") is not None and loc.get("parse_ok"):
            file_f1.append(float(loc["file_f1"]))
        if loc.get("localization_score") is not None:
            scores.append(float(loc["localization_score"]))
        tokens = row.get("tokens") if isinstance(row.get("tokens"), MappingABC) else {}
        if budget.get("obs_tokens_used") is not None:
            obs_used.append(float(budget["obs_tokens_used"]))
        elif tokens.get("observation_token_count") is not None:
            obs_used.append(float(tokens["observation_token_count"]))
        if tokens.get("policy_token_count") is not None:
            policy_tokens.append(float(tokens["policy_token_count"]))
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "n_episodes": n,
        "n_finish": n_finish,
        "n_budget_exhausted": n_exhausted,
        "parse_ok_rate": (n_parse_ok / n) if n else None,
        "budget_exhaustion_rate": (n_exhausted / n) if n else None,
        "n_symbol_scored": n_symbol_scored,
        "n_symbol_unavailable": n_symbol_unavailable,
        "mean_file_f1_parse_ok": _mean(file_f1),
        "mean_symbol_f1_scored": _mean(symbol_f1),
        "mean_localization_score": _mean(scores),
        "mean_obs_tokens_used": _mean(obs_used),
        "mean_policy_token_count": _mean(policy_tokens),
    }


def _headers_from_event(event: Mapping[str, Any]) -> dict[str, str]:
    text = event.get("observation")
    if not isinstance(text, str):
        text = event.get("observation_preview")
    if not isinstance(text, str):
        return {}
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() == "---":
            break
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(": ")
        if not sep or key in headers:
            continue
        headers[key] = value
    return headers


def _response_token_count(
    segments: Sequence[Mapping[str, Any]],
    extra_fields: Mapping[str, Any],
) -> int | None:
    if extra_fields.get("policy_token_count") is not None and extra_fields.get(
        "observation_token_count"
    ) is not None:
        return int(extra_fields["policy_token_count"]) + int(
            extra_fields["observation_token_count"]
        )
    if not segments:
        return None
    return sum(len(item.get("token_ids") or []) for item in segments)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
