"""Research episode records. Not a substitute for veRL token trajectories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping, Sequence

from budget_coder_rl.budget.state import BUDGET_ACCOUNTING_VERSION

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
    accounting = extra_fields.get("budget_accounting_version")
    repo_obs = extra_fields.get("repo_observation_tokens")
    if repo_obs is None:
        repo_obs = extra_fields.get("tool_observation_token_count")
    total_env = extra_fields.get("total_env_tokens")
    if total_env is None:
        total_env = extra_fields.get("observation_token_count")
    obs_used = extra_fields.get("obs_tokens_used")
    if obs_used is None:
        if accounting == BUDGET_ACCOUNTING_VERSION and repo_obs is not None:
            obs_used = repo_obs
        elif total_env is not None:
            obs_used = total_env
        else:
            obs_used = sum(
                len(item.get("token_ids") or [])
                for item in segments
                if item.get("kind") == "observation"
            )
    metadata = extra_fields.get("budget_metadata_tokens")
    if metadata is None and repo_obs is not None and total_env is not None:
        metadata = int(total_env) - int(repo_obs)
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
            "sampling_seed": extra_fields.get("sampling_seed"),
            "budget_accounting_version": accounting,
        },
        "termination": extra_fields.get("termination"),
        "final_submission": extra_fields.get("final_submission"),
        "budget": {
            "budget_accounting_version": accounting,
            "obs_tokens_used": obs_used,
            "obs_tokens_limit": extra_fields.get("obs_tokens_limit"),
            "obs_tokens_remaining": extra_fields.get("obs_tokens_remaining"),
            "repo_observation_tokens": repo_obs,
            "budget_metadata_tokens": metadata,
            "total_env_tokens": total_env,
            "budget_exhausted": extra_fields.get("budget_exhausted"),
            "budget_visible": extra_fields.get("budget_visible"),
        },
        "tokens": {
            "prompt_token_count": extra_fields.get("prompt_token_count"),
            "policy_token_count": extra_fields.get("policy_token_count"),
            "observation_token_count": extra_fields.get("observation_token_count", total_env),
            "tool_observation_token_count": extra_fields.get(
                "tool_observation_token_count", repo_obs
            ),
            "repo_observation_tokens": repo_obs,
            "budget_metadata_tokens": metadata,
            "total_env_tokens": total_env,
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
    read_keys: list[tuple[str, int | None, int | None]] = []
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
            read_keys.append(
                (
                    str(path or ""),
                    _optional_int(args.get("start_line")),
                    _optional_int(args.get("end_line")),
                )
            )
        elif name == "tree":
            n_tree += 1
            path = args.get("path")
            if path:
                tree_paths.add(str(path))
    query_counts = Counter(search_queries)
    n_repeated_query = int(sum(count - 1 for count in query_counts.values() if count > 1))
    read_counts = Counter(read_keys)
    n_repeated_read = int(sum(count - 1 for count in read_counts.values() if count > 1))
    return {
        "n_search": n_search,
        "n_read": n_read,
        "n_tree": n_tree,
        "n_empty_search_hits": n_empty_search,
        "n_repeated_search_queries": n_repeated_query,
        "n_repeated_reads": n_repeated_read,
        "unique_search_queries": len(set(search_queries)),
        "unique_search_paths": len(search_paths),
        "unique_read_paths": len(read_paths),
        "unique_tree_paths": len(tree_paths),
        "unique_inspected_files": len(read_paths),
        "read_paths": sorted(read_paths),
        "search_paths": sorted(search_paths),
    }


def summarize_episodes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_finish = 0
    n_exhausted = 0
    n_max_turns = 0
    n_response_length = 0
    n_parse_ok = 0
    n_symbol_scored = 0
    n_symbol_unavailable = 0
    n_empty_submission = 0
    n_invalid_tool = 0
    file_p: list[float] = []
    file_r: list[float] = []
    file_f1: list[float] = []
    symbol_p: list[float] = []
    symbol_r: list[float] = []
    symbol_f1: list[float] = []
    scores: list[float] = []
    repo_obs: list[float] = []
    metadata_tokens: list[float] = []
    total_env: list[float] = []
    policy_tokens: list[float] = []
    prompt_tokens: list[float] = []
    utilizations: list[float] = []
    n_search: list[float] = []
    n_read: list[float] = []
    n_tree: list[float] = []
    n_empty_search: list[float] = []
    n_repeated_search: list[float] = []
    n_repeated_read: list[float] = []
    unique_files: list[float] = []
    n_tool_ok: list[float] = []
    n_events: list[float] = []
    for row in rows:
        termination = row.get("termination")
        if termination == "finish":
            n_finish += 1
        elif termination == "max_turns":
            n_max_turns += 1
        elif termination == "response_length":
            n_response_length += 1
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
            if loc.get("symbol_precision") is not None:
                symbol_p.append(float(loc["symbol_precision"]))
            if loc.get("symbol_recall") is not None:
                symbol_r.append(float(loc["symbol_recall"]))
        elif status == "unavailable":
            n_symbol_unavailable += 1
        if loc.get("file_f1") is not None and loc.get("parse_ok"):
            file_f1.append(float(loc["file_f1"]))
        if loc.get("file_precision") is not None and loc.get("parse_ok"):
            file_p.append(float(loc["file_precision"]))
        if loc.get("file_recall") is not None and loc.get("parse_ok"):
            file_r.append(float(loc["file_recall"]))
        if loc.get("localization_score") is not None:
            scores.append(float(loc["localization_score"]))
        if loc.get("submission_missing") or _empty_submission(row):
            n_empty_submission += 1
        counts = row.get("counts") if isinstance(row.get("counts"), MappingABC) else {}
        if int(counts.get("n_protocol_errors") or 0) or int(counts.get("n_tool_errors") or 0):
            n_invalid_tool += 1
        tokens = row.get("tokens") if isinstance(row.get("tokens"), MappingABC) else {}
        repo = budget.get("repo_observation_tokens")
        if repo is None:
            repo = tokens.get("repo_observation_tokens")
        if repo is None:
            repo = tokens.get("tool_observation_token_count")
        if repo is None:
            repo = budget.get("obs_tokens_used")
        if repo is not None:
            repo_obs.append(float(repo))
        meta = budget.get("budget_metadata_tokens")
        if meta is None:
            meta = tokens.get("budget_metadata_tokens")
        if meta is not None:
            metadata_tokens.append(float(meta))
        env = budget.get("total_env_tokens")
        if env is None:
            env = tokens.get("total_env_tokens")
        if env is None:
            env = tokens.get("observation_token_count")
        if env is not None:
            total_env.append(float(env))
        limit = budget.get("obs_tokens_limit")
        if repo is not None and limit:
            utilizations.append(float(repo) / float(limit))
        if tokens.get("policy_token_count") is not None:
            policy_tokens.append(float(tokens["policy_token_count"]))
        if tokens.get("prompt_token_count") is not None:
            prompt_tokens.append(float(tokens["prompt_token_count"]))
        behavior = row.get("behavior") if isinstance(row.get("behavior"), MappingABC) else {}
        if behavior.get("n_search") is not None:
            n_search.append(float(behavior["n_search"]))
        if behavior.get("n_read") is not None:
            n_read.append(float(behavior["n_read"]))
        if behavior.get("n_tree") is not None:
            n_tree.append(float(behavior["n_tree"]))
        if behavior.get("n_empty_search_hits") is not None:
            n_empty_search.append(float(behavior["n_empty_search_hits"]))
        if behavior.get("n_repeated_search_queries") is not None:
            n_repeated_search.append(float(behavior["n_repeated_search_queries"]))
        if behavior.get("n_repeated_reads") is not None:
            n_repeated_read.append(float(behavior["n_repeated_reads"]))
        if behavior.get("unique_inspected_files") is not None:
            unique_files.append(float(behavior["unique_inspected_files"]))
        elif behavior.get("unique_read_paths") is not None:
            unique_files.append(float(behavior["unique_read_paths"]))
        if counts.get("n_tool_ok") is not None:
            n_tool_ok.append(float(counts["n_tool_ok"]))
        if counts.get("n_events") is not None:
            n_events.append(float(counts["n_events"]))
    return {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "n_episodes": n,
        "n_finish": n_finish,
        "n_max_turns": n_max_turns,
        "n_response_length": n_response_length,
        "n_budget_exhausted": n_exhausted,
        "n_empty_submission": n_empty_submission,
        "n_invalid_tool": n_invalid_tool,
        "parse_ok_rate": (n_parse_ok / n) if n else None,
        "budget_exhaustion_rate": (n_exhausted / n) if n else None,
        "invalid_tool_rate": (n_invalid_tool / n) if n else None,
        "empty_submission_rate": (n_empty_submission / n) if n else None,
        "finish_rate": (n_finish / n) if n else None,
        "max_turn_rate": (n_max_turns / n) if n else None,
        "n_symbol_scored": n_symbol_scored,
        "n_symbol_unavailable": n_symbol_unavailable,
        "symbol_evaluable_rate": (n_symbol_scored / n) if n else None,
        "mean_file_precision_parse_ok": _mean(file_p),
        "median_file_precision_parse_ok": _median(file_p),
        "mean_file_recall_parse_ok": _mean(file_r),
        "median_file_recall_parse_ok": _median(file_r),
        "mean_file_f1_parse_ok": _mean(file_f1),
        "median_file_f1_parse_ok": _median(file_f1),
        "mean_symbol_precision_scored": _mean(symbol_p),
        "median_symbol_precision_scored": _median(symbol_p),
        "mean_symbol_recall_scored": _mean(symbol_r),
        "median_symbol_recall_scored": _median(symbol_r),
        "mean_symbol_f1_scored": _mean(symbol_f1),
        "median_symbol_f1_scored": _median(symbol_f1),
        "mean_localization_score": _mean(scores),
        "median_localization_score": _median(scores),
        "mean_obs_tokens_used": _mean(repo_obs),
        "median_obs_tokens_used": _median(repo_obs),
        "mean_repo_observation_tokens": _mean(repo_obs),
        "median_repo_observation_tokens": _median(repo_obs),
        "mean_budget_metadata_tokens": _mean(metadata_tokens),
        "median_budget_metadata_tokens": _median(metadata_tokens),
        "mean_total_env_tokens": _mean(total_env),
        "median_total_env_tokens": _median(total_env),
        "mean_policy_token_count": _mean(policy_tokens),
        "median_policy_token_count": _median(policy_tokens),
        "mean_prompt_token_count": _mean(prompt_tokens),
        "mean_budget_utilization": _mean(utilizations),
        "median_budget_utilization": _median(utilizations),
        "mean_n_search": _mean(n_search),
        "mean_n_read": _mean(n_read),
        "mean_n_tree": _mean(n_tree),
        "mean_n_empty_search_hits": _mean(n_empty_search),
        "mean_n_repeated_search_queries": _mean(n_repeated_search),
        "mean_n_repeated_reads": _mean(n_repeated_read),
        "mean_unique_inspected_files": _mean(unique_files),
        "mean_n_tool_ok": _mean(n_tool_ok),
        "mean_n_events": _mean(n_events),
        "histogram_localization_score": _histogram(scores),
        "histogram_repo_observation_tokens": _histogram(repo_obs),
        "histogram_budget_utilization": _histogram(utilizations),
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
    policy = extra_fields.get("policy_token_count")
    env = extra_fields.get("total_env_tokens")
    if env is None:
        env = extra_fields.get("observation_token_count")
    if policy is not None and env is not None:
        return int(policy) + int(env)
    if not segments:
        return None
    return sum(len(item.get("token_ids") or []) for item in segments)


def _empty_submission(row: Mapping[str, Any]) -> bool:
    submission = row.get("final_submission")
    if not isinstance(submission, MappingABC):
        return True
    locations = submission.get("locations")
    return not isinstance(locations, list) or len(locations) == 0


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _histogram(values: Sequence[float], *, bins: int = 8) -> dict[str, Any] | None:
    if not values:
        return None
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return {"min": lo, "max": hi, "bins": [{"lo": lo, "hi": hi, "n": len(values)}]}
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        index = int((value - lo) / width)
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    return {
        "min": lo,
        "max": hi,
        "bins": [
            {
                "lo": lo + i * width,
                "hi": lo + (i + 1) * width,
                "n": counts[i],
            }
            for i in range(bins)
        ],
    }
