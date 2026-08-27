"""M7A offline invalid-action forensics.

Diagnoses frozen E017/E018 research JSONL. Does not modify parse_action,
prompt, reward, AgentLoop, or any Stage-1 frozen experiment artifact.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from budget_coder_rl.eval.episode import action_counts
from budget_coder_rl.eval.m5a import _truthy
from budget_coder_rl.protocol.parser import (
    FINAL_CLOSE,
    FINAL_OPEN,
    ProtocolError,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    parse_action,
)

SCHEMA_VERSION = "bcrl-m7a-v1"
MILESTONE = "M7A"
EXPERIMENT_ID = "M7A"
EXAMPLE_CHAR_LIMIT = 700
EXAMPLES_PER_BUCKET = 3
E017_REPORTED_INVALID_ACTION_RATE = 0.952
E017_REPORTED_PARSE_OK_RATE = 0.178
E017_RATE_TOLERANCE = 0.005
PRIMARY_EVAL_BUDGET = 4096
E018_COMPARE_CONDITIONS = ("B1", "M_scaled")

RECOVERABLE = "recoverable_protocol_mismatch"
UNRECOVERABLE = "genuine_unrecoverable_error"
TOOL_FAILURE = "parsed_ok_tool_failure"
NOT_INVALID = "not_invalid"

TAXONOMY_BUCKETS = (
    "no_recognizable_action",
    "framing_unbalanced_tags",
    "framing_wrong_envelope",
    "surrounding_prose",
    "multiple_actions",
    "malformed_json",
    "unknown_tool",
    "wrong_schema",
    "bad_args",
    "tool_semantic_misuse",
    "runtime_infra",
    "other_protocol",
)

METRIC_SEMANTICS = {
    "schema_version": SCHEMA_VERSION,
    "invalid_action_rate": {
        "training_key": "bcrl/invalid_action_rate",
        "eval_key": "invalid_tool_rate",
        "grain": "episode",
        "denominator": (
            "Number of trajectories in the aggregation unit "
            "(training: 32 rollouts per optimizer step, then mean over steps; "
            "eval: scored episodes in the cell)."
        ),
        "numerator": (
            "Episodes with n_protocol_errors > 0 OR n_tool_errors > 0. "
            "A single bad turn marks the whole episode."
        ),
        "includes_tool_errors": True,
        "not_per_turn": True,
        "not_the_complement_of_parse_ok": True,
    },
    "parse_ok_rate": {
        "training_key": "bcrl/parse_ok_rate",
        "eval_key": "parse_ok_rate",
        "grain": "episode",
        "meaning": (
            "Final localization parse: termination == 'finish' and "
            "final_submission.locations is a JSON array. Independent of "
            "whether earlier turns had protocol or tool errors."
        ),
        "jsonl_string_trap": (
            "E017 research JSONL often stores localization.parse_ok as the "
            "string 'False'/'True' (veRL extra_fields). Training step "
            "metrics use m5a._truthy (correct). Naive bool('False') is True. "
            "M7A always uses _truthy."
        ),
    },
    "protocol_error_count": {
        "training_key": "bcrl/protocol_error_count",
        "grain": "event_count_per_step",
        "meaning": "Sum of protocol-error turns in the step, not a rate.",
    },
    "event_level_invalid_rate": {
        "grain": "assistant_turn",
        "denominator": "assistant turns / events",
        "numerator": "events with error_kind in {protocol, tool}",
        "note": "This is the M7A diagnostic rate; it is not logged in E017 W&B.",
    },
}


def truthy(value: Any) -> bool:
    """Same contract as training ``bcrl/parse_ok_rate``."""
    return _truthy(value)


def episode_parse_ok(row: Mapping[str, Any]) -> bool:
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    if "parse_ok" in loc:
        return truthy(loc.get("parse_ok"))
    return truthy(row.get("parse_ok"))


def localization_score(row: Mapping[str, Any]) -> float | None:
    loc = row.get("localization") if isinstance(row.get("localization"), MappingABC) else {}
    value = loc.get("localization_score")
    if value is None:
        value = row.get("localization_score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def episode_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("events")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, MappingABC)]


def counts_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    stored = row.get("counts") if isinstance(row.get("counts"), MappingABC) else {}
    events = episode_events(row)
    computed = action_counts(events, termination=row.get("termination"))
    if stored:
        merged = dict(computed)
        for key in (
            "n_events",
            "n_tool_ok",
            "n_protocol_errors",
            "n_tool_errors",
            "n_finish",
        ):
            if key in stored:
                try:
                    merged[key] = int(stored[key])
                except (TypeError, ValueError):
                    pass
        return merged
    return computed


def episode_is_invalid(row: Mapping[str, Any]) -> bool:
    """Training ``bcrl/invalid_action_rate`` numerator predicate."""
    counts = counts_from_row(row)
    return int(counts.get("n_protocol_errors") or 0) > 0 or int(
        counts.get("n_tool_errors") or 0
    ) > 0


def event_is_invalid(event: Mapping[str, Any]) -> bool:
    kind = event.get("error_kind")
    return kind in {"protocol", "tool"}


def naive_bool(value: Any) -> bool:
    """Documents the JSONL string trap. Not used for M7A rates."""
    return bool(value)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            if row.get("error") and "events" not in row:
                continue
            yield row


def mean_step_bcrl_rates(path: Path) -> dict[str, float | None]:
    invalid: list[float] = []
    parse_ok: list[float] = []
    protocol_counts: list[float] = []
    n_steps = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            metrics = row.get("metrics") if isinstance(row, dict) else None
            if not isinstance(metrics, MappingABC):
                continue
            n_steps += 1
            inv = metrics.get("bcrl/invalid_action_rate")
            pok = metrics.get("bcrl/parse_ok_rate")
            proto = metrics.get("bcrl/protocol_error_count")
            if inv is not None:
                invalid.append(float(inv))
            if pok is not None:
                parse_ok.append(float(pok))
            if proto is not None:
                protocol_counts.append(float(proto))
    return {
        "n_steps": n_steps,
        "mean_invalid_action_rate": _mean(invalid),
        "mean_parse_ok_rate": _mean(parse_ok),
        "mean_protocol_error_count": _mean(protocol_counts),
    }


def tag_counts(text: str) -> dict[str, int]:
    body = text if isinstance(text, str) else ""
    n_tool_open = body.count(TOOL_CALL_OPEN)
    n_tool_close = body.count(TOOL_CALL_CLOSE)
    n_final_open = body.count(FINAL_OPEN)
    n_final_close = body.count(FINAL_CLOSE)
    return {
        "n_tool_open": n_tool_open,
        "n_tool_close": n_tool_close,
        "n_final_open": n_final_open,
        "n_final_close": n_final_close,
        "n_open": n_tool_open + n_final_open,
        "n_close": n_tool_close + n_final_close,
    }


def classify_event(event: Mapping[str, Any]) -> str | None:
    """Taxonomy label for one turn. None if the turn is not invalid."""
    kind = event.get("error_kind")
    if kind == "tool":
        return "tool_semantic_misuse"
    if kind == "runtime" or str(event.get("termination") or "") == "operational_error":
        return "runtime_infra"
    if kind != "protocol":
        return None
    raw = event.get("raw_action")
    text = raw if isinstance(raw, str) else ""
    stripped = text.strip()
    code = str(event.get("parse_error_code") or event.get("error_code") or "")
    tags = tag_counts(stripped)
    if not stripped or code == "empty_action":
        return "no_recognizable_action"
    if tags["n_open"] > 1:
        return "multiple_actions"
    if tags["n_open"] == 0:
        obj = _load_json_object(stripped)
        if obj is not None and _looks_like_action_object(obj):
            return "framing_wrong_envelope"
        return "no_recognizable_action"
    if tags["n_tool_open"] == 1:
        if tags["n_tool_close"] != 1 or tags["n_final_close"] != 0:
            return "framing_unbalanced_tags"
        inner = _inner_complete(stripped, TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
        if inner is None:
            return "surrounding_prose"
        return _classify_inner("tool", inner, code)
    if tags["n_final_open"] == 1:
        if tags["n_final_close"] != 1 or tags["n_tool_close"] != 0:
            return "framing_unbalanced_tags"
        inner = _inner_complete(stripped, FINAL_OPEN, FINAL_CLOSE)
        if inner is None:
            return "surrounding_prose"
        return _classify_inner("final", inner, code)
    return "other_protocol"


def try_recover_action(raw: str) -> str | None:
    """At most one unique bounded unwrap. Returns text that ``parse_action`` accepts.

    Never mutates the production parser. Returns None when recovery is not
    unique or would require guessing schema/intent.
    """
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if _parse_ok(stripped):
        return None
    extracted = _unwrap_unique_complete_block(stripped)
    if extracted is not None and extracted.strip() != stripped and _parse_ok(extracted):
        return extracted
    closed = _unwrap_unique_unclosed_tag(stripped)
    if closed is not None and _parse_ok(closed):
        return closed
    wrapped = _unwrap_untagged_json(stripped)
    if wrapped is not None and _parse_ok(wrapped):
        return wrapped
    return None


def recoverability_label(event: Mapping[str, Any]) -> str:
    kind = event.get("error_kind")
    if kind == "tool":
        return TOOL_FAILURE
    if kind != "protocol":
        return NOT_INVALID
    recovered = try_recover_action(str(event.get("raw_action") or ""))
    if recovered is not None:
        return RECOVERABLE
    return UNRECOVERABLE


def analyze_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    events = episode_events(row)
    counts = counts_from_row(row)
    parse_ok = episode_parse_ok(row)
    invalid_ep = episode_is_invalid(row)
    event_rows: list[dict[str, Any]] = []
    first_invalid_turn: int | None = None
    n_invalid_events = 0
    n_recoverable = 0
    n_unrecoverable = 0
    n_tool_fail = 0
    for index, event in enumerate(events):
        if not event_is_invalid(event):
            continue
        n_invalid_events += 1
        turn = event.get("turn")
        try:
            turn_i = int(turn) if turn is not None else index + 1
        except (TypeError, ValueError):
            turn_i = index + 1
        if first_invalid_turn is None:
            first_invalid_turn = turn_i
        bucket = classify_event(event) or "other_protocol"
        rec = recoverability_label(event)
        if rec == RECOVERABLE:
            n_recoverable += 1
        elif rec == UNRECOVERABLE:
            n_unrecoverable += 1
        elif rec == TOOL_FAILURE:
            n_tool_fail += 1
        event_rows.append(
            {
                "turn": turn_i,
                "taxonomy": bucket,
                "recoverability": rec,
                "error_kind": event.get("error_kind"),
                "parse_error_code": event.get("parse_error_code") or event.get("error_code"),
                "raw_action": event.get("raw_action"),
            }
        )
    if n_invalid_events == 0:
        episode_rec = NOT_INVALID
    elif n_recoverable == n_invalid_events:
        episode_rec = "all_invalid_recoverable"
    elif n_recoverable > 0:
        episode_rec = "any_invalid_recoverable"
    else:
        episode_rec = "none_invalid_recoverable"
    loc_score = localization_score(row)
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    budget = row.get("budget") if isinstance(row.get("budget"), MappingABC) else {}
    obs_limit = condition.get("obs_tokens_limit")
    if obs_limit is None:
        obs_limit = budget.get("obs_tokens_limit")
    return {
        "instance_id": identity.get("instance_id") or row.get("instance_id"),
        "termination": row.get("termination"),
        "parse_ok": parse_ok,
        "invalid": invalid_ep,
        "localization_score": loc_score,
        "n_events": int(counts.get("n_events") or len(events)),
        "n_protocol_errors": int(counts.get("n_protocol_errors") or 0),
        "n_tool_errors": int(counts.get("n_tool_errors") or 0),
        "n_invalid_events": n_invalid_events,
        "n_recoverable": n_recoverable,
        "n_unrecoverable": n_unrecoverable,
        "n_tool_fail": n_tool_fail,
        "first_invalid_turn": first_invalid_turn,
        "episode_recoverability": episode_rec,
        "condition_id": condition.get("condition_id") or row.get("condition_id"),
        "obs_tokens_limit": obs_limit,
        "invalid_events": event_rows,
    }


def analyze_corpus(
    rows: Iterator[Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    source: str,
    keep_examples: bool = True,
) -> dict[str, Any]:
    n_episodes = 0
    n_invalid_episodes = 0
    n_parse_ok = 0
    n_events = 0
    n_invalid_events = 0
    n_protocol_events = 0
    n_tool_events = 0
    n_recoverable = 0
    n_unrecoverable = 0
    n_tool_fail = 0
    n_all_rec = 0
    n_any_rec = 0
    n_none_rec = 0
    n_runtime = 0
    taxonomy_event: Counter[str] = Counter()
    taxonomy_first: Counter[str] = Counter()
    rec_event: Counter[str] = Counter()
    first_turn: Counter[int] = Counter()
    invalid_counts: Counter[int] = Counter()
    parse_codes: Counter[str] = Counter()
    termination: Counter[str] = Counter()
    cross_parse_invalid: Counter[str] = Counter()
    cross_term_invalid: Counter[str] = Counter()
    score_by_invalid: dict[str, list[float]] = {"invalid": [], "valid": []}
    score_by_parse: dict[str, list[float]] = {"parse_ok": [], "parse_fail": []}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rec_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        n_episodes += 1
        if str(row.get("termination") or "") == "operational_error":
            n_runtime += 1
        analyzed = analyze_episode(row)
        n_events += int(analyzed["n_events"])
        n_invalid_events += int(analyzed["n_invalid_events"])
        n_protocol_events += int(analyzed["n_protocol_errors"])
        n_tool_events += int(analyzed["n_tool_errors"])
        n_recoverable += int(analyzed["n_recoverable"])
        n_unrecoverable += int(analyzed["n_unrecoverable"])
        n_tool_fail += int(analyzed["n_tool_fail"])
        invalid = bool(analyzed["invalid"])
        parse_ok = bool(analyzed["parse_ok"])
        if invalid:
            n_invalid_episodes += 1
        if parse_ok:
            n_parse_ok += 1
        term = str(analyzed.get("termination") or "unknown")
        termination[term] += 1
        cross_parse_invalid[f"parse_ok={parse_ok}|invalid={invalid}"] += 1
        cross_term_invalid[f"termination={term}|invalid={invalid}"] += 1
        score = analyzed.get("localization_score")
        if score is not None:
            score_by_invalid["invalid" if invalid else "valid"].append(float(score))
            score_by_parse["parse_ok" if parse_ok else "parse_fail"].append(float(score))
        rec_ep = analyzed["episode_recoverability"]
        if rec_ep == "all_invalid_recoverable":
            n_all_rec += 1
            n_any_rec += 1
        elif rec_ep == "any_invalid_recoverable":
            n_any_rec += 1
        elif rec_ep == "none_invalid_recoverable":
            n_none_rec += 1
        invalid_counts[int(analyzed["n_invalid_events"])] += 1
        if analyzed["first_invalid_turn"] is not None:
            first_turn[int(analyzed["first_invalid_turn"])] += 1
            first_bucket = None
            for item in analyzed["invalid_events"]:
                first_bucket = item["taxonomy"]
                break
            if first_bucket:
                taxonomy_first[str(first_bucket)] += 1
        for item in analyzed["invalid_events"]:
            bucket = str(item["taxonomy"])
            taxonomy_event[bucket] += 1
            rec_event[str(item["recoverability"])] += 1
            code = item.get("parse_error_code")
            if code:
                parse_codes[str(code)] += 1
            if keep_examples:
                _maybe_keep_example(
                    examples[bucket],
                    analyzed,
                    item,
                )
                _maybe_keep_example(
                    rec_examples[str(item["recoverability"])],
                    analyzed,
                    item,
                )

    n_inv_ep = n_invalid_episodes or None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "n_episodes": n_episodes,
        "n_events": n_events,
        "n_runtime_infra_episodes": n_runtime,
        "episode_invalid_rate": (n_invalid_episodes / n_episodes) if n_episodes else None,
        "episode_parse_ok_rate": (n_parse_ok / n_episodes) if n_episodes else None,
        "event_invalid_rate": (n_invalid_events / n_events) if n_events else None,
        "n_invalid_episodes": n_invalid_episodes,
        "n_parse_ok_episodes": n_parse_ok,
        "n_invalid_events": n_invalid_events,
        "n_protocol_events": n_protocol_events,
        "n_tool_error_events": n_tool_events,
        "n_valid_events": n_events - n_invalid_events,
        "recoverability_event": {
            RECOVERABLE: n_recoverable,
            UNRECOVERABLE: n_unrecoverable,
            TOOL_FAILURE: n_tool_fail,
            "share_of_invalid_events": _shares(
                {
                    RECOVERABLE: n_recoverable,
                    UNRECOVERABLE: n_unrecoverable,
                    TOOL_FAILURE: n_tool_fail,
                },
                n_invalid_events,
            ),
            "share_of_protocol_events": _shares(
                {RECOVERABLE: n_recoverable, UNRECOVERABLE: n_unrecoverable},
                n_protocol_events,
            ),
        },
        "recoverability_episode": {
            "n_invalid_episodes": n_invalid_episodes,
            "all_invalid_recoverable": n_all_rec,
            "any_invalid_recoverable": n_any_rec,
            "none_invalid_recoverable": n_none_rec,
            "share_of_invalid_episodes": _shares(
                {
                    "all_invalid_recoverable": n_all_rec,
                    "any_invalid_recoverable": n_any_rec,
                    "none_invalid_recoverable": n_none_rec,
                },
                n_inv_ep or 0,
            ),
        },
        "taxonomy_event_counts": dict(taxonomy_event),
        "taxonomy_event_share_of_invalid": _shares(taxonomy_event, n_invalid_events),
        "taxonomy_first_invalid_counts": dict(taxonomy_first),
        "parse_error_codes": dict(parse_codes),
        "first_invalid_turn": _counter_to_sorted_dict(first_turn),
        "invalid_count_per_trajectory": _counter_to_sorted_dict(invalid_counts),
        "termination_counts": dict(termination),
        "cross_parse_ok_invalid": dict(cross_parse_invalid),
        "cross_termination_invalid": dict(cross_term_invalid),
        "mean_localization_score_invalid": _mean(score_by_invalid["invalid"]),
        "mean_localization_score_valid_protocol": _mean(score_by_invalid["valid"]),
        "mean_localization_score_parse_ok": _mean(score_by_parse["parse_ok"]),
        "mean_localization_score_parse_fail": _mean(score_by_parse["parse_fail"]),
        "examples": {key: list(val) for key, val in examples.items()},
        "recoverability_examples": {key: list(val) for key, val in rec_examples.items()},
        "recoverability_rules": recoverability_rules_text(),
    }
    payload["gate"] = gate_recommendation(payload)
    return payload


def recoverability_rules_text() -> dict[str, Any]:
    return {
        RECOVERABLE: (
            "Protocol-error turn whose raw_action has exactly one unique "
            "bounded unwrap, after which frozen parse_action succeeds. "
            "Allowed unwraps: (1) extract the single complete <tool_call> or "
            "<final> block from surrounding prose; (2) wrap a single untagged "
            "JSON object as tool_call iff keys are name+arguments and not "
            "locations, or as final iff keys include locations and not name; "
            "(3) close exactly one unclosed open tag when the inner text is "
            "strict JSON. At most one layer. No key renaming, no filling "
            "missing fields, no choosing among multiple blocks."
        ),
        UNRECOVERABLE: (
            "Protocol-error turn that does not have a unique unwrap accepted "
            "by frozen parse_action."
        ),
        TOOL_FAILURE: (
            "Turn already accepted by parse_action; tool execution failed "
            "(path_not_found, invalid_range, ...). Not a parser mismatch."
        ),
    }


def gate_recommendation(corpus: Mapping[str, Any]) -> dict[str, Any]:
    n_invalid = int(corpus.get("n_invalid_events") or 0)
    n_protocol = int(corpus.get("n_protocol_events") or 0)
    rec = corpus.get("recoverability_event") or {}
    n_recoverable = int(rec.get(RECOVERABLE) or 0)
    taxonomy = corpus.get("taxonomy_event_counts") or {}
    tool_n = int(taxonomy.get("tool_semantic_misuse") or 0)
    no_act = int(taxonomy.get("no_recognizable_action") or 0)
    multi = int(taxonomy.get("multiple_actions") or 0)
    unknown = int(taxonomy.get("unknown_tool") or 0)
    prose = int(taxonomy.get("surrounding_prose") or 0)
    framing = int(taxonomy.get("framing_unbalanced_tags") or 0) + int(
        taxonomy.get("framing_wrong_envelope") or 0
    )
    protocol_share = (n_protocol / n_invalid) if n_invalid else 0.0
    recoverable_of_protocol = (n_recoverable / n_protocol) if n_protocol else 0.0
    tool_share = (tool_n / n_invalid) if n_invalid else 0.0
    promptish = no_act + multi + unknown
    promptish_share = (promptish / n_invalid) if n_invalid else 0.0
    framing_share = ((prose + framing) / n_invalid) if n_invalid else 0.0

    if n_invalid == 0:
        decision = "no_invalid_events"
        reason = "No invalid events in corpus."
    elif protocol_share >= 0.5 and recoverable_of_protocol >= 0.45:
        decision = "parser_level_protocol_repair"
        reason = (
            "Protocol errors are the majority of invalid events and a large "
            "share uniquely unwrap to a legal action under bounded parsing."
        )
    elif tool_share >= 0.5:
        decision = "stop_parser_direction"
        reason = (
            "Most invalid events already parse; failures are tool semantics. "
            "Parser repair would not address the dominant mass."
        )
    elif promptish_share >= 0.45 and recoverable_of_protocol < 0.30:
        decision = "prompt_or_template_alignment_or_stop"
        reason = (
            "Dominant failures are no recognizable action, multiple actions, "
            "or unknown tools, and unique unwrap recovery is low."
        )
    elif framing_share >= 0.40 and recoverable_of_protocol >= 0.30:
        decision = "parser_level_protocol_repair"
        reason = (
            "Framing / surrounding-prose mismatches dominate and many are "
            "uniquely recoverable without guessing intent."
        )
    else:
        decision = "mixed_do_not_start_m7b_from_one_lever"
        reason = (
            "Invalid mass is split across recoverable protocol mismatch, "
            "genuine unrecoverable protocol errors, and tool misuse."
        )
    return {
        "decision": decision,
        "reason": reason,
        "protocol_share_of_invalid_events": protocol_share,
        "recoverable_share_of_protocol_events": recoverable_of_protocol,
        "tool_semantic_share_of_invalid_events": tool_share,
        "promptish_share_of_invalid_events": promptish_share,
        "framing_prose_share_of_invalid_events": framing_share,
        "start_m7b": False,
        "note": "M7A is diagnostic only. Do not change the production parser.",
    }


def e017_self_check(
    corpus: Mapping[str, Any],
    step_rates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ep_invalid = corpus.get("episode_invalid_rate")
    ep_parse = corpus.get("episode_parse_ok_rate")
    step_invalid = None if step_rates is None else step_rates.get("mean_invalid_action_rate")
    step_parse = None if step_rates is None else step_rates.get("mean_parse_ok_rate")
    checks = {
        "episodes_match_reported_invalid": _close(
            ep_invalid, E017_REPORTED_INVALID_ACTION_RATE
        ),
        "episodes_match_reported_parse_ok": _close(
            ep_parse, E017_REPORTED_PARSE_OK_RATE
        ),
        "step_bcrl_match_reported_invalid": _close(
            step_invalid, E017_REPORTED_INVALID_ACTION_RATE
        )
        if step_invalid is not None
        else None,
        "step_bcrl_match_reported_parse_ok": _close(
            step_parse, E017_REPORTED_PARSE_OK_RATE
        )
        if step_parse is not None
        else None,
        "episodes_match_step_bcrl_invalid": _close(ep_invalid, step_invalid),
        "episodes_match_step_bcrl_parse_ok": _close(ep_parse, step_parse),
        "episode_invalid_rate": ep_invalid,
        "episode_parse_ok_rate": ep_parse,
        "step_mean_invalid_action_rate": step_invalid,
        "step_mean_parse_ok_rate": step_parse,
        "reported_invalid_action_rate": E017_REPORTED_INVALID_ACTION_RATE,
        "reported_parse_ok_rate": E017_REPORTED_PARSE_OK_RATE,
        "tolerance": E017_RATE_TOLERANCE,
    }
    flags = [
        checks["episodes_match_reported_invalid"],
        checks["episodes_match_reported_parse_ok"],
        checks["episodes_match_step_bcrl_invalid"],
        checks["episodes_match_step_bcrl_parse_ok"],
    ]
    if step_invalid is not None:
        flags.append(checks["step_bcrl_match_reported_invalid"])
        flags.append(checks["step_bcrl_match_reported_parse_ok"])
    checks["pass"] = all(item is not False for item in flags)
    return checks


def filter_e018_cell(
    rows: Iterator[Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    condition_id: str,
    budget: int = PRIMARY_EVAL_BUDGET,
) -> Iterator[dict[str, Any]]:
    from budget_coder_rl.eval.e018 import condition_id_from_row
    from budget_coder_rl.eval.m6 import obs_limit_from_row

    for row in rows:
        if condition_id_from_row(row) != condition_id:
            continue
        limit = obs_limit_from_row(row)
        if limit != int(budget):
            continue
        yield dict(row)


def render_summary(payload: Mapping[str, Any]) -> str:
    e017 = payload.get("e017") or {}
    e018 = payload.get("e018") or {}
    semantics = payload.get("metric_semantics") or METRIC_SEMANTICS
    check = payload.get("e017_self_check") or {}
    gate = e017.get("gate") or {}
    lines = [
        "# M7A — Invalid-Action Forensics",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- status: **diagnostic only** (no parser / prompt / reward / RL change)",
        f"- primary corpus: E017 train episodes (`n={e017.get('n_episodes')}`)",
        f"- E017 self-check pass: **{check.get('pass')}**",
        f"- recommended M7B lever: `{gate.get('decision')}`",
        "- start M7B from this audit: **false**",
        "",
        "This note explains what E017 `invalid_action≈0.952` measures. It does "
        "not rewrite E017 or E018 scientific conclusions.",
        "",
        "## 0. What 95.2% is",
        "",
        f"- **0.952 is episode-level any-error**, not per-turn illegal actions. "
        f"{e017.get('n_invalid_episodes')} / {e017.get('n_episodes')} training "
        "rollouts have at least one protocol or tool error.",
        f"- **Per-turn invalid is {_fmt(e017.get('event_invalid_rate'))}** "
        f"({e017.get('n_invalid_events')} / {e017.get('n_events')} assistant turns).",
        f"- Of those invalid turns: protocol "
        f"{_fmt((e017.get('n_protocol_events') or 0) / (e017.get('n_invalid_events') or 1))} "
        f"({e017.get('n_protocol_events')}), already-parsed tool failures "
        f"{_fmt((e017.get('n_tool_error_events') or 0) / (e017.get('n_invalid_events') or 1))} "
        f"({e017.get('n_tool_error_events')}). Runtime/infra episodes: "
        f"{e017.get('n_runtime_infra_episodes')}.",
        "- Largest taxonomy buckets (share of invalid turns): "
        "`multiple_actions`, `framing_unbalanced_tags`, `tool_semantic_misuse`.",
        f"- Unique bounded unwrap recovers "
        f"{_fmt(((e017.get('recoverability_event') or {}).get('share_of_invalid_events') or {}).get('recoverable_protocol_mismatch'))} "
        "of invalid turns "
        f"({_fmt(((e017.get('recoverability_event') or {}).get('share_of_protocol_events') or {}).get('recoverable_protocol_mismatch'))} "
        "of protocol turns). Most remaining protocol failures are multiple "
        "open tags or unclosed tags whose inner JSON is itself illegal.",
        "- `parse_ok=0.178` is final-submission parse, not the complement of "
        "invalid_action. "
        f"{(e017.get('cross_parse_ok_invalid') or {}).get('parse_ok=True|invalid=True', 0)} "
        "E017 episodes are invalid **and** parse_ok "
        "(errors then a legal `<final>`).",
        "- E018 B1/M_scaled @4096 episode invalid is ~0.26 and event invalid "
        "~0.11. The 0.95 figure is the **training** any-error rate, not the "
        "held-out eval table.",
        "",
        "## 1. Metric semantics",
        "",
        "### `bcrl/invalid_action_rate` (training) / `invalid_tool_rate` (eval)",
        "",
        "- Grain: **episode**, not per-turn.",
        "- Denominator: trajectories in the unit (32/step, then mean over 275 "
        "steps; equal n so mean-of-rates equals pooled episode rate).",
        "- Numerator: episode has **any** `error_kind=protocol` **or** "
        "`error_kind=tool` turn.",
        "- Tool execution failures (`path_not_found`, …) count, even when "
        "`parse_action` succeeded.",
        "",
        "### `bcrl/parse_ok_rate`",
        "",
        "- Grain: **episode**.",
        "- Meaning: valid **final** submission (`termination=finish` and "
        "`locations` is a list). Protocol errors do not terminate the session.",
        "- **Not** `1 - invalid_action_rate`.",
        "",
        "### JSONL trap",
        "",
        semantics.get("parse_ok_rate", {}).get("jsonl_string_trap", ""),
        "",
        "### E017 reported vs recomputed",
        "",
        _md_kv_table(
            [
                ("reported invalid_action_rate", E017_REPORTED_INVALID_ACTION_RATE),
                (
                    "step_bcrl mean invalid",
                    check.get("step_mean_invalid_action_rate"),
                ),
                ("episodes.jsonl episode invalid", check.get("episode_invalid_rate")),
                ("reported parse_ok_rate", E017_REPORTED_PARSE_OK_RATE),
                ("step_bcrl mean parse_ok", check.get("step_mean_parse_ok_rate")),
                ("episodes.jsonl episode parse_ok", check.get("episode_parse_ok_rate")),
                ("episode-level event invalid rate", e017.get("event_invalid_rate")),
            ]
        ),
        "",
        "E018 held-out `invalid_tool_rate` at B_obs=4096 is **not** 0.95; that "
        "figure is the training episode-any-error rate. Do not splice it into "
        "the E018 eval table.",
        "",
        "## 2. E017 aggregate",
        "",
        _corpus_markdown(e017),
        "",
        "## 3. Taxonomy (invalid events)",
        "",
        _counts_table(
            e017.get("taxonomy_event_counts") or {},
            e017.get("taxonomy_event_share_of_invalid") or {},
        ),
        "",
        "First-invalid-turn taxonomy:",
        "",
        _counts_table(
            e017.get("taxonomy_first_invalid_counts") or {},
            _shares(
                e017.get("taxonomy_first_invalid_counts") or {},
                int(e017.get("n_invalid_episodes") or 0),
            ),
        ),
        "",
        "Error codes on invalid events (protocol `parse_error_code` plus tool `error_code`):",
        "",
        _simple_count_table(e017.get("parse_error_codes") or {}),
        "",
        "## 4. First-invalid-turn / invalid-count-per-trajectory",
        "",
        "First invalid turn (among invalid episodes):",
        "",
        _simple_count_table(e017.get("first_invalid_turn") or {}),
        "",
        "Invalid event count per trajectory (all episodes, including zeros):",
        "",
        _simple_count_table(e017.get("invalid_count_per_trajectory") or {}),
        "",
        "## 5. Descriptive relation to parse_ok / termination / score",
        "",
        _simple_count_table(e017.get("cross_parse_ok_invalid") or {}),
        "",
        _simple_count_table(e017.get("cross_termination_invalid") or {}),
        "",
        _md_kv_table(
            [
                (
                    "mean loc score | invalid episode",
                    e017.get("mean_localization_score_invalid"),
                ),
                (
                    "mean loc score | no protocol/tool error",
                    e017.get("mean_localization_score_valid_protocol"),
                ),
                (
                    "mean loc score | parse_ok",
                    e017.get("mean_localization_score_parse_ok"),
                ),
                (
                    "mean loc score | not parse_ok",
                    e017.get("mean_localization_score_parse_fail"),
                ),
            ]
        ),
        "",
        "The clean-episode mean score is lower than the invalid-episode mean "
        "because many error-free rollouts budget-exhaust after a large first "
        "tool observation and never emit `<final>` (score 0). This is "
        "descriptive, not a claim that protocol errors help localization.",
        "",
        "## 6. Recoverability (offline; production parser unchanged)",
        "",
        str((e017.get("recoverability_rules") or {}).get(RECOVERABLE) or ""),
        "",
        _md_kv_table(
            [
                (
                    "recoverable_protocol_mismatch / invalid events",
                    (e017.get("recoverability_event") or {})
                    .get("share_of_invalid_events", {})
                    .get(RECOVERABLE),
                ),
                (
                    "genuine_unrecoverable_error / invalid events",
                    (e017.get("recoverability_event") or {})
                    .get("share_of_invalid_events", {})
                    .get(UNRECOVERABLE),
                ),
                (
                    "parsed_ok_tool_failure / invalid events",
                    (e017.get("recoverability_event") or {})
                    .get("share_of_invalid_events", {})
                    .get(TOOL_FAILURE),
                ),
                (
                    "recoverable / protocol events",
                    (e017.get("recoverability_event") or {})
                    .get("share_of_protocol_events", {})
                    .get(RECOVERABLE),
                ),
                (
                    "episodes where ALL invalid turns recover",
                    (e017.get("recoverability_episode") or {})
                    .get("share_of_invalid_episodes", {})
                    .get("all_invalid_recoverable"),
                ),
                (
                    "episodes with ANY recoverable invalid turn",
                    (e017.get("recoverability_episode") or {})
                    .get("share_of_invalid_episodes", {})
                    .get("any_invalid_recoverable"),
                ),
            ]
        ),
        "",
        "## 7. E018 supplement (same taxonomy, B1 / M_scaled @4096)",
        "",
        _e018_markdown(e018),
        "",
        "## 8. Representative raw examples",
        "",
        "Raw `raw_action` strings are truncated, not rewritten. Full copies "
        "live in `examples.json`.",
        "",
        _examples_markdown(e017.get("examples") or {}),
        "",
        "Recoverability examples:",
        "",
        _examples_markdown(e017.get("recoverability_examples") or {}),
        "",
        "## 9. M7B gate",
        "",
        f"- decision: `{gate.get('decision')}`",
        f"- {gate.get('reason')}",
        f"- protocol share of invalid events: {_fmt(gate.get('protocol_share_of_invalid_events'))}",
        f"- recoverable share of protocol events: {_fmt(gate.get('recoverable_share_of_protocol_events'))}",
        f"- tool-semantic share of invalid events: {_fmt(gate.get('tool_semantic_share_of_invalid_events'))}",
        f"- prompt-like share (no_action + multiple_actions + unknown_tool): {_fmt(gate.get('promptish_share_of_invalid_events'))}",
        f"- framing/prose share: {_fmt(gate.get('framing_prose_share_of_invalid_events'))}",
        "- Parser-only unique unwrap does **not** cover a majority of invalid "
        "turns. Multiple open tags are not uniquely recoverable. Unbalanced "
        "tags often have broken JSON, so closing the tag is not enough. "
        "Tool `path_not_found` / `path_escape` already parsed.",
        "- If a later milestone continues: do not treat parser repair, "
        "prompt/template alignment, or stopping as a single sufficient lever. "
        "Do not start M7B from this audit. Do not retune parser/prompt/reward.",
        "",
        "## Not verified",
        "",
        "- Causal effect of a future parser change on localization reward.",
        "- Token-level training trajectories (research JSONL only).",
        "",
        "## Risks",
        "",
        "- Recoverability is conservative; schema aliases such as "
        "`max_depth`→`depth` are counted unrecoverable by design.",
        "- Episode-level 0.952 still inflates vs per-turn rate; both numbers "
        "are reported.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _classify_inner(kind: str, inner: str, code: str) -> str:
    if code == "malformed_json":
        return "malformed_json"
    if code == "unknown_tool":
        return "unknown_tool"
    if code == "duplicate_keys":
        return "wrong_schema"
    obj = _load_json_object(inner.strip())
    if obj is None:
        if code == "malformed_json":
            return "malformed_json"
        return "malformed_json" if not inner.strip() else "other_protocol"
    if kind == "tool" and "locations" in obj and "name" not in obj:
        return "framing_wrong_envelope"
    if kind == "final" and "name" in obj and "locations" not in obj:
        return "framing_wrong_envelope"
    if code == "invalid_arguments":
        return _schema_or_args(obj, code)
    if code == "malformed_final":
        return "wrong_schema"
    if code:
        return "other_protocol"
    return "other_protocol"


def _schema_or_args(obj: Mapping[str, Any], code: str) -> str:
    del code
    if not isinstance(obj, MappingABC):
        return "wrong_schema"
    extra_top = set(obj) - {"name", "arguments"}
    if extra_top or "arguments" not in obj or "name" not in obj:
        if "locations" in obj:
            return "framing_wrong_envelope"
        return "wrong_schema"
    arguments = obj.get("arguments")
    if not isinstance(arguments, dict):
        return "bad_args"
    name = obj.get("name")
    allowed = {
        "tree": {"path", "depth"},
        "search": {"query", "path", "max_results"},
        "read": {"path", "start_line", "end_line"},
    }
    if name not in allowed:
        return "unknown_tool"
    extra = set(arguments) - allowed[str(name)]
    missing_read = []
    if name == "read":
        missing_read = [key for key in ("path", "start_line", "end_line") if key not in arguments]
    if extra or (name == "search" and "query" not in arguments) or missing_read:
        return "wrong_schema"
    return "bad_args"


def _looks_like_action_object(obj: Mapping[str, Any]) -> bool:
    keys = set(obj)
    if "name" in keys or "arguments" in keys or "locations" in keys:
        return True
    if "path" in keys and keys <= {"path", "symbol"}:
        return True
    return False


def _inner_complete(text: str, open_tag: str, close_tag: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith(open_tag) or not stripped.endswith(close_tag):
        return None
    inner = stripped[len(open_tag) : len(stripped) - len(close_tag)]
    return inner


def _unwrap_unique_complete_block(stripped: str) -> str | None:
    tags = tag_counts(stripped)
    if tags["n_tool_open"] == 1 and tags["n_tool_close"] == 1 and tags["n_final_open"] == 0:
        start = stripped.find(TOOL_CALL_OPEN)
        end = stripped.find(TOOL_CALL_CLOSE, start + len(TOOL_CALL_OPEN))
        if start < 0 or end < 0:
            return None
        return stripped[start : end + len(TOOL_CALL_CLOSE)]
    if tags["n_final_open"] == 1 and tags["n_final_close"] == 1 and tags["n_tool_open"] == 0:
        start = stripped.find(FINAL_OPEN)
        end = stripped.find(FINAL_CLOSE, start + len(FINAL_OPEN))
        if start < 0 or end < 0:
            return None
        return stripped[start : end + len(FINAL_CLOSE)]
    return None


def _unwrap_unique_unclosed_tag(stripped: str) -> str | None:
    tags = tag_counts(stripped)
    if tags["n_tool_open"] == 1 and tags["n_tool_close"] == 0 and tags["n_final_open"] == 0 and tags["n_final_close"] == 0:
        start = stripped.find(TOOL_CALL_OPEN)
        inner = stripped[start + len(TOOL_CALL_OPEN) :].strip()
        if _load_json_object(inner) is None:
            return None
        return f"{TOOL_CALL_OPEN}\n{inner}\n{TOOL_CALL_CLOSE}"
    if tags["n_final_open"] == 1 and tags["n_final_close"] == 0 and tags["n_tool_open"] == 0 and tags["n_tool_close"] == 0:
        start = stripped.find(FINAL_OPEN)
        inner = stripped[start + len(FINAL_OPEN) :].strip()
        if _load_json_object(inner) is None:
            return None
        return f"{FINAL_OPEN}\n{inner}\n{FINAL_CLOSE}"
    return None


def _unwrap_untagged_json(stripped: str) -> str | None:
    tags = tag_counts(stripped)
    if tags["n_open"] or tags["n_close"]:
        return None
    obj = _load_json_object(stripped)
    if obj is None:
        return None
    keys = set(obj)
    tool_like = "name" in keys and "arguments" in keys
    final_like = "locations" in keys
    if tool_like and not final_like:
        return f"{TOOL_CALL_OPEN}\n{stripped}\n{TOOL_CALL_CLOSE}"
    if final_like and not tool_like:
        return f"{FINAL_OPEN}\n{stripped}\n{FINAL_CLOSE}"
    return None


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and not isinstance(payload, bool):
        return payload
    return None


def _parse_ok(text: str) -> bool:
    try:
        parse_action(text)
    except ProtocolError:
        return False
    return True


def _maybe_keep_example(
    bucket: list[dict[str, Any]],
    analyzed: Mapping[str, Any],
    item: Mapping[str, Any],
) -> None:
    if len(bucket) >= EXAMPLES_PER_BUCKET:
        return
    raw = item.get("raw_action")
    text = raw if isinstance(raw, str) else ""
    if len(text) > EXAMPLE_CHAR_LIMIT:
        text = text[:EXAMPLE_CHAR_LIMIT] + "\n...[truncated]..."
    bucket.append(
        {
            "instance_id": analyzed.get("instance_id"),
            "turn": item.get("turn"),
            "termination": analyzed.get("termination"),
            "parse_ok": analyzed.get("parse_ok"),
            "taxonomy": item.get("taxonomy"),
            "recoverability": item.get("recoverability"),
            "parse_error_code": item.get("parse_error_code"),
            "raw_action": text,
        }
    )


def _shares(counts: Mapping[str, Any], denom: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in counts.items():
        out[str(key)] = (int(value) / denom) if denom else None
    return out


def _counter_to_sorted_dict(counter: Mapping[int, int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _mean(values: Sequence[float]) -> float | None:
    present = [float(item) for item in values]
    if not present:
        return None
    return sum(present) / len(present)


def _close(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right)) <= E017_RATE_TOLERANCE


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_kv_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| item | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def _counts_table(counts: Mapping[str, Any], shares: Mapping[str, Any]) -> str:
    if not counts:
        return "_none_"
    lines = ["| category | n | share |", "| --- | ---: | ---: |"]
    for key in TAXONOMY_BUCKETS:
        if key not in counts:
            continue
        lines.append(
            f"| `{key}` | {int(counts[key])} | {_fmt(shares.get(key))} |"
        )
    for key, value in counts.items():
        if key in TAXONOMY_BUCKETS:
            continue
        lines.append(f"| `{key}` | {int(value)} | {_fmt(shares.get(key))} |")
    return "\n".join(lines)


def _simple_count_table(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "_none_"
    lines = ["| key | n |", "| --- | ---: |"]
    items = list(counts.items())
    try:
        items.sort(key=lambda pair: int(pair[0]))
    except (TypeError, ValueError):
        items.sort(key=lambda pair: str(pair[0]))
    for key, value in items:
        lines.append(f"| `{key}` | {int(value)} |")
    return "\n".join(lines)


def _corpus_markdown(corpus: Mapping[str, Any]) -> str:
    if not corpus:
        return "_E017 corpus missing._"
    rec = corpus.get("recoverability_event") or {}
    shares = rec.get("share_of_invalid_events") or {}
    return "\n".join(
        [
            _md_kv_table(
                [
                    ("n_episodes", corpus.get("n_episodes")),
                    ("n_events (assistant turns)", corpus.get("n_events")),
                    ("episode invalid rate", corpus.get("episode_invalid_rate")),
                    ("episode parse_ok rate", corpus.get("episode_parse_ok_rate")),
                    ("event invalid rate", corpus.get("event_invalid_rate")),
                    ("n_protocol_events", corpus.get("n_protocol_events")),
                    ("n_tool_error_events", corpus.get("n_tool_error_events")),
                    ("runtime_infra episodes", corpus.get("n_runtime_infra_episodes")),
                    (RECOVERABLE, rec.get(RECOVERABLE)),
                    (UNRECOVERABLE, rec.get(UNRECOVERABLE)),
                    (TOOL_FAILURE, rec.get(TOOL_FAILURE)),
                    (f"{RECOVERABLE} share", shares.get(RECOVERABLE)),
                    (f"{UNRECOVERABLE} share", shares.get(UNRECOVERABLE)),
                    (f"{TOOL_FAILURE} share", shares.get(TOOL_FAILURE)),
                ]
            ),
        ]
    )


def _e018_markdown(e018: Mapping[str, Any]) -> str:
    cells = e018.get("cells") if isinstance(e018.get("cells"), MappingABC) else e018
    if not cells:
        return "_E018 supplement not run._"
    lines = [
        "| cell | n | episode invalid | event invalid | parse_ok | recoverable/invalid | unrecoverable/invalid | tool_fail/invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, corpus in cells.items():
        rec = (corpus or {}).get("recoverability_event") or {}
        shares = rec.get("share_of_invalid_events") or {}
        lines.append(
            "| {key} | {n} | {einv} | {ev} | {pok} | {rec} | {un} | {tool} |".format(
                key=key,
                n=_fmt((corpus or {}).get("n_episodes"), 0)
                if isinstance((corpus or {}).get("n_episodes"), int)
                else _fmt((corpus or {}).get("n_episodes")),
                einv=_fmt((corpus or {}).get("episode_invalid_rate")),
                ev=_fmt((corpus or {}).get("event_invalid_rate")),
                pok=_fmt((corpus or {}).get("episode_parse_ok_rate")),
                rec=_fmt(shares.get(RECOVERABLE)),
                un=_fmt(shares.get(UNRECOVERABLE)),
                tool=_fmt(shares.get(TOOL_FAILURE)),
            )
        )
    return "\n".join(lines)


def _examples_markdown(examples: Mapping[str, Any]) -> str:
    if not examples:
        return "_none_"
    blocks: list[str] = []
    keys = [key for key in TAXONOMY_BUCKETS if key in examples]
    keys.extend(key for key in examples if key not in keys)
    for key in keys:
        items = examples.get(key) or []
        if not items:
            continue
        blocks.append(f"### `{key}`")
        for item in items:
            raw = str(item.get("raw_action") or "")
            blocks.append(
                f"- instance `{item.get('instance_id')}` turn {item.get('turn')} "
                f"code=`{item.get('parse_error_code')}` rec=`{item.get('recoverability')}`"
            )
            blocks.append("```")
            blocks.append(raw)
            blocks.append("```")
    return "\n".join(blocks) if blocks else "_none_"


__all__ = [
    "E017_REPORTED_INVALID_ACTION_RATE",
    "E017_REPORTED_PARSE_OK_RATE",
    "METRIC_SEMANTICS",
    "analyze_corpus",
    "analyze_episode",
    "classify_event",
    "e017_self_check",
    "episode_is_invalid",
    "episode_parse_ok",
    "event_is_invalid",
    "filter_e018_cell",
    "gate_recommendation",
    "iter_jsonl",
    "mean_step_bcrl_rates",
    "recoverability_label",
    "render_summary",
    "try_recover_action",
    "truthy",
]
