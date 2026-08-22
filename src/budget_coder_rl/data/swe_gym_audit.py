"""SWE-Gym semantic integrity and leakage audit (M1B).

Audits metadata in place. Does not drop rows, parse gold patches into
oracle locations, create splits, or build an RL dataset.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_SHA256,
    HF_REPO_ID,
    HF_REVISION,
    is_null,
)
from budget_coder_rl.data.swe_gym_fields import PRIVILEGED_FIELDS, agent_task_view

EVIDENCE_MAX_EXAMPLES = 5
EVIDENCE_MAX_CHARS = 200
SHAPE_LABELS: tuple[str, ...] = (
    "pytest_nodeid",
    "path_like",
    "dotted_unittest",
    "other",
)
DOTTED_UNITTEST_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

KNOWN_UPSTREAM_MALFORMED_ID = "conan-io__conan-11594"
KNOWN_UPSTREAM_MALFORMED_GENERIC_FLAGS: tuple[str, ...] = (
    "f2p_unbalanced_brackets",
    "p2p_unbalanced_brackets",
    "f2p_unbalanced_parens",
    "p2p_unbalanced_parens",
    "f2p_embedded_newline",
    "p2p_embedded_newline",
)

# Heuristic structural suspicion only. Not confirmed malformed.
# Pytest parametrized nodeids may contain [], (), and special characters.
HEURISTIC_FLAG_ORDER: tuple[str, ...] = (
    "f2p_non_string_entry",
    "p2p_non_string_entry",
    "f2p_empty_entry",
    "p2p_empty_entry",
    "f2p_whitespace_padded",
    "p2p_whitespace_padded",
    "f2p_embedded_newline",
    "p2p_embedded_newline",
    "f2p_unbalanced_brackets",
    "p2p_unbalanced_brackets",
    "f2p_unbalanced_parens",
    "p2p_unbalanced_parens",
    "f2p_duplicate_entries",
    "p2p_duplicate_entries",
    "f2p_empty",
    "f2p_p2p_overlap",
    "patch_empty",
    "test_patch_empty",
    "patch_not_unified_diff",
    "test_patch_not_unified_diff",
    "patch_binary_or_null",
    "test_patch_binary_or_null",
)
DATASET_PROPERTY_ORDER: tuple[str, ...] = (
    "p2p_empty",
    "duplicate_problem_statement",
    "duplicate_patch",
)
OBSERVATION_ORDER: tuple[str, ...] = (
    "hints_text_nonempty",
    "hints_text_whitespace_only",
    "f2p_selector_verbatim_in_problem_statement",
    "selector_control_chars",
)
HEURISTIC_BY_REPO_FLAGS: tuple[str, ...] = (
    "p2p_unbalanced_brackets",
    "p2p_unbalanced_parens",
    "f2p_unbalanced_brackets",
    "f2p_unbalanced_parens",
)
FLAG_CLASS_NOTES: dict[str, str] = {
    "heuristic_structural_suspicion": (
        "Count-mismatch / coarse structural heuristics. "
        "n_heuristic_suspicion_rows is not a count of confirmed malformed rows. "
        "Pytest parametrized IDs may legitimately contain [], (), and special characters."
    ),
    "dataset_correlation_property": (
        "Dataset or correlation properties (empty P2P, exact duplicate text/patch). "
        "Not a drop filter; all rows are retained."
    ),
    "observational_signal": (
        "Observational signals only (hints presence, verbatim F2P in the issue, "
        "control characters in selectors). Not leakage verdicts."
    ),
}
CONTROL_CHAR_LABELS: dict[int, str] = {
    0: "NUL",
    9: "tab",
    10: "newline",
    13: "carriage_return",
}
LITERAL_ESCAPE_RE = re.compile(
    r"\\x[01][0-9A-Fa-f]|\\u00[01][0-9A-Fa-f]|\\[nrt0]"
)


def audit_jsonl_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "interim" / "swe_gym" / "m1b_audit.jsonl"


def audit_summary_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "manifests" / "swe_gym_m1b_audit_summary.json"


def parse_test_list(value: Any) -> list[Any] | None:
    """Parse F2P/P2P to a list without stringifying element types."""
    if is_null(value):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return parse_test_list(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [value]
        if parsed is None:
            return None
        if isinstance(parsed, list):
            return list(parsed)
        return [parsed]
    return [value]


def classify_selector_shape(selector: str) -> str:
    if "::" in selector:
        return "pytest_nodeid"
    if "/" in selector or selector.endswith(".py"):
        return "path_like"
    if selector and DOTTED_UNITTEST_RE.fullmatch(selector):
        return "dotted_unittest"
    return "other"


def _truncate_evidence(text: str) -> str:
    if len(text) <= EVIDENCE_MAX_CHARS:
        return text
    return f"{text[:EVIDENCE_MAX_CHARS]}... <truncated, {len(text)} chars total>"


def _has_embedded_newline(text: str) -> bool:
    return "\n" in text or "\r" in text


def _has_other_control_chars(text: str) -> bool:
    return any(ord(char) < 32 and char not in "\n\r" for char in text)


def inspect_selector_controls(text: str) -> tuple[Counter[str], Counter[str]]:
    """Split actual C0 bytes from literal backslash-escape text (e.g. '\\x00')."""
    actual: Counter[str] = Counter()
    for char in text:
        code = ord(char)
        if code < 32:
            actual[CONTROL_CHAR_LABELS.get(code, f"U+{code:04X}")] += 1
    literal: Counter[str] = Counter()
    for match in LITERAL_ESCAPE_RE.finditer(text):
        literal[match.group()] += 1
    return actual, literal


def selector_anomaly_tags(selector: Any) -> tuple[list[str], list[str], str | None]:
    """Return (anomaly_tags, observation_tags, shape). Tags are unprefixed."""
    if not isinstance(selector, str):
        return (["non_string_entry"], [], None)
    tags: list[str] = []
    observations: list[str] = []
    if selector == "":
        tags.append("empty_entry")
    if selector != selector.strip():
        tags.append("whitespace_padded")
    if _has_embedded_newline(selector):
        tags.append("embedded_newline")
    if _has_other_control_chars(selector):
        observations.append("control_chars")
    if selector.count("[") != selector.count("]"):
        tags.append("unbalanced_brackets")
    if selector.count("(") != selector.count(")"):
        tags.append("unbalanced_parens")
    return (tags, observations, classify_selector_shape(selector))


def looks_like_unified_diff(text: str) -> bool:
    has_diff_git = "diff --git " in text
    has_minus = text.startswith("--- ") or "\n--- " in text
    has_plus = text.startswith("+++ ") or "\n+++ " in text
    return has_diff_git and has_minus and has_plus


def looks_binary_or_null(text: str) -> bool:
    if "\x00" in text:
        return True
    if "GIT binary patch" in text:
        return True
    if "Binary files " in text:
        return True
    return False


def normalize_problem_statement(value: Any) -> str:
    if is_null(value):
        return ""
    text = str(value).strip()
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _add_evidence(store: dict[str, list[str]], flag: str, example: str) -> None:
    bucket = store.setdefault(flag, [])
    if len(bucket) >= EVIDENCE_MAX_EXAMPLES:
        return
    clipped = _truncate_evidence(example)
    if clipped not in bucket:
        bucket.append(clipped)


def _sorted_unique(values: Iterable[str], order: Sequence[str]) -> list[str]:
    present = set(values)
    ordered = [name for name in order if name in present]
    extras = sorted(present.difference(order))
    return ordered + extras


def _sorted_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter) if counter[key]}


def _partition_tags(tags: Iterable[str]) -> tuple[list[str], list[str]]:
    property_set = set(DATASET_PROPERTY_ORDER)
    heuristic_set = set(HEURISTIC_FLAG_ORDER)
    heuristic: list[str] = []
    properties: list[str] = []
    extras: list[str] = []
    for name in tags:
        if name in property_set:
            properties.append(name)
        elif name in heuristic_set:
            heuristic.append(name)
        else:
            extras.append(name)
    heuristic.extend(extras)
    return (
        _sorted_unique(heuristic, HEURISTIC_FLAG_ORDER),
        _sorted_unique(properties, DATASET_PROPERTY_ORDER),
    )


def audit_test_list(values: Any, *, prefix: str) -> dict[str, Any]:
    parsed = parse_test_list(values)
    flags: list[str] = []
    observations: list[str] = []
    evidence: dict[str, list[str]] = {}
    shape_counts: Counter[str] = Counter()
    string_entries: list[str] = []

    if parsed is None or len(parsed) == 0:
        flags.append(f"{prefix}_empty")
        return {
            "flags": flags,
            "observations": observations,
            "evidence": evidence,
            "shape_counts": dict(shape_counts),
            "string_entries": string_entries,
        }

    for item in parsed:
        tags, obs_tags, shape = selector_anomaly_tags(item)
        if shape is not None:
            shape_counts[shape] += 1
        if isinstance(item, str):
            string_entries.append(item)
            preview = item
        else:
            preview = f"{type(item).__name__}:{item!r}"
        for tag in tags:
            flag = f"{prefix}_{tag}"
            if flag not in flags:
                flags.append(flag)
            _add_evidence(evidence, flag, preview)
        if obs_tags:
            if "selector_control_chars" not in observations:
                observations.append("selector_control_chars")
            _add_evidence(evidence, "selector_control_chars", preview)

    if len(string_entries) != len(set(string_entries)):
        flag = f"{prefix}_duplicate_entries"
        flags.append(flag)
        seen: set[str] = set()
        for item in string_entries:
            if item in seen:
                _add_evidence(evidence, flag, item)
            else:
                seen.add(item)

    return {
        "flags": flags,
        "observations": observations,
        "evidence": evidence,
        "shape_counts": {key: int(shape_counts[key]) for key in sorted(shape_counts)},
        "string_entries": string_entries,
    }


def audit_patch_text(text: Any, *, field: str) -> dict[str, Any]:
    flags: list[str] = []
    evidence: dict[str, list[str]] = {}
    if is_null(text) or (isinstance(text, str) and text == ""):
        flag = f"{field}_empty"
        flags.append(flag)
        _add_evidence(evidence, flag, "empty")
        return {"flags": flags, "evidence": evidence}

    blob = str(text)
    if not looks_like_unified_diff(blob):
        flag = f"{field}_not_unified_diff"
        flags.append(flag)
        missing: list[str] = []
        if "diff --git " not in blob:
            missing.append("diff --git")
        if not (blob.startswith("--- ") or "\n--- " in blob):
            missing.append("---")
        if not (blob.startswith("+++ ") or "\n+++ " in blob):
            missing.append("+++")
        _add_evidence(evidence, flag, "missing: " + ", ".join(missing) if missing else "incomplete unified diff")
    if looks_binary_or_null(blob):
        flag = f"{field}_binary_or_null"
        flags.append(flag)
        if "\x00" in blob:
            _add_evidence(evidence, flag, "contains NUL")
        elif "GIT binary patch" in blob:
            _add_evidence(evidence, flag, "GIT binary patch")
        else:
            _add_evidence(evidence, flag, "Binary files marker")
    return {"flags": flags, "evidence": evidence}


def _merge_evidence(
    target: dict[str, list[str]], source: Mapping[str, Sequence[str]]
) -> None:
    for key, values in source.items():
        for value in values:
            _add_evidence(target, key, value)


def _row_mapping(frame: Any, index: Any) -> dict[str, Any]:
    row = frame.loc[index]
    return {str(column): row[column] for column in frame.columns}


def _hints_observation(value: Any) -> str | None:
    if is_null(value):
        return None
    text = str(value)
    if text == "":
        return None
    if text.strip() == "":
        return "hints_text_whitespace_only"
    return "hints_text_nonempty"


def audit_row_local(row: Mapping[str, Any]) -> dict[str, Any]:
    """Per-row checks that do not need table-wide duplicate maps."""
    flags: list[str] = []
    observations: list[str] = []
    evidence: dict[str, list[str]] = {}
    shape_counts = {
        "FAIL_TO_PASS": {label: 0 for label in SHAPE_LABELS},
        "PASS_TO_PASS": {label: 0 for label in SHAPE_LABELS},
    }

    f2p = audit_test_list(row.get("FAIL_TO_PASS"), prefix="f2p")
    p2p = audit_test_list(row.get("PASS_TO_PASS"), prefix="p2p")
    flags.extend(f2p["flags"])
    flags.extend(p2p["flags"])
    observations.extend(f2p["observations"])
    observations.extend(p2p["observations"])
    _merge_evidence(evidence, f2p["evidence"])
    _merge_evidence(evidence, p2p["evidence"])
    for label, count in f2p["shape_counts"].items():
        shape_counts["FAIL_TO_PASS"][label] = int(count)
    for label, count in p2p["shape_counts"].items():
        shape_counts["PASS_TO_PASS"][label] = int(count)

    overlap = sorted(set(f2p["string_entries"]) & set(p2p["string_entries"]))
    if overlap:
        flags.append("f2p_p2p_overlap")
        for item in overlap:
            _add_evidence(evidence, "f2p_p2p_overlap", item)

    problem = row.get("problem_statement")
    problem_text = "" if is_null(problem) else str(problem)
    for selector in f2p["string_entries"]:
        if selector and selector in problem_text:
            if "f2p_selector_verbatim_in_problem_statement" not in observations:
                observations.append("f2p_selector_verbatim_in_problem_statement")
            _add_evidence(
                evidence, "f2p_selector_verbatim_in_problem_statement", selector
            )

    hints_obs = _hints_observation(row.get("hints_text"))
    if hints_obs is not None:
        observations.append(hints_obs)

    patch_audit = audit_patch_text(row.get("patch"), field="patch")
    test_patch_audit = audit_patch_text(row.get("test_patch"), field="test_patch")
    flags.extend(patch_audit["flags"])
    flags.extend(test_patch_audit["flags"])
    _merge_evidence(evidence, patch_audit["evidence"])
    _merge_evidence(evidence, test_patch_audit["evidence"])

    compact_shapes = {
        field: {key: value for key, value in counts.items() if value}
        for field, counts in shape_counts.items()
    }
    return {
        "flags": flags,
        "observations": observations,
        "evidence": evidence,
        "selector_shape_counts": compact_shapes,
        "f2p_string_entries": f2p["string_entries"],
        "p2p_string_entries": p2p["string_entries"],
    }


def _empty_shape() -> dict[str, int]:
    return {label: 0 for label in SHAPE_LABELS}


def _group_size_distribution(sizes: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(size) for size in sizes)
    return {str(size): int(counts[size]) for size in sorted(counts)}


def _duplicate_stats(groups: Mapping[Any, Sequence[str]]) -> dict[str, int]:
    dup_groups = [list(members) for members in groups.values() if len(members) > 1]
    n_rows = sum(len(members) for members in dup_groups)
    n_extra = sum(len(members) - 1 for members in dup_groups)
    return {
        "n_groups": len(dup_groups),
        "n_rows": n_rows,
        "n_extra": n_extra,
    }


def _human_review_notes(
    *,
    heuristic_counts: Mapping[str, int],
    property_counts: Mapping[str, int],
    observation_counts: Mapping[str, int],
    n_heuristic: int,
    n_rows: int,
    n_identical_repo_commit_ps: int = 0,
) -> list[str]:
    notes: list[str] = [
        "M1B does not drop rows. All input instances are retained.",
        FLAG_CLASS_NOTES["heuristic_structural_suspicion"],
        FLAG_CLASS_NOTES["dataset_correlation_property"],
        FLAG_CLASS_NOTES["observational_signal"],
        "Issue mentions of file/function/test names are not automatic leakage.",
    ]
    if heuristic_counts.get("p2p_unbalanced_brackets", 0) or heuristic_counts.get(
        "f2p_unbalanced_brackets", 0
    ):
        notes.append(
            "Unbalanced [] / () flags are count-mismatch heuristics, not confirmed "
            "malformed selectors. Pytest parametrized IDs may contain brackets."
        )
    if property_counts.get("p2p_empty", 0):
        notes.append(
            "Empty PASS_TO_PASS lists are a dataset property, not a structural "
            "anomaly or download defect. They are not filtered."
        )
    if observation_counts.get("selector_control_chars", 0):
        notes.append(
            "selector_control_chars counts actual C0 bytes in selectors. "
            "Literal backslash-escape text is reported separately. UUID/NUL "
            "parametrize nodeids may be legitimate."
        )
    if property_counts.get("duplicate_problem_statement", 0):
        notes.append(
            "Exact-duplicate normalized problem_statement groups are correlation "
            "properties, not a drop filter."
        )
    if n_identical_repo_commit_ps:
        notes.append(
            "At least one identical (repo, base_commit, problem_statement) tuple "
            "exists; inspect before treating those instance_ids as independent."
        )
    if observation_counts.get("f2p_selector_verbatim_in_problem_statement", 0):
        notes.append(
            "F2P identifiers that appear verbatim in problem_statement are "
            "observational only; they may be legitimate issue text."
        )
    notes.append(
        f"{n_heuristic} of {n_rows} instances have at least one heuristic "
        "structural suspicion (not confirmed malformed)."
    )
    return notes


def audit_frame(frame: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit every row. Returns (per-instance records, summary). Never drops rows."""
    indices = list(frame.index)
    n_rows = len(indices)
    locals_by_index: dict[Any, dict[str, Any]] = {}
    instance_ids: list[str] = []
    ps_groups: dict[str, list[str]] = defaultdict(list)
    patch_groups: dict[str, list[str]] = defaultdict(list)
    repo_commit_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    triple_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    shape_totals = {
        "FAIL_TO_PASS": _empty_shape(),
        "PASS_TO_PASS": _empty_shape(),
    }

    for index in indices:
        row = _row_mapping(frame, index)
        local = audit_row_local(row)
        instance_id = "" if is_null(row.get("instance_id")) else str(row["instance_id"])
        repo = "" if is_null(row.get("repo")) else str(row["repo"])
        base_commit = "" if is_null(row.get("base_commit")) else str(row["base_commit"])
        patch_text = "" if is_null(row.get("patch")) else str(row["patch"])
        ps_norm = normalize_problem_statement(row.get("problem_statement"))
        local["instance_id"] = instance_id
        local["repo"] = repo
        locals_by_index[index] = local
        instance_ids.append(instance_id)
        if instance_id:
            ps_groups[ps_norm].append(instance_id)
            patch_groups[patch_text].append(instance_id)
            repo_commit_groups[(repo, base_commit)].append(instance_id)
            triple_groups[(repo, base_commit, ps_norm)].append(instance_id)
        for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            for label, count in local["selector_shape_counts"].get(field, {}).items():
                shape_totals[field][label] = shape_totals[field].get(label, 0) + int(count)

    ps_dup_ids = {
        member
        for members in ps_groups.values()
        if len(members) > 1
        for member in members
    }
    patch_dup_ids = {
        member
        for members in patch_groups.values()
        if len(members) > 1
        for member in members
    }
    ps_peers = {
        member: sorted(peer for peer in members if peer != member)
        for members in ps_groups.values()
        if len(members) > 1
        for member in members
    }
    patch_peers = {
        member: sorted(peer for peer in members if peer != member)
        for members in patch_groups.values()
        if len(members) > 1
        for member in members
    }

    records: list[dict[str, Any]] = []
    heuristic_counter: Counter[str] = Counter()
    property_counter: Counter[str] = Counter()
    observation_counter: Counter[str] = Counter()
    by_repo_counters: dict[str, Counter[str]] = {
        name: Counter() for name in HEURISTIC_BY_REPO_FLAGS
    }
    actual_occ: Counter[str] = Counter()
    actual_inst: Counter[str] = Counter()
    literal_occ: Counter[str] = Counter()
    literal_inst: Counter[str] = Counter()
    n_rows_with_actual_c0 = 0
    n_rows_with_literal_escape = 0
    for index in indices:
        local = locals_by_index[index]
        instance_id = local["instance_id"]
        mixed_tags = list(local["flags"])
        evidence = dict(local["evidence"])
        if instance_id in ps_dup_ids:
            mixed_tags.append("duplicate_problem_statement")
            for peer in ps_peers.get(instance_id, []):
                _add_evidence(evidence, "duplicate_problem_statement", peer)
        if instance_id in patch_dup_ids:
            mixed_tags.append("duplicate_patch")
            for peer in patch_peers.get(instance_id, []):
                _add_evidence(evidence, "duplicate_patch", peer)
        flags, dataset_properties = _partition_tags(mixed_tags)
        observations = _sorted_unique(local["observations"], OBSERVATION_ORDER)
        evidence = {key: evidence[key] for key in sorted(evidence)}
        for flag in flags:
            heuristic_counter[flag] += 1
            if flag in by_repo_counters:
                by_repo_counters[flag][local["repo"]] += 1
        for prop in dataset_properties:
            property_counter[prop] += 1
        for obs in observations:
            observation_counter[obs] += 1
        row_actual: Counter[str] = Counter()
        row_literal: Counter[str] = Counter()
        for selector in local["f2p_string_entries"] + local["p2p_string_entries"]:
            actual, literal = inspect_selector_controls(selector)
            row_actual.update(actual)
            row_literal.update(literal)
        if row_actual:
            n_rows_with_actual_c0 += 1
            actual_occ.update(row_actual)
            for label in row_actual:
                actual_inst[label] += 1
        if row_literal:
            n_rows_with_literal_escape += 1
            literal_occ.update(row_literal)
            for label in row_literal:
                literal_inst[label] += 1
        records.append(
            {
                "instance_id": instance_id,
                "repo": local["repo"],
                "flags": flags,
                "dataset_properties": dataset_properties,
                "observations": observations,
                "evidence": evidence,
                "selector_shape_counts": local["selector_shape_counts"],
            }
        )

    records.sort(key=lambda item: item["instance_id"])
    n_heuristic = sum(1 for item in records if item["flags"])
    by_id = {item["instance_id"]: item for item in records}
    known = by_id.get(KNOWN_UPSTREAM_MALFORMED_ID)
    known_flags: list[str] = []
    captured = False
    if known is not None:
        known_flags = _sorted_unique(
            (flag for flag in known["flags"] if flag in KNOWN_UPSTREAM_MALFORMED_GENERIC_FLAGS),
            KNOWN_UPSTREAM_MALFORMED_GENERIC_FLAGS,
        )
        captured = bool(known_flags)

    rc_sizes = [len(members) for members in repo_commit_groups.values()]
    heuristic_counts = {
        name: int(heuristic_counter[name])
        for name in _sorted_unique(heuristic_counter, HEURISTIC_FLAG_ORDER)
    }
    property_counts = {
        name: int(property_counter[name])
        for name in _sorted_unique(property_counter, DATASET_PROPERTY_ORDER)
    }
    observation_counts = {
        name: int(observation_counter[name])
        for name in _sorted_unique(observation_counter, OBSERVATION_ORDER)
    }
    heuristic_by_repo = {
        flag: _sorted_counts(by_repo_counters[flag]) for flag in HEURISTIC_BY_REPO_FLAGS
    }
    triple_dup_groups = []
    for (repo, commit, _ps_norm), members in sorted(
        triple_groups.items(),
        key=lambda item: (item[0][0], item[0][1], tuple(sorted(item[1]))),
    ):
        if len(members) > 1:
            triple_dup_groups.append(
                {
                    "repo": repo,
                    "base_commit": commit,
                    "instance_ids": sorted(members),
                    "n": len(members),
                }
            )
    triple_stats = _duplicate_stats(triple_groups)
    triple_stats["groups"] = triple_dup_groups

    summary: dict[str, Any] = {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": HF_REVISION,
        "sha256": EXPECTED_SHA256,
        "n_rows": n_rows,
        "n_rows_written": n_rows,
        "rows_dropped": 0,
        "n_heuristic_suspicion_rows": n_heuristic,
        "heuristic_suspicion_is_not_confirmed_malformed": True,
        "flag_class_notes": dict(FLAG_CLASS_NOTES),
        "heuristic_suspicion_counts": heuristic_counts,
        "dataset_property_counts": property_counts,
        "observation_counts": observation_counts,
        "heuristic_suspicion_by_repo": heuristic_by_repo,
        "selector_control_chars": {
            "n_instances_with_actual_c0": n_rows_with_actual_c0,
            "n_instances_with_literal_escape_sequence": n_rows_with_literal_escape,
            "actual_control_chars": {
                "instance_counts": _sorted_counts(actual_inst),
                "occurrence_counts": _sorted_counts(actual_occ),
            },
            "literal_escape_sequences": {
                "instance_counts": _sorted_counts(literal_inst),
                "occurrence_counts": _sorted_counts(literal_occ),
                "note": (
                    "Two-character (or longer) C0-related backslash sequences "
                    "in selector text (\\n, \\t, \\0, \\x00-\\x1f), not decoded "
                    "control bytes and not unicode letters."
                ),
            },
        },
        "selector_shape": {
            "FAIL_TO_PASS": dict(shape_totals["FAIL_TO_PASS"]),
            "PASS_TO_PASS": dict(shape_totals["PASS_TO_PASS"]),
        },
        "duplicates": {
            "problem_statement": _duplicate_stats(ps_groups),
            "patch": _duplicate_stats(patch_groups),
            "repo_base_commit_problem_statement": triple_stats,
        },
        "repo_base_commit": {
            "n_groups": len(repo_commit_groups),
            "group_size_distribution": _group_size_distribution(rc_sizes),
            "max_group_size": max(rc_sizes) if rc_sizes else 0,
            "n_singletons": sum(1 for size in rc_sizes if size == 1),
        },
        "known_upstream_malformed": {
            "instance_id": KNOWN_UPSTREAM_MALFORMED_ID,
            "present_in_input": known is not None,
            "captured_by_generic_rule": captured,
            "matching_flags": known_flags,
        },
        "privileged_fields": list(PRIVILEGED_FIELDS),
        "agent_task_input_fields": list(agent_task_view({"problem_statement": ""}).keys()),
        "human_review_notes": _human_review_notes(
            heuristic_counts=heuristic_counts,
            property_counts=property_counts,
            observation_counts=observation_counts,
            n_heuristic=n_heuristic,
            n_rows=n_rows,
            n_identical_repo_commit_ps=int(triple_stats["n_groups"]),
        ),
    }
    return records, summary


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            n_written += 1
    return n_written
