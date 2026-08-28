"""M7C matched train-vs-dev prompt-path audit and replay analysis.

Diagnostic only. Does not modify parser, prompt, reward, AgentLoop, or
frozen E017/E018 artifacts. GPU replay is a separate runner.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.budget.state import BudgetState, resolve_episode_budget
from budget_coder_rl.data.swe_gym_fields import POLICY_FORBIDDEN_DERIVED_FIELDS
from budget_coder_rl.data.swe_gym_materialize import (
    DATA_SOURCE,
    DEV_PARQUET_RELPATH,
    EXTRA_INFO_KEYS,
    POLICY_COLUMNS,
    TRAIN_PARQUET_RELPATH,
    oracle_parquet_path,
)
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING, sha256_ids
from budget_coder_rl.eval.m4a import PRIVILEGED_LEAK_MARKERS, load_json
from budget_coder_rl.eval.m5a import default_output_dir
from budget_coder_rl.eval.m6 import PRIVILEGED_EXTRA_KEYS, extra_info_leakage_errors
from budget_coder_rl.eval.m7a import (
    classify_event,
    episode_events,
    episode_is_invalid,
    episode_parse_ok,
    event_is_invalid,
    localization_score,
    naive_bool,
    truthy,
)
from budget_coder_rl.eval.m7b import first_turn_event
from budget_coder_rl.protocol.prompt import (
    build_stage1_messages,
    extract_issue_text,
    policy_safe_repo,
)

SCHEMA_VERSION = "bcrl-m7c-v1"
MILESTONE = "M7C"
EXPERIMENT_ID = "M7C"
CONFIG_RELPATH = "configs/experiments/stage1_m7c.json"
AGENT_LOOP_CONFIG_RELPATH = "configs/agent_loop/repo_exploration_m3c.yaml"
TRAIN_CANDIDATES_RELPATH = "data/manifests/m5_scaled_train_candidates.json"
DEV_TASKS_RELPATH = "data/manifests/m3b_baseline_tasks.json"
TRAJECTORY_RELPATH = "trajectories/m7c/M7C"
N_SUBSET = 64
N_AUDIT = 8
SEED_BASE = 20260828
SEED_FORMULA = "SEED_BASE + subset_index"
OBS_TOKENS_LIMIT = 4096
BUDGET_VISIBLE = True
MAX_TURNS = 6
MAX_NEW_TOKENS_PER_TURN = 2048
VALIDATE = False
VLLM_ROLLOUT_N = 1
APPLY_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {}
AGENT_NAME = "repo_exploration"
DATASET_CLASS = "verl.utils.dataset.rl_dataset.RLHFDataset"
POLICY = "Qwen/Qwen3-4B-Instruct-2507"
N_GPUS = 2
TENSOR_MODEL_PARALLEL_SIZE = 1
VERDICT_STRENGTHENED_DELTA = 0.20
VERDICT_NOT_REPRODUCED_ABS = 0.10
TRACKED_TAXONOMY = (
    "multiple_actions",
    "framing_unbalanced_tags",
    "tool_semantic_misuse",
    "other_protocol",
    "other_tool_error",
)
VERDICTS = (
    "H3_causally_strengthened",
    "H3_not_reproduced",
    "execution_path_confound_found",
)
FORBIDDEN_OUTPUT_IDS = (
    "E011",
    "E012",
    "E013",
    "E014",
    "E015",
    "E016",
    "E017",
    "E018",
    "M7A",
    "M7B",
)
SOURCE_AUDIT_RELPATHS = (
    "src/budget_coder_rl/agent_loop/repo_exploration.py",
    "src/budget_coder_rl/protocol/prompt.py",
    "src/budget_coder_rl/agent_loop/tokenization.py",
    "scripts/eval/gpu_runtime.py",
)
SPLIT_BRANCH_RE = re.compile(
    r"""(?:if\s+split\b|split\s*==\s*['\"](?:train|dev)['\"]|split\s*!=\s*['\"](?:train|dev)['\"])""",
    re.MULTILINE,
)
METADATA_LEAK_PATTERNS = (
    r"\bsplit\s*[:=]\s*(?:train|dev)\b",
    r"\bextra_info\b",
    r"\bglobal_step\b",
    r"\bsampling_seed\b",
    r"\bcondition_id\b",
    r"\breward_model\b",
    r"\bground_truth\b",
    r"\bcorrelation_group",
    r"\buid\b",
)
THINK_MARKERS = ("<think>", "</think>")


def default_m7c_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_trace_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / TRAJECTORY_RELPATH


def default_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / CONFIG_RELPATH


def replay_seed(subset_index: int, *, base: int = SEED_BASE) -> int:
    return int(base) + int(subset_index)


def as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    blob = ",".join(str(int(item)) for item in token_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_ordered_ids(path: Path) -> list[str]:
    payload = load_json(Path(path))
    ids = [str(item) for item in payload.get("ordered_ids") or []]
    if not ids:
        raise ValueError(f"{path} has empty ordered_ids")
    return ids


def select_subset(ordered_ids: Sequence[str], n: int) -> list[str]:
    if int(n) <= 0:
        raise ValueError("subset n must be positive")
    selected = [str(item) for item in list(ordered_ids)[: int(n)]]
    if len(selected) < int(n):
        raise ValueError(f"need {n} ids, found {len(selected)}")
    return selected


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            return [f"refusing to write into {experiment_id} artifact directory {forbidden}"]
    return []


def synthetic_policy_row(
    *,
    problem_statement: str,
    repo: str,
    instance_id: str,
    split: str,
    index: int = 0,
    base_commit: str = "0" * 40,
    version: str = "m7c-synthetic",
    data_source: str = DATA_SOURCE,
) -> dict[str, Any]:
    if split not in {"train", "dev"}:
        raise ValueError(f"split must be train|dev, got {split!r}")
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": problem_statement}],
        "reward_model": {"style": "rule", "ground_truth": instance_id},
        "extra_info": {
            "index": int(index),
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "version": version,
            "split": split,
        },
    }


def dataset_to_agent_kwargs(row: Mapping[str, Any]) -> dict[str, Any]:
    """Mimic RLHFDataset.__getitem__ + eval build_batch non-tensor fields."""
    extra = as_mapping(row.get("extra_info"))
    prompt = row.get("prompt")
    raw_prompt = row.get("raw_prompt")
    if raw_prompt is None:
        raw_prompt = prompt
    return {
        "raw_prompt": raw_prompt,
        "extra_info": extra,
        "agent_name": AGENT_NAME,
        "data_source": row.get("data_source"),
        "reward_model": row.get("reward_model"),
    }


def build_matched_extra_info(
    source: Mapping[str, Any],
    *,
    sampling_seed: int,
    obs_tokens_limit: int = OBS_TOKENS_LIMIT,
    budget_visible: bool = BUDGET_VISIBLE,
) -> dict[str, Any]:
    """Single extra_info builder for both splits. No condition_id / policy."""
    extra = dict(source)
    for key in PRIVILEGED_EXTRA_KEYS:
        extra.pop(key, None)
    extra["budget_visible"] = bool(budget_visible)
    extra["obs_tokens_limit"] = int(obs_tokens_limit)
    extra["sampling_seed"] = int(sampling_seed)
    extra.pop("condition_id", None)
    extra.pop("policy", None)
    leaks = extra_info_leakage_errors(extra)
    if leaks:
        raise ValueError(f"policy extra_info leaked privileged fields: {leaks}")
    return extra


def assemble_first_turn(
    kwargs: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_turns: int = MAX_TURNS,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from budget_coder_rl.agent_loop.tokenization import encode_chat_messages

    extra = as_mapping(kwargs.get("extra_info"))
    issue = extract_issue_text(kwargs.get("raw_prompt"))
    obs_limit, visible = resolve_episode_budget(
        extra,
        default_limit=OBS_TOKENS_LIMIT,
        default_visible=BUDGET_VISIBLE,
    )
    budget = BudgetState(
        obs_tokens_used=0,
        obs_tokens_limit=obs_limit,
        turns_used=0,
        turns_limit=int(max_turns),
    )
    messages = build_stage1_messages(
        issue,
        repo=policy_safe_repo(extra),
        budget_state=budget if visible else None,
        budget_visible=visible,
    )
    template_kwargs = dict(apply_chat_template_kwargs or APPLY_CHAT_TEMPLATE_KWARGS)
    prompt_ids = encode_chat_messages(
        tokenizer,
        messages,
        apply_chat_template_kwargs=template_kwargs or None,
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        **template_kwargs,
    )
    if not isinstance(rendered, str):
        rendered = str(rendered)
    try:
        decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    except TypeError:
        decoded = tokenizer.decode(prompt_ids)
    system = str(messages[0].get("content") or "")
    user = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    think_in_rendered = any(marker in rendered for marker in THINK_MARKERS)
    think_in_decoded = any(marker in decoded for marker in THINK_MARKERS)
    return {
        "problem_statement": issue,
        "system_prompt": system,
        "system_prompt_sha256": sha256_text(system),
        "user_prompt": user,
        "messages": [dict(item) for item in messages],
        "rendered": rendered,
        "rendered_sha256": sha256_text(rendered),
        "prompt_ids": [int(item) for item in prompt_ids],
        "prompt_ids_sha256": sha256_token_ids(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "decoded": decoded,
        "repo": policy_safe_repo(extra),
        "split": extra.get("split"),
        "instance_id": extra.get("instance_id"),
        "sampling_seed": extra.get("sampling_seed"),
        "budget_visible": visible,
        "obs_tokens_limit": obs_limit,
        "max_turns": int(max_turns),
        "apply_chat_template_kwargs": dict(template_kwargs),
        "tools": None,
        "add_generation_prompt": True,
        "contains_think_tag": bool(think_in_rendered or think_in_decoded),
        "agent_name": kwargs.get("agent_name") or AGENT_NAME,
    }


def first_generation_prompt_ids(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    sampling_seed: int,
    obs_tokens_limit: int = OBS_TOKENS_LIMIT,
    budget_visible: bool = BUDGET_VISIBLE,
) -> list[int]:
    kwargs = dataset_to_agent_kwargs(row)
    kwargs["extra_info"] = build_matched_extra_info(
        as_mapping(kwargs.get("extra_info")),
        sampling_seed=int(sampling_seed),
        obs_tokens_limit=int(obs_tokens_limit),
        budget_visible=bool(budget_visible),
    )
    ctx = assemble_first_turn(kwargs, tokenizer)
    return list(ctx["prompt_ids"])


def coerce_prompt_messages(prompt: Any) -> list[Any]:
    if prompt is None:
        return []
    if hasattr(prompt, "tolist"):
        try:
            prompt = prompt.tolist()
        except (TypeError, ValueError):
            pass
    if isinstance(prompt, list):
        return list(prompt)
    return []


def policy_row_schema_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for column in POLICY_COLUMNS:
        if column not in row:
            errors.append(f"missing policy column {column}")
    extra = as_mapping(row.get("extra_info"))
    for key in EXTRA_INFO_KEYS:
        if key not in extra:
            errors.append(f"missing extra_info.{key}")
    forbidden = set(POLICY_FORBIDDEN_DERIVED_FIELDS)
    for key in extra:
        if key in forbidden:
            errors.append(f"forbidden extra_info key {key}")
    prompt = coerce_prompt_messages(row.get("prompt"))
    if not prompt:
        errors.append("prompt must be a non-empty list of messages")
    else:
        first = prompt[0] if isinstance(prompt[0], MappingABC) else as_mapping(prompt[0])
        if str(first.get("role") or "") != "user":
            errors.append("prompt[0].role must be user")
    reward = as_mapping(row.get("reward_model"))
    if str(reward.get("style") or "") != "rule":
        errors.append("reward_model.style must be rule")
    extra_id = str(extra.get("instance_id") or "")
    gt = reward.get("ground_truth")
    if extra_id and gt is not None and str(gt) != extra_id:
        errors.append("reward_model.ground_truth must be opaque instance_id")
    return errors


def leakage_scan(
    *,
    rendered: str,
    decoded: str,
    system_prompt: str,
    user_prompt: str,
    problem_statement: str,
    extra_info: Mapping[str, Any],
) -> list[str]:
    """Flag evaluator/split metadata that should not enter model context.

    Natural occurrences of the words train/dev inside problem_statement are
    allowed. Split must not appear outside the issue body.
    """
    errors: list[str] = []
    extra = as_mapping(extra_info)
    leaks = extra_info_leakage_errors(extra)
    errors.extend(leaks)
    blob = "\n".join([system_prompt, rendered, decoded])
    for marker in PRIVILEGED_LEAK_MARKERS:
        if marker in blob:
            errors.append(f"{marker} appeared in rendered/decoded prompt")
    for field in POLICY_FORBIDDEN_DERIVED_FIELDS:
        needle = f"LEAK_{field.upper()}_SENTINEL"
        if needle in blob:
            errors.append(f"sentinel for {field} appeared in prompt")
    user_without_issue = user_prompt
    if problem_statement and problem_statement in user_without_issue:
        user_without_issue = user_without_issue.replace(problem_statement, "", 1)
    non_issue_blob = system_prompt + "\n" + user_without_issue
    split = str(extra.get("split") or "")
    if split and re.search(
        rf"\bsplit\s*[:=]\s*{re.escape(split)}\b",
        non_issue_blob,
        flags=re.IGNORECASE,
    ):
        errors.append(f"labeled split {split!r} appeared outside problem_statement")
    for pattern in METADATA_LEAK_PATTERNS:
        if re.search(pattern, non_issue_blob, flags=re.IGNORECASE):
            errors.append(f"metadata pattern {pattern} in non-issue prompt text")
    instance_id = str(extra.get("instance_id") or "")
    if instance_id and instance_id not in problem_statement and instance_id in non_issue_blob:
        errors.append("instance_id appeared outside problem_statement")
    base_commit = str(extra.get("base_commit") or "")
    if base_commit and len(base_commit) >= 8 and base_commit in blob:
        errors.append("base_commit appeared in model-visible prompt")
    seed = extra.get("sampling_seed")
    if seed is not None and f"sampling_seed" in non_issue_blob:
        errors.append("sampling_seed key appeared in model-visible prompt")
    return errors


def scan_source_split_branches(repo_root: Path) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for relpath in SOURCE_AUDIT_RELPATHS:
        path = Path(repo_root) / relpath
        if not path.is_file():
            hits.append({"path": relpath, "error": "missing"})
            continue
        text = path.read_text(encoding="utf-8")
        for match in SPLIT_BRANCH_RE.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            line = text.splitlines()[line_no - 1].strip()
            hits.append(
                {
                    "path": relpath,
                    "line": line_no,
                    "text": line,
                    "in_prompt_builder": relpath.endswith("prompt.py"),
                    "in_agent_loop": relpath.endswith("repo_exploration.py"),
                }
            )
    prompt_hits = [item for item in hits if item.get("in_prompt_builder") or item.get("in_agent_loop")]
    return {
        "files": list(SOURCE_AUDIT_RELPATHS),
        "hits": hits,
        "n_hits": len(hits),
        "model_visible_hits": prompt_hits,
        "n_model_visible_hits": len(prompt_hits),
        "note": (
            "Logging extra_info.split into episode JSONL is allowed. "
            "A branch in prompt.py or AgentLoop run() that changes messages "
            "by split would be a confound."
        ),
    }


def non_task_identity(train_ctx: Mapping[str, Any], dev_ctx: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "system_prompt_sha256",
        "obs_tokens_limit",
        "budget_visible",
        "max_turns",
        "apply_chat_template_kwargs",
        "tools",
        "add_generation_prompt",
        "contains_think_tag",
    )
    equal = {key: train_ctx.get(key) == dev_ctx.get(key) for key in keys}
    ids_equal = list(train_ctx.get("prompt_ids") or []) == list(dev_ctx.get("prompt_ids") or [])
    messages_equal = train_ctx.get("messages") == dev_ctx.get("messages")
    return {
        "field_equal": equal,
        "system_prompt_identical": bool(equal["system_prompt_sha256"]),
        "prompt_ids_equal": ids_equal,
        "messages_equal": messages_equal,
        "train_prompt_ids_sha256": train_ctx.get("prompt_ids_sha256"),
        "dev_prompt_ids_sha256": dev_ctx.get("prompt_ids_sha256"),
        "train_prompt_token_count": train_ctx.get("prompt_token_count"),
        "dev_prompt_token_count": dev_ctx.get("prompt_token_count"),
    }


def coarsen_taxonomy(bucket: str | None, *, error_kind: Any = None) -> str | None:
    if bucket is None and error_kind not in {"protocol", "tool"}:
        return None
    if error_kind == "tool":
        if bucket == "tool_semantic_misuse":
            return "tool_semantic_misuse"
        return "other_tool_error"
    if bucket in {"multiple_actions", "framing_unbalanced_tags", "tool_semantic_misuse"}:
        return bucket
    return "other_protocol"


def compact_episode_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    events = episode_events(row)
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
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
        bucket = coarsen_taxonomy(classify_event(event), error_kind=kind)
        if bucket:
            taxonomy[bucket] += 1
    first = first_turn_event(row)
    first_invalid = bool(first is not None and event_is_invalid(first))
    first_protocol = bool(first is not None and first.get("error_kind") == "protocol")
    first_taxonomy = None
    if first is not None and first_invalid:
        first_taxonomy = coarsen_taxonomy(
            classify_event(first),
            error_kind=first.get("error_kind"),
        )
    loc = localization_score(row)
    instance_id = identity.get("instance_id") or row.get("instance_id")
    repo = identity.get("repo") or row.get("repo")
    split = identity.get("split") or row.get("split")
    return {
        "instance_id": instance_id,
        "repo": repo,
        "split": split,
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
        "termination": row.get("termination"),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    nums = [float(item) for item in values if item is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def analyze_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    metrics = [compact_episode_metrics(row) for row in rows]
    n_episodes = len(metrics)
    n_events = sum(int(item["n_events"]) for item in metrics)
    n_invalid_events = sum(int(item["n_invalid_events"]) for item in metrics)
    n_first_present = sum(1 for item in metrics if item["first_turn_present"])
    n_first_protocol = sum(1 for item in metrics if item["first_turn_protocol"])
    n_first_invalid = sum(1 for item in metrics if item["first_turn_invalid"])
    n_episode_invalid = sum(1 for item in metrics if item["invalid"])
    n_parse_ok = sum(1 for item in metrics if item["parse_ok"])
    event_taxonomy: Counter[str] = Counter()
    first_taxonomy: Counter[str] = Counter()
    for item in metrics:
        for key, count in (item.get("taxonomy") or {}).items():
            event_taxonomy[str(key)] += int(count)
        if item.get("first_turn_taxonomy"):
            first_taxonomy[str(item["first_turn_taxonomy"])] += 1
    loc_values = [item["localization_score"] for item in metrics]
    return {
        "split": split,
        "n_episodes": n_episodes,
        "n_events": n_events,
        "n_invalid_events": n_invalid_events,
        "n_first_turn_present": n_first_present,
        "episode_invalid_rate": (n_episode_invalid / n_episodes) if n_episodes else None,
        "event_invalid_rate": (n_invalid_events / n_events) if n_events else None,
        "first_turn_protocol_rate": (n_first_protocol / n_episodes) if n_episodes else None,
        "first_turn_invalid_rate": (n_first_invalid / n_episodes) if n_episodes else None,
        "parse_ok_rate": (n_parse_ok / n_episodes) if n_episodes else None,
        "mean_localization_score": _mean(loc_values),
        "mean_turn_count": _mean([float(item["n_events"]) for item in metrics]) if metrics else None,
        "event_taxonomy": {key: event_taxonomy[key] for key in TRACKED_TAXONOMY},
        "first_turn_taxonomy": {key: first_taxonomy[key] for key in TRACKED_TAXONOMY},
        "denominators": {
            "episode_invalid_rate": n_episodes,
            "event_invalid_rate": n_events,
            "first_turn_protocol_rate": n_episodes,
            "parse_ok_rate": n_episodes,
            "note": (
                "first-turn rates use one slot per episode. "
                "event_invalid_rate uses assistant-turn count."
            ),
        },
        "n_parse_ok_string_false": sum(
            1
            for row in rows
            if isinstance((row.get("localization") or {}).get("parse_ok"), str)
            and str((row.get("localization") or {}).get("parse_ok")).lower() == "false"
        ),
    }


def compare_splits(train: Mapping[str, Any], dev: Mapping[str, Any]) -> dict[str, Any]:
    def _delta(key: str) -> float | None:
        left = train.get(key)
        right = dev.get(key)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    primary = _delta("first_turn_protocol_rate")
    return {
        "event_invalid_delta_train_minus_dev": _delta("event_invalid_rate"),
        "episode_invalid_delta_train_minus_dev": _delta("episode_invalid_rate"),
        "first_turn_protocol_delta_train_minus_dev": primary,
        "parse_ok_delta_train_minus_dev": _delta("parse_ok_rate"),
        "localization_delta_train_minus_dev": _delta("mean_localization_score"),
        "thresholds": {
            "H3_causally_strengthened": (
                f"first_turn_protocol_delta >= {VERDICT_STRENGTHENED_DELTA}"
            ),
            "H3_not_reproduced": (
                f"abs(first_turn_protocol_delta) < {VERDICT_NOT_REPRODUCED_ABS}"
            ),
        },
    }


def decide_verdict(
    *,
    allow_replay: bool,
    confound_reasons: Sequence[str],
    comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not allow_replay or confound_reasons:
        return {
            "verdict": "execution_path_confound_found",
            "ambiguous_band": False,
            "primary_delta": None,
            "reasons": list(confound_reasons),
        }
    if comparison is None:
        return {
            "verdict": None,
            "ambiguous_band": False,
            "primary_delta": None,
            "reasons": ["replay comparison not available"],
        }
    delta = comparison.get("first_turn_protocol_delta_train_minus_dev")
    if delta is None:
        return {
            "verdict": None,
            "ambiguous_band": False,
            "primary_delta": None,
            "reasons": ["missing first-turn protocol delta"],
        }
    value = float(delta)
    if value >= VERDICT_STRENGTHENED_DELTA:
        verdict = "H3_causally_strengthened"
        ambiguous = False
    elif abs(value) < VERDICT_NOT_REPRODUCED_ABS:
        verdict = "H3_not_reproduced"
        ambiguous = False
    else:
        ambiguous = True
        if value >= 0:
            verdict = "H3_causally_strengthened"
        else:
            verdict = "H3_not_reproduced"
    return {
        "verdict": verdict,
        "ambiguous_band": ambiguous,
        "primary_delta": value,
        "reasons": [],
    }


def sampling_contract() -> dict[str, Any]:
    return {
        "temperature": QWEN3_SAMPLING["temperature"],
        "top_p": QWEN3_SAMPLING["top_p"],
        "top_k": QWEN3_SAMPLING["top_k"],
        "do_sample": QWEN3_SAMPLING["do_sample"],
        "n": VLLM_ROLLOUT_N,
        "validate": VALIDATE,
        "vllm_rollout_n": VLLM_ROLLOUT_N,
        "group_n": 1,
        "lora": None,
        "policy": POLICY,
        "budget_visible": BUDGET_VISIBLE,
        "obs_tokens_limit": OBS_TOKENS_LIMIT,
        "max_turns": MAX_TURNS,
        "max_new_tokens_per_turn": MAX_NEW_TOKENS_PER_TURN,
        "seed_base": SEED_BASE,
        "seed_formula": SEED_FORMULA,
        "apply_chat_template_kwargs": dict(APPLY_CHAT_TEMPLATE_KWARGS),
        "enable_thinking": None,
        "tools": None,
        "add_generation_prompt": True,
        "agent_name": AGENT_NAME,
        "agent_loop_config": AGENT_LOOP_CONFIG_RELPATH,
    }


def historical_path_notes() -> dict[str, Any]:
    return {
        "shared_generation_chain": (
            "M1E parquet → RLHFDataset(return_raw_chat=True) → DataProto "
            "raw_prompt+extra_info → RepoExplorationAgentLoop.run → "
            "build_stage1_messages → apply_chat_template(tools=None) → "
            "first generate"
        ),
        "known_unmatched_not_confound": {
            "e017_rollout_n": 4,
            "e018_vllm_rollout_n": 1,
            "e017_sampling_seed": "stripped_unseeded",
            "e018_sampling_seed": "20260827 + task_index",
            "e017_lora": "fresh_then_evolving",
            "e018_B1_lora": None,
            "e018_M_scaled_lora": "adapter_123_frozen_step_275",
            "task_pool": "train_2193 vs held_out_dev_244",
        },
        "m7c_matches": {
            "policy": POLICY,
            "lora": None,
            "rollout_n": 1,
            "sampling": dict(QWEN3_SAMPLING),
            "seed_rule": SEED_FORMULA,
            "extra_info_builder": "build_matched_extra_info (both splits)",
        },
    }


def build_execution_contract(
    *,
    tokenizer_facts: Mapping[str, Any] | None = None,
    source_scan: Mapping[str, Any] | None = None,
    subset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "experiment_id": EXPERIMENT_ID,
        "diagnostic_only": True,
        "do_not_modify_e017_e018": True,
        "dataset_class": DATASET_CLASS,
        "policy_columns": list(POLICY_COLUMNS),
        "extra_info_keys_materialized": list(EXTRA_INFO_KEYS),
        "data_source": DATA_SOURCE,
        "agent_name_in_dataset": False,
        "agent_name_runtime": AGENT_NAME,
        "reward_model_ground_truth": "instance_id (evaluator lookup only)",
        "train_parquet": TRAIN_PARQUET_RELPATH,
        "dev_parquet": DEV_PARQUET_RELPATH,
        "sampling": sampling_contract(),
        "tokenizer": dict(tokenizer_facts or {}),
        "source_scan": dict(source_scan or {}),
        "subset": dict(subset or {}),
        "historical_paths": historical_path_notes(),
        "verdict_thresholds": {
            "H3_causally_strengthened_delta": VERDICT_STRENGTHENED_DELTA,
            "H3_not_reproduced_abs": VERDICT_NOT_REPRODUCED_ABS,
        },
        "forbidden_output_ids": list(FORBIDDEN_OUTPUT_IDS),
    }


def tokenizer_facts(tokenizer: Any, tokenizer_path: str | None = None) -> dict[str, Any]:
    config_rev = None
    path = Path(tokenizer_path) if tokenizer_path else None
    if path is not None and (path / "tokenizer_config.json").is_file():
        try:
            payload = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
            config_rev = {
                "tokenizer_class": payload.get("tokenizer_class"),
                "model_max_length": payload.get("model_max_length"),
                "extra_special_tokens": payload.get("extra_special_tokens"),
                "chat_template_present": bool(payload.get("chat_template")),
            }
        except (OSError, json.JSONDecodeError):
            config_rev = {"error": "unreadable tokenizer_config.json"}
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "path": tokenizer_path,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0) or None,
        "len": len(tokenizer) if tokenizer is not None else None,
        "apply_chat_template_kwargs_effective": dict(APPLY_CHAT_TEMPLATE_KWARGS),
        "enable_thinking_in_repo": False,
        "tokenizer_config": config_rev,
    }


def load_policy_rows_by_ids(parquet_path: Path, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
    import pandas as pd

    wanted = [str(item) for item in instance_ids]
    frame = pd.read_parquet(Path(parquet_path))
    by_id: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        extra = as_mapping(record.get("extra_info"))
        instance_id = str(extra.get("instance_id") or "")
        if instance_id in wanted:
            record["extra_info"] = extra
            by_id[instance_id] = record
    missing = [item for item in wanted if item not in by_id]
    if missing:
        raise ValueError(f"instance_ids not in {parquet_path}: {missing[:8]}")
    return [by_id[item] for item in wanted]


def load_covariates(
    repo_root: Path,
    instance_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    wanted = {str(item) for item in instance_ids}
    out: dict[str, dict[str, Any]] = {item: {} for item in wanted}
    features_path = Path(repo_root) / "data/interim/swe_gym/m1d_features.jsonl"
    if features_path.is_file():
        with features_path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                instance_id = str(row.get("instance_id") or "")
                if instance_id not in wanted:
                    continue
                search_space = as_mapping(row.get("search_space"))
                target_scope = as_mapping(row.get("target_scope"))
                out[instance_id].update(
                    {
                        "repo_tracked_files": search_space.get("repo_tracked_files"),
                        "repo_python_files": search_space.get("repo_python_files"),
                        "gold_file_count_m1d": target_scope.get("base_changed_file_count"),
                        "oracle_symbol_count_m1d": target_scope.get("oracle_symbol_count"),
                        "gold_basename_mentioned": row.get("gold_basename_mentioned"),
                        "gold_full_path_mentioned": row.get("gold_full_path_mentioned"),
                    }
                )
    oracle_path = oracle_parquet_path(Path(repo_root))
    if oracle_path.is_file():
        import pandas as pd

        frame = pd.read_parquet(oracle_path)
        for record in frame.to_dict(orient="records"):
            instance_id = str(record.get("instance_id") or "")
            if instance_id not in wanted:
                continue
            files = record.get("base_changed_files")
            symbols = record.get("oracle_symbols")
            out[instance_id]["symbol_applicable"] = record.get("symbol_applicable")
            try:
                out[instance_id]["gold_file_count"] = None if files is None else len(files)
            except TypeError:
                out[instance_id]["gold_file_count"] = None
            try:
                out[instance_id]["oracle_symbol_count"] = (
                    None if symbols is None else len(symbols)
                )
            except TypeError:
                out[instance_id]["oracle_symbol_count"] = None
    return out


def subset_tasks(
    *,
    repo_root: Path,
    n: int = N_SUBSET,
) -> dict[str, Any]:
    train_ids = select_subset(
        load_ordered_ids(Path(repo_root) / TRAIN_CANDIDATES_RELPATH),
        n,
    )
    dev_ids = select_subset(
        load_ordered_ids(Path(repo_root) / DEV_TASKS_RELPATH),
        n,
    )
    train_tasks = [
        {
            "subset_index": index,
            "instance_id": instance_id,
            "split": "train",
            "sampling_seed": replay_seed(index),
            "repo": instance_id.split("__", 1)[0].replace("_", "/", 1),
        }
        for index, instance_id in enumerate(train_ids)
    ]
    dev_tasks = [
        {
            "subset_index": index,
            "instance_id": instance_id,
            "split": "dev",
            "sampling_seed": replay_seed(index),
            "repo": instance_id.split("__", 1)[0].replace("_", "/", 1),
        }
        for index, instance_id in enumerate(dev_ids)
    ]
    covariates = load_covariates(Path(repo_root), train_ids + dev_ids)
    for task in train_tasks + dev_tasks:
        extra = covariates.get(str(task["instance_id"])) or {}
        task.update(extra)
    return {
        "n": int(n),
        "selection": "frozen ordered_ids prefix; no result-based resampling",
        "train_source": TRAIN_CANDIDATES_RELPATH,
        "dev_source": DEV_TASKS_RELPATH,
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "train_ids_sha256": sha256_ids(train_ids),
        "dev_ids_sha256": sha256_ids(dev_ids),
        "train_repo_counts": dict(Counter(item["repo"] for item in train_tasks)),
        "dev_repo_counts": dict(Counter(item["repo"] for item in dev_tasks)),
        "train_tasks": train_tasks,
        "dev_tasks": dev_tasks,
        "seed_base": SEED_BASE,
        "seed_formula": SEED_FORMULA,
    }


def write_prompt_audit_sample(dest: Path, ctx: Mapping[str, Any]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "problem_statement.txt").write_text(
        str(ctx.get("problem_statement") or ""), encoding="utf-8"
    )
    (dest / "system_prompt.txt").write_text(
        str(ctx.get("system_prompt") or ""), encoding="utf-8"
    )
    (dest / "rendered.txt").write_text(str(ctx.get("rendered") or ""), encoding="utf-8")
    (dest / "decoded.txt").write_text(str(ctx.get("decoded") or ""), encoding="utf-8")
    (dest / "messages.json").write_text(
        json.dumps(ctx.get("messages") or [], indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (dest / "prompt_ids.json").write_text(
        json.dumps(
            {
                "prompt_ids": ctx.get("prompt_ids"),
                "prompt_ids_sha256": ctx.get("prompt_ids_sha256"),
                "prompt_token_count": ctx.get("prompt_token_count"),
                "system_prompt_sha256": ctx.get("system_prompt_sha256"),
                "rendered_sha256": ctx.get("rendered_sha256"),
                "split": ctx.get("split"),
                "instance_id": ctx.get("instance_id"),
                "seed": ctx.get("sampling_seed"),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def audit_prompt_path(
    *,
    repo_root: Path,
    tokenizer: Any,
    tokenizer_path: str | None,
    output_dir: Path,
    n_audit: int = N_AUDIT,
) -> dict[str, Any]:
    source_scan = scan_source_split_branches(repo_root)
    facts = tokenizer_facts(tokenizer, tokenizer_path)
    train_ids = select_subset(
        load_ordered_ids(Path(repo_root) / TRAIN_CANDIDATES_RELPATH),
        n_audit,
    )
    dev_ids = select_subset(
        load_ordered_ids(Path(repo_root) / DEV_TASKS_RELPATH),
        n_audit,
    )
    train_parquet = Path(repo_root) / TRAIN_PARQUET_RELPATH
    dev_parquet = Path(repo_root) / DEV_PARQUET_RELPATH
    train_rows = load_policy_rows_by_ids(train_parquet, train_ids)
    dev_rows = load_policy_rows_by_ids(dev_parquet, dev_ids)
    schema_errors: list[str] = []
    for row in train_rows + dev_rows:
        schema_errors.extend(policy_row_schema_errors(row))

    samples: list[dict[str, Any]] = []
    leak_errors: list[str] = []
    system_hashes: set[str] = set()
    for split, rows, ids in (
        ("train", train_rows, train_ids),
        ("dev", dev_rows, dev_ids),
    ):
        for index, (row, instance_id) in enumerate(zip(rows, ids)):
            kwargs = dataset_to_agent_kwargs(row)
            kwargs["extra_info"] = build_matched_extra_info(
                as_mapping(kwargs["extra_info"]),
                sampling_seed=replay_seed(index),
            )
            ctx = assemble_first_turn(kwargs, tokenizer)
            system_hashes.add(str(ctx["system_prompt_sha256"]))
            leaks = leakage_scan(
                rendered=str(ctx["rendered"]),
                decoded=str(ctx["decoded"]),
                system_prompt=str(ctx["system_prompt"]),
                user_prompt=str(ctx["user_prompt"]),
                problem_statement=str(ctx["problem_statement"]),
                extra_info=as_mapping(kwargs["extra_info"]),
            )
            leak_errors.extend(f"{split}/{instance_id}: {item}" for item in leaks)
            dest = output_dir / "prompt_audits" / split / str(instance_id)
            write_prompt_audit_sample(dest, ctx)
            samples.append(
                {
                    "split": split,
                    "instance_id": instance_id,
                    "seed": replay_seed(index),
                    "prompt_token_count": ctx["prompt_token_count"],
                    "prompt_ids_sha256": ctx["prompt_ids_sha256"],
                    "system_prompt_sha256": ctx["system_prompt_sha256"],
                    "contains_think_tag": ctx["contains_think_tag"],
                    "n_leak_errors": len(leaks),
                }
            )

    fake_issue = "synthetic M7C issue body that must not mention gold files"
    fake_repo = "acme/widget"
    train_row = synthetic_policy_row(
        problem_statement=fake_issue,
        repo=fake_repo,
        instance_id="acme__widget-1",
        split="train",
        index=0,
    )
    dev_row = synthetic_policy_row(
        problem_statement=fake_issue,
        repo=fake_repo,
        instance_id="acme__widget-1",
        split="dev",
        index=99,
    )
    train_ids_out = first_generation_prompt_ids(train_row, tokenizer, sampling_seed=SEED_BASE)
    dev_ids_out = first_generation_prompt_ids(dev_row, tokenizer, sampling_seed=SEED_BASE)
    train_kwargs = dataset_to_agent_kwargs(train_row)
    train_kwargs["extra_info"] = build_matched_extra_info(
        as_mapping(train_kwargs["extra_info"]),
        sampling_seed=SEED_BASE,
    )
    dev_kwargs = dataset_to_agent_kwargs(dev_row)
    dev_kwargs["extra_info"] = build_matched_extra_info(
        as_mapping(dev_kwargs["extra_info"]),
        sampling_seed=SEED_BASE,
    )
    train_ctx = assemble_first_turn(train_kwargs, tokenizer)
    dev_ctx = assemble_first_turn(dev_kwargs, tokenizer)
    identity = non_task_identity(train_ctx, dev_ctx)
    identity["prompt_ids_equal"] = train_ids_out == dev_ids_out

    confound_reasons: list[str] = []
    if schema_errors:
        confound_reasons.append(f"schema errors: {schema_errors[:8]}")
    if leak_errors:
        confound_reasons.append(f"leakage: {leak_errors[:8]}")
    if not identity["system_prompt_identical"]:
        confound_reasons.append("synthetic train/dev system prompts differ")
    if not identity["prompt_ids_equal"]:
        confound_reasons.append("synthetic train/dev first-generation prompt_ids differ")
    if not identity["messages_equal"]:
        confound_reasons.append("synthetic train/dev runtime messages differ")
    if len(system_hashes) != 1:
        confound_reasons.append(
            f"audit samples produced {len(system_hashes)} distinct system-prompt hashes"
        )
    if source_scan.get("n_model_visible_hits"):
        confound_reasons.append(
            f"split-dependent branch in prompt/AgentLoop: {source_scan['model_visible_hits']}"
        )
    allow_replay = not confound_reasons
    audit = {
        "schema_version": SCHEMA_VERSION,
        "n_audit_per_split": int(n_audit),
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "schema_errors": schema_errors,
        "leak_errors": leak_errors,
        "samples": samples,
        "system_prompt_sha256_unique": sorted(system_hashes),
        "synthetic_equivalence": identity,
        "source_scan": source_scan,
        "tokenizer": facts,
        "contains_think_tag_any": any(item.get("contains_think_tag") for item in samples),
        "allow_replay": allow_replay,
        "confound_reasons": confound_reasons,
        "gate": "stop_gpu_if_not_allow_replay",
    }
    return audit


def _md_kv_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| item | value |", "| --- | --- |"]
    for key, value in rows:
        if isinstance(value, float):
            cell = f"{value:.6f}".rstrip("0").rstrip(".")
        else:
            cell = "—" if value is None else str(value)
        lines.append(f"| {key} | {cell} |")
    return "\n".join(lines)


def render_summary(payload: Mapping[str, Any]) -> str:
    audit = payload.get("prompt_path_audit") or {}
    contract = payload.get("execution_contract") or {}
    train = payload.get("train") or {}
    dev = payload.get("dev") or {}
    comparison = payload.get("comparison") or {}
    decision = payload.get("decision") or {}
    q1 = payload.get("q1") or ""
    q2 = payload.get("q2") or ""
    lines = [
        "# M7C — Matched Train-vs-Dev Causal Replay & Prompt-Path Audit",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- status: **diagnostic only** (no parser / prompt / reward / RL change)",
        "- frozen E017 / E018 artifacts: **not modified**",
        f"- allow_replay: **{audit.get('allow_replay')}**",
        f"- verdict: **{decision.get('verdict')}**",
        f"- ambiguous_band: **{decision.get('ambiguous_band')}**",
        "",
        "This note isolates task/input distribution from execution-path confounds.",
        "It is not a mathematical causal proof.",
        "",
        "## Q1",
        "",
        "> Train 和 dev 是否通过完全相同的数据加载 / prompt construction / "
        "chat-template / AgentLoop 路径进入模型？",
        "",
        q1 or "not answered",
        "",
        "## Q2",
        "",
        "> 在 same frozen Base policy + same execution + same sampling 下，"
        "train task distribution 是否仍显著产生更高 invalid-action / "
        "first-turn protocol failure？",
        "",
        q2 or "not answered (replay not run or not scored)",
        "",
        "## Gate",
        "",
        "Do not start parser, prompt, reward, or training intervention.",
        "",
        "## 1. Prompt / data path",
        "",
        _md_kv_table(
            [
                ("dataset_class", contract.get("dataset_class")),
                ("agent_name_runtime", contract.get("agent_name_runtime")),
                ("B_obs", (contract.get("sampling") or {}).get("obs_tokens_limit")),
                ("budget_visible", (contract.get("sampling") or {}).get("budget_visible")),
                ("max_turns", (contract.get("sampling") or {}).get("max_turns")),
                ("apply_chat_template_kwargs", (contract.get("sampling") or {}).get("apply_chat_template_kwargs")),
                ("enable_thinking_in_repo", False),
                ("synthetic prompt_ids equal", (audit.get("synthetic_equivalence") or {}).get("prompt_ids_equal")),
                ("system prompt identical", (audit.get("synthetic_equivalence") or {}).get("system_prompt_identical")),
                ("n_model_visible split hits", (audit.get("source_scan") or {}).get("n_model_visible_hits")),
                ("contains_think_tag_any", audit.get("contains_think_tag_any")),
                ("n_leak_errors", len(audit.get("leak_errors") or [])),
                ("allow_replay", audit.get("allow_replay")),
            ]
        ),
        "",
        "Confound reasons:",
        "",
    ]
    reasons = audit.get("confound_reasons") or decision.get("reasons") or []
    if reasons:
        lines.extend(f"- {item}" for item in reasons)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 2. Matched Base replay",
            "",
            _md_kv_table(
                [
                    ("policy", POLICY),
                    ("lora", None),
                    ("n_train", train.get("n_episodes")),
                    ("n_dev", dev.get("n_episodes")),
                    ("train event invalid", train.get("event_invalid_rate")),
                    ("dev event invalid", dev.get("event_invalid_rate")),
                    ("train episode invalid", train.get("episode_invalid_rate")),
                    ("dev episode invalid", dev.get("episode_invalid_rate")),
                    ("train first-turn protocol", train.get("first_turn_protocol_rate")),
                    ("dev first-turn protocol", dev.get("first_turn_protocol_rate")),
                    ("train parse_ok", train.get("parse_ok_rate")),
                    ("dev parse_ok", dev.get("parse_ok_rate")),
                    ("train loc reward", train.get("mean_localization_score")),
                    ("dev loc reward", dev.get("mean_localization_score")),
                    (
                        "first-turn protocol Δ (train−dev)",
                        comparison.get("first_turn_protocol_delta_train_minus_dev"),
                    ),
                    ("thresholds", f">={VERDICT_STRENGTHENED_DELTA} / |Δ|<{VERDICT_NOT_REPRODUCED_ABS}"),
                ]
            ),
            "",
            "### First-turn taxonomy (counts)",
            "",
            _md_kv_table(
                [
                    (f"train {key}", (train.get("first_turn_taxonomy") or {}).get(key, 0))
                    for key in TRACKED_TAXONOMY
                ]
                + [
                    (f"dev {key}", (dev.get("first_turn_taxonomy") or {}).get(key, 0))
                    for key in TRACKED_TAXONOMY
                ]
            ),
            "",
            "## 3. Verdict",
            "",
            f"**{decision.get('verdict')}**",
            "",
            "Pre-registered. Do not retune the threshold after seeing numbers.",
            "If `ambiguous_band` is true, the nearest-neighbor label is reported "
            "without claiming a clean separation.",
            "",
            "## Not verified",
            "",
            "- This replay uses Base policy, not E017 late LoRA.",
            "- Subsets are round-robin prefixes, not the full 2193/244 pools.",
            "- n=64 is an existence check, not a high-power estimate.",
            "",
            "## Next step",
            "",
            "Stop. Do not start intervention. Wait for review.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(row)
    return rows


def split_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for row in rows:
        identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
        split = str(identity.get("split") or row.get("split") or "")
        if split == "train":
            train.append(dict(row))
        elif split == "dev":
            dev.append(dict(row))
    return train, dev


def analyze_replay(
    rows: Sequence[Mapping[str, Any]],
    *,
    audit: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    train_rows, dev_rows = split_rows(rows)
    train = analyze_split_rows(train_rows, split="train")
    dev = analyze_split_rows(dev_rows, split="dev")
    comparison = compare_splits(train, dev)
    decision = decide_verdict(
        allow_replay=bool(audit.get("allow_replay", True)),
        confound_reasons=list(audit.get("confound_reasons") or []),
        comparison=comparison,
    )
    if audit.get("allow_replay") is False:
        q1 = (
            "No. Prompt/data-path audit found an unexplained model-visible "
            f"train/dev difference: {audit.get('confound_reasons')}"
        )
        q2 = "Not run. GPU matched replay was gated off."
    else:
        q1 = (
            "Yes, on the M7C matched path: same RLHFDataset class, same AgentLoop, "
            "same build_stage1_messages, same apply_chat_template kwargs {}, "
            "same tokenizer. Synthetic same-issue train vs dev extra_info.split "
            "yields identical runtime messages and prompt token IDs "
            f"(sha256={((audit.get('synthetic_equivalence') or {}).get('train_prompt_ids_sha256'))}). "
            "extra_info.split is stored for logging and does not enter the prompt."
        )
        delta = comparison.get("first_turn_protocol_delta_train_minus_dev")
        q2 = (
            f"Matched Base replay first-turn protocol: train="
            f"{train.get('first_turn_protocol_rate')} vs dev="
            f"{dev.get('first_turn_protocol_rate')} (Δ={delta}). "
            f"Verdict {decision.get('verdict')} "
            f"(ambiguous_band={decision.get('ambiguous_band')})."
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_path_audit": dict(audit),
        "execution_contract": dict(contract),
        "train": train,
        "dev": dev,
        "comparison": comparison,
        "decision": decision,
        "q1": q1,
        "q2": q2,
        "taxonomy": {
            "train_event": train.get("event_taxonomy"),
            "dev_event": dev.get("event_taxonomy"),
            "train_first_turn": train.get("first_turn_taxonomy"),
            "dev_first_turn": dev.get("first_turn_taxonomy"),
            "buckets": list(TRACKED_TAXONOMY),
        },
    }
    payload["summary_markdown"] = render_summary(payload)
    return payload


__all__ = [
    "AGENT_NAME",
    "APPLY_CHAT_TEMPLATE_KWARGS",
    "EXPERIMENT_ID",
    "N_AUDIT",
    "N_SUBSET",
    "OBS_TOKENS_LIMIT",
    "SCHEMA_VERSION",
    "SEED_BASE",
    "TRACKED_TAXONOMY",
    "VALIDATE",
    "VERDICT_NOT_REPRODUCED_ABS",
    "VERDICT_STRENGTHENED_DELTA",
    "VLLM_ROLLOUT_N",
    "analyze_replay",
    "analyze_split_rows",
    "assemble_first_turn",
    "audit_prompt_path",
    "build_execution_contract",
    "build_matched_extra_info",
    "coarsen_taxonomy",
    "compact_episode_metrics",
    "compare_splits",
    "dataset_to_agent_kwargs",
    "decide_verdict",
    "default_m7c_output_dir",
    "default_trace_dir",
    "first_generation_prompt_ids",
    "forbidden_output_dir_errors",
    "iter_jsonl",
    "leakage_scan",
    "naive_bool",
    "policy_row_schema_errors",
    "replay_seed",
    "sampling_contract",
    "select_subset",
    "subset_tasks",
    "synthetic_policy_row",
    "tokenizer_facts",
    "truthy",
]
