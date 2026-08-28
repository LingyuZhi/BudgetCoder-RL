"""M7D trainer-rollout-path forensic helpers.

Diagnostic only. Does not modify parser, prompt, reward, AgentLoop, or
frozen E017/E018/M7C artifacts. Does not train or call optimizer.step.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from budget_coder_rl.data.swe_gym_materialize import TRAIN_PARQUET_RELPATH
from budget_coder_rl.data.swe_gym_repos import bcrl_data_root
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING, sha256_ids
from budget_coder_rl.eval.m4a import load_json
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID, VLLM_LORA_NAME
from budget_coder_rl.eval.m5a import default_output_dir
from budget_coder_rl.eval.m6 import extra_info_leakage_errors
from budget_coder_rl.eval.m7a import (
    classify_event,
    episode_events,
    episode_is_invalid,
    episode_parse_ok,
    event_is_invalid,
    localization_score,
)
from budget_coder_rl.eval.m7b import first_turn_event
from budget_coder_rl.eval.m7c import (
    AGENT_LOOP_CONFIG_RELPATH,
    AGENT_NAME,
    APPLY_CHAT_TEMPLATE_KWARGS,
    BUDGET_VISIBLE,
    DATASET_CLASS,
    MAX_NEW_TOKENS_PER_TURN,
    MAX_TURNS,
    OBS_TOKENS_LIMIT,
    POLICY,
    PRIVILEGED_EXTRA_KEYS,
    TRAIN_CANDIDATES_RELPATH,
    VALIDATE,
    VLLM_ROLLOUT_N,
    as_mapping,
    assemble_first_turn,
    dataset_to_agent_kwargs,
    load_ordered_ids,
    load_policy_rows_by_ids,
    select_subset,
    sha256_text,
    sha256_token_ids,
    synthetic_policy_row,
)

SCHEMA_VERSION = "bcrl-m7d-v1"
MILESTONE = "M7D"
EXPERIMENT_ID = "M7D"
CONFIG_RELPATH = "configs/experiments/stage1_m7d.json"
TRAJECTORY_RELPATH = "trajectories/m7d/M7D"
N_SUBSET = 16
GROUP_N = 4
SEED_POLICY = "e017_unseeded"
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET = "all-linear"
E017_CHECKPOINT_MARKER = "stage1_m5_scaled_e017"
LOW_FIRST_TURN = 0.10
HIGH_FIRST_TURN = 0.20
N_GPUS = 2
TENSOR_MODEL_PARALLEL_SIZE = 1
CELLS = ("A", "B", "C", "D")
CELL_SPECS = {
    "A": {"path": "standalone_eval", "lora": None, "group_n": 1},
    "B": {"path": "trainer_rollout", "lora": None, "group_n": 1},
    "C": {"path": "trainer_rollout", "lora": None, "group_n": 4},
    "D": {"path": "trainer_rollout", "lora": "fresh_zero_init", "group_n": 4},
}
E017_FINGERPRINT_BUCKETS = ("multiple_actions", "framing_unbalanced_tags")
FIRST_GEN_BUCKETS = (
    "valid_single_action",
    "multiple_actions",
    "framing_unbalanced_tags",
    "extra_prose",
    "malformed_json",
    "no_action",
    "other_protocol",
)
M7A_TO_FIRST_GEN = {
    "multiple_actions": "multiple_actions",
    "framing_unbalanced_tags": "framing_unbalanced_tags",
    "surrounding_prose": "extra_prose",
    "malformed_json": "malformed_json",
    "no_recognizable_action": "no_action",
}
VERDICTS = (
    "G4_sibling_path_implicated",
    "trainer_rollout_path_implicated",
    "fresh_lora_runtime_implicated",
    "first_request_divergence_found",
    "E017_historical_pathology_not_reproduced",
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
    "M7C",
)
GET_GEN_BATCH_REWARD_KEYS = ("data_source", "reward_model", "extra_info", "uid")
EXPECTED_SAMPLING = {
    "temperature": QWEN3_SAMPLING["temperature"],
    "top_p": QWEN3_SAMPLING["top_p"],
    "top_k": QWEN3_SAMPLING["top_k"],
    "do_sample": True,
    "validate": False,
    "n": 1,
}


def default_m7d_output_dir(repo_root: Path) -> Path:
    return default_output_dir(Path(repo_root), EXPERIMENT_ID)


def default_trace_dir(data_root: Path | None = None) -> Path:
    return Path(data_root or bcrl_data_root()) / TRAJECTORY_RELPATH


def default_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / CONFIG_RELPATH


def forbidden_output_dir_errors(output_dir: Path, repo_root: Path) -> list[str]:
    resolved = Path(output_dir).resolve()
    errors: list[str] = []
    for experiment_id in FORBIDDEN_OUTPUT_IDS:
        forbidden = (Path(repo_root) / "outputs" / "experiments" / experiment_id).resolve()
        if resolved == forbidden:
            errors.append(
                f"refusing to write into {experiment_id} artifact directory {forbidden}"
            )
    if E017_CHECKPOINT_MARKER in str(resolved):
        errors.append(f"refusing to write into E017 checkpoint path {resolved}")
    return errors


def object_array(values: Sequence[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array


def build_unseeded_extra_info(
    source: Mapping[str, Any],
    *,
    obs_tokens_limit: int = OBS_TOKENS_LIMIT,
    budget_visible: bool = BUDGET_VISIBLE,
) -> dict[str, Any]:
    """E017-faithful extra_info: budget knobs, no sampling_seed."""
    extra = dict(source)
    for key in PRIVILEGED_EXTRA_KEYS:
        extra.pop(key, None)
    extra["budget_visible"] = bool(budget_visible)
    extra["obs_tokens_limit"] = int(obs_tokens_limit)
    extra.pop("sampling_seed", None)
    extra.pop("condition_id", None)
    extra.pop("policy", None)
    leaks = extra_info_leakage_errors(extra)
    if leaks:
        raise ValueError(f"policy extra_info leaked privileged fields: {leaks}")
    return extra


def first_generation_prompt_ids_unseeded(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    obs_tokens_limit: int = OBS_TOKENS_LIMIT,
    budget_visible: bool = BUDGET_VISIBLE,
) -> list[int]:
    kwargs = dataset_to_agent_kwargs(row)
    kwargs["extra_info"] = build_unseeded_extra_info(
        as_mapping(kwargs.get("extra_info")),
        obs_tokens_limit=int(obs_tokens_limit),
        budget_visible=bool(budget_visible),
    )
    ctx = assemble_first_turn(kwargs, tokenizer)
    return list(ctx["prompt_ids"])


def sampling_contract(*, group_n: int = 1, lora: str | None = None) -> dict[str, Any]:
    return {
        "temperature": QWEN3_SAMPLING["temperature"],
        "top_p": QWEN3_SAMPLING["top_p"],
        "top_k": QWEN3_SAMPLING["top_k"],
        "do_sample": QWEN3_SAMPLING["do_sample"],
        "n": VLLM_ROLLOUT_N,
        "validate": VALIDATE,
        "vllm_rollout_n": VLLM_ROLLOUT_N,
        "group_n": int(group_n),
        "lora": lora,
        "policy": POLICY,
        "budget_visible": BUDGET_VISIBLE,
        "obs_tokens_limit": OBS_TOKENS_LIMIT,
        "max_turns": MAX_TURNS,
        "max_new_tokens_per_turn": MAX_NEW_TOKENS_PER_TURN,
        "seed_policy": SEED_POLICY,
        "apply_chat_template_kwargs": dict(APPLY_CHAT_TEMPLATE_KWARGS),
        "enable_thinking": None,
        "tools": None,
        "add_generation_prompt": True,
        "agent_name": AGENT_NAME,
        "agent_loop_config": AGENT_LOOP_CONFIG_RELPATH,
        "dataset_class": DATASET_CLASS,
    }


def canonicalize_sampling_params(
    params: Mapping[str, Any] | None,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    payload = dict(params or {})
    n_value = payload.get("n")
    if n_value is None:
        n_value = 1
    recorded = {
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "top_k": payload.get("top_k"),
        "n": int(n_value),
        "validate": bool(payload.get("validate", False)),
        "do_sample": payload.get("do_sample"),
        "seed": payload.get("seed", seed),
        "repetition_penalty": payload.get("repetition_penalty"),
        "max_tokens": payload.get("max_tokens"),
    }
    return recorded


def extract_seed_report(record: Mapping[str, Any]) -> dict[str, Any]:
    sampling = as_mapping(record.get("sampling_params") or record.get("canonical_sampling"))
    extra = as_mapping(record.get("extra_info"))
    return {
        "extra_info_sampling_seed": extra.get("sampling_seed", record.get("sampling_seed")),
        "dataset_index": record.get("dataset_index"),
        "sibling_index": record.get("sibling_index"),
        "rollout_n": record.get("rollout_n"),
        "sampling_params_seed": sampling.get("seed", record.get("sampling_params_seed")),
        "engine_seed": record.get("engine_seed"),
        "seed_policy": record.get("seed_policy") or SEED_POLICY,
    }


def lora_runtime_metadata(
    *,
    cell: str,
    attached: bool,
    lora_int_id: int | None,
    lora_name: str | None = None,
    lora_path: str | None = None,
    listed_ids: Sequence[Any] | None = None,
    checkpoint_path: str | None = None,
    adapter_digest: str | None = None,
    lora_b_max_abs: float | None = None,
) -> dict[str, Any]:
    spec = CELL_SPECS[str(cell)]
    expect_attached = spec["lora"] is not None
    listed = [int(item) for item in list(listed_ids or []) if str(item).isdigit() or isinstance(item, int)]
    errors: list[str] = []
    if bool(attached) != bool(expect_attached):
        errors.append(
            f"cell {cell} LoRA attached={attached} expected={expect_attached}"
        )
    if expect_attached:
        if checkpoint_path and E017_CHECKPOINT_MARKER in str(checkpoint_path):
            errors.append("fresh LoRA cell must not load E017 checkpoint")
        if lora_int_id is not None and int(lora_int_id) != int(VLLM_LORA_INT_ID):
            errors.append(
                f"unexpected lora_int_id {lora_int_id}, expected {VLLM_LORA_INT_ID}"
            )
    else:
        if listed:
            errors.append(f"Base cell listed residual adapters {listed}")
        if checkpoint_path:
            errors.append("Base cell must not have a LoRA checkpoint path")
    return {
        "cell": cell,
        "lora_attached": bool(attached),
        "expect_attached": bool(expect_attached),
        "lora_int_id": lora_int_id,
        "lora_name": lora_name,
        "lora_path": lora_path,
        "listed_lora_ids": listed,
        "checkpoint_path": checkpoint_path,
        "adapter_digest": adapter_digest,
        "lora_b_max_abs": lora_b_max_abs,
        "vllm_lora_int_id_contract": int(VLLM_LORA_INT_ID),
        "vllm_lora_name_contract": VLLM_LORA_NAME,
        "errors": errors,
        "ok": not errors,
    }


def assign_logical_uids(n: int, *, factory=uuid4) -> np.ndarray:
    return np.array([str(factory()) for _ in range(int(n))], dtype=object)


def repeat_non_tensor(
    non_tensor: Mapping[str, Any],
    *,
    group_n: int,
) -> dict[str, np.ndarray]:
    """Match veRL DataProto.repeat(..., interleave=True) object-array semantics."""
    if int(group_n) <= 0:
        raise ValueError("group_n must be positive")
    repeated: dict[str, np.ndarray] = {}
    for key, value in non_tensor.items():
        array = np.asarray(value, dtype=object)
        repeated[str(key)] = np.repeat(array, int(group_n), axis=0)
    return repeated


def apply_get_gen_batch_semantics(
    non_tensor: Mapping[str, Any],
) -> dict[str, Any]:
    """Mirror RayPPOTrainer._get_gen_batch: all non-tensor keys survive."""
    keys = set(non_tensor)
    reward_keys = set(GET_GEN_BATCH_REWARD_KEYS) & keys
    popped = {key: non_tensor[key] for key in keys if key not in reward_keys}
    popped.update({key: non_tensor[key] for key in reward_keys})
    return popped


def trajectory_info_from_index(
    index: Sequence[Any],
    *,
    step: int = 0,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Synchronous copy of veRL get_trajectory_info sibling numbering."""
    info: list[dict[str, Any]] = []
    rollout_n = 0
    values = list(index)
    for i, sample_index in enumerate(values):
        if i > 0 and values[i - 1] == sample_index:
            rollout_n += 1
        else:
            rollout_n = 0
        info.append(
            {
                "step": int(step),
                "sample_index": sample_index,
                "rollout_n": int(rollout_n),
                "validate": bool(validate),
            }
        )
    return info


def expand_trainer_siblings(
    *,
    extras: Sequence[Mapping[str, Any]],
    raw_prompts: Sequence[Any],
    indices: Sequence[Any] | None = None,
    group_n: int,
    uid_factory=uuid4,
) -> dict[str, Any]:
    n = len(extras)
    if n != len(raw_prompts):
        raise ValueError("extras/raw_prompts length mismatch")
    if indices is None:
        index_values: list[Any] = list(range(n))
    else:
        index_values = list(indices)
        if len(index_values) != n:
            raise ValueError("indices length mismatch")
    uids = assign_logical_uids(n, factory=uid_factory)
    non_tensor = {
        "extra_info": object_array([dict(item) for item in extras]),
        "raw_prompt": object_array(list(raw_prompts)),
        "uid": uids,
        "index": object_array(index_values),
        "agent_name": object_array([AGENT_NAME] * n),
    }
    after_gen = apply_get_gen_batch_semantics(non_tensor)
    expanded = repeat_non_tensor(after_gen, group_n=int(group_n))
    traj = trajectory_info_from_index(
        list(expanded["index"]),
        step=0,
        validate=False,
    )
    return {
        "n_logical": n,
        "group_n": int(group_n),
        "n_rows": n * int(group_n),
        "logical_uids": [str(item) for item in uids],
        "expanded": expanded,
        "trajectory_info": traj,
    }


def sibling_group_errors(expansion: Mapping[str, Any]) -> list[str]:
    group_n = int(expansion["group_n"])
    n_logical = int(expansion["n_logical"])
    expanded = expansion["expanded"]
    traj = list(expansion["trajectory_info"])
    errors: list[str] = []
    if int(expansion["n_rows"]) != n_logical * group_n:
        errors.append("expanded row count != n_logical * group_n")
    if len(traj) != n_logical * group_n:
        errors.append("trajectory_info length mismatch")
    uids = [str(item) for item in list(expanded["uid"])]
    extras = list(expanded["extra_info"])
    for logical in range(n_logical):
        start = logical * group_n
        chunk_uids = uids[start : start + group_n]
        if len(set(chunk_uids)) != 1:
            errors.append(f"logical {logical} uid not shared across siblings")
        expected_rollout = list(range(group_n))
        got_rollout = [int(item["rollout_n"]) for item in traj[start : start + group_n]]
        if got_rollout != expected_rollout:
            errors.append(
                f"logical {logical} rollout_n {got_rollout} != {expected_rollout}"
            )
        if extras:
            first = extras[start]
            instance = as_mapping(first).get("instance_id")
            for offset in range(group_n):
                sibling = as_mapping(extras[start + offset])
                if sibling.get("instance_id") != instance:
                    errors.append(f"logical {logical} sibling instance_id drifted")
                    break
    return errors


def probe_repeat_aliasing(
    *,
    extras: Sequence[Mapping[str, Any]],
    raw_prompts: Sequence[Any],
    group_n: int = GROUP_N,
) -> dict[str, Any]:
    extra_copies = [dict(item) for item in extras]
    prompts = list(raw_prompts)
    expansion = expand_trainer_siblings(
        extras=extra_copies,
        raw_prompts=prompts,
        group_n=int(group_n),
    )
    expanded = expansion["expanded"]
    extra_rows = list(expanded["extra_info"])
    prompt_rows = list(expanded["raw_prompt"])
    groups: list[dict[str, Any]] = []
    any_extra_alias = False
    any_prompt_alias = False
    any_nested_alias = False
    for logical, extra in enumerate(extra_copies):
        start = logical * int(group_n)
        sibling_extras = extra_rows[start : start + int(group_n)]
        sibling_prompts = prompt_rows[start : start + int(group_n)]
        extra_alias = bool(sibling_extras) and all(
            item is sibling_extras[0] for item in sibling_extras
        )
        prompt_alias = bool(sibling_prompts) and all(
            item is sibling_prompts[0] for item in sibling_prompts
        )
        nested_alias = False
        if extra_alias and sibling_extras:
            shared = sibling_extras[0]
            if isinstance(shared, dict):
                nested_alias = any(shared.get(key) is extra.get(key) for key in extra)
        any_extra_alias = any_extra_alias or extra_alias
        any_prompt_alias = any_prompt_alias or prompt_alias
        any_nested_alias = any_nested_alias or nested_alias
        groups.append(
            {
                "logical_index": logical,
                "extra_info_aliased": extra_alias,
                "raw_prompt_aliased": prompt_alias,
                "nested_value_aliased": nested_alias,
                "uid": str(expanded["uid"][start]),
            }
        )
    return {
        "group_n": int(group_n),
        "n_logical": len(extra_copies),
        "any_extra_info_aliased": any_extra_alias,
        "any_raw_prompt_aliased": any_prompt_alias,
        "any_nested_value_aliased": any_nested_alias,
        "groups": groups,
        "note": (
            "np.repeat on object arrays reuses references. This is veRL "
            "DataProto.repeat(interleave=True) semantics. M7D reports, does not fix."
        ),
    }


def map_first_generation_bucket(event: Mapping[str, Any] | None) -> str:
    if event is None or not event_is_invalid(event):
        return "valid_single_action"
    if event.get("error_kind") != "protocol":
        return "other_protocol"
    bucket = classify_event(event)
    return M7A_TO_FIRST_GEN.get(str(bucket or ""), "other_protocol")


def first_generation_from_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    first = first_turn_event(row)
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    condition = row.get("condition") if isinstance(row.get("condition"), MappingABC) else {}
    raw = None
    if first is not None:
        raw = first.get("raw_action")
    bucket = map_first_generation_bucket(first)
    protocol = bool(first is not None and first.get("error_kind") == "protocol")
    return {
        "instance_id": identity.get("instance_id") or row.get("instance_id"),
        "cell": (row.get("m7d") or {}).get("cell") if isinstance(row.get("m7d"), MappingABC) else row.get("cell"),
        "sibling_index": (row.get("m7d") or {}).get("sibling_index") if isinstance(row.get("m7d"), MappingABC) else row.get("sibling_index"),
        "uid": (row.get("m7d") or {}).get("uid") if isinstance(row.get("m7d"), MappingABC) else row.get("uid"),
        "raw_action": raw,
        "bucket": bucket,
        "first_turn_protocol": protocol,
        "first_turn_invalid": bool(first is not None and event_is_invalid(first)),
        "sampling": as_mapping(condition.get("sampling")),
        "sampling_seed": condition.get("sampling_seed"),
    }


def compact_episode_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    events = episode_events(row)
    identity = row.get("identity") if isinstance(row.get("identity"), MappingABC) else {}
    m7d = row.get("m7d") if isinstance(row.get("m7d"), MappingABC) else {}
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
        bucket = classify_event(event) or "other_protocol"
        taxonomy[bucket] += 1
    first = first_generation_from_episode(row)
    loc = localization_score(row)
    return {
        "instance_id": identity.get("instance_id") or row.get("instance_id"),
        "cell": m7d.get("cell") or row.get("cell"),
        "sibling_index": m7d.get("sibling_index"),
        "uid": m7d.get("uid"),
        "parse_ok": episode_parse_ok(row),
        "invalid": episode_is_invalid(row),
        "localization_score": loc,
        "n_events": len(events),
        "n_invalid_events": n_invalid,
        "n_protocol_events": n_protocol,
        "n_tool_error_events": n_tool,
        "taxonomy": dict(taxonomy),
        "first_turn_protocol": first["first_turn_protocol"],
        "first_turn_invalid": first["first_turn_invalid"],
        "first_bucket": first["bucket"],
        "error_row": bool(row.get("error") and "events" not in row),
    }


def analyze_cell_rows(rows: Sequence[Mapping[str, Any]], *, cell: str) -> dict[str, Any]:
    metrics = [compact_episode_metrics(row) for row in rows]
    n_episodes = len(metrics)
    n_events = sum(int(item["n_events"]) for item in metrics)
    n_invalid_events = sum(int(item["n_invalid_events"]) for item in metrics)
    n_first_protocol = sum(1 for item in metrics if item["first_turn_protocol"])
    n_invalid = sum(1 for item in metrics if item["invalid"])
    n_parse_ok = sum(1 for item in metrics if item["parse_ok"])
    n_tool = sum(int(item["n_tool_error_events"]) for item in metrics)
    first_buckets: Counter[str] = Counter(item["first_bucket"] for item in metrics)
    loc_values = [
        float(item["localization_score"])
        for item in metrics
        if item["localization_score"] is not None
    ]
    event_taxonomy: Counter[str] = Counter()
    for item in metrics:
        event_taxonomy.update(item["taxonomy"])
    n_fingerprint = int(first_buckets.get("multiple_actions") or 0) + int(
        first_buckets.get("framing_unbalanced_tags") or 0
    )
    first_rate = (n_first_protocol / n_episodes) if n_episodes else None
    return {
        "cell": cell,
        "n_episodes": n_episodes,
        "n_events": n_events,
        "n_invalid_events": n_invalid_events,
        "event_invalid_rate": (n_invalid_events / n_events) if n_events else None,
        "episode_invalid_rate": (n_invalid / n_episodes) if n_episodes else None,
        "first_turn_protocol_rate": first_rate,
        "parse_ok_rate": (n_parse_ok / n_episodes) if n_episodes else None,
        "tool_semantic_misuse_event_rate": (n_tool / n_events) if n_events else None,
        "mean_turn_count": (n_events / n_episodes) if n_episodes else None,
        "mean_localization_score": (sum(loc_values) / len(loc_values)) if loc_values else None,
        "first_generation_taxonomy": {key: int(first_buckets.get(key) or 0) for key in FIRST_GEN_BUCKETS},
        "event_taxonomy": dict(event_taxonomy),
        "e017_fingerprint_first_count": n_fingerprint,
        "denominators": {
            "first_turn_protocol_rate": n_episodes,
            "event_invalid_rate": n_events,
            "episode_invalid_rate": n_episodes,
            "parse_ok_rate": n_episodes,
        },
        "low": cell_is_low(first_rate, n_fingerprint),
        "high": cell_is_high(first_rate, n_fingerprint),
    }


def cell_is_low(first_rate: float | None, fingerprint_count: int) -> bool:
    if first_rate is None:
        return False
    return float(first_rate) < LOW_FIRST_TURN and int(fingerprint_count) == 0


def cell_is_high(first_rate: float | None, fingerprint_count: int) -> bool:
    if first_rate is None:
        return False
    return float(first_rate) >= HIGH_FIRST_TURN or int(fingerprint_count) > 0


def build_first_request_record(
    *,
    cell: str,
    logical_task_index: int,
    sibling_index: int,
    instance_id: str,
    uid: str | None,
    dataset_index: Any,
    extra_info: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    tokenizer: Any,
    sampling_params: Mapping[str, Any] | None = None,
    engine_seed: int | None = None,
    lora_meta: Mapping[str, Any] | None = None,
    model_identifier: str = POLICY,
    agent_loop_kwargs: Mapping[str, Any] | None = None,
    request_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = assemble_first_turn(kwargs, tokenizer)
    extra = as_mapping(extra_info)
    sampling = canonicalize_sampling_params(
        sampling_params or sampling_contract(group_n=CELL_SPECS[cell]["group_n"]),
        seed=None,
    )
    spec = CELL_SPECS[cell]
    record = {
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "path": spec["path"],
        "instance_id": instance_id,
        "logical_task_index": int(logical_task_index),
        "trainer_uid": uid,
        "sibling_index": int(sibling_index),
        "global_index": int(logical_task_index) * int(spec["group_n"]) + int(sibling_index),
        "dataset_index": dataset_index,
        "rollout_n": int(sibling_index),
        "sampling_seed": extra.get("sampling_seed"),
        "seed_policy": SEED_POLICY,
        "prompt_ids_sha256": ctx["prompt_ids_sha256"],
        "prompt_token_count": ctx["prompt_token_count"],
        "rendered_sha256": ctx["rendered_sha256"],
        "sampling_params": sampling,
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
        "validate": bool(sampling.get("validate")),
        "effective_n": int(sampling.get("n") or 1),
        "lora_attached": spec["lora"] is not None,
        "lora": dict(lora_meta or {}),
        "model_identifier": model_identifier,
        "engine_seed": engine_seed,
        "agent_loop_kwargs": {
            "agent_name": kwargs.get("agent_name") or AGENT_NAME,
            "obs_tokens_limit": extra.get("obs_tokens_limit"),
            "budget_visible": extra.get("budget_visible"),
            "max_turns": MAX_TURNS,
            **dict(agent_loop_kwargs or {}),
        },
        "request_kwargs": dict(request_kwargs or {}),
        "extra_info": {
            "instance_id": extra.get("instance_id"),
            "repo": extra.get("repo"),
            "split": extra.get("split"),
            "sampling_seed": extra.get("sampling_seed"),
            "obs_tokens_limit": extra.get("obs_tokens_limit"),
            "budget_visible": extra.get("budget_visible"),
        },
    }
    return record


def compare_prompt_identity(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_task: dict[str, dict[str, list[str]]] = {}
    for record in records:
        instance_id = str(record.get("instance_id") or "")
        cell = str(record.get("cell") or "")
        by_task.setdefault(instance_id, {}).setdefault(cell, []).append(
            str(record.get("prompt_ids_sha256") or "")
        )
    divergences: list[dict[str, Any]] = []
    for instance_id, cells in sorted(by_task.items()):
        hashes = {
            cell: values[0] if values else ""
            for cell, values in cells.items()
        }
        unique = set(hashes.values())
        if len(unique) > 1:
            divergences.append(
                {
                    "where": "first_request.prompt_ids_sha256",
                    "instance_id": instance_id,
                    "field": "prompt_ids_sha256",
                    "per_cell": hashes,
                    "why": "same-task prompt token IDs differ across cells before generation",
                    "intentional": False,
                    "potential_impact_on_e017": (
                        "If trainer construction changes first-generation tokens, "
                        "E017 first-turn pathology may be a request-assembly bug."
                    ),
                }
            )
        for cell, values in cells.items():
            if len(set(values)) > 1:
                divergences.append(
                    {
                        "where": "first_request.prompt_ids_sha256",
                        "instance_id": instance_id,
                        "field": f"prompt_ids_sha256[{cell}].siblings",
                        "values": values,
                        "why": "siblings of one task have different first prompt IDs",
                        "intentional": False,
                        "potential_impact_on_e017": (
                            "Sibling prompt divergence would mean G=4 is not "
                            "same-prompt group sampling."
                        ),
                    }
                )
    return {
        "n_records": len(records),
        "n_tasks": len(by_task),
        "identical": not divergences,
        "divergences": divergences,
    }


def compare_sampling_identity(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    divergences: list[dict[str, Any]] = []
    for record in records:
        for field, expected in (
            ("temperature", EXPECTED_SAMPLING["temperature"]),
            ("top_p", EXPECTED_SAMPLING["top_p"]),
            ("top_k", EXPECTED_SAMPLING["top_k"]),
            ("validate", EXPECTED_SAMPLING["validate"]),
            ("effective_n", EXPECTED_SAMPLING["n"]),
        ):
            value = record.get(field)
            if field == "validate":
                ok = bool(value) is bool(expected)
            else:
                try:
                    ok = abs(float(value) - float(expected)) < 1e-9
                except (TypeError, ValueError):
                    ok = value == expected
            if not ok:
                divergences.append(
                    {
                        "where": "first_request.sampling",
                        "instance_id": record.get("instance_id"),
                        "cell": record.get("cell"),
                        "field": field,
                        "value": value,
                        "expected": expected,
                        "why": "sampling knob differs from locked Qwen3 / vLLM n=1 contract",
                        "intentional": False,
                        "potential_impact_on_e017": (
                            "A silent sampling override could explain first-turn protocol failure."
                        ),
                    }
                )
        seed = extract_seed_report(record)
        if seed.get("extra_info_sampling_seed") is not None:
            divergences.append(
                {
                    "where": "first_request.seed",
                    "instance_id": record.get("instance_id"),
                    "cell": record.get("cell"),
                    "field": "extra_info.sampling_seed",
                    "value": seed.get("extra_info_sampling_seed"),
                    "expected": None,
                    "why": "M7D seed policy is E017-faithful unseeded",
                    "intentional": False,
                    "potential_impact_on_e017": (
                        "Seeding trainer rollouts would no longer match E017."
                    ),
                }
            )
    return {"identical": not divergences, "divergences": divergences}


def subset_tasks(*, repo_root: Path, n: int = N_SUBSET) -> dict[str, Any]:
    train_ids = select_subset(
        load_ordered_ids(Path(repo_root) / TRAIN_CANDIDATES_RELPATH),
        n,
    )
    tasks = [
        {
            "subset_index": index,
            "instance_id": instance_id,
            "split": "train",
            "sampling_seed": None,
            "repo": instance_id.split("__", 1)[0].replace("_", "/", 1),
        }
        for index, instance_id in enumerate(train_ids)
    ]
    return {
        "n": int(n),
        "selection": "frozen ordered_ids prefix; no result-based resampling",
        "train_source": TRAIN_CANDIDATES_RELPATH,
        "split": "train",
        "train_ids": train_ids,
        "train_ids_sha256": sha256_ids(train_ids),
        "train_repo_counts": dict(Counter(item["repo"] for item in tasks)),
        "train_tasks": tasks,
        "seed_policy": SEED_POLICY,
        "cells": {key: dict(value) for key, value in CELL_SPECS.items()},
    }


def representative_examples(
    outputs: Sequence[Mapping[str, Any]],
    *,
    per_bucket: int = 2,
    char_limit: int = 700,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in FIRST_GEN_BUCKETS}
    for row in outputs:
        bucket = str(row.get("bucket") or "other_protocol")
        if bucket not in grouped:
            bucket = "other_protocol"
        if len(grouped[bucket]) >= per_bucket:
            continue
        raw = row.get("raw_action")
        text = raw if isinstance(raw, str) else ""
        grouped[bucket].append(
            {
                "instance_id": row.get("instance_id"),
                "cell": row.get("cell"),
                "sibling_index": row.get("sibling_index"),
                "raw_preview": text[:char_limit],
            }
        )
    return grouped


def decide_verdict(
    *,
    request_divergences: Sequence[Mapping[str, Any]],
    cell_stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if request_divergences:
        return {
            "verdict": "first_request_divergence_found",
            "reasons": [
                f"{item.get('field')}: {item.get('why')}" for item in request_divergences
            ],
            "n_divergences": len(request_divergences),
        }
    flags = {}
    for cell in CELLS:
        stats = dict(cell_stats.get(cell) or {})
        flags[cell] = {
            "low": bool(stats.get("low")),
            "high": bool(stats.get("high")),
            "first_turn_protocol_rate": stats.get("first_turn_protocol_rate"),
            "e017_fingerprint_first_count": stats.get("e017_fingerprint_first_count"),
        }
    a, b, c, d = flags["A"], flags["B"], flags["C"], flags["D"]
    if a["low"] and b["low"] and c["high"]:
        verdict = "G4_sibling_path_implicated"
        reasons.append("A/B low, C high: G=4 sibling path implicated")
    elif a["low"] and b["high"] and c["high"]:
        verdict = "trainer_rollout_path_implicated"
        reasons.append("A low, B/C high: trainer rollout path implicated independent of G=4")
    elif a["low"] and b["low"] and c["low"] and d["high"]:
        verdict = "fresh_lora_runtime_implicated"
        reasons.append("A/B/C low, D high: fresh LoRA runtime implicated")
    elif a["low"] and b["low"] and c["low"] and d["low"]:
        verdict = "E017_historical_pathology_not_reproduced"
        reasons.append(
            "current code/config does not reproduce historical E017 first-turn pathology"
        )
    else:
        verdict = "E017_historical_pathology_not_reproduced"
        reasons.append(
            "cell pattern did not match a primary tree branch; "
            "defaulting to not-reproduced with recorded flags"
        )
        reasons.append(json.dumps(flags, ensure_ascii=True, sort_keys=True))
    return {"verdict": verdict, "reasons": reasons, "cell_flags": flags}


def build_execution_cells() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "experiment_id": EXPERIMENT_ID,
        "not_training": True,
        "no_optimizer_step": True,
        "seed_policy": SEED_POLICY,
        "sampling": sampling_contract(),
        "cells": {key: dict(value) for key, value in CELL_SPECS.items()},
        "engine_isolation": {
            "base_session_cells": ["A", "B", "C"],
            "lora_session_cells": ["D"],
            "e017_checkpoint_forbidden": True,
        },
        "thresholds": {
            "low_first_turn_protocol": LOW_FIRST_TURN,
            "high_first_turn_protocol": HIGH_FIRST_TURN,
            "e017_fingerprint_buckets": list(E017_FINGERPRINT_BUCKETS),
        },
        "verdicts": list(VERDICTS),
    }


def audit_first_requests(
    *,
    repo_root: Path,
    tokenizer: Any,
    n: int = N_SUBSET,
) -> dict[str, Any]:
    subset = subset_tasks(repo_root=repo_root, n=int(n))
    parquet = Path(repo_root) / TRAIN_PARQUET_RELPATH
    rows = load_policy_rows_by_ids(parquet, subset["train_ids"]) if parquet.is_file() else []
    records: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    raw_prompts: list[Any] = []
    synthetic_used = False
    if not rows:
        synthetic_used = True
        for task in subset["train_tasks"]:
            row = synthetic_policy_row(
                problem_statement=f"synthetic issue for {task['instance_id']}",
                repo=str(task["repo"]),
                instance_id=str(task["instance_id"]),
                split="train",
                index=int(task["subset_index"]),
            )
            rows.append(row)
    for task, row in zip(subset["train_tasks"], rows):
        kwargs = dataset_to_agent_kwargs(row)
        extra = build_unseeded_extra_info(as_mapping(kwargs.get("extra_info")))
        kwargs["extra_info"] = extra
        extras.append(extra)
        raw_prompts.append(kwargs.get("raw_prompt"))
        for cell, spec in CELL_SPECS.items():
            lora_meta = lora_runtime_metadata(
                cell=cell,
                attached=spec["lora"] is not None,
                lora_int_id=VLLM_LORA_INT_ID if spec["lora"] else None,
                listed_ids=[VLLM_LORA_INT_ID] if spec["lora"] else [],
                checkpoint_path=None,
            )
            for sibling in range(int(spec["group_n"])):
                records.append(
                    build_first_request_record(
                        cell=cell,
                        logical_task_index=int(task["subset_index"]),
                        sibling_index=sibling,
                        instance_id=str(task["instance_id"]),
                        uid=None,
                        dataset_index=extra.get("index"),
                        extra_info=extra,
                        kwargs=kwargs,
                        tokenizer=tokenizer,
                        lora_meta=lora_meta,
                    )
                )
    expansion = expand_trainer_siblings(
        extras=extras,
        raw_prompts=raw_prompts,
        indices=[item.get("index") for item in extras],
        group_n=GROUP_N,
    )
    aliasing = probe_repeat_aliasing(extras=extras, raw_prompts=raw_prompts, group_n=GROUP_N)
    expansion_errors = sibling_group_errors(expansion)
    prompt_cmp = compare_prompt_identity(records)
    sampling_cmp = compare_sampling_identity(records)
    divergences = list(prompt_cmp["divergences"]) + list(sampling_cmp["divergences"])
    allow_gpu = not divergences
    return {
        "schema_version": SCHEMA_VERSION,
        "n_tasks": subset["n"],
        "synthetic_rows": synthetic_used,
        "n_first_request_records": len(records),
        "records": records,
        "prompt_identity": prompt_cmp,
        "sampling_identity": sampling_cmp,
        "expansion_errors": expansion_errors,
        "aliasing": aliasing,
        "divergences": divergences,
        "allow_gpu": allow_gpu and not expansion_errors,
        "subset": {
            "n": subset["n"],
            "train_ids": subset["train_ids"],
            "train_ids_sha256": subset["train_ids_sha256"],
            "seed_policy": subset["seed_policy"],
        },
    }


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def split_rows_by_cell(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELLS}
    for row in rows:
        if row.get("error") and "events" not in row:
            continue
        m7d = row.get("m7d") if isinstance(row.get("m7d"), MappingABC) else {}
        cell = str(m7d.get("cell") or row.get("cell") or "")
        if cell in out:
            out[cell].append(dict(row))
    return out


def analyze_replay(
    episodes: Sequence[Mapping[str, Any]],
    *,
    first_requests: Sequence[Mapping[str, Any]],
    first_outputs: Sequence[Mapping[str, Any]] | None = None,
    aliasing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_cmp = compare_prompt_identity(first_requests)
    sampling_cmp = compare_sampling_identity(first_requests)
    request_divergences = list(prompt_cmp["divergences"]) + list(sampling_cmp["divergences"])
    by_cell = split_rows_by_cell(episodes)
    cell_stats = {
        cell: analyze_cell_rows(by_cell[cell], cell=cell) for cell in CELLS
    }
    outputs = list(first_outputs or [])
    if not outputs:
        outputs = [first_generation_from_episode(row) for row in episodes if not (row.get("error") and "events" not in row)]
    taxonomy = {
        cell: cell_stats[cell]["first_generation_taxonomy"] for cell in CELLS
    }
    examples = representative_examples(outputs)
    decision = decide_verdict(
        request_divergences=request_divergences,
        cell_stats=cell_stats,
    )
    first_divergence = None
    if request_divergences:
        first_divergence = dict(request_divergences[0])
        if aliasing and (aliasing.get("any_extra_info_aliased") or aliasing.get("any_raw_prompt_aliased")):
            first_divergence.setdefault(
                "related_aliasing",
                {
                    "extra_info": aliasing.get("any_extra_info_aliased"),
                    "raw_prompt": aliasing.get("any_raw_prompt_aliased"),
                },
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "cells": cell_stats,
        "taxonomy": taxonomy,
        "examples": examples,
        "prompt_identity": prompt_cmp,
        "sampling_identity": sampling_cmp,
        "decision": decision,
        "first_divergence": first_divergence,
        "aliasing": dict(aliasing or {}),
        "n_episodes": sum(len(by_cell[cell]) for cell in CELLS),
        "n_first_requests": len(first_requests),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _md_kv_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| item | value |", "| --- | --- |"]
    for key, value in rows:
        text = _fmt(value).replace("|", "\\|")
        lines.append(f"| {key} | {text} |")
    return "\n".join(lines)


def render_summary(payload: Mapping[str, Any]) -> str:
    decision = as_mapping(payload.get("decision"))
    cells = as_mapping(payload.get("cells"))
    audit = as_mapping(payload.get("audit") or payload.get("prompt_identity"))
    divergence = payload.get("first_divergence")
    aliasing = as_mapping(payload.get("aliasing"))
    lines = [
        "# M7D — E017 Trainer Rollout Path Forensic Closure",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        "- status: **diagnostic only** (no parser / prompt / reward / RL change)",
        "- frozen E017 / E018 / M7C artifacts: **not modified**",
        f"- verdict: **{decision.get('verdict')}**",
        f"- seed policy: `{SEED_POLICY}`",
        "",
        "M7D asks why E017 trainer rollouts showed ~0.66 first-turn protocol "
        "failure while matched standalone Base replay showed ~0.02.",
        "",
        "## Gate",
        "",
        "Stop after this SUMMARY. Do not start parser, prompt, reward, or training intervention.",
        "",
        "## Verdict",
        "",
        _md_kv_table(
            [
                ("verdict", decision.get("verdict")),
                ("prompt identity identical", (payload.get("prompt_identity") or {}).get("identical") if isinstance(payload.get("prompt_identity"), MappingABC) else audit.get("identical")),
                ("sampling identity identical", (payload.get("sampling_identity") or {}).get("identical")),
            ]
        ),
        "",
        "Reasons:",
        "",
    ]
    reasons = decision.get("reasons") or []
    if reasons:
        lines.extend(f"- {item}" for item in reasons)
    else:
        lines.append("- none")
    if decision.get("verdict") == "E017_historical_pathology_not_reproduced":
        lines.extend(
            [
                "",
                "Current code/config cannot reproduce the historical E017 first-turn "
                "pathology. E017 scientific interpretation should carry an unresolved "
                "training-rollout-integrity limitation.",
            ]
        )
    if divergence:
        lines.extend(
            [
                "",
                "## First request divergence",
                "",
                _md_kv_table(
                    [
                        ("where", as_mapping(divergence).get("where")),
                        ("field", as_mapping(divergence).get("field")),
                        ("why", as_mapping(divergence).get("why")),
                        ("intentional", as_mapping(divergence).get("intentional")),
                        (
                            "potential impact on E017",
                            as_mapping(divergence).get("potential_impact_on_e017"),
                        ),
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## First-request / aliasing",
            "",
            _md_kv_table(
                [
                    ("extra_info aliased across G=4 siblings", aliasing.get("any_extra_info_aliased")),
                    ("raw_prompt aliased across G=4 siblings", aliasing.get("any_raw_prompt_aliased")),
                    ("n_first_requests", payload.get("n_first_requests")),
                ]
            ),
            "",
            "## First-generation rates",
            "",
        ]
    )
    rate_rows = []
    for cell in CELLS:
        stats = as_mapping(cells.get(cell))
        rate_rows.append((f"{cell} n_episodes", stats.get("n_episodes")))
        rate_rows.append((f"{cell} first-turn protocol", stats.get("first_turn_protocol_rate")))
        tax = as_mapping(stats.get("first_generation_taxonomy"))
        rate_rows.append((f"{cell} multiple_actions", tax.get("multiple_actions")))
        rate_rows.append(
            (f"{cell} framing_unbalanced_tags", tax.get("framing_unbalanced_tags"))
        )
        rate_rows.append((f"{cell} event invalid", stats.get("event_invalid_rate")))
        rate_rows.append((f"{cell} episode invalid", stats.get("episode_invalid_rate")))
        rate_rows.append((f"{cell} parse_ok", stats.get("parse_ok_rate")))
        rate_rows.append((f"{cell} loc reward", stats.get("mean_localization_score")))
        rate_rows.append((f"{cell} low/high", f"{stats.get('low')}/{stats.get('high')}"))
    lines.append(_md_kv_table(rate_rows))
    examples = as_mapping(payload.get("examples"))
    if examples:
        lines.extend(["", "## Representative first-generation previews", ""])
        for bucket in E017_FINGERPRINT_BUCKETS + ("valid_single_action",):
            rows = list(examples.get(bucket) or [])
            if not rows:
                continue
            lines.append(f"### {bucket}")
            for item in rows:
                preview = str(as_mapping(item).get("raw_preview") or "").replace("\n", " ")
                lines.append(
                    f"- {as_mapping(item).get('cell')} "
                    f"{as_mapping(item).get('instance_id')} "
                    f"sib={as_mapping(item).get('sibling_index')}: `{preview[:240]}`"
                )
            lines.append("")
    traj = payload.get("trajectory_path")
    if traj:
        lines.extend(
            [
                "## Provenance",
                "",
                f"- GPU trajectories: `{traj}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Stop",
            "",
            "Do not continue invalid-action forensic expansion. Do not start intervention.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AGENT_LOOP_CONFIG_RELPATH",
    "CELL_SPECS",
    "CELLS",
    "CONFIG_RELPATH",
    "E017_CHECKPOINT_MARKER",
    "E017_FINGERPRINT_BUCKETS",
    "EXPERIMENT_ID",
    "FIRST_GEN_BUCKETS",
    "FORBIDDEN_OUTPUT_IDS",
    "GROUP_N",
    "HIGH_FIRST_TURN",
    "LORA_ALPHA",
    "LORA_RANK",
    "LORA_TARGET",
    "LOW_FIRST_TURN",
    "MILESTONE",
    "N_GPUS",
    "N_SUBSET",
    "OBS_TOKENS_LIMIT",
    "SCHEMA_VERSION",
    "SEED_POLICY",
    "TENSOR_MODEL_PARALLEL_SIZE",
    "TRAJECTORY_RELPATH",
    "VALIDATE",
    "VERDICTS",
    "VLLM_ROLLOUT_N",
    "analyze_cell_rows",
    "analyze_replay",
    "apply_get_gen_batch_semantics",
    "assign_logical_uids",
    "audit_first_requests",
    "build_execution_cells",
    "build_first_request_record",
    "build_unseeded_extra_info",
    "canonicalize_sampling_params",
    "cell_is_high",
    "cell_is_low",
    "compare_prompt_identity",
    "compare_sampling_identity",
    "compact_episode_metrics",
    "decide_verdict",
    "default_m7d_output_dir",
    "default_trace_dir",
    "expand_trainer_siblings",
    "extract_seed_report",
    "first_generation_from_episode",
    "first_generation_prompt_ids_unseeded",
    "forbidden_output_dir_errors",
    "iter_jsonl",
    "lora_runtime_metadata",
    "map_first_generation_bucket",
    "probe_repeat_aliasing",
    "render_summary",
    "repeat_non_tensor",
    "sampling_contract",
    "sibling_group_errors",
    "subset_tasks",
    "trajectory_info_from_index",
]
