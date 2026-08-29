"""GRPO sibling-expansion / DataProto.repeat grouping helpers.

Documents veRL ``np.repeat`` object-array aliasing. Does not change trainer
behavior.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

AGENT_NAME = "repo_exploration"
GROUP_N = 4
GET_GEN_BATCH_REWARD_KEYS = ("data_source", "reward_model", "extra_info", "uid")


def as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def object_array(values: Sequence[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = list(values)
    return array


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
            }
        )
    return {
        "any_extra_info_aliased": any_extra_alias,
        "any_raw_prompt_aliased": any_prompt_alias,
        "any_nested_value_aliased": any_nested_alias,
        "groups": groups,
        "n_logical": len(extra_copies),
        "group_n": int(group_n),
    }
