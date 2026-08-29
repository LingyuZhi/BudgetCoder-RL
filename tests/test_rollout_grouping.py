"""GRPO sibling-expansion / DataProto.repeat grouping contracts. CPU-only."""

from __future__ import annotations

from budget_coder_rl.train.rollout_grouping import (
    GROUP_N,
    apply_get_gen_batch_semantics,
    assign_logical_uids,
    expand_trainer_siblings,
    probe_repeat_aliasing,
    sibling_group_errors,
    trajectory_info_from_index,
)


def test_g4_sibling_expansion_count_and_uid():
    extras = [
        {"instance_id": "a", "index": 10},
        {"instance_id": "b", "index": 11},
    ]
    prompts = [[{"role": "user", "content": "issue a"}], [{"role": "user", "content": "issue b"}]]
    expansion = expand_trainer_siblings(
        extras=extras,
        raw_prompts=prompts,
        indices=[10, 11],
        group_n=GROUP_N,
    )
    assert expansion["n_rows"] == 8
    assert expansion["n_logical"] == 2
    assert sibling_group_errors(expansion) == []
    uids = [str(item) for item in expansion["expanded"]["uid"]]
    assert uids[0] == uids[1] == uids[2] == uids[3]
    assert uids[4] == uids[5] == uids[6] == uids[7]
    assert uids[0] != uids[4]
    rollouts = [item["rollout_n"] for item in expansion["trajectory_info"]]
    assert rollouts == [0, 1, 2, 3, 0, 1, 2, 3]


def test_sibling_message_objects_alias_under_np_repeat():
    extras = [{"instance_id": "a", "tags": ["keep"]}]
    prompts = [[{"role": "user", "content": "hello"}]]
    aliasing = probe_repeat_aliasing(extras=extras, raw_prompts=prompts, group_n=4)
    assert aliasing["any_extra_info_aliased"] is True
    assert aliasing["any_raw_prompt_aliased"] is True
    expansion = expand_trainer_siblings(extras=extras, raw_prompts=prompts, group_n=4)
    sibling_extras = list(expansion["expanded"]["extra_info"])
    assert sibling_extras[0] is sibling_extras[1]
    sibling_extras[0]["tags"].append("mutated")
    assert "mutated" in sibling_extras[3]["tags"]


def test_get_gen_batch_preserves_keys():
    non_tensor = {
        "raw_prompt": ["p"],
        "extra_info": [{"instance_id": "a"}],
        "uid": ["u"],
        "index": [0],
        "agent_name": ["repo_exploration"],
        "data_source": ["swe"],
        "reward_model": [{"ground_truth": "a"}],
    }
    survived = apply_get_gen_batch_semantics(non_tensor)
    assert set(survived) == set(non_tensor)
    assert "extra_info" in survived and "raw_prompt" in survived and "uid" in survived
    uids = assign_logical_uids(3)
    assert len(set(str(item) for item in uids)) == 3
    info = trajectory_info_from_index([1, 1, 1, 1], validate=False)
    assert [item["rollout_n"] for item in info] == [0, 1, 2, 3]
