"""M3C diagnostic manifests, group stats, and offline behavior (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m3c import (
    CANDIDATE_BUDGETS,
    GROUP_N,
    GROUP_SEED_BASE,
    OVERLONG_INSTANCE_IDS,
    assign_budget_regimes,
    build_diagnostic_manifest,
    build_train_candidate_manifest,
    e001_budget_quantile_report,
    episode_behavior_flags,
    group_reward_stats,
    group_seed,
    grouped_rows,
    aggregate_group_stats,
    is_starvation,
    n8_probe_ids,
    quantile,
    select_representative_groups,
    sha256_ids,
)


def test_group_seed_formula():
    assert group_seed(0, 0) == GROUP_SEED_BASE
    assert group_seed(0, 3) == GROUP_SEED_BASE + 3
    assert group_seed(2, 1) == GROUP_SEED_BASE + 2 * 8 + 1
    assert group_seed(0, 7) == GROUP_SEED_BASE + 7


def test_diagnostic_manifest_is_repo_round_robin_without_oracle():
    rows = [
        {"instance_id": f"r{i}__{j}", "repo": f"owner/r{i}"}
        for i in range(3)
        for j in range(4)
    ]
    payload = build_diagnostic_manifest(rows, primary_n=5, expected_n=12, group_n=4)
    blob = str(payload)
    assert payload["oracle_used"] is False
    assert payload["gold_used"] is False
    assert "oracle_symbols" not in blob
    assert "base_changed_files" not in blob
    assert "patch" not in blob
    assert payload["ordered_ids"][0] == "r0__0"
    assert payload["n_primary"] == 5
    assert payload["tasks"][0]["group_seeds"][0] == GROUP_SEED_BASE
    assert payload["tasks"][1]["group_seeds"][2] == group_seed(1, 2)
    assert len(payload["tasks"][0]["group_seeds"]) == 8
    assert payload["ordered_ids_sha256"] == sha256_ids(payload["ordered_ids"])
    source = Path(__file__).resolve().parents[1] / "src" / "budget_coder_rl" / "eval" / "m3c.py"
    text = source.read_text(encoding="utf-8")
    assert "load_evaluator_oracle" not in text
    assert "oracle_parquet" not in text


def test_diagnostic_skips_overlong_without_replacement():
    overlong = "r0__0"
    rows = [
        {"instance_id": f"r{i}__{j}", "repo": f"owner/r{i}"}
        for i in range(3)
        for j in range(4)
    ]
    payload = build_diagnostic_manifest(
        rows,
        primary_n=5,
        expected_n=12,
        overlong_ids=[overlong],
    )
    assert payload["skipped_overlong"] == [overlong]
    assert payload["n_primary_runnable"] == 4
    assert payload["primary_ids"][0] == overlong
    assert payload["tasks"][0]["skipped_overlong"] is True


def test_quantile_and_e001_style_report():
    values = [float(i) for i in range(1, 11)]
    assert quantile(values, 0.0) == 1.0
    assert quantile(values, 1.0) == 10.0
    assert abs(quantile(values, 0.5) - 5.5) < 1e-12
    rows = []
    for index, used in enumerate([100, 2048, 4096, 8192]):
        rows.append(
            {
                "identity": {"instance_id": f"v{index}"},
                "condition": {"budget_visible": True},
                "termination": "finish",
                "budget": {
                    "budget_visible": True,
                    "repo_observation_tokens": used,
                    "obs_tokens_used": used,
                    "obs_tokens_limit": 8192,
                    "budget_exhausted": used == 8192,
                },
                "localization": {"localization_score": 0.0},
            }
        )
        rows.append(
            {
                "identity": {"instance_id": f"h{index}"},
                "condition": {"budget_visible": False},
                "termination": "budget_exhausted" if used == 100 else "finish",
                "budget": {
                    "budget_visible": False,
                    "repo_observation_tokens": 0 if used == 100 else used,
                    "obs_tokens_used": 0 if used == 100 else used,
                    "obs_tokens_limit": 8192,
                    "budget_exhausted": used == 100,
                },
                "localization": {"localization_score": 0.0},
            }
        )
    report = e001_budget_quantile_report(rows)
    assert report["candidates"] == list(CANDIDATE_BUDGETS)
    assert report["visible"]["n"] == 4
    assert report["visible"]["frac_ge"]["2048"] == 0.75
    assert any(item["repo_observation_tokens"] == 0 for item in report["exhausted"])


def test_group_reward_stats_flags():
    mixed = group_reward_stats([0.0, 0.0, 0.5, 0.5])
    assert mixed["mixed"] is True
    assert mixed["zero_variance"] is False
    assert mixed["all_zero"] is False
    assert mixed["distinct_count"] == 2
    zeros = group_reward_stats([0.0, 0.0, 0.0, 0.0])
    assert zeros["zero_variance"] is True
    assert zeros["all_zero"] is True
    assert zeros["mixed"] is False
    high = group_reward_stats([0.8, 0.8, 0.8, 0.8])
    assert high["all_equal_positive"] is True
    assert high["all_high"] is True


def _member(instance_id: str, group_index: int, score: float, **kwargs) -> dict:
    query = kwargs.get("query", "Foo")
    match_count = kwargs.get("match_count", 2)
    read_path = kwargs.get("read_path")
    events = [
        {
            "action_name": "search",
            "action_arguments": {"query": query, "path": "."},
            "observation": (
                "# bcrl-obs-v1\ntool: search\nstatus: ok\n"
                f"match_count: {match_count}\n---\n"
                "src/foo.py:10:class Foo\n"
            ),
        }
    ]
    if kwargs.get("repeat_search"):
        events.append(
            {
                "action_name": "search",
                "action_arguments": {"query": query, "path": "."},
                "observation": (
                    "# bcrl-obs-v1\ntool: search\nstatus: ok\n"
                    f"match_count: {match_count}\n---\n"
                    "src/foo.py:10:class Foo\n"
                ),
            }
        )
    if read_path:
        events.append(
            {
                "action_name": "read",
                "action_arguments": {"path": read_path, "start_line": 1, "end_line": 20},
            }
        )
    events.append({"action_name": "finish"})
    n_read = 1 if read_path else 0
    n_search = 2 if kwargs.get("repeat_search") else 1
    return {
        "identity": {"instance_id": instance_id, "repo": "a/a"},
        "condition": {
            "budget_visible": True,
            "obs_tokens_limit": 4096,
            "group_index": group_index,
            "sampling_seed": group_seed(0, group_index),
        },
        "group": {"group_index": group_index},
        "termination": "finish",
        "localization": {"localization_score": score, "parse_ok": True, "file_f1": score},
        "budget": {
            "repo_observation_tokens": 100,
            "obs_tokens_limit": 4096,
            "budget_visible": True,
        },
        "events": events,
        "behavior": {
            "n_search": n_search,
            "n_read": n_read,
            "n_empty_search_hits": 0 if match_count else 1,
            "n_repeated_search_queries": 1 if kwargs.get("repeat_search") else 0,
            "read_paths": [read_path] if read_path else [],
        },
    }


def test_grouped_rows_and_behavior_conversion():
    rows = [
        _member("t1", 0, 0.0, match_count=2, repeat_search=True),
        _member("t1", 1, 0.0, match_count=0),
        _member("t1", 2, 0.7, match_count=2, read_path="src/foo.py"),
        _member("t1", 3, 0.7, match_count=2, read_path="src/foo.py"),
    ]
    groups = grouped_rows(rows, group_n=4)
    assert len(groups) == 1
    assert groups[0]["complete"] is True
    stats = groups[0]["stats"]
    assert stats["mixed"] is True
    assert stats["distinct_count"] == 2
    agg = aggregate_group_stats(groups)
    assert agg["n_complete"] == 1
    assert agg["mixed_fraction"] == 1.0
    converted = episode_behavior_flags(rows[2])
    assert converted["search_to_read_after_nonempty"] is True
    assert converted["read_count"] == 1
    wasted = episode_behavior_flags(rows[0])
    assert wasted["repeated_search"] is True
    assert wasted["finish_with_zero_read"] is True
    empty = episode_behavior_flags(rows[1])
    assert empty["zero_hit_search"] is True
    selected = select_representative_groups(groups, rows, n_target=2)
    assert selected[0]["instance_id"] == "t1"
    assert selected[0]["contrast"]["hypothesis_aligned"] is True


def test_regime_assignment_starvation_vs_binding():
    starved = {
        "n_episodes": 10,
        "budget_exhaustion_rate": 0.6,
        "mean_repo_observation_tokens": 80.0,
        "n_zero_c_obs": 4,
        "mean_budget_utilization": 0.04,
        "mean_localization_score": 0.05,
    }
    binding = {
        "n_episodes": 10,
        "budget_exhaustion_rate": 0.22,
        "mean_repo_observation_tokens": 1600.0,
        "n_zero_c_obs": 0,
        "mean_budget_utilization": 0.78,
        "mean_localization_score": 0.18,
    }
    medium = {
        "n_episodes": 10,
        "budget_exhaustion_rate": 0.08,
        "mean_repo_observation_tokens": 2400.0,
        "n_zero_c_obs": 0,
        "mean_budget_utilization": 0.58,
        "mean_localization_score": 0.20,
    }
    loose = {
        "n_episodes": 10,
        "budget_exhaustion_rate": 0.01,
        "mean_repo_observation_tokens": 2700.0,
        "n_zero_c_obs": 0,
        "mean_budget_utilization": 0.33,
        "mean_localization_score": 0.21,
    }
    assert is_starvation(starved) is True
    assert is_starvation(binding) is False
    starved_regimes = assign_budget_regimes({2048: starved, 4096: medium, 8192: loose})
    assert starved_regimes["tight"] is None
    assert starved_regimes["tight_starvation"] is True
    assert starved_regimes["primary_training_B_obs"] == 4096
    binding_regimes = assign_budget_regimes({2048: binding, 4096: medium, 8192: loose})
    assert binding_regimes["tight"] == 2048
    assert binding_regimes["eval_budget_set"] == [2048, 4096, 8192]


def test_train_candidate_rule_does_not_drop_zero_variance():
    rows = [
        {"instance_id": f"r{i}__{j}", "repo": f"owner/r{i}"}
        for i in range(3)
        for j in range(4)
    ]
    eligible = [row["instance_id"] for row in rows if row["instance_id"] != "r0__3"]
    rule = (
        "Start from M1E train. Exclude overlong prompt. Require symbol_applicable. "
        "Do not drop a task solely for n=4 zero-variance. Repo-round-robin to N."
    )
    payload = build_train_candidate_manifest(
        rows,
        eligible_ids=eligible,
        skipped={"overlong": [], "symbol_unavailable": ["r0__3"]},
        rule_text=rule,
        target_n=6,
        expected_n=12,
    )
    assert payload["gold_used_for_cherry_pick"] is False
    assert payload["zero_variance_used_as_drop"] is False
    assert payload["n_selected"] == 6
    assert "r0__3" not in payload["ordered_ids"]
    assert payload["ordered_ids_sha256"] == sha256_ids(payload["ordered_ids"])


def test_n8_probe_prefers_mixed_then_zero_variance():
    groups = [
        {
            "instance_id": "z",
            "complete": True,
            "stats": {"mixed": False, "zero_variance": True},
        },
        {
            "instance_id": "m",
            "complete": True,
            "stats": {"mixed": True, "zero_variance": False},
        },
        {
            "instance_id": "a",
            "complete": True,
            "stats": {"mixed": False, "zero_variance": False},
        },
    ]
    assert n8_probe_ids(groups, n_target=2) == ["m", "z"]


def test_m3c_yaml_matches_frozen_training_scaffold():
    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[1]
    yaml_path = repo / "configs" / "agent" / "repo_exploration.yaml"
    configs = OmegaConf.load(str(yaml_path))
    assert configs[0].name == "repo_exploration"
    assert configs[0].budget_visible is True
    assert configs[0].max_turns == 6
    assert configs[0].max_new_tokens_per_turn == 2048
    assert configs[0].obs_tokens_limit == 4096
    assert OVERLONG_INSTANCE_IDS == frozenset({"Project-MONAI__MONAI-6344"})


def test_stage1_freeze_json_contract():
    path = Path(__file__).resolve().parents[1] / "configs" / "historical" / "stage1_m3c_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "bcrl-stage1-m3c-freeze-v1"
    assert payload["not_trained"] is True
    assert payload["reward_loop_wired"] is False
    assert payload["grpo_optimizer"] is False
    assert payload["lora_update"] is False
    assert payload["primary_training_B_obs"] == 4096
    assert payload["budget_regimes"] == {"tight": 2048, "medium": 4096, "loose": 8192}
    assert payload["final_evaluation_budget_set"] == [2048, 4096, 8192]
    assert payload["budget_visible"] is True
    assert payload["budget_accounting_version"] == "bcrl-bobs-v2"
    assert payload["vllm_rollout_n"] == 1
    assert payload["proposed_grpo_rollout_n"] == 4
    assert payload["validate"] is False
    assert payload["sampling"]["temperature"] == 0.7
    assert payload["sampling"]["top_p"] == 0.8
    assert payload["sampling"]["top_k"] == 20
    assert payload["envelope"] == {
        "prompt_length": 16384,
        "response_length": 16384,
        "max_model_len": 32768,
    }
    assert payload["overlong_train_sample_policy"]["truncation"] == "error"
    assert payload["overlong_train_sample_policy"]["filter_overlong_prompts"] is False
    assert "Project-MONAI__MONAI-6344" in payload["overlong_train_sample_policy"]["excluded_instance_ids"]
    assert payload["train_candidate_manifest"]["n_selected"] == 256
    assert payload["e007_group_signal"]["grpo_signal_plausible"] is True
    assert payload["e007_group_signal"]["needs_n8_probe"] is False
