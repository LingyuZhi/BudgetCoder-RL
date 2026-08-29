"""M6 frozen held-out-task eval contract, pairing, bootstrap, case ranking."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.m3b import load_manifest
from budget_coder_rl.eval.m6 import (
    BUDGETS,
    CASE_RULES,
    EVAL_NAME,
    EXPECTED_EVAL_SHA256,
    EXPECTED_M3C_FREEZE_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    N_DEV_TASKS,
    PAIRED_SEED_BASE,
    PRIMARY_TRAINING_BUDGET,
    bootstrap_mean_ci,
    build_policy_extra_info,
    case_candidates,
    checkpoint_path_errors,
    compare_pair,
    condition_id_from_row,
    contract_errors,
    eval_seed,
    extra_info_leakage_errors,
    iter_eval_jobs,
    jobs_for_phase,
    load_correlation_groups,
    load_eval_contract,
    load_tasks,
    lock_errors,
    main_table,
    paired_cells,
    paired_summary_stats,
    per_repo_breakdown,
    select_case_studies,
)
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _episode(
    instance_id: str,
    *,
    condition_id: str,
    budget: int,
    score: float,
    repo: str = "pandas-dev/pandas",
    parse_ok: bool = True,
    invalid: bool = False,
    seed: int = 1,
) -> dict:
    visible = condition_id != "B0"
    policy = "rl" if condition_id == "M1" else "base"
    return {
        "identity": {"instance_id": instance_id, "repo": repo, "split": "dev"},
        "condition": {
            "condition_id": condition_id,
            "policy": policy,
            "budget_visible": visible,
            "obs_tokens_limit": budget,
            "sampling_seed": seed,
        },
        "termination": "finish" if parse_ok else "max_turns",
        "localization": {
            "parse_ok": parse_ok,
            "localization_score": score,
            "file_f1": score,
            "symbol_status": "scored",
            "symbol_f1": score,
            "n_gold_files": 2,
        },
        "budget": {
            "budget_visible": visible,
            "obs_tokens_limit": budget,
            "repo_observation_tokens": 1000 + int(score * 10),
            "obs_tokens_used": 1000,
            "budget_exhausted": False,
        },
        "counts": {
            "n_events": 4,
            "n_protocol_errors": 1 if invalid else 0,
            "n_tool_errors": 0,
        },
        "tokens": {"policy_token_count": 80},
        "behavior": {"n_search": 2, "n_read": 1},
        "events": [
            {
                "turn": 1,
                "action_name": "search",
                "action_arguments": {"query": "foo" if condition_id != "M1" else "Foo.bar"},
            },
            {
                "turn": 2,
                "action_name": "read",
                "action_arguments": {"path": "src/foo.py", "start_line": 1, "end_line": 20},
            },
        ],
        "final_submission": {"locations": [{"path": "src/foo.py", "symbol": "Foo.bar"}]},
    }


def test_eval_contract_and_lock():
    payload = load_eval_contract(REPO_ROOT)
    assert contract_errors(payload, repo_root=REPO_ROOT) == []
    assert lock_errors(REPO_ROOT) == []
    assert sha256_file(REPO_ROOT / "configs/historical/stage1_m6_eval.json") == EXPECTED_EVAL_SHA256
    assert payload["eval_name"] == EVAL_NAME
    assert "held-out-repo" not in payload["eval_name"].lower()
    assert payload["not_held_out_repository_test"] is True
    assert payload["n_tasks"] == N_DEV_TASKS
    assert tuple(payload["budgets"]) == BUDGETS
    assert payload["canonical_rl_step"] == 32
    assert payload["evaluate_intermediate_checkpoints"] is False
    assert payload["historical_runs_in_main_table"] is False
    assert payload["case_selection"]["hand_pick_pretty_examples"] is False
    freeze = REPO_ROOT / "configs/historical/stage1_m3c_freeze.json"
    assert sha256_file(freeze) == EXPECTED_M3C_FREEZE_SHA256


def test_task_set_matches_frozen_m3b_dev():
    manifest = load_manifest(REPO_ROOT / "data/manifests/m3b_baseline_tasks.json")
    assert manifest["ordered_ids_sha256"] == EXPECTED_ORDERED_IDS_SHA256
    tasks = load_tasks(REPO_ROOT)
    assert len(tasks) == 244
    assert tasks[0]["sampling_seed"] == PAIRED_SEED_BASE
    assert tasks[4]["sampling_seed"] == eval_seed(4)
    jobs = iter_eval_jobs(tasks)
    assert len(jobs) == 244 * 3 * 3
    same_task = [job for job in jobs if job["instance_id"] == tasks[0]["instance_id"]]
    seeds = {job["sampling_seed"] for job in same_task}
    assert seeds == {tasks[0]["sampling_seed"]}
    assert {job["condition_id"] for job in same_task} == {"B0", "B1", "M1"}
    groups = load_correlation_groups(REPO_ROOT)
    assert len(groups) == 244
    assert len(set(groups.values())) == 211


def test_extra_info_must_not_carry_oracle():
    clean = {
        "instance_id": "a__1",
        "repo": "a/a",
        "budget_visible": True,
        "obs_tokens_limit": 4096,
        "sampling_seed": 1,
        "condition_id": "B1",
        "policy": "base",
    }
    assert extra_info_leakage_errors(clean) == []
    dirty = dict(clean)
    dirty["oracle_symbols"] = ["x"]
    assert extra_info_leakage_errors(dirty)
    dirty2 = dict(clean)
    dirty2["correlation_group_id"] = "cg:a"
    assert extra_info_leakage_errors(dirty2)


def test_paired_win_tie_and_cluster_bootstrap():
    rows = []
    for index, instance_id in enumerate(["a__1", "a__2", "b__1"]):
        repo = "a/a" if instance_id.startswith("a") else "b/b"
        rows.append(_episode(instance_id, condition_id="B1", budget=4096, score=0.1, repo=repo, seed=index))
        rows.append(_episode(instance_id, condition_id="M1", budget=4096, score=0.6 if index < 2 else 0.1, repo=repo, seed=index))
        rows.append(_episode(instance_id, condition_id="B0", budget=4096, score=0.2, repo=repo, seed=index))
    assert condition_id_from_row(rows[0]) == "B1"
    pairs = paired_cells(rows, left_id="B1", right_id="M1", budget=4096)
    assert len(pairs) == 3
    stats = paired_summary_stats(
        pairs,
        left_id="B1",
        right_id="M1",
        group_ids={"a__1": "g1", "a__2": "g1", "b__1": "g2"},
        n_boot=200,
        seed=20260827,
    )
    assert stats["n_M1_win"] == 2
    assert stats["n_tie"] == 1
    assert stats["n_B1_win"] == 0
    assert abs(stats["mean_delta"] - (0.5 + 0.5 + 0.0) / 3) < 1e-9
    assert stats["bootstrap"]["low"] is not None
    assert stats["bootstrap"]["high"] is not None
    assert stats["bootstrap"]["method"] == "correlation_group_cluster"
    ci_again = bootstrap_mean_ci(
        pairs,
        group_ids={"a__1": "g1", "a__2": "g1", "b__1": "g2"},
        n_boot=200,
        seed=20260827,
    )
    assert ci_again == stats["bootstrap"]
    vis = paired_cells(rows, left_id="B0", right_id="B1", budget=4096)
    vis_stats = paired_summary_stats(vis, left_id="B0", right_id="B1", group_ids={})
    assert vis_stats["n_pairs"] == 3
    repos = per_repo_breakdown(pairs)
    assert {item["repo"] for item in repos} == {"a/a", "b/b"}
    bokeh_like = [item for item in repos if item["n"] < 8]
    assert all(item["significance_ok"] is False for item in bokeh_like)


def test_case_ranking_is_programmatic_and_cross_repo():
    pairs = []
    specs = [
        ("rl/r1", "repo/one", 0.0, 0.8),
        ("rl/r2", "repo/two", 0.05, 0.7),
        ("base/b1", "repo/three", 0.8, 0.0),
        ("base/b2", "repo/one", 0.9, 0.1),
        ("fail/f1", "repo/four", 0.0, 0.0),
        ("fail/f2", "repo/five", 0.05, 0.0),
    ]
    rows = []
    for instance_id, repo, b1, m1 in specs:
        rows.append(_episode(instance_id, condition_id="B1", budget=4096, score=b1, repo=repo))
        rows.append(_episode(instance_id, condition_id="M1", budget=4096, score=m1, repo=repo, parse_ok=m1 > 0))
    pairs = paired_cells(rows, left_id="B1", right_id="M1", budget=4096)
    pools = case_candidates(pairs, rules=CASE_RULES)
    assert pools["rl_win"]
    assert pools["base_win"]
    assert pools["both_fail"]
    selected = select_case_studies(pairs, rules=CASE_RULES)
    chosen = selected["selected"]
    assert chosen["rl_win"]["instance_id"] == "rl/r1"
    assert chosen["base_win"] is not None
    assert chosen["both_fail"] is not None
    repos = {
        chosen["rl_win"]["repo"],
        chosen["base_win"]["repo"],
        chosen["both_fail"]["repo"],
    }
    assert len(repos) == 3
    again = select_case_studies(pairs, rules=CASE_RULES)
    assert again["selected"]["rl_win"]["instance_id"] == chosen["rl_win"]["instance_id"]
    assert again["selected"]["base_win"]["instance_id"] == chosen["base_win"]["instance_id"]


def test_main_table_has_nine_cells():
    rows = []
    for condition_id in ("B0", "B1", "M1"):
        for budget in BUDGETS:
            rows.append(
                _episode("x__1", condition_id=condition_id, budget=budget, score=0.2)
            )
    table = main_table(rows)
    assert len(table) == 9
    assert {item["condition_id"] for item in table} == {"B0", "B1", "M1"}


def test_contract_rejects_intermediate_checkpoints():
    payload = json.loads(
        (REPO_ROOT / "configs/historical/stage1_m6_eval.json").read_text(encoding="utf-8")
    )
    payload["evaluate_intermediate_checkpoints"] = True
    payload["canonical_rl_step"] = 24
    errors = contract_errors(payload, repo_root=None)
    assert any("intermediate" in item or "canonical_rl_step" in item for item in errors)


def test_compare_pair_names():
    left = _episode("z__1", condition_id="B1", budget=2048, score=0.2)
    right = _episode("z__1", condition_id="M1", budget=2048, score=0.4)
    cmp_ = compare_pair(left, right, left_name="B1", right_name="M1")
    assert cmp_["winner"] == "M1"
    assert abs(cmp_["delta_localization_score"] - 0.2) < 1e-9


def test_policy_extra_info_and_forbidden_checkpoints():
    extra = build_policy_extra_info(
        {
            "instance_id": "a__1",
            "repo": "a/a",
            "base_commit": "abc",
            "split": "dev",
            "oracle_symbols": ["leak"],
            "correlation_group_id": "cg:a",
        },
        {
            "condition_id": "B1",
            "policy": "base",
            "budget_visible": True,
            "obs_tokens_limit": 4096,
            "sampling_seed": 20260827,
        },
    )
    assert extra_info_leakage_errors(extra) == []
    assert "oracle_symbols" not in extra
    assert "correlation_group_id" not in extra
    assert extra["condition_id"] == "B1"
    assert extra["obs_tokens_limit"] == 4096
    assert checkpoint_path_errors("/tmp/global_step_32/actor") == []
    assert checkpoint_path_errors("/tmp/global_step_8/actor")
    assert checkpoint_path_errors("/tmp/global_step_16/actor")
    assert checkpoint_path_errors("/tmp/global_step_24/actor")
    tasks = load_tasks(REPO_ROOT)
    smoke = jobs_for_phase(tasks, "smoke")
    assert len(smoke) == 2 * 3 * 1
    assert {job["obs_tokens_limit"] for job in smoke} == {PRIMARY_TRAINING_BUDGET}
    assert {job["condition_id"] for job in smoke} == {"B0", "B1", "M1"}
    assert len(jobs_for_phase(tasks, "base")) == 244 * 2 * 3
    assert len(jobs_for_phase(tasks, "rl")) == 244 * 1 * 3
    assert len(jobs_for_phase(tasks, "all")) == 244 * 3 * 3
