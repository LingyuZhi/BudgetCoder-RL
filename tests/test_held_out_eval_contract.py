"""E018 scaled-M6 eval overlay; E015 freeze and E017 checkpoints stay immutable."""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.eval.e018 import (
    BUDGETS,
    CANONICAL_RL_STEP,
    EXPECTED_OVERLAY_SHA256,
    EXPERIMENT_ID,
    WANDB_EXPERIMENT_NAME,
    actor_dir_errors,
    case_candidates,
    checkpoint_path_errors,
    condition_id_from_row,
    consume_e018_overlay,
    default_e018_output_dir,
    default_overlay_path,
    forbidden_output_dir_errors,
    jobs_for_phase,
    latest_iteration_errors,
    main_table,
    overlay_errors,
    overlay_lock_errors,
    paired_cells,
    reliability_class,
    reuse_base_audit,
    scientific_conclusion,
    select_case_studies,
    tag_source,
    treatment_integrity_errors,
)
from budget_coder_rl.eval.m6 import (
    EXPECTED_EVAL_SHA256,
    N_DEV_TASKS,
    PAIRED_SEED_BASE,
    load_tasks,
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
    source_experiment: str = "E018",
) -> dict:
    visible = condition_id != "B0"
    policy = "rl" if condition_id in {"M_scaled", "M1", "M1_proto"} else "base"
    return {
        "identity": {"instance_id": instance_id, "repo": repo, "split": "dev"},
        "source_experiment": source_experiment,
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
                "action_arguments": {"query": "foo" if "scaled" not in condition_id else "Foo.bar"},
            }
        ],
        "final_submission": {"locations": [{"path": "src/foo.py", "symbol": "Foo.bar"}]},
    }


def test_e015_freeze_unchanged_and_overlay_is_namespace_only():
    assert sha256_file(REPO_ROOT / "configs/historical/stage1_m6_eval.json") == EXPECTED_EVAL_SHA256
    overlay_path = default_overlay_path(REPO_ROOT)
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay_errors(overlay, repo_root=REPO_ROOT) == []
    digest = sha256_file(overlay_path)
    if EXPECTED_OVERLAY_SHA256 != "0" * 64:
        assert digest == EXPECTED_OVERLAY_SHA256
    assert overlay_lock_errors(REPO_ROOT) == []
    consumed = consume_e018_overlay(repo_root=REPO_ROOT, overlay=overlay)
    assert consumed["canonical_rl_step"] == CANONICAL_RL_STEP == 275
    assert consumed["wandb_experiment_name"] == WANDB_EXPERIMENT_NAME
    assert overlay["eval_only"] is True
    assert overlay["not_training"] is True
    assert overlay["do_not_enter_m7"] is True
    assert overlay["experiment_id"] == EXPERIMENT_ID
    assert overlay["frozen_from_parent"]["paired_seed_base"] == PAIRED_SEED_BASE
    ids = [item["id"] for item in overlay["overrides"]["conditions"]]
    assert ids == ["B0", "B1", "M_scaled"]


def test_overlay_rejects_research_knob_changes():
    overlay = json.loads(default_overlay_path(REPO_ROOT).read_text(encoding="utf-8"))
    bad = json.loads(json.dumps(overlay))
    bad["overrides"]["sampling"] = {"temperature": 0.0}
    errors = overlay_errors(bad, repo_root=None)
    assert errors
    bad2 = json.loads(json.dumps(overlay))
    bad2["overrides"]["checkpoint"]["global_step"] = 256
    errors = overlay_errors(bad2, repo_root=None)
    assert any("275" in item for item in errors)
    bad3 = json.loads(json.dumps(overlay))
    bad3["do_not_enter_m7"] = False
    errors = overlay_errors(bad3, repo_root=None)
    assert any("m7" in item for item in errors)


def test_checkpoint_gate_requires_275_and_refuses_256():
    assert checkpoint_path_errors(
        "/data/x/checkpoints/stage1_m5_scaled_e017/global_step_275/actor"
    ) == []
    assert checkpoint_path_errors(
        "/data/x/checkpoints/stage1_m5_scaled_e017/global_step_256/actor"
    )
    assert checkpoint_path_errors(
        "/data/x/checkpoints/stage1_m5_e014/global_step_32/actor"
    )
    assert actor_dir_errors(
        Path("/data/x/checkpoints/stage1_m5_scaled_e017/global_step_275")
    )


def test_latest_iteration_must_be_275(tmp_path: Path):
    root = tmp_path / "stage1_m5_scaled_e017"
    actor = root / "global_step_275" / "actor"
    actor.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("256\n", encoding="utf-8")
    assert latest_iteration_errors(actor)
    (root / "latest_checkpointed_iteration.txt").write_text("275\n", encoding="utf-8")
    assert latest_iteration_errors(actor) == []


def test_forbids_historical_output_dirs():
    repo = REPO_ROOT
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E015", repo)
    assert forbidden_output_dir_errors(repo / "outputs/experiments/E017", repo)
    assert forbidden_output_dir_errors(default_e018_output_dir(repo), repo) == []


def test_jobs_and_explicit_condition_ids():
    tasks = load_tasks(REPO_ROOT)
    assert len(tasks) == N_DEV_TASKS
    smoke = jobs_for_phase(tasks, "smoke")
    assert {job["condition_id"] for job in smoke} == {"B0", "B1", "M_scaled"}
    assert {job["obs_tokens_limit"] for job in smoke} == {4096}
    assert len(smoke) == 2 * 3 * 1
    reuse_smoke = jobs_for_phase(tasks, "smoke", reuse_base=True)
    assert {job["condition_id"] for job in reuse_smoke} == {"M_scaled"}
    assert jobs_for_phase(tasks, "base", reuse_base=True) == []
    assert len(jobs_for_phase(tasks, "base")) == 244 * 2 * 3
    assert len(jobs_for_phase(tasks, "rl")) == 244 * 1 * 3
    row = _episode("a__1", condition_id="M_scaled", budget=4096, score=0.2)
    assert condition_id_from_row(row) == "M_scaled"
    proto = tag_source(
        _episode("a__1", condition_id="M1", budget=4096, score=0.2, source_experiment="E015"),
        "E015",
        condition_id="M1_proto",
    )
    assert condition_id_from_row(proto) == "M1_proto"


def test_paired_scaled_vs_b1_does_not_collapse_to_m1():
    rows = []
    for index, instance_id in enumerate(["a__1", "a__2", "b__1"]):
        repo = "a/a" if instance_id.startswith("a") else "b/b"
        rows.append(_episode(instance_id, condition_id="B1", budget=4096, score=0.1, repo=repo, seed=index))
        rows.append(
            _episode(
                instance_id,
                condition_id="M_scaled",
                budget=4096,
                score=0.6 if index < 2 else 0.1,
                repo=repo,
                seed=index,
            )
        )
    pairs = paired_cells(rows, left_id="B1", right_id="M_scaled", budget=4096)
    assert len(pairs) == 3
    assert pairs[0]["winner"] in {"M_scaled", "tie"}
    table = main_table(rows, condition_ids=("B1", "M_scaled"))
    assert {item["condition_id"] for item in table} == {"B1", "M_scaled"}


def test_case_ranking_generic_ids():
    rows = []
    specs = [
        ("rl/r1", "repo/one", 0.0, 0.8),
        ("base/b1", "repo/three", 0.8, 0.0),
        ("fail/f1", "repo/four", 0.0, 0.0),
    ]
    for instance_id, repo, b1, scaled in specs:
        rows.append(_episode(instance_id, condition_id="B1", budget=4096, score=b1, repo=repo))
        rows.append(
            _episode(
                instance_id,
                condition_id="M_scaled",
                budget=4096,
                score=scaled,
                repo=repo,
                parse_ok=scaled > 0,
            )
        )
    pairs = paired_cells(rows, left_id="B1", right_id="M_scaled", budget=4096)
    pools = case_candidates(pairs, left_id="B1", right_id="M_scaled")
    assert pools["rl_win"]
    assert pools["base_win"]
    assert pools["both_fail"]
    selected = select_case_studies(pairs, left_id="B1", right_id="M_scaled")
    assert selected["selected"]["rl_win"]["instance_id"] == "rl/r1"


def test_treatment_integrity_requires_engine_path_not_http():
    good = {
        "checkpoint_actor_dir": "/data/x/checkpoints/stage1_m5_scaled_e017/global_step_275/actor",
        "load_ok": True,
        "update_weights_ok": True,
        "listed_lora_ids": [123],
        "lora_as_adapter": True,
        "lora_request_attached": True,
        "lora_int_id": 123,
        "load_fingerprint": {"digest": "abc"},
        "sync_payload": {
            "peft_config_present": True,
            "n_adapter_tensors": 4,
            "digest": "abc",
            "lora_b_max_abs": 0.2,
            "adapter_nonzero": True,
            "tensors": {"x.lora_B": {"sha256": "abc", "max_abs": 0.2}},
        },
        "http_saw_adapter": False,
    }
    assert treatment_integrity_errors(good) == []
    bad_http_only = dict(good)
    bad_http_only["listed_lora_ids"] = []
    bad_http_only["lora_request_attached"] = False
    bad_http_only["http_saw_adapter"] = True
    assert treatment_integrity_errors(bad_http_only)
    bad_output_diff = dict(good)
    bad_output_diff["used_output_difference_as_proof"] = True
    assert any("output difference" in item for item in treatment_integrity_errors(bad_output_diff))
    bad_step = dict(good)
    bad_step["checkpoint_actor_dir"] = "/data/x/checkpoints/stage1_m5_scaled_e017/global_step_256/actor"
    assert treatment_integrity_errors(bad_step)


def test_reuse_audit_fails_closed_without_e015_scored(tmp_path: Path):
    audit = reuse_base_audit(
        REPO_ROOT,
        e015_provenance={"budget_coder_rl": {"commit": "deadbeef", "dirty": True}},
        e015_scored_path=tmp_path / "missing.jsonl",
    )
    assert audit["allow_reuse"] is False
    assert audit["decision"] == "rerun_b0_b1"


def test_conclusion_and_reliability_classes():
    def stats(mean: float, low: float, high: float) -> dict:
        return {"mean_delta": mean, "bootstrap": {"low": low, "high": high}}

    positive = {str(b): stats(0.05, 0.01, 0.09) for b in BUDGETS}
    assert scientific_conclusion(positive) == "POSITIVE"
    weak = {str(b): stats(0.01, -0.01, 0.03) for b in BUDGETS}
    assert scientific_conclusion(weak) == "WEAK POSITIVE"
    null = {str(b): stats(0.0, -0.02, 0.02) for b in BUDGETS}
    assert scientific_conclusion(null) == "NULL"
    negative = {str(b): stats(-0.04, -0.07, -0.01) for b in BUDGETS}
    assert scientific_conclusion(negative) == "NEGATIVE"

    def cells(parse_ok: float, invalid: float) -> list[dict]:
        return [
            {
                "obs_tokens_limit": budget,
                "parse_ok_rate": parse_ok,
                "invalid_tool_rate": invalid,
            }
            for budget in BUDGETS
        ]

    assert reliability_class(conclusion="POSITIVE", b1_cells=cells(0.3, 0.4), scaled_cells=cells(0.5, 0.2)) == "A"
    assert reliability_class(conclusion="POSITIVE", b1_cells=cells(0.3, 0.4), scaled_cells=cells(0.3, 0.4)) == "B"
    assert reliability_class(conclusion="NULL", b1_cells=cells(0.3, 0.4), scaled_cells=cells(0.5, 0.2)) == "C"
    assert reliability_class(conclusion="NULL", b1_cells=cells(0.3, 0.4), scaled_cells=cells(0.3, 0.4)) == "D"
    assert reliability_class(conclusion="NEGATIVE", b1_cells=cells(0.3, 0.2), scaled_cells=cells(0.2, 0.5)) == "E"
