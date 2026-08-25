"""M3B task order, paired comparison, and taxonomy (no gold in selection)."""

from __future__ import annotations

from pathlib import Path

from budget_coder_rl.eval.m3b import (
    PAIRED_SEED_BASE,
    build_manifest,
    compare_pair,
    first_pass_taxonomy,
    paired_seed,
    paired_summary,
    repo_round_robin_ids,
    select_review_cases,
    sha256_ids,
)


def test_repo_round_robin_is_deterministic_and_repo_aware():
    rows = [
        {"instance_id": "b__2", "repo": "b/b"},
        {"instance_id": "a__2", "repo": "a/a"},
        {"instance_id": "a__1", "repo": "a/a"},
        {"instance_id": "b__1", "repo": "b/b"},
        {"instance_id": "c__1", "repo": "c/c"},
    ]
    ordered = repo_round_robin_ids(rows)
    assert ordered == ["a__1", "b__1", "c__1", "a__2", "b__2"]
    assert repo_round_robin_ids(list(reversed(rows))) == ordered


def test_manifest_does_not_touch_oracle_fields():
    rows = [
        {"instance_id": f"r{i}__{j}", "repo": f"owner/r{i}"}
        for i in range(3)
        for j in range(4)
    ]
    payload = build_manifest(rows, primary_n=5, expected_n=12)
    blob = str(payload)
    assert payload["oracle_used"] is False
    assert payload["gold_used"] is False
    assert "oracle_symbols" not in blob
    assert "base_changed_files" not in blob
    assert "patch" not in blob
    assert payload["n_primary"] == 5
    assert payload["n_remainder"] == 7
    assert payload["n_repos_primary"] == 3
    assert payload["ordered_ids_sha256"] == sha256_ids(payload["ordered_ids"])
    assert payload["tasks"][0]["sampling_seed"] == PAIRED_SEED_BASE
    assert payload["tasks"][4]["sampling_seed"] == paired_seed(4)
    source = Path(__file__).resolve().parents[1] / "src" / "budget_coder_rl" / "eval" / "m3b.py"
    text = source.read_text(encoding="utf-8")
    assert "load_evaluator_oracle" not in text
    assert "oracle_parquet" not in text


def test_paired_summary_win_tie_loss():
    def row(instance_id: str, visible: bool, score: float, actions: list[str]) -> dict:
        return {
            "identity": {"instance_id": instance_id, "repo": "a/a"},
            "condition": {"budget_visible": visible, "sampling_seed": 1},
            "termination": "finish",
            "localization": {
                "parse_ok": True,
                "localization_score": score,
                "file_f1": score,
                "symbol_status": "scored",
                "symbol_f1": score,
            },
            "budget": {
                "budget_visible": visible,
                "repo_observation_tokens": 10 if not visible else 12,
                "obs_tokens_used": 10 if not visible else 12,
                "obs_tokens_limit": 8192,
                "budget_metadata_tokens": 0 if not visible else 2,
                "total_env_tokens": 10 if not visible else 14,
            },
            "tokens": {"policy_token_count": 8},
            "counts": {"n_events": len(actions), "n_protocol_errors": 0, "n_tool_errors": 0},
            "events": [{"action_name": name} for name in actions],
            "behavior": {"n_search": 1, "n_read": 1, "n_tree": 0},
        }

    rows = [
        row("x", False, 0.5, ["tree", "finish"]),
        row("x", True, 0.8, ["search", "finish"]),
        row("y", False, 0.4, ["tree", "finish"]),
        row("y", True, 0.4, ["tree", "finish"]),
        row("z", False, 0.9, ["read", "finish"]),
        row("z", True, 0.2, ["read", "finish"]),
    ]
    summary = paired_summary(rows)
    assert summary["n_completed_pairs"] == 3
    assert summary["n_visible_win"] == 1
    assert summary["n_tie"] == 1
    assert summary["n_hidden_win"] == 1
    assert summary["n_action_sequence_equal"] == 2
    cmp = compare_pair(rows[0], rows[1])
    assert cmp["winner"] == "visible"
    assert abs(cmp["delta_localization_score"] - 0.3) < 1e-9


def test_taxonomy_and_review_selection_are_deterministic():
    def episode(*, instance_id: str, visible: bool, score: float, **kwargs) -> dict:
        events = kwargs.get("events") or [
            {
                "action_name": "search",
                "action_arguments": {"query": "Nope"},
                "observation": "# bcrl-obs-v1\ntool: search\nstatus: ok\nmatch_count: 0\n---\n",
            },
            {"action_name": "finish", "action_arguments": {"locations": []}},
        ]
        return {
            "identity": {"instance_id": instance_id, "repo": "a/a"},
            "condition": {"budget_visible": visible},
            "termination": kwargs.get("termination", "finish"),
            "final_submission": kwargs.get("final_submission", {"locations": []}),
            "localization": {
                "parse_ok": kwargs.get("parse_ok", True),
                "localization_score": score,
                "file_f1": kwargs.get("file_f1", score),
                "symbol_f1": kwargs.get("symbol_f1", 0.0),
                "symbol_status": "scored",
            },
            "budget": {
                "budget_visible": visible,
                "budget_exhausted": kwargs.get("budget_exhausted", False),
                "obs_tokens_limit": 8192,
                "repo_observation_tokens": kwargs.get("used", 100),
                "obs_tokens_used": kwargs.get("used", 100),
            },
            "counts": {
                "n_events": len(events),
                "n_protocol_errors": kwargs.get("n_protocol_errors", 0),
                "n_tool_errors": 0,
            },
            "events": events,
            "behavior": {
                "n_search": sum(1 for event in events if event.get("action_name") == "search"),
                "n_read": sum(1 for event in events if event.get("action_name") == "read"),
                "n_empty_search_hits": 1,
                "n_repeated_search_queries": 0,
                "n_repeated_reads": 0,
            },
        }

    rows = []
    for index in range(6):
        rows.append(episode(instance_id=f"id{index}", visible=False, score=0.0))
        rows.append(
            episode(
                instance_id=f"id{index}",
                visible=True,
                score=0.8 if index == 0 else 0.0,
                used=8000,
            )
        )
    rows.append(
        episode(
            instance_id="exh",
            visible=False,
            score=0.0,
            termination="budget_exhausted",
            budget_exhausted=True,
        )
    )
    rows.append(
        episode(
            instance_id="exh",
            visible=True,
            score=0.0,
            termination="budget_exhausted",
            budget_exhausted=True,
        )
    )
    labels = first_pass_taxonomy(rows[0])
    assert "wrong_search_query" in labels["labels"]
    assert labels["failure_class"] == "exploration_policy"
    first = select_review_cases(rows, n_target=8, seed=20260825)
    second = select_review_cases(rows, n_target=8, seed=20260825)
    assert [item["instance_id"] for item in first] == [item["instance_id"] for item in second]
    assert any(item["reason"] == "budget_exhausted" for item in first)
