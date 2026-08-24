"""Episode JSONL assembly, behavior stats, and provenance."""

from __future__ import annotations

from pathlib import Path

from budget_coder_rl.eval.episode import (
    EPISODE_SCHEMA_VERSION,
    behavior_stats,
    build_episode_record,
    summarize_episodes,
)
from budget_coder_rl.eval.localization import evaluate_episode
from budget_coder_rl.eval.oracle import OracleRow
from budget_coder_rl.eval.provenance import collect_run_provenance, git_info


def test_behavior_stats_from_events():
    events = [
        {
            "action_name": "search",
            "action_arguments": {"query": "Foo", "path": "src"},
            "observation": "# bcrl-obs-v1\ntool: search\nstatus: ok\nmatch_count: 0\n---\n",
            "error_kind": None,
        },
        {
            "action_name": "search",
            "action_arguments": {"query": "Foo", "path": "src"},
            "observation": "# bcrl-obs-v1\ntool: search\nstatus: ok\nmatch_count: 2\n---\n",
            "error_kind": None,
        },
        {
            "action_name": "read",
            "action_arguments": {"path": "src/foo.py", "start_line": 1, "end_line": 2},
            "error_kind": None,
        },
    ]
    stats = behavior_stats(events)
    assert stats["n_search"] == 2
    assert stats["n_empty_search_hits"] == 1
    assert stats["n_repeated_search_queries"] == 1
    assert stats["unique_read_paths"] == 1
    assert stats["read_paths"] == ["src/foo.py"]


def test_build_episode_record_joins_localization_without_gold_lists():
    extra = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "split": "train",
        "termination": "finish",
        "final_submission": {"locations": [{"path": "src/foo.py"}]},
        "budget_visible": False,
        "obs_tokens_limit": 100,
        "obs_tokens_used": 40,
        "obs_tokens_remaining": 60,
        "budget_exhausted": False,
        "prompt_token_count": 10,
        "policy_token_count": 5,
        "observation_token_count": 40,
        "tool_observation_token_count": 40,
        "max_turns": 6,
        "events": [
            {
                "turn": 1,
                "action_name": "tree",
                "action_arguments": {"path": "."},
                "observation": "# bcrl-obs-v1\nsecret-should-compact\n",
                "observation_preview": "preview",
                "inserted": True,
            }
        ],
        "segments": [
            {"kind": "assistant", "token_ids": [1, 2]},
            {"kind": "observation", "token_ids": [3] * 40},
        ],
        "trace_role": "research_debug_not_training_tokens",
    }
    oracle = OracleRow(
        instance_id="owner__repo-1",
        split="train",
        base_changed_files=("src/foo.py",),
        oracle_symbols=(),
        symbol_applicable=False,
    )
    loc = evaluate_episode(
        termination="finish",
        submission=extra["final_submission"],
        oracle=oracle,
    ).as_dict()
    record = build_episode_record(extra, localization=loc, provenance={"k": 1})
    assert record["schema_version"] == EPISODE_SCHEMA_VERSION
    assert record["localization"]["symbol_status"] == "unavailable"
    assert record["localization"]["localization_score"] == 1.0
    assert "src/foo.py" not in str(record["localization"])
    assert "observation" not in record["events"][0]
    assert record["events"][0]["observation_preview"] == "preview"
    assert "secret-should-compact" not in str(record["events"])
    summary = summarize_episodes([record])
    assert summary["n_episodes"] == 1
    assert summary["n_finish"] == 1
    assert summary["n_symbol_unavailable"] == 1


def test_provenance_records_git_commit(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    info = git_info(repo_root)
    assert info["commit"]
    assert len(info["commit"]) >= 7
    payload = collect_run_provenance(
        repo_root,
        agent_loop_config=repo_root / "configs" / "agent_loop" / "repo_exploration_m3a.yaml",
        model_path="unused",
    )
    assert payload["budget_coder_rl"]["commit"] == info["commit"]
    assert payload["m1e_dataset_manifest"]["sha256"]
    assert payload["agent_loop_config"]["sha256"]
    assert "dirty" in payload["budget_coder_rl"]
