"""AgentLoop observation-token budget + hidden/visible encoding contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from verl.workers.rollout.replica import TokenOutput

from budget_coder_rl.budget.state import BUDGET_OBS_VERSION
from budget_coder_rl.data.swe_gym_fields import POLICY_FORBIDDEN_DERIVED_FIELDS
from budget_coder_rl.protocol.observation import OBS_VERSION
from test_repo_exploration_agent_loop import (
    AGENT_LOOP_CONFIG,
    FakeServerManager,
    _build_loop,
    _encode_action,
    _final,
    _prepare_workspace,
    _resolve_tokenizer_path,
    _run,
    _tool,
)


def _obs_ids(output) -> list[list[int]]:
    return [
        list(item["token_ids"])
        for item in output.extra_fields["segments"]
        if item["kind"] == "observation"
    ]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    path = _resolve_tokenizer_path()
    if path is None:
        pytest.skip(
            "no local Qwen tokenizer found; set BCRL_TOKENIZER_PATH to a local snapshot"
        )
    return AutoTokenizer.from_pretrained(path)


def test_hidden_unlimited_matches_inserted_observation_ids(tokenizer, tmp_path: Path):
    env, extra = _prepare_workspace(tmp_path)
    tree = _tool("tree", {"path": ".", "depth": 2})
    finish = _final({"locations": [{"path": "pkg.py"}]})
    output = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(token_ids=_encode_action(tokenizer, tree)),
                    TokenOutput(token_ids=_encode_action(tokenizer, finish)),
                ]
            ),
            env,
        ),
        extra,
    )
    obs = _obs_ids(output)
    assert len(obs) == 1
    assert output.extra_fields["obs_tokens_used"] == len(obs[0])
    assert output.extra_fields["observation_token_count"] == len(obs[0])
    assert output.extra_fields["tool_observation_token_count"] == len(obs[0])
    assert output.extra_fields["obs_tokens_limit"] is None
    assert output.extra_fields["budget_visible"] is False
    assert output.extra_fields["termination"] == "finish"
    decoded = tokenizer.decode(obs[0], skip_special_tokens=True)
    assert OBS_VERSION in decoded
    assert BUDGET_OBS_VERSION not in decoded
    assert "obs_tokens_remaining" not in decoded
    assert output.extra_fields["events"][0]["inserted"] is True
    assert output.extra_fields["events"][1]["inserted"] is False


def test_accounting_equals_concat_observation_segments(tokenizer, tmp_path: Path):
    env, extra = _prepare_workspace(tmp_path)
    actions = [
        _tool("tree", {"path": ".", "depth": 2}),
        _tool("search", {"query": "version"}),
        _final({"locations": [{"path": "pkg.py", "symbol": "x"}]}),
    ]
    output = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [TokenOutput(token_ids=_encode_action(tokenizer, text)) for text in actions]
            ),
            env,
            obs_tokens_limit=50_000,
            budget_visible=False,
        ),
        extra,
    )
    obs = _obs_ids(output)
    total = sum(len(ids) for ids in obs)
    assert output.extra_fields["obs_tokens_used"] == total
    assert output.response_mask.count(0) == total
    assert output.extra_fields["policy_token_count"] == output.response_mask.count(1)
    assert extra.get("split") == output.extra_fields["split"]


def test_exact_limit_inserts_minus_one_exhausts(tokenizer, tmp_path: Path):
    env, extra = _prepare_workspace(tmp_path)
    tree = _tool("tree", {"path": ".", "depth": 2})
    finish = _final({"locations": [{"path": "pkg.py"}]})
    tree_ids = _encode_action(tokenizer, tree)
    finish_ids = _encode_action(tokenizer, finish)
    probe_out = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [TokenOutput(token_ids=list(tree_ids)), TokenOutput(token_ids=list(finish_ids))]
            ),
            env,
        ),
        extra,
    )
    cost = len(_obs_ids(probe_out)[0])

    exact_out = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [TokenOutput(token_ids=list(tree_ids)), TokenOutput(token_ids=list(finish_ids))]
            ),
            env,
            obs_tokens_limit=cost,
        ),
        extra,
    )
    assert exact_out.extra_fields["termination"] == "finish"
    assert exact_out.extra_fields["obs_tokens_used"] == cost
    assert exact_out.extra_fields["obs_tokens_remaining"] == 0
    assert len(_obs_ids(exact_out)) == 1

    under_out = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [TokenOutput(token_ids=list(tree_ids)), TokenOutput(token_ids=list(finish_ids))]
            ),
            env,
            obs_tokens_limit=cost - 1,
        ),
        extra,
    )
    assert under_out.extra_fields["termination"] == "budget_exhausted"
    assert under_out.extra_fields["budget_exhausted"] is True
    assert _obs_ids(under_out) == []
    assert under_out.response_ids == list(tree_ids)
    assert under_out.extra_fields["final_submission"] is None
    assert under_out.extra_fields["events"][0]["inserted"] is False
    assert under_out.extra_fields["events"][0]["would_be_observation_token_count"] == cost
    assert OBS_VERSION not in tokenizer.decode(
        under_out.response_ids, skip_special_tokens=True
    )


def test_remaining_zero_allows_finish_but_not_another_tool(tokenizer, tmp_path: Path):
    env, extra = _prepare_workspace(tmp_path)
    tree = _tool("tree", {"path": ".", "depth": 2})
    search = _tool("search", {"query": "version"})
    finish = _final({"locations": [{"path": "pkg.py"}]})
    tree_ids = _encode_action(tokenizer, tree)
    search_ids = _encode_action(tokenizer, search)
    finish_ids = _encode_action(tokenizer, finish)
    cost = len(
        _obs_ids(
            _run(
                _build_loop(
                    tokenizer,
                    FakeServerManager(
                        [
                            TokenOutput(token_ids=list(tree_ids)),
                            TokenOutput(token_ids=list(finish_ids)),
                        ]
                    ),
                    env,
                ),
                extra,
            )
        )[0]
    )

    finish_run = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(token_ids=list(tree_ids)),
                    TokenOutput(token_ids=list(finish_ids)),
                ]
            ),
            env,
            obs_tokens_limit=cost,
        ),
        extra,
    )
    assert finish_run.extra_fields["termination"] == "finish"
    assert finish_run.extra_fields["obs_tokens_remaining"] == 0

    search_run = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(token_ids=list(tree_ids)),
                    TokenOutput(token_ids=list(search_ids)),
                ]
            ),
            env,
            obs_tokens_limit=cost,
        ),
        extra,
    )
    assert search_run.extra_fields["termination"] == "budget_exhausted"
    kinds = [item["kind"] for item in search_run.extra_fields["segments"]]
    assert kinds == ["assistant", "observation", "assistant"]
    assert search_run.extra_fields["final_submission"] is None


def test_protocol_error_observation_consumes_budget_finish_does_not(
    tokenizer, tmp_path: Path
):
    env, extra = _prepare_workspace(tmp_path)
    bad = "<tool_call>{bad}</tool_call>"
    good = _final({"locations": [{"path": "pkg.py"}]})
    output = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(
                        token_ids=_encode_action(tokenizer, bad, must_parse=False)
                    ),
                    TokenOutput(token_ids=_encode_action(tokenizer, good)),
                ]
            ),
            env,
            obs_tokens_limit=50_000,
        ),
        extra,
    )
    obs = _obs_ids(output)
    assert len(obs) == 1
    assert output.extra_fields["obs_tokens_used"] == len(obs[0])
    assert output.extra_fields["termination"] == "finish"
    assert output.extra_fields["events"][1]["action_name"] == "finish"
    assert output.extra_fields["events"][1]["observation_token_count"] is None


def test_visible_remaining_after_matches_inserted_ids(tokenizer, tmp_path: Path):
    env, extra = _prepare_workspace(tmp_path)
    extra = dict(extra)
    extra["budget_visible"] = True
    extra["obs_tokens_limit"] = 8000
    tree = _tool("tree", {"path": ".", "depth": 2})
    finish = _final({"locations": [{"path": "pkg.py"}]})
    output = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(token_ids=_encode_action(tokenizer, tree)),
                    TokenOutput(token_ids=_encode_action(tokenizer, finish)),
                ]
            ),
            env,
            obs_tokens_limit=8000,
            budget_visible=True,
        ),
        extra,
    )
    obs = _obs_ids(output)[0]
    decoded = tokenizer.decode(obs, skip_special_tokens=True)
    assert BUDGET_OBS_VERSION in decoded
    assert OBS_VERSION in decoded
    remaining = None
    used = None
    for line in decoded.splitlines():
        if line.startswith("obs_tokens_remaining: "):
            remaining = int(line.split(": ", 1)[1])
        if line.startswith("obs_tokens_used: "):
            used = int(line.split(": ", 1)[1])
    assert used == len(obs)
    assert remaining == 8000 - len(obs)
    assert output.extra_fields["obs_tokens_used"] == len(obs)
    assert (
        output.extra_fields["tool_observation_token_count"]
        < output.extra_fields["observation_token_count"]
    )
    prompt_text = tokenizer.decode(output.prompt_ids, skip_special_tokens=True)
    assert BUDGET_OBS_VERSION in prompt_text
    assert "Issue:" in prompt_text


def test_hidden_and_visible_share_tools_and_v1_body(tokenizer, tmp_path: Path):
    tree = _tool("tree", {"path": ".", "depth": 2})
    finish = _final({"locations": [{"path": "pkg.py"}]})
    env, extra = _prepare_workspace(tmp_path)
    hidden = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(token_ids=_encode_action(tokenizer, tree)),
                    TokenOutput(token_ids=_encode_action(tokenizer, finish)),
                ]
            ),
            env,
            obs_tokens_limit=8000,
            budget_visible=False,
        ),
        extra,
    )
    extra_v = dict(extra)
    extra_v["budget_visible"] = True
    visible = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(token_ids=_encode_action(tokenizer, tree)),
                    TokenOutput(token_ids=_encode_action(tokenizer, finish)),
                ]
            ),
            env,
            obs_tokens_limit=8000,
            budget_visible=True,
        ),
        extra_v,
    )
    assert hidden.extra_fields["events"][0]["action_name"] == "tree"
    assert visible.extra_fields["events"][0]["action_name"] == "tree"
    assert (
        hidden.extra_fields["events"][0]["action_arguments"]
        == visible.extra_fields["events"][0]["action_arguments"]
    )
    hidden_obs = hidden.extra_fields["events"][0]["observation"]
    visible_obs = visible.extra_fields["events"][0]["observation"]
    assert hidden_obs == visible_obs
    assert hidden_obs.startswith(f"# {OBS_VERSION}")
    hidden_ids = tokenizer.decode(_obs_ids(hidden)[0], skip_special_tokens=True)
    visible_ids = tokenizer.decode(_obs_ids(visible)[0], skip_special_tokens=True)
    assert BUDGET_OBS_VERSION not in hidden_ids
    assert BUDGET_OBS_VERSION in visible_ids


def test_oracle_sentinels_do_not_enter_prompt_or_observations(tokenizer, tmp_path: Path):
    env, extra = _prepare_workspace(tmp_path)
    extra = dict(extra)
    extra["obs_tokens_limit"] = 8000
    extra["budget_visible"] = False
    extra["oracle_symbols"] = "LEAK_ORACLE_SYMBOLS_SENTINEL"
    extra["base_changed_files"] = "LEAK_BASE_CHANGED_FILES_SENTINEL"
    extra["patch"] = "LEAK_PATCH_SENTINEL"
    for field in list(POLICY_FORBIDDEN_DERIVED_FIELDS)[:8]:
        extra[field] = f"LEAK_{field.upper()}_SENTINEL"
    output = _run(
        _build_loop(
            tokenizer,
            FakeServerManager(
                [
                    TokenOutput(
                        token_ids=_encode_action(
                            tokenizer, _final({"locations": [{"path": "pkg.py"}]})
                        )
                    )
                ]
            ),
            env,
            obs_tokens_limit=8000,
        ),
        extra,
    )
    blob = tokenizer.decode(output.prompt_ids, skip_special_tokens=True)
    assert "LEAK_ORACLE_SYMBOLS_SENTINEL" not in blob
    assert "LEAK_BASE_CHANGED_FILES_SENTINEL" not in blob
    assert "LEAK_PATCH_SENTINEL" not in blob
    assert extra["base_commit"] not in blob
    keys = list(output.extra_fields)
    assert "oracle_symbols" not in keys
    assert "base_changed_files" not in keys


def test_agent_loop_source_does_not_import_evaluator_oracle():
    text = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "budget_coder_rl"
        / "agent_loop"
        / "repo_exploration.py"
    ).read_text(encoding="utf-8")
    assert "budget_coder_rl.eval" not in text
    assert "oracle_symbols" not in text
    assert "base_changed_files" not in text
    assert "evaluator_oracle" not in text
    assert "localization_score" not in text


def test_m3a_yaml_has_smoke_limit_not_frozen_training_budget():
    configs = OmegaConf.load(
        str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "agent_loop"
            / "repo_exploration_m3a.yaml"
        )
    )
    assert configs[0].name == "repo_exploration"
    assert configs[0].obs_tokens_limit == 8192
    assert configs[0].budget_visible is False
    assert configs[0].max_new_tokens_per_turn == 2048
    m2 = OmegaConf.load(str(AGENT_LOOP_CONFIG))
    assert "obs_tokens_limit" not in m2[0]
