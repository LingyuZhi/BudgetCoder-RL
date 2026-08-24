"""Exact-token / mask contract for RepoExplorationAgentLoop."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from budget_coder_rl.agent_loop.repo_exploration import (
    PromptTooLongError,
    RepoExplorationAgentLoop,
)
from budget_coder_rl.agent_loop.tokenization import encode_chat_messages
from budget_coder_rl.data.swe_gym_repos import cache_key, resolve_commit, run_git
from budget_coder_rl.env import RepoEnvironment, TaskRef
from budget_coder_rl.protocol.prompt import build_stage1_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_LOOP_CONFIG = REPO_ROOT / "configs" / "agent_loop" / "repo_exploration.yaml"

A1_IDS = [101, 102, 103, 104]
A2_IDS = [201, 202, 203]
A3_IDS = [301, 302, 303, 304, 305]
OBS_IDS = [801, 802, 803]
PROMPT_IDS = [0, 10, 11, 12]
REENCODE_IDS = [9999, 9998, 9997]


def _resolve_tokenizer_path() -> str | None:
    env_path = os.environ.get("BCRL_TOKENIZER_PATH")
    if env_path:
        return env_path
    candidates: list[Path] = []
    data_root = Path(
        os.environ.get("BCRL_DATA_ROOT", os.path.expanduser("~/my_data/budget-coder-rl"))
    )
    preferred = data_root / "models" / "Qwen3-4B-Instruct-2507"
    if preferred.is_dir():
        candidates.append(preferred)
    if (data_root / "models").is_dir():
        candidates.extend(sorted((data_root / "models").glob("*")))
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    for repo_dir in sorted(hub.glob("models--Qwen--*")):
        candidates.extend(sorted((repo_dir / "snapshots").glob("*")))
    for cand in candidates:
        if (cand / "tokenizer_config.json").exists():
            return str(cand)
    return None


class FakeServerManager:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)}
        )
        return self._outputs.pop(0)


class AdversarialTokenizer:
    """decode/encode is intentionally not an identity; chat-template ids are independent."""

    def __init__(self, decode_map: dict[tuple[int, ...], str]):
        self._decode_map = dict(decode_map)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        **kwargs,
    ):
        if tools is not None:
            raise AssertionError("M2C must not pass HF tools= into the chat template")
        if not tokenize:
            return "adversarial"
        roles = [item.get("role") for item in messages]
        contents = [item.get("content", "") for item in messages]
        if roles == ["user"] and contents == [""] and not add_generation_prompt:
            return [0, 1, 2]
        if roles == ["user", "user"] and contents == ["", ""] and not add_generation_prompt:
            return [0, 1, 2, 3, 4]
        if roles and roles[0] == "system":
            return list(PROMPT_IDS)
        return [0, *OBS_IDS]

    def decode(self, ids, skip_special_tokens=True):
        key = tuple(ids)
        if key not in self._decode_map:
            raise AssertionError(f"unexpected decode ids {list(ids)}")
        return self._decode_map[key]

    def encode(self, text, add_special_tokens=False):
        return list(REENCODE_IDS)


@pytest.fixture(scope="module")
def tokenizer():
    path = _resolve_tokenizer_path()
    if path is None:
        pytest.skip(
            "no local Qwen tokenizer found; set BCRL_TOKENIZER_PATH to a local snapshot"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _init_src(src: Path) -> None:
    src.mkdir()
    run_git(["init"], cwd=src, allow_network=True, timeout=30)
    run_git(["config", "user.email", "test@example.com"], cwd=src, allow_network=True)
    run_git(["config", "user.name", "Test"], cwd=src, allow_network=True)
    run_git(["config", "commit.gpgsign", "false"], cwd=src, allow_network=True)


def _prepare_workspace(tmp_path: Path):
    src = tmp_path / "src"
    _init_src(src)
    (src / "pkg.py").write_text("version = 1\n", encoding="utf-8")
    (src / "nested").mkdir()
    (src / "nested" / "keep.txt").write_text("stable\n", encoding="utf-8")
    run_git(["add", "."], cwd=src, allow_network=True)
    run_git(["commit", "-m", "base"], cwd=src, allow_network=True)
    sha = resolve_commit(src, "HEAD")
    assert sha is not None
    repos_root = tmp_path / "mirrors"
    repos_root.mkdir()
    dest = repos_root / cache_key("owner/repo")
    run_git(["clone", "--bare", str(src), str(dest)], cwd=repos_root, allow_network=True)
    env = RepoEnvironment(repos_root=repos_root, snapshots_root=tmp_path / "snapshots")
    extra = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": sha,
        "split": "train",
    }
    return env, extra


def _tool(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


def _encode_action(tokenizer, text: str, *, must_parse: bool = True) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if must_parse:
        from budget_coder_rl.protocol.parser import parse_action

        parse_action(decoded)
    return ids


def _build_loop(
    tokenizer,
    server,
    env: RepoEnvironment,
    *,
    prompt_length: int = 2048,
    response_length: int = 2048,
    max_turns: int = 6,
) -> RepoExplorationAgentLoop:
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    trainer_config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                }
            }
        }
    )
    return RepoExplorationAgentLoop(
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(OmegaConf.create({})),
        repo_environment=env,
        max_turns=max_turns,
    )


def _run(loop: RepoExplorationAgentLoop, extra_info: dict, issue: str = "locate version"):
    messages = [{"role": "user", "content": issue}]
    return loop.loop.run_until_complete(
        loop.run({"temperature": 1.0}, raw_prompt=messages, extra_info=extra_info)
    )


def test_multi_turn_exact_ids_and_masks(tokenizer, tmp_path: Path):
    from verl.workers.rollout.replica import TokenOutput

    env, extra = _prepare_workspace(tmp_path)
    a1 = _tool("tree", {"path": ".", "depth": 2})
    a2 = _tool("search", {"query": "version"})
    a3 = _final({"locations": [{"path": "pkg.py", "symbol": "Scripted.placeholder"}]})
    a1_ids = _encode_action(tokenizer, a1)
    a2_ids = _encode_action(tokenizer, a2)
    a3_ids = _encode_action(tokenizer, a3)
    server = FakeServerManager(
        [
            TokenOutput(token_ids=a1_ids),
            TokenOutput(token_ids=a2_ids),
            TokenOutput(token_ids=a3_ids),
        ]
    )
    loop = _build_loop(tokenizer, server, env)
    output = _run(loop, extra)

    assert len(output.response_ids) == len(output.response_mask)
    assert output.extra_fields["termination"] == "finish"
    assert output.extra_fields["final_submission"] == {
        "locations": [{"path": "pkg.py", "symbol": "Scripted.placeholder"}]
    }

    segments = output.extra_fields["segments"]
    kinds = [item["kind"] for item in segments]
    assert kinds == ["assistant", "observation", "assistant", "observation", "assistant"]
    assert segments[0]["token_ids"] == a1_ids
    assert segments[2]["token_ids"] == a2_ids
    assert segments[4]["token_ids"] == a3_ids

    o1 = segments[1]["token_ids"]
    o2 = segments[3]["token_ids"]
    assert o1 and o2
    assert output.response_ids == a1_ids + o1 + a2_ids + o2 + a3_ids
    assert output.response_mask == (
        [1] * len(a1_ids)
        + [0] * len(o1)
        + [1] * len(a2_ids)
        + [0] * len(o2)
        + [1] * len(a3_ids)
    )

    assert server.calls[0]["prompt_ids"] == output.prompt_ids
    assert server.calls[1]["prompt_ids"] == output.prompt_ids + a1_ids + o1
    assert server.calls[2]["prompt_ids"] == output.prompt_ids + a1_ids + o1 + a2_ids + o2
    decoded_o1 = tokenizer.decode(o1)
    assert "bcrl-obs-v1" in decoded_o1
    assert "tool: tree" in decoded_o1


def test_protocol_error_keeps_generated_ids_and_continues(tokenizer, tmp_path: Path):
    from verl.workers.rollout.replica import TokenOutput

    env, extra = _prepare_workspace(tmp_path)
    bad = "<tool_call>{bad}</tool_call>"
    good = _final({"locations": [{"path": "pkg.py"}]})
    bad_ids = _encode_action(tokenizer, bad, must_parse=False)
    good_ids = _encode_action(tokenizer, good)
    server = FakeServerManager(
        [TokenOutput(token_ids=bad_ids), TokenOutput(token_ids=good_ids)]
    )
    loop = _build_loop(tokenizer, server, env)
    output = _run(loop, extra)
    segments = output.extra_fields["segments"]
    assert segments[0]["token_ids"] == bad_ids
    assert output.response_ids[: len(bad_ids)] == bad_ids
    assert output.response_mask[: len(bad_ids)] == [1] * len(bad_ids)
    assert segments[1]["kind"] == "observation"
    assert output.response_mask[len(bad_ids) : len(bad_ids) + len(segments[1]["token_ids"])] == (
        [0] * len(segments[1]["token_ids"])
    )
    assert output.extra_fields["termination"] == "finish"
    assert output.extra_fields["events"][0]["error_kind"] == "protocol"


def test_adversarial_tokenizer_does_not_reencode_generated_ids(tmp_path: Path):
    from verl.workers.rollout.replica import TokenOutput

    env, extra = _prepare_workspace(tmp_path)
    a1 = _tool("tree", {"path": ".", "depth": 2})
    a2 = _tool("search", {"query": "version"})
    a3 = _final({"locations": [{"path": "pkg.py", "symbol": "pkg"}]})
    decode_map = {
        tuple(A1_IDS): a1,
        tuple(A2_IDS): a2,
        tuple(A3_IDS): a3,
    }
    tokenizer = AdversarialTokenizer(decode_map)
    server = FakeServerManager(
        [
            TokenOutput(token_ids=list(A1_IDS)),
            TokenOutput(token_ids=list(A2_IDS)),
            TokenOutput(token_ids=list(A3_IDS)),
        ]
    )
    loop = _build_loop(tokenizer, server, env)
    output = _run(loop, extra)
    segments = output.extra_fields["segments"]
    assert segments[0]["token_ids"] == A1_IDS
    assert segments[2]["token_ids"] == A2_IDS
    assert segments[4]["token_ids"] == A3_IDS
    assert REENCODE_IDS != A1_IDS
    joined = output.response_ids
    assert joined[: len(A1_IDS)] == A1_IDS
    assert REENCODE_IDS not in [joined[i : i + len(REENCODE_IDS)] for i in range(len(joined))]
    assert output.prompt_ids == PROMPT_IDS
    o1 = segments[1]["token_ids"]
    assert o1 == OBS_IDS
    assert server.calls[1]["prompt_ids"] == PROMPT_IDS + A1_IDS + OBS_IDS
    reconstructed = tokenizer.encode(tokenizer.decode(A1_IDS))
    assert reconstructed == REENCODE_IDS
    assert reconstructed != A1_IDS


def test_overlong_initial_prompt_hard_fails(tokenizer, tmp_path: Path):
    from verl.workers.rollout.replica import TokenOutput

    env, extra = _prepare_workspace(tmp_path)
    server = FakeServerManager([TokenOutput(token_ids=[1])])
    loop = _build_loop(tokenizer, server, env, prompt_length=8)
    with pytest.raises(PromptTooLongError, match="Silent truncation is not allowed"):
        _run(loop, extra)
    assert server.calls == []


def test_encode_chat_messages_matches_agent_loop_apply_chat_template(tokenizer, tmp_path: Path):
    from verl.workers.rollout.replica import TokenOutput

    env, extra = _prepare_workspace(tmp_path)
    server = FakeServerManager(
        [TokenOutput(token_ids=_encode_action(tokenizer, _final({"locations": []})))]
    )
    loop = _build_loop(tokenizer, server, env)
    messages = build_stage1_messages("short issue", repo="owner/repo")
    via_helper = encode_chat_messages(tokenizer, messages)
    via_loop = loop.loop.run_until_complete(loop.apply_chat_template(messages))
    assert via_helper == via_loop


def test_hydra_yaml_instantiates_repo_exploration_loop(tokenizer, tmp_path: Path):
    import hydra
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.rollout.replica import TokenOutput

    env, extra = _prepare_workspace(tmp_path)
    configs = OmegaConf.load(str(AGENT_LOOP_CONFIG))
    assert configs[0].name == "repo_exploration"
    server = FakeServerManager(
        [TokenOutput(token_ids=_encode_action(tokenizer, _final({"locations": [{"path": "pkg.py"}]})))]
    )
    trainer_config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"prompt_length": 2048, "response_length": 2048}}}
    )
    loop = hydra.utils.instantiate(
        configs[0],
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(OmegaConf.create({})),
        tools=ToolListWrap([]),
        repo_environment=env,
    )
    assert isinstance(loop, RepoExplorationAgentLoop)
    assert loop.max_turns == 6
    output = _run(loop, extra)
    assert output.extra_fields["termination"] == "finish"
    assert extra["instance_id"] == output.extra_fields["instance_id"]
