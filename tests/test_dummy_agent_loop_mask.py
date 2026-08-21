"""Unit tests for the M0 DummyTwoTurnAgentLoop token/mask contract.

The LLM server is mocked (deterministic token ids); the tokenizer is real.
The tokenizer path is parameterized via ``BCRL_TOKENIZER_PATH`` with local
auto-discovery fallback, so a Qwen3-4B -> Qwen3-1.7B model fallback never
invalidates these tests.
"""

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from budget_coder_rl.agent_loop.dummy import DummyTwoTurnAgentLoop

TURN1_IDS = [101, 102, 103, 104, 105]
TURN2_IDS = [201, 202, 203]


def _resolve_tokenizer_path() -> str | None:
    env_path = os.environ.get("BCRL_TOKENIZER_PATH")
    if env_path:
        return env_path

    candidates: list[Path] = []
    data_root = Path(
        os.environ.get("BCRL_DATA_ROOT", os.path.expanduser("~/my_data/budget-coder-rl"))
    )
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
    """Mock LLMServerClient: returns fixed TokenOutput per call, records calls."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)}
        )
        return self._outputs.pop(0)


@pytest.fixture(scope="module")
def tokenizer():
    path = _resolve_tokenizer_path()
    if path is None:
        pytest.skip(
            "no local Qwen tokenizer found; set BCRL_TOKENIZER_PATH to a local snapshot"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _build_agent_loop(tokenizer, server_manager) -> DummyTwoTurnAgentLoop:
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    trainer_config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"prompt_length": 512, "response_length": 512}}}
    )
    return DummyTwoTurnAgentLoop(
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(OmegaConf.create({})),
    )


def _run_dummy_loop(tokenizer):
    from verl.workers.rollout.replica import TokenOutput

    server = FakeServerManager(
        [TokenOutput(token_ids=TURN1_IDS), TokenOutput(token_ids=TURN2_IDS)]
    )
    agent_loop = _build_agent_loop(tokenizer, server)
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    output = agent_loop.loop.run_until_complete(
        agent_loop.run({"temperature": 1.0}, raw_prompt=messages)
    )
    return output, server


def test_agent_loop_output_contract(tokenizer):
    """Test 2 — response_ids / response_mask non-empty and equal length."""
    output, _ = _run_dummy_loop(tokenizer)

    assert len(output.response_ids) > 0
    assert len(output.response_mask) > 0
    assert len(output.response_ids) == len(output.response_mask)
    assert len(output.prompt_ids) > 0
    assert output.num_turns == 4


def test_mask_segmentation_and_exact_token_ids(tokenizer):
    """Test 3 — LLM tokens mask=1, env observation tokens mask=0, exact ids kept."""
    output, server = _run_dummy_loop(tokenizer)

    segments = output.extra_fields["dummy_segments"]
    turn1_ids = segments["turn1_ids"]
    obs_ids = segments["obs_ids"]
    turn2_ids = segments["turn2_ids"]

    # exact sampled token ids are preserved, never re-encoded
    assert turn1_ids == TURN1_IDS
    assert turn2_ids == TURN2_IDS
    assert output.response_ids == TURN1_IDS + obs_ids + TURN2_IDS

    # mask segmentation: assistant spans = 1, environment span = 0
    n1, n_obs, n2 = len(turn1_ids), len(obs_ids), len(turn2_ids)
    assert n_obs > 0
    assert output.response_mask == [1] * n1 + [0] * n_obs + [1] * n2

    # observation tokens really decode to the fake env observation
    decoded_obs = tokenizer.decode(obs_ids)
    assert "ENV_OBSERVATION" in decoded_obs

    # turn-2 context is the exact accumulated token ids (token-in/token-out)
    assert (
        server.calls[1]["prompt_ids"]
        == server.calls[0]["prompt_ids"] + TURN1_IDS + obs_ids
    )

    # per-turn generation cap is threaded into sampling params
    assert server.calls[0]["sampling_params"]["max_tokens"] == 256
    assert server.calls[1]["sampling_params"]["max_tokens"] > 0
