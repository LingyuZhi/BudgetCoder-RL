"""M0 smoke agent loop: two real LLM generations around one fake environment observation.

Token contract (M0-E):

- assistant-generated tokens  -> response_mask = 1
- environment observation tokens -> response_mask = 0
- ``response_ids`` are the exact rollout token ids: LLM turns are token-in/token-out
  (``server_manager.generate`` consumes and produces token ids directly), and the
  observation is encoded exactly once. Nothing is ever decoded and re-encoded.

Scope note: the fake observation is inserted as a *generic environment observation*
carried by a ``role="user"`` message. M0 does NOT exercise the real tool-role /
tool-parser path (that is the ``ToolAgentLoop``-style path, targeted at M2).
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FAKE_ENV_OBSERVATION = (
    "ENV_OBSERVATION:\nYou have one step remaining. Continue and give the final answer."
)


class DummyTwoTurnAgentLoop(AgentLoopBase):
    """Two-turn agent loop with a fixed fake environment observation in between.

    Flow: prompt -> LLM generation #1 (mask=1) -> fake env observation (mask=0)
    -> LLM generation #2 (mask=1) -> AgentLoopOutput.

    The full per-segment token ids are stored in
    ``extra_fields["dummy_segments"]`` so the integration smoke can assert full
    per-token equality between the trainer-side ``responses`` tensor and the
    exact rollout token ids.
    """

    def __init__(self, *args, max_new_tokens_per_turn: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_new_tokens_per_turn = max_new_tokens_per_turn

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self.apply_chat_template(messages)

        # Turn 1: real LLM generation (mask=1), token-in/token-out.
        turn1_params = {**sampling_params, "max_tokens": self.max_new_tokens_per_turn}
        with simple_timer("generate_sequences", metrics):
            out1: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn1_params,
            )
        turn1_ids = list(out1.token_ids)

        # Fake environment observation (mask=0): encoded exactly once and appended
        # as observation tokens, never merged into assistant-generated tokens.
        obs_ids = await self.apply_chat_template(
            [{"role": "user", "content": FAKE_ENV_OBSERVATION}],
            remove_system_prompt=True,
        )

        # Turn 2: continue generation from the exact accumulated token ids.
        remaining = self.response_length - len(turn1_ids) - len(obs_ids)
        if remaining <= 0:
            raise ValueError(
                f"M0 dummy loop ran out of response budget before turn 2: "
                f"turn1={len(turn1_ids)}, obs={len(obs_ids)}, "
                f"response_length={self.response_length}. Increase response_length "
                f"or decrease max_new_tokens_per_turn."
            )
        turn2_params = {
            **sampling_params,
            "max_tokens": min(self.max_new_tokens_per_turn, remaining),
        }
        with simple_timer("generate_sequences", metrics):
            out2: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + turn1_ids + obs_ids,
                sampling_params=turn2_params,
            )
        turn2_ids = list(out2.token_ids)

        num_preempted = -1
        for out in (out1, out2):
            if out.num_preempted is not None:
                num_preempted = max(0, num_preempted) + out.num_preempted
        metrics["num_preempted"] = num_preempted

        response_ids = turn1_ids + obs_ids + turn2_ids
        response_mask = [1] * len(turn1_ids) + [0] * len(obs_ids) + [1] * len(turn2_ids)
        assert len(response_ids) == len(response_mask)
        assert len(response_ids) <= self.response_length

        response_logprobs = None
        if out1.log_probs and out2.log_probs:
            response_logprobs = (
                list(out1.log_probs) + [0.0] * len(obs_ids) + list(out2.log_probs)
            )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=4,  # user, assistant, env observation, assistant
            metrics=metrics,
        )
        output.extra_fields.update(
            {
                # keep schema consistent with built-in agent loops
                "turn_scores": [],
                "tool_rewards": [],
                # full segments (not truncated) for exact-token verification
                "dummy_segments": {
                    "turn1_ids": turn1_ids,
                    "obs_ids": list(obs_ids),
                    "turn2_ids": turn2_ids,
                },
            }
        )
        return output
