"""Stage-1 RepoExplorationAgentLoop: M2A+M2B on the M0 exact-token path.

Replaces DummyTwoTurnAgentLoop's fake observation with a real
``ExplorationSession``. LLM ``TokenOutput.token_ids`` are the training
ground truth. Observations are encoded once via ``apply_chat_template``
(``role=user``, ``remove_system_prompt=True``) and appended. History is
never decoded and re-encoded.

Does not look up evaluator oracles or compute reward.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from budget_coder_rl.env import ExplorationSession, RepoEnvironment
from budget_coder_rl.protocol.prompt import (
    build_stage1_messages,
    extract_issue_text,
    policy_safe_repo,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_NEW_TOKENS_PER_TURN = 256


class PromptTooLongError(ValueError):
    """Initial prompt exceeds rollout.prompt_length. Truncation is forbidden."""


class RepoExplorationAgentLoop(AgentLoopBase):
    """Multi-turn repository exploration on veRL token-in/token-out generate."""

    def __init__(
        self,
        *args,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_new_tokens_per_turn: int = DEFAULT_MAX_NEW_TOKENS_PER_TURN,
        repo_environment: RepoEnvironment | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_turns = int(max_turns)
        self.max_new_tokens_per_turn = int(max_new_tokens_per_turn)
        self.repo_environment = repo_environment

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = _coerce_mapping(kwargs.get("extra_info"))
        instance_id = str(extra_info.get("instance_id") or "")
        issue = extract_issue_text(kwargs.get("raw_prompt"))
        messages = build_stage1_messages(issue, repo=policy_safe_repo(extra_info))
        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self.apply_chat_template(messages)
        if len(prompt_ids) > self.prompt_length:
            raise PromptTooLongError(
                f"initial prompt has {len(prompt_ids)} tokens, exceeds "
                f"prompt_length={self.prompt_length} "
                f"(instance_id={instance_id!r}). Silent truncation is not allowed."
            )

        env = self.repo_environment or RepoEnvironment()
        workspace = env.prepare_from_extra_info(extra_info)
        session = ExplorationSession(workspace)

        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = []
        segments: list[dict[str, Any]] = []
        research_events: list[dict[str, Any]] = []
        termination: str | None = None
        submission: dict[str, Any] | None = None
        assistant_turns = 0
        observation_turns = 0
        num_preempted = -1

        for _turn in range(self.max_turns):
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                termination = "response_length"
                break

            turn_params = {
                **sampling_params,
                "max_tokens": min(self.max_new_tokens_per_turn, remaining),
            }
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params=turn_params,
                )
            if output.num_preempted is not None:
                num_preempted = max(0, num_preempted) + output.num_preempted

            gen_ids = list(output.token_ids)
            response_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            assistant_turns += 1
            segments.append({"kind": "assistant", "token_ids": list(gen_ids)})
            if response_logprobs is not None:
                if output.log_probs:
                    response_logprobs.extend(list(output.log_probs))
                elif gen_ids:
                    response_logprobs = None

            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            step = session.step(text)
            research_events.append(
                {
                    "turn": step.turn,
                    "action_type": step.action_type,
                    "raw_action": text,
                    "observation": step.observation,
                    "terminal": step.terminal,
                    "error_kind": step.error_kind,
                    "termination": step.termination,
                    "submission": step.submission,
                }
            )

            if step.terminal:
                termination = step.termination or "finish"
                submission = step.submission
                break

            obs_ids = await self.apply_chat_template(
                [{"role": "user", "content": step.observation}],
                remove_system_prompt=True,
            )
            obs_ids = list(obs_ids)
            if len(response_ids) + len(obs_ids) > self.response_length:
                termination = "response_length"
                break
            response_ids.extend(obs_ids)
            response_mask.extend([0] * len(obs_ids))
            observation_turns += 1
            segments.append({"kind": "observation", "token_ids": list(obs_ids)})
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(obs_ids))
        else:
            if termination is None:
                termination = "max_turns"

        metrics["num_preempted"] = num_preempted
        assert len(response_ids) == len(response_mask)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=1 + assistant_turns + observation_turns,
            metrics=metrics,
        )
        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "instance_id": instance_id,
                "repo": extra_info.get("repo"),
                "base_commit": extra_info.get("base_commit"),
                "final_submission": submission,
                "termination": termination,
                "segments": segments,
                "events": research_events,
                # Research/debug only. Never rebuild RL token trajectories from this.
                "trace_role": "research_debug_not_training_tokens",
            }
        )
        return output


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_container"):
        try:
            value = value.to_container()
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (dict, str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    return {}
