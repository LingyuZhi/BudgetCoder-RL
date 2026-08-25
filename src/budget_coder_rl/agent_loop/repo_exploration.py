"""Stage-1 RepoExplorationAgentLoop: M2A+M2B on the M0 exact-token path.

Replaces DummyTwoTurnAgentLoop's fake observation with a real
``ExplorationSession``. LLM ``TokenOutput.token_ids`` are the training
ground truth. Observations are encoded once via ``apply_chat_template``
(``role=user``, ``remove_system_prompt=True``) and appended. History is
never decoded and re-encoded.

Does not look up evaluator oracles or compute reward. Primary observation
budget (``bcrl-bobs-v2``) counts inserted ``# bcrl-obs-v1`` tokens only.
Visible ``# bcrl-budget-v1`` envelope tokens are logged separately.
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

from budget_coder_rl.agent_loop.tokenization import decode_for_parser
from budget_coder_rl.budget.state import (
    BUDGET_ACCOUNTING_VERSION,
    BudgetState,
    resolve_episode_budget,
    wrap_observation_with_budget,
)
from budget_coder_rl.env import ExplorationSession, RepoEnvironment
from budget_coder_rl.protocol.parser import (
    FinalAction,
    ProtocolError,
    ToolCall,
    locations_payload,
    parse_action,
)
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
        obs_tokens_limit: int | None = None,
        budget_visible: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_turns = int(max_turns)
        self.max_new_tokens_per_turn = int(max_new_tokens_per_turn)
        self.repo_environment = repo_environment
        self.obs_tokens_limit, self.budget_visible = resolve_episode_budget(
            {
                "obs_tokens_limit": obs_tokens_limit,
                "budget_visible": budget_visible,
            },
            default_limit=None,
            default_visible=False,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = _coerce_mapping(kwargs.get("extra_info"))
        instance_id = str(extra_info.get("instance_id") or "")
        obs_tokens_limit, budget_visible = resolve_episode_budget(
            extra_info,
            default_limit=self.obs_tokens_limit,
            default_visible=self.budget_visible,
        )
        budget = BudgetState(
            obs_tokens_used=0,
            obs_tokens_limit=obs_tokens_limit,
            turns_used=0,
            turns_limit=self.max_turns,
        )
        issue = extract_issue_text(kwargs.get("raw_prompt"))
        messages = build_stage1_messages(
            issue,
            repo=policy_safe_repo(extra_info),
            budget_state=budget if budget_visible else None,
            budget_visible=budget_visible,
        )
        request_id = uuid4().hex
        metrics: dict[str, Any] = {}
        sampling_seed = _coerce_optional_seed(extra_info.get("sampling_seed"))
        sampling_record = {
            "temperature": sampling_params.get("temperature"),
            "top_p": sampling_params.get("top_p"),
            "top_k": sampling_params.get("top_k"),
        }
        if sampling_seed is not None:
            sampling_record["seed"] = sampling_seed

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
        inserted_env_tokens = 0
        inserted_repo_obs_tokens = 0
        num_preempted = -1

        for _turn in range(self.max_turns):
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                termination = "response_length"
                break

            max_tokens = min(self.max_new_tokens_per_turn, remaining)
            generate_prefix_n = len(prompt_ids) + len(response_ids)
            turn_params = {
                **sampling_params,
                "max_tokens": max_tokens,
            }
            turn_params.pop("do_sample", None)
            if sampling_seed is not None:
                turn_params["seed"] = sampling_seed
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
            budget.turns_used = assistant_turns
            segments.append({"kind": "assistant", "token_ids": list(gen_ids)})
            if response_logprobs is not None:
                if output.log_probs:
                    response_logprobs.extend(list(output.log_probs))
                elif gen_ids:
                    response_logprobs = None

            text = decode_for_parser(self.tokenizer, gen_ids)
            step = session.step(text)
            action_name, action_arguments, parse_error_code = _research_action(text)
            obs_headers = _observation_headers(step.observation)
            budget_before = budget.as_dict()
            research_events.append(
                {
                    "turn": step.turn,
                    "action_type": step.action_type,
                    "action_name": action_name,
                    "action_arguments": action_arguments,
                    "raw_action": text,
                    "generated_token_count": len(gen_ids),
                    "stop_reason": output.stop_reason,
                    "max_tokens": max_tokens,
                    "generate_prefix_n": generate_prefix_n,
                    "parse_error_code": parse_error_code,
                    "observation": step.observation,
                    "observation_preview": _preview(step.observation),
                    "observation_token_count": None,
                    "tool_observation_token_count": None,
                    "inserted": None,
                    "budget_before": budget_before,
                    "budget_after": budget.as_dict(),
                    "tool_status": obs_headers.get("status"),
                    "tool": obs_headers.get("tool"),
                    "error_code": obs_headers.get("code") or parse_error_code,
                    "error_kind": step.error_kind,
                    "terminal": step.terminal,
                    "termination": step.termination,
                    "submission": step.submission,
                    "cumulative_response_tokens": len(response_ids),
                }
            )

            if step.terminal:
                termination = step.termination or "finish"
                submission = step.submission
                research_events[-1]["inserted"] = False
                research_events[-1]["budget_after"] = budget.as_dict()
                break

            obs_ids, _user_content, tool_obs_n = await self._encode_observation(
                step.observation,
                budget=budget,
                visible=budget_visible,
            )
            event = research_events[-1]
            event["observation_token_count"] = len(obs_ids)
            event["tool_observation_token_count"] = tool_obs_n
            event["budget_metadata_token_count"] = len(obs_ids) - tool_obs_n
            event["would_be_observation_token_count"] = len(obs_ids)

            if not budget.can_insert(tool_obs_n):
                termination = "budget_exhausted"
                event["inserted"] = False
                event["budget_after"] = budget.as_dict()
                break
            if len(response_ids) + len(obs_ids) > self.response_length:
                termination = "response_length"
                event["inserted"] = False
                event["budget_after"] = budget.as_dict()
                break

            response_ids.extend(obs_ids)
            response_mask.extend([0] * len(obs_ids))
            observation_turns += 1
            budget.consume(tool_obs_n)
            inserted_env_tokens += len(obs_ids)
            inserted_repo_obs_tokens += tool_obs_n
            segments.append({"kind": "observation", "token_ids": list(obs_ids)})
            event["inserted"] = True
            event["budget_after"] = budget.as_dict()
            event["cumulative_response_tokens"] = len(response_ids)
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(obs_ids))
        else:
            if termination is None:
                termination = "max_turns"

        metrics["num_preempted"] = num_preempted
        assert len(response_ids) == len(response_mask)
        policy_token_count = int(sum(response_mask))
        inserted_env_check = sum(
            len(item["token_ids"]) for item in segments if item["kind"] == "observation"
        )
        assert inserted_env_tokens == inserted_env_check
        assert inserted_repo_obs_tokens == budget.obs_tokens_used
        inserted_metadata_tokens = inserted_env_tokens - inserted_repo_obs_tokens
        if inserted_metadata_tokens < 0:
            raise RuntimeError("budget metadata token count went negative")

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
                "split": extra_info.get("split"),
                "final_submission": submission,
                "termination": termination,
                "segments": segments,
                "events": research_events,
                "unpadded_prompt_ids": list(prompt_ids),
                "prompt_token_count": len(prompt_ids),
                "policy_token_count": policy_token_count,
                "observation_token_count": inserted_env_tokens,
                "tool_observation_token_count": inserted_repo_obs_tokens,
                "repo_observation_tokens": inserted_repo_obs_tokens,
                "budget_metadata_tokens": inserted_metadata_tokens,
                "total_env_tokens": inserted_env_tokens,
                "obs_tokens_used": budget.obs_tokens_used,
                "obs_tokens_limit": obs_tokens_limit,
                "obs_tokens_remaining": budget.obs_tokens_remaining,
                "budget_accounting_version": BUDGET_ACCOUNTING_VERSION,
                "budget_visible": budget_visible,
                "budget_exhausted": termination == "budget_exhausted",
                "sampling_params": sampling_record,
                "sampling_seed": sampling_seed,
                "max_turns": self.max_turns,
                "max_new_tokens_per_turn": self.max_new_tokens_per_turn,
                "model_name_or_path": getattr(self.tokenizer, "name_or_path", None),
                # Research/debug only. Never rebuild RL token trajectories from this.
                "trace_role": "research_debug_not_training_tokens",
            }
        )
        return output

    async def _encode_user_message(self, content: str) -> list[int]:
        ids = await self.apply_chat_template(
            [{"role": "user", "content": content}],
            remove_system_prompt=True,
        )
        return list(ids)

    async def _encode_observation(
        self,
        v1_text: str,
        *,
        budget: BudgetState,
        visible: bool,
    ) -> tuple[list[int], str, int]:
        v1_ids = await self._encode_user_message(v1_text)
        v1_n = len(v1_ids)
        if not visible:
            return v1_ids, v1_text, v1_n

        if budget.obs_tokens_limit is None:
            raise RuntimeError("visible observation encode requires obs_tokens_limit")
        displayed_used = budget.obs_tokens_used + v1_n
        tentative = BudgetState(
            obs_tokens_used=displayed_used,
            obs_tokens_limit=budget.obs_tokens_limit,
            turns_used=budget.turns_used,
            turns_limit=budget.turns_limit,
        )
        content = wrap_observation_with_budget(v1_text, tentative)
        ids = await self._encode_user_message(content)
        return ids, content, v1_n


def _research_action(text: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    try:
        parsed = parse_action(text)
    except ProtocolError as exc:
        return None, None, exc.code
    if isinstance(parsed, ToolCall):
        return parsed.name, dict(parsed.arguments), None
    if isinstance(parsed, FinalAction):
        return "finish", locations_payload(parsed), None
    return None, None, None


def _observation_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in str(text).splitlines():
        if line.strip() == "---":
            break
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(": ")
        if not sep or key in headers:
            continue
        headers[key] = value
    return headers


def _preview(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated preview]..."


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


def _coerce_optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("sampling_seed must be an int or None, not bool")
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            raise ValueError("sampling_seed must be an int or None, not bool")
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "none", "null"}:
            return None
        return int(stripped)
    return int(value)
