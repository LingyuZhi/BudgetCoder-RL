"""Chat-template encoding shared by AgentLoop audit and tests.

Mirrors pinned ``AgentLoopBase.apply_chat_template`` when ``processor is None``.
Never passes HF ``tools=``: Qwen's tools branch injects a multi-call protocol
that conflicts with the frozen M2B one-action / ``<final>`` contract.
"""

from __future__ import annotations

from typing import Any

from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids


def encode_chat_messages(
    tokenizer: Any,
    messages: list[dict],
    *,
    remove_system_prompt: bool = False,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
    system_prompt_ids: list[int] | None = None,
) -> list[int]:
    kwargs = dict(apply_chat_template_kwargs or {})
    tokenized = apply_chat_template(
        tokenizer,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=True,
        **kwargs,
    )
    prompt_ids = list(normalize_token_ids(tokenized))
    if remove_system_prompt:
        prefix = (
            list(system_prompt_ids)
            if system_prompt_ids is not None
            else list(initialize_system_prompt(tokenizer, **kwargs))
        )
        prompt_ids = prompt_ids[len(prefix) :]
    return prompt_ids
