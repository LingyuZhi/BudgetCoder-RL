"""Chat-template encoding shared by AgentLoop audit and tests.

Mirrors pinned ``AgentLoopBase.apply_chat_template`` when ``processor is None``.
Never passes HF ``tools=``: Qwen's tools branch injects a multi-call protocol
that conflicts with the frozen M2B one-action / ``<final>`` contract.
"""

from __future__ import annotations

from typing import Any, Sequence

from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids

# Qwen3-Instruct added tokens used by the Stage-1 parser. They are *not*
# ``special=True``; ``skip_special_tokens=True`` must keep them. EOS/pad
# (``<|im_end|>``, ``<|endoftext|>``) *are* special and must be stripped
# before ``parse_action`` fullmatch.
PARSER_TAG_STRINGS = ("<tool_call>", "</tool_call>", "<final>", "</final>")


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


def decode_for_parser(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode generated IDs for ``parse_action`` only.

    Training truth remains the raw token IDs. This must keep parser tags and
    drop chat EOS/pad special tokens. Qwen marks ``<tool_call>`` as added but
    ``special=False``, so ``skip_special_tokens=True`` is the correct setting.
    Do not decode/re-encode history to rebuild policy tokens.
    """
    ids = list(token_ids)
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)
