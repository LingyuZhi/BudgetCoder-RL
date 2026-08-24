"""Postprocess / TITO checks for real AgentLoop DataProto rows.

Used by M2D smoke and CPU tests. Training truth is still the token-ID arrays,
not decoded text or JSONL traces.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_EOS_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")


def concat_segment_ids(segments: Sequence[Mapping[str, Any]]) -> list[int]:
    out: list[int] = []
    for item in segments:
        out.extend(list(item["token_ids"]))
    return out


def expected_response_mask(segments: Sequence[Mapping[str, Any]]) -> list[int]:
    mask: list[int] = []
    for item in segments:
        bit = 1 if item["kind"] == "assistant" else 0
        mask.extend([bit] * len(item["token_ids"]))
    return mask


def unpadded_left(ids, attention) -> list[int]:
    values = _as_int_list(ids)
    attn = _as_int_list(attention)
    if len(values) != len(attn):
        raise ValueError(f"id/attention length mismatch {len(values)} vs {len(attn)}")
    return [token for token, flag in zip(values, attn) if int(flag) != 0]


def unpadded_right(ids, attention) -> list[int]:
    values = _as_int_list(ids)
    attn = _as_int_list(attention)
    if len(values) != len(attn):
        raise ValueError(f"id/attention length mismatch {len(values)} vs {len(attn)}")
    end = len(attn)
    while end > 0 and int(attn[end - 1]) == 0:
        end -= 1
    return values[:end]


def segment_decomposition(segments: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    assistant_n = 0
    observation_n = 0
    for item in segments:
        n = len(item["token_ids"])
        if item["kind"] == "assistant":
            assistant_n += 1
            parts.append(f"A{assistant_n}({n} tokens, mask=1)")
        else:
            observation_n += 1
            parts.append(f"O{observation_n}({n} tokens, mask=0)")
    return " → ".join(parts)


def inspect_turn_boundary(
    tokenizer: Any,
    *,
    prompt_ids: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
    eos_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Debug-only decode of turn edges. Never used to rebuild training IDs."""
    eos_ids = set(eos_token_ids or _eos_ids(tokenizer))
    assistants = [item for item in segments if item["kind"] == "assistant"]
    observations = [item for item in segments if item["kind"] == "observation"]
    turns: list[dict[str, Any]] = []
    prefix = list(prompt_ids)
    for index, item in enumerate(segments):
        ids = list(item["token_ids"])
        record: dict[str, Any] = {
            "index": index,
            "kind": item["kind"],
            "n_tokens": len(ids),
            "first_ids": ids[:8],
            "last_ids": ids[-8:],
            "decode_skip_special_true": tokenizer.decode(ids, skip_special_tokens=True),
            "decode_skip_special_false": tokenizer.decode(ids, skip_special_tokens=False),
        }
        if item["kind"] == "assistant":
            record["ends_with_eos"] = bool(ids) and int(ids[-1]) in eos_ids
            record["last_token_is_im_end"] = bool(ids) and _token_string(
                tokenizer, ids[-1]
            ) == "<|im_end|>"
        else:
            false_text = record["decode_skip_special_false"]
            record["has_im_start_user"] = "<|im_start|>user" in false_text
            record["has_im_start_assistant"] = "<|im_start|>assistant" in false_text
            record["has_im_end"] = "<|im_end|>" in false_text
        turns.append(record)
        prefix.extend(ids)

    legal = True
    notes: list[str] = []
    if assistants and observations:
        a1 = assistants[0]
        o1 = observations[0]
        a1_ids = list(a1["token_ids"])
        o1_false = tokenizer.decode(list(o1["token_ids"]), skip_special_tokens=False)
        if a1_ids and int(a1_ids[-1]) not in eos_ids:
            legal = False
            notes.append(
                "assistant turn 1 does not end with EOS; next user observation "
                "may glue onto the assistant span"
            )
        if "<|im_start|>user" not in o1_false:
            legal = False
            notes.append("observation 1 decode(skip_special=False) lacks <|im_start|>user")
        if "<|im_start|>assistant" not in o1_false:
            legal = False
            notes.append(
                "observation 1 decode(skip_special=False) lacks generation prompt "
                "<|im_start|>assistant"
            )
    elif assistants and not observations:
        notes.append("no observation segment; single-turn finish or empty tools")

    return {
        "prompt_n_tokens": len(prompt_ids),
        "n_assistant": len(assistants),
        "n_observation": len(observations),
        "legal_qwen_chat_continuation": legal,
        "notes": notes,
        "turns": turns,
    }


def verify_padded_sample(
    *,
    prompt_width: int,
    response_width: int,
    prompts_row,
    responses_row,
    response_mask_row,
    attention_mask_row,
    unpadded_prompt_ids: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    prompt_ids = _as_int_list(unpadded_prompt_ids)
    expected_response = concat_segment_ids(segments)
    expected_mask = expected_response_mask(segments)
    valid_len = len(expected_response)

    if len(prompts_row) != prompt_width:
        errors.append(
            f"prompts width {len(prompts_row)} != prompt_length {prompt_width}"
        )
    if len(responses_row) != response_width:
        errors.append(
            f"responses width {len(responses_row)} != response_length {response_width}"
        )
    if len(response_mask_row) != response_width:
        errors.append("response_mask width != response_length")
    if len(attention_mask_row) != prompt_width + response_width:
        errors.append(
            f"attention_mask width {len(attention_mask_row)} != "
            f"prompt+response {prompt_width + response_width}"
        )
    if valid_len > response_width:
        errors.append(
            f"unpadded response {valid_len} exceeds response_length {response_width}"
        )
        return errors

    recovered_prompt = unpadded_left(prompts_row, attention_mask_row[:prompt_width])
    if recovered_prompt != prompt_ids:
        errors.append("padded prompt IDs != AgentLoop unpadded prompt_ids")
    if len(prompt_ids) > prompt_width:
        errors.append("silent prompt truncation would have been required")
    attn_prompt = int(sum(_as_int_list(attention_mask_row[:prompt_width])))
    if attn_prompt != len(prompt_ids):
        errors.append(
            f"prompt attention sum {attn_prompt} != unpadded prompt {len(prompt_ids)}"
        )

    recovered_response = unpadded_right(
        responses_row, attention_mask_row[prompt_width:]
    )
    if recovered_response != expected_response:
        errors.append("padded responses != concat(segment token_ids)")
    attn_response = int(sum(_as_int_list(attention_mask_row[prompt_width:])))
    if attn_response != valid_len:
        errors.append(
            f"response attention sum {attn_response} != segment total {valid_len}"
        )

    mask = _as_int_list(response_mask_row)
    if mask[:valid_len] != expected_mask:
        errors.append("response_mask assistant/observation bits != segments")
    if mask[valid_len:] != [0] * (response_width - valid_len):
        errors.append("response_mask padding is not all 0")
    response_attn = _as_int_list(attention_mask_row[prompt_width:])
    if response_attn[valid_len:] != [0] * (response_width - valid_len):
        errors.append("attention_mask response padding is not all 0")
    for index, (bit, attn) in enumerate(zip(mask, response_attn)):
        if int(attn) == 0 and int(bit) != 0:
            errors.append(f"padding position {index} has response_mask=1")
            break
    return errors


def _as_int_list(values) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(item) for item in list(values)]


def _eos_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for token in _EOS_TOKEN_STRINGS:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            continue
        value = convert(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if value is None or value == unk:
            continue
        ids.append(int(value))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        ids.append(int(pad))
    return list(dict.fromkeys(ids))


def _token_string(tokenizer: Any, token_id: int) -> str:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return ""
    value = convert(int(token_id))
    return str(value) if value is not None else ""
