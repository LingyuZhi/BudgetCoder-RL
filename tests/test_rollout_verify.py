"""CPU checks for DataProto padding / TITO helpers. No Ray/vLLM."""

from budget_coder_rl.agent_loop.rollout_verify import (
    concat_segment_ids,
    expected_response_mask,
    inspect_turn_boundary,
    segment_decomposition,
    unpadded_left,
    unpadded_right,
    verify_padded_sample,
)


def test_segment_decomposition_and_mask():
    segments = [
        {"kind": "assistant", "token_ids": [1, 2, 3]},
        {"kind": "observation", "token_ids": [8, 9]},
        {"kind": "assistant", "token_ids": [4]},
    ]
    assert concat_segment_ids(segments) == [1, 2, 3, 8, 9, 4]
    assert expected_response_mask(segments) == [1, 1, 1, 0, 0, 1]
    assert segment_decomposition(segments) == (
        "A1(3 tokens, mask=1) → O1(2 tokens, mask=0) → A2(1 tokens, mask=1)"
    )


def test_verify_padded_sample_accepts_left_prompt_right_response_pad():
    prompt_ids = [11, 12, 13]
    segments = [
        {"kind": "assistant", "token_ids": [21, 22]},
        {"kind": "observation", "token_ids": [31]},
        {"kind": "assistant", "token_ids": [41, 42, 43]},
    ]
    prompt_width = 6
    response_width = 8
    prompts = [0, 0, 0, 11, 12, 13]
    prompt_attn = [0, 0, 0, 1, 1, 1]
    response = [21, 22, 31, 41, 42, 43, 0, 0]
    response_attn = [1, 1, 1, 1, 1, 1, 0, 0]
    mask = [1, 1, 0, 1, 1, 1, 0, 0]
    attention = prompt_attn + response_attn
    errors = verify_padded_sample(
        prompt_width=prompt_width,
        response_width=response_width,
        prompts_row=prompts,
        responses_row=response,
        response_mask_row=mask,
        attention_mask_row=attention,
        unpadded_prompt_ids=prompt_ids,
        segments=segments,
    )
    assert errors == []
    assert unpadded_left(prompts, prompt_attn) == prompt_ids
    assert unpadded_right(response, response_attn) == concat_segment_ids(segments)


def test_verify_padded_sample_detects_mask_and_truncation():
    segments = [{"kind": "assistant", "token_ids": [7, 8]}]
    errors = verify_padded_sample(
        prompt_width=3,
        response_width=4,
        prompts_row=[0, 1, 2],
        responses_row=[7, 8, 0, 0],
        response_mask_row=[1, 1, 0, 1],
        attention_mask_row=[0, 1, 1, 1, 1, 0, 0],
        unpadded_prompt_ids=[1, 2],
        segments=segments,
    )
    assert any("padding" in item for item in errors)

    truncated = verify_padded_sample(
        prompt_width=2,
        response_width=2,
        prompts_row=[1, 2],
        responses_row=[7, 8],
        response_mask_row=[1, 1],
        attention_mask_row=[1, 1, 1, 1],
        unpadded_prompt_ids=[9, 1, 2],
        segments=segments,
    )
    assert any("prompt" in item for item in truncated)


class _FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            return "assistant-text"
        if ids and ids[0] == 100:
            return "<|im_start|>user\nobs\n<|im_end|>\n<|im_start|>assistant\n"
        return "raw"

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(token, 0)

    def convert_ids_to_tokens(self, token_id):
        return {151645: "<|im_end|>"}.get(int(token_id), "x")


def test_inspect_turn_boundary_flags_missing_eos():
    tokenizer = _FakeTok()
    segments = [
        {"kind": "assistant", "token_ids": [1, 2, 3]},
        {"kind": "observation", "token_ids": [100, 101]},
    ]
    dump = inspect_turn_boundary(
        tokenizer,
        prompt_ids=[0, 1],
        segments=segments,
        eos_token_ids=[151645],
    )
    assert dump["legal_qwen_chat_continuation"] is False
    assert dump["turns"][0]["ends_with_eos"] is False
    assert dump["turns"][1]["has_im_start_user"] is True

    legal = inspect_turn_boundary(
        tokenizer,
        prompt_ids=[0, 1],
        segments=[
            {"kind": "assistant", "token_ids": [1, 2, 151645]},
            {"kind": "observation", "token_ids": [100, 101]},
        ],
        eos_token_ids=[151645],
    )
    assert legal["legal_qwen_chat_continuation"] is True
    assert legal["turns"][0]["ends_with_eos"] is True
