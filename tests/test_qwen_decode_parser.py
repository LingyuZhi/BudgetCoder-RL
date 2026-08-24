"""Qwen decode vs M2B parser tags. CPU-only; no Ray/vLLM/GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from budget_coder_rl.agent_loop.tokenization import (
    PARSER_TAG_STRINGS,
    decode_for_parser,
)
from budget_coder_rl.protocol.parser import (
    FINAL_CLOSE,
    FINAL_OPEN,
    ProtocolError,
    ToolCall,
    parse_action,
)


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


@pytest.fixture(scope="module")
def tokenizer():
    path = _resolve_tokenizer_path()
    if path is None:
        pytest.skip(
            "no local Qwen tokenizer found; set BCRL_TOKENIZER_PATH to a local snapshot"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def _tool_text() -> str:
    payload = json.dumps(
        {"name": "search", "arguments": {"query": "version", "path": "."}},
        separators=(",", ":"),
    )
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final_text() -> str:
    return '<final>\n{"locations":[{"path":"pkg.py","symbol":"Foo.bar"}]}\n</final>'


def test_tool_call_added_token_is_not_special(tokenizer):
    added = tokenizer.added_tokens_decoder
    open_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    close_id = tokenizer.convert_tokens_to_ids("</tool_call>")
    assert open_id == 151657
    assert close_id == 151658
    assert added[open_id].content == "<tool_call>"
    assert added[close_id].content == "</tool_call>"
    assert added[open_id].special is False
    assert added[close_id].special is False
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert added[im_end_id].special is True
    assert added[eot_id].special is True


def test_skip_special_tokens_keeps_parser_tags(tokenizer):
    text = _tool_text()
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    kept = tokenizer.decode(ids, skip_special_tokens=True)
    dropped = tokenizer.decode(ids, skip_special_tokens=False)
    assert "<tool_call>" in kept
    assert "</tool_call>" in kept
    for tag in PARSER_TAG_STRINGS[:2]:
        assert tag in kept
    parsed = parse_action(decode_for_parser(tokenizer, ids))
    assert isinstance(parsed, ToolCall)
    assert parsed.name == "search"
    assert kept.strip() == dropped.strip() or "<tool_call>" in dropped


def test_final_tags_survive_decode_for_parser(tokenizer):
    text = _final_text()
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    decoded = decode_for_parser(tokenizer, ids)
    assert FINAL_OPEN in decoded
    assert FINAL_CLOSE in decoded
    parsed = parse_action(decoded)
    assert parsed.locations[0].path == "pkg.py"


def test_decode_for_parser_strips_trailing_eos(tokenizer):
    text = _tool_text()
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    with_eos = ids + [im_end]
    with_eot = ids + [eot]
    assert parse_action(decode_for_parser(tokenizer, with_eos)).name == "search"
    assert parse_action(decode_for_parser(tokenizer, with_eot)).name == "search"
    raw = tokenizer.decode(with_eos, skip_special_tokens=False)
    assert "<|im_end|>" in raw
    with pytest.raises(ProtocolError):
        parse_action(raw)


def test_empty_ids_decode_to_empty_string(tokenizer):
    assert decode_for_parser(tokenizer, []) == ""
