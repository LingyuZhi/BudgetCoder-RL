"""Frozen Stage-1 observation text.

The rendered string is the exact future trajectory observation. M2B does
not compute tokenizer token_cost.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

OBS_VERSION = "bcrl-obs-v1"


def format_error(*, tool: str, code: str, message: str) -> str:
    return _render(
        tool=tool,
        status="error",
        headers=(("code", code), ("message", _one_line(message))),
        body=None,
    )


def format_tree(
    *,
    path: str,
    depth: int,
    truncated: bool,
    entries: Sequence[tuple[str, str]],
) -> str:
    body = "\n".join(f"{kind} {rel}" for kind, rel in entries)
    return _render(
        tool="tree",
        status="ok",
        headers=(
            ("truncated", _bool(truncated)),
            ("path", path),
            ("depth", str(depth)),
            ("entry_count", str(len(entries))),
        ),
        body=body,
    )


def format_search(
    *,
    query: str,
    path: str,
    truncated: bool,
    hits: Sequence[tuple[str, int, str]],
) -> str:
    body = "\n".join(f"{rel}:{line_no}:{line}" for rel, line_no, line in hits)
    return _render(
        tool="search",
        status="ok",
        headers=(
            ("truncated", _bool(truncated)),
            ("query", query),
            ("path", path),
            ("match_count", str(len(hits))),
        ),
        body=body,
    )


def format_read(
    *,
    path: str,
    start_line: int,
    end_line: int,
    truncated: bool,
    numbered_lines: Sequence[str],
) -> str:
    body = "\n".join(numbered_lines)
    return _render(
        tool="read",
        status="ok",
        headers=(
            ("truncated", _bool(truncated)),
            ("path", path),
            ("start_line", str(start_line)),
            ("end_line", str(end_line)),
        ),
        body=body,
    )


def format_finish(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    locations = payload.get("locations")
    count = len(locations) if isinstance(locations, list) else 0
    return _render(
        tool="finish",
        status="ok",
        headers=(("location_count", str(count)),),
        body=body,
    )


def _render(
    *,
    tool: str,
    status: str,
    headers: Sequence[tuple[str, str]],
    body: str | None,
) -> str:
    lines = [f"# {OBS_VERSION}", f"tool: {tool}", f"status: {status}"]
    for key, value in headers:
        lines.append(f"{key}: {_one_line(value)}")
    if body is not None:
        lines.append("---")
        if body:
            lines.extend(body.split("\n"))
    return "\n".join(lines) + "\n"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _one_line(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")
