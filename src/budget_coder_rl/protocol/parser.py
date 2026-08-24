"""Strict Stage-1 action parser.

One turn is exactly one ``<tool_call>`` or exactly one ``<final>``.
No silent fallback, no heuristic repair, no filesystem access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from budget_coder_rl.data.swe_gym_repos import is_safe_repo_path

TOOL_NAMES = frozenset({"tree", "search", "read"})
TREE_DEFAULT_PATH = "."
TREE_DEFAULT_DEPTH = 2
SEARCH_DEFAULT_PATH = "."
SEARCH_DEFAULT_MAX_RESULTS = 50

_TOOL_CALL_RE = re.compile(r"^<tool_call>\s*(.*?)\s*</tool_call>$", re.DOTALL)
_FINAL_RE = re.compile(r"^<final>\s*(.*?)\s*</final>$", re.DOTALL)
_CONTROL_NEWLINE = re.compile(r"[\r\n]")


class ProtocolError(Exception):
    """Malformed model output. Session maps this to an error observation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Location:
    path: str
    symbol: str | None = None


@dataclass(frozen=True)
class FinalAction:
    locations: tuple[Location, ...]


def parse_action(text: str) -> ToolCall | FinalAction:
    """Parse one turn. Raises ``ProtocolError``; never returns a repaired action."""
    if not isinstance(text, str):
        raise ProtocolError("malformed_action", "action text must be a string")
    stripped = text.strip()
    if not stripped:
        raise ProtocolError("empty_action", "action text is empty")

    n_tool_open = stripped.count("<tool_call>")
    n_tool_close = stripped.count("</tool_call>")
    n_final_open = stripped.count("<final>")
    n_final_close = stripped.count("</final>")
    n_open = n_tool_open + n_final_open
    if n_open > 1:
        raise ProtocolError(
            "multiple_actions", "turn must contain exactly one action block"
        )
    if n_open == 0:
        raise ProtocolError("malformed_action", "turn has no <tool_call> or <final>")

    if n_tool_open == 1:
        if n_tool_close != 1 or n_final_close != 0:
            raise ProtocolError("malformed_action", "unbalanced <tool_call> tags")
        match = _TOOL_CALL_RE.fullmatch(stripped)
        if match is None:
            raise ProtocolError(
                "multiple_actions",
                "extra text around <tool_call> is not allowed",
            )
        return _parse_tool_call(_load_strict_json(match.group(1)))

    if n_final_close != 1 or n_tool_close != 0:
        raise ProtocolError("malformed_action", "unbalanced <final> tags")
    match = _FINAL_RE.fullmatch(stripped)
    if match is None:
        raise ProtocolError(
            "multiple_actions",
            "extra text around <final> is not allowed",
        )
    return _parse_final(_load_strict_json(match.group(1)))


def _load_strict_json(text: str) -> Any:
    def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ProtocolError(
                    "duplicate_keys",
                    f"duplicate JSON key {key!r}",
                )
            out[key] = value
        return out

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except ProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise ProtocolError("malformed_json", f"malformed JSON: {exc.msg}") from exc


def _reject_nonfinite(token: str) -> None:
    raise ProtocolError(
        "malformed_json",
        f"non-finite JSON number {token} is not allowed",
    )


def _parse_tool_call(payload: Any) -> ToolCall:
    obj = _require_object(payload, "tool_call")
    extra = set(obj) - {"name", "arguments"}
    if extra:
        raise ProtocolError(
            "invalid_arguments",
            f"unexpected tool_call keys: {sorted(extra)}",
        )
    if "name" not in obj or "arguments" not in obj:
        raise ProtocolError(
            "invalid_arguments",
            "tool_call requires name and arguments",
        )
    name = obj["name"]
    if not isinstance(name, str) or _CONTROL_NEWLINE.search(name):
        raise ProtocolError("invalid_arguments", "tool name must be a single-line string")
    if name not in TOOL_NAMES:
        raise ProtocolError("unknown_tool", f"unknown tool {name!r}")
    arguments = obj["arguments"]
    if not isinstance(arguments, dict) or isinstance(arguments, bool):
        raise ProtocolError("invalid_arguments", "arguments must be a JSON object")
    if name == "tree":
        parsed = _parse_tree_args(arguments)
    elif name == "search":
        parsed = _parse_search_args(arguments)
    else:
        parsed = _parse_read_args(arguments)
    return ToolCall(name=name, arguments=parsed)


def _parse_tree_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    extra = set(arguments) - {"path", "depth"}
    if extra:
        raise ProtocolError(
            "invalid_arguments",
            f"unexpected tree arguments: {sorted(extra)}",
        )
    path = TREE_DEFAULT_PATH
    if "path" in arguments:
        path = _require_path_string(arguments["path"], "path")
    depth = TREE_DEFAULT_DEPTH
    if "depth" in arguments:
        depth = _require_int(arguments["depth"], "depth")
    return {"path": path, "depth": depth}


def _parse_search_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    extra = set(arguments) - {"query", "path", "max_results"}
    if extra:
        raise ProtocolError(
            "invalid_arguments",
            f"unexpected search arguments: {sorted(extra)}",
        )
    if "query" not in arguments:
        raise ProtocolError("invalid_arguments", "search requires query")
    query = arguments["query"]
    if not isinstance(query, str):
        raise ProtocolError("invalid_arguments", "query must be a string")
    if _CONTROL_NEWLINE.search(query):
        raise ProtocolError("invalid_arguments", "query must be a single-line string")
    path = SEARCH_DEFAULT_PATH
    if "path" in arguments:
        path = _require_path_string(arguments["path"], "path")
    max_results = SEARCH_DEFAULT_MAX_RESULTS
    if "max_results" in arguments:
        max_results = _require_int(arguments["max_results"], "max_results")
    return {"query": query, "path": path, "max_results": max_results}


def _parse_read_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    extra = set(arguments) - {"path", "start_line", "end_line"}
    if extra:
        raise ProtocolError(
            "invalid_arguments",
            f"unexpected read arguments: {sorted(extra)}",
        )
    missing = [key for key in ("path", "start_line", "end_line") if key not in arguments]
    if missing:
        raise ProtocolError(
            "invalid_arguments",
            f"read requires {missing}",
        )
    return {
        "path": _require_path_string(arguments["path"], "path"),
        "start_line": _require_int(arguments["start_line"], "start_line"),
        "end_line": _require_int(arguments["end_line"], "end_line"),
    }


def _parse_final(payload: Any) -> FinalAction:
    try:
        obj = _require_object(payload, "final")
        extra = set(obj) - {"locations"}
        if extra:
            raise ProtocolError(
                "malformed_final",
                f"unexpected final keys: {sorted(extra)}",
            )
        if "locations" not in obj:
            raise ProtocolError("malformed_final", "final requires locations")
        raw_locations = obj["locations"]
        if not isinstance(raw_locations, list):
            raise ProtocolError("malformed_final", "locations must be a JSON array")
        locations: list[Location] = []
        for index, item in enumerate(raw_locations):
            locations.append(_parse_location(item, index))
        return FinalAction(locations=tuple(locations))
    except ProtocolError as exc:
        if exc.code == "malformed_final":
            raise
        raise ProtocolError("malformed_final", exc.message) from exc


def _parse_location(item: Any, index: int) -> Location:
    obj = _require_object(item, f"locations[{index}]")
    extra = set(obj) - {"path", "symbol"}
    if extra:
        raise ProtocolError(
            "malformed_final",
            f"unexpected location keys at {index}: {sorted(extra)}",
        )
    if "path" not in obj:
        raise ProtocolError("malformed_final", f"locations[{index}] missing path")
    path = obj["path"]
    if not isinstance(path, str) or not path or _CONTROL_NEWLINE.search(path):
        raise ProtocolError(
            "malformed_final",
            f"locations[{index}].path must be a nonempty single-line string",
        )
    if not is_safe_repo_path(path):
        raise ProtocolError(
            "malformed_final",
            f"locations[{index}].path is not a safe repo-relative path",
        )
    symbol: str | None = None
    if "symbol" in obj:
        symbol = obj["symbol"]
        if not isinstance(symbol, str) or not symbol or _CONTROL_NEWLINE.search(symbol):
            raise ProtocolError(
                "malformed_final",
                f"locations[{index}].symbol must be a nonempty single-line string",
            )
    return Location(path=path, symbol=symbol)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or isinstance(value, bool):
        raise ProtocolError("invalid_arguments", f"{label} must be a JSON object")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("invalid_arguments", f"{field} must be an integer")
    return value


def _require_path_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or _CONTROL_NEWLINE.search(value):
        raise ProtocolError(
            "invalid_arguments",
            f"{field} must be a nonempty single-line string",
        )
    return value


def locations_payload(action: FinalAction) -> dict[str, Any]:
    """Preserve submission order. Do not sort, dedup, or canonicalize."""
    items: list[dict[str, str]] = []
    for location in action.locations:
        item = {"path": location.path}
        if location.symbol is not None:
            item["symbol"] = location.symbol
        items.append(item)
    return {"locations": items}
