"""CPU/mock exploration session over an M2A workspace.

Maps protocol/tool misuse to deterministic observations. Infrastructure
failures (``RepoWorkspaceError``, unexpected ``OSError``) propagate.
Does not score localization, access oracle labels, or track budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from budget_coder_rl.env.repo_workspace import RepoWorkspace
from budget_coder_rl.env.tools import ExplorationTools, ToolError
from budget_coder_rl.protocol.observation import format_error, format_finish
from budget_coder_rl.protocol.parser import (
    FinalAction,
    ProtocolError,
    locations_payload,
    parse_action,
)


@dataclass(frozen=True)
class StepResult:
    observation: str
    terminal: bool
    submission: dict[str, Any] | None
    error_kind: str | None
    turn: int
    action_type: str
    termination: str | None = None


@dataclass
class ExplorationSession:
    """Scripted/mock policy loop: ``step(raw_text)`` once per turn."""

    workspace: RepoWorkspace
    tools: ExplorationTools = field(init=False)
    events: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    terminal: bool = False
    submission: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.tools = ExplorationTools(self.workspace.view())

    def step(self, raw_text: str) -> StepResult:
        if self.terminal:
            raise RuntimeError("session already terminated")
        self.turn += 1
        try:
            action = parse_action(raw_text)
        except ProtocolError as exc:
            observation = format_error(
                tool="protocol",
                code=exc.code,
                message=exc.message,
            )
            result = StepResult(
                observation=observation,
                terminal=False,
                submission=None,
                error_kind="protocol",
                turn=self.turn,
                action_type="protocol",
            )
            self._record(raw_text, result)
            return result

        if isinstance(action, FinalAction):
            payload = locations_payload(action)
            observation = format_finish(payload)
            self.terminal = True
            self.submission = payload
            result = StepResult(
                observation=observation,
                terminal=True,
                submission=payload,
                error_kind=None,
                turn=self.turn,
                action_type="finish",
                termination="finish",
            )
            self._record(raw_text, result)
            return result

        try:
            observation = self.tools.execute(action.name, action.arguments)
        except ToolError as exc:
            observation = format_error(
                tool=action.name,
                code=exc.code,
                message=exc.message,
            )
            result = StepResult(
                observation=observation,
                terminal=False,
                submission=None,
                error_kind="tool",
                turn=self.turn,
                action_type=action.name,
            )
            self._record(raw_text, result)
            return result

        result = StepResult(
            observation=observation,
            terminal=False,
            submission=None,
            error_kind=None,
            turn=self.turn,
            action_type=action.name,
        )
        self._record(raw_text, result)
        return result

    def _record(self, raw_text: str, result: StepResult) -> None:
        self.events.append(
            {
                "turn": result.turn,
                "raw_action": raw_text,
                "action_type": result.action_type,
                "observation": result.observation,
                "terminal": result.terminal,
                "error_kind": result.error_kind,
                "termination": result.termination,
                "submission": result.submission,
            }
        )
