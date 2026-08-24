"""Stage 1 repository environment (M2A workspace + M2B exploration scaffold).

Read-only snapshot workspace over M1 bare object stores, plus structured
tree/search/read/finish tools. No AgentLoop, reward, or budget.
"""

from budget_coder_rl.env.repo_workspace import (
    CommitNotFoundError,
    InvalidCommitRefError,
    PathEscapeError,
    RepoEnvironment,
    RepoFileView,
    RepoUnavailableError,
    RepoWorkspace,
    RepoWorkspaceError,
    TaskRef,
    WorkspaceIdentityError,
    SNAPSHOT_RELPATH,
    require_full_commit_sha,
    snapshot_repo_key,
    swe_gym_snapshots_root,
)
from budget_coder_rl.env.session import ExplorationSession, StepResult
from budget_coder_rl.env.tools import ExplorationTools, ToolError

__all__ = [
    "CommitNotFoundError",
    "ExplorationSession",
    "ExplorationTools",
    "InvalidCommitRefError",
    "PathEscapeError",
    "RepoEnvironment",
    "RepoFileView",
    "RepoUnavailableError",
    "RepoWorkspace",
    "RepoWorkspaceError",
    "SNAPSHOT_RELPATH",
    "StepResult",
    "TaskRef",
    "ToolError",
    "WorkspaceIdentityError",
    "require_full_commit_sha",
    "snapshot_repo_key",
    "swe_gym_snapshots_root",
]
