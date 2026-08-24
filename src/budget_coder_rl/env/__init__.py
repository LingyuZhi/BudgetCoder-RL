"""Stage 1 repository environment (M2A).

Read-only snapshot workspace over M1 bare object stores. No exploration
tools, AgentLoop, reward, or budget.
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

__all__ = [
    "CommitNotFoundError",
    "InvalidCommitRefError",
    "PathEscapeError",
    "RepoEnvironment",
    "RepoFileView",
    "RepoUnavailableError",
    "RepoWorkspace",
    "RepoWorkspaceError",
    "SNAPSHOT_RELPATH",
    "TaskRef",
    "WorkspaceIdentityError",
    "require_full_commit_sha",
    "snapshot_repo_key",
    "swe_gym_snapshots_root",
]
