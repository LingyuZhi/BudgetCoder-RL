def test_package_import():
    import budget_coder_rl  # noqa: F401


def test_agent_loop_import():
    from budget_coder_rl.agent_loop import (  # noqa: F401
        DummyTwoTurnAgentLoop,
        RepoExplorationAgentLoop,
    )


def test_env_import():
    from budget_coder_rl.env import RepoEnvironment, TaskRef  # noqa: F401


def test_protocol_and_session_import():
    from budget_coder_rl.env import ExplorationSession, ExplorationTools  # noqa: F401
    from budget_coder_rl.protocol import parse_action  # noqa: F401
