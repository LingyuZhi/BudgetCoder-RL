#!/usr/bin/env python
"""CPU-only M2C smoke: real M1 row + M2A/M2B + fake TITO AgentLoop.

Proves dataset kwargs reach ``RepoExplorationAgentLoop`` via the pinned
veRL hydra registration path. Does not start Ray, vLLM, or Qwen generate.

Usage (pinned RL conda env):

    python scripts/smoke/smoke_repo_exploration_agent_loop.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_exploration_scaffold import (  # noqa: E402
    first_tree_file,
    select_smoke_task,
)
from smoke_repo_workspace import load_task_rows  # noqa: E402
from smoke_rlhf_dataset import build_dataset, resolve_tokenizer_path  # noqa: E402

from budget_coder_rl.agent_loop.repo_exploration import (  # noqa: E402
    RepoExplorationAgentLoop,
)
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    cache_path_for_repo,
    is_git_dir,
    swe_gym_repos_root,
)
from budget_coder_rl.env import ExplorationSession, RepoEnvironment  # noqa: E402
from budget_coder_rl.protocol import parse_action  # noqa: E402

AGENT_LOOP_CONFIG = REPO_ROOT / "configs" / "agent" / "repo_exploration.yaml"
TRACE_NOTE = (
    "Research/debug artifact. AgentLoopOutput token arrays are the training "
    "truth. Do not rebuild RL token trajectories from this JSONL."
)


class FakeServerManager:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {"prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)}
        )
        return self._outputs.pop(0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--repos-root", type=Path, default=None)
    parser.add_argument("--snapshots-root", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--raw-parquet", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke" / "m2c_agent_loop_trace.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke" / "m2c_smoke_report.json",
    )
    return parser.parse_args(argv)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def _tool(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict[str, Any]) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


def encode_action(tokenizer, text: str) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    parse_action(decoded)
    return ids


def find_dataset_row(dataset, instance_id: str) -> dict[str, Any]:
    for index in range(len(dataset)):
        item = dataset[index]
        extra = _as_mapping(item.get("extra_info"))
        if str(extra.get("instance_id") or "") == instance_id:
            return item
    raise SystemExit(f"instance {instance_id} not in RLHFDataset")


def mask_segments(response_mask: list[int], segments: list[dict[str, Any]]) -> list[int]:
    expected: list[int] = []
    for item in segments:
        bit = 1 if item["kind"] == "assistant" else 0
        expected.extend([bit] * len(item["token_ids"]))
    return expected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    if tokenizer_path is None:
        print("HARD FAIL: no local Qwen tokenizer; set BCRL_TOKENIZER_PATH", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer
    from verl.workers.rollout.replica import TokenOutput

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_path = (
        args.train.resolve()
        if args.train is not None
        else repo_root / "data" / "processed" / "swe_gym" / "train.parquet"
    )
    if not train_path.is_file():
        print(f"HARD FAIL: missing M1E train parquet {train_path}", file=sys.stderr)
        return 1

    cache_dir = repo_root / "outputs" / "smoke" / "rlhf_dataset_cache" / "m2c"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(train_path, tokenizer, cache_dir)
    rows = load_task_rows(repo_root, args.train, args.raw_parquet)
    task = select_smoke_task(rows)
    item = find_dataset_row(dataset, task.instance_id)
    extra = _as_mapping(item.get("extra_info"))
    raw_prompt = item.get("raw_prompt")

    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(args.data_root)
    )
    store = cache_path_for_repo(task.repo, repos_root)
    if not is_git_dir(store):
        print(f"HARD FAIL: local object store missing: {store}", file=sys.stderr)
        return 1

    env = RepoEnvironment(
        repos_root=repos_root,
        snapshots_root=(
            args.snapshots_root.expanduser() if args.snapshots_root is not None else None
        ),
        data_root=args.data_root,
    )
    workspace = env.prepare(task)
    workspace.validate()
    probe = ExplorationSession(workspace)
    tree_obs = probe.step(_tool("tree", {"path": ".", "depth": 2}))
    if tree_obs.error_kind is not None:
        print(f"HARD FAIL: probe tree failed:\n{tree_obs.observation}", file=sys.stderr)
        return 1
    rel = first_tree_file(tree_obs.observation)
    filename = rel.rsplit("/", 1)[-1]
    query = filename[:-3] if filename.endswith(".py") else filename[:32]
    actions = [
        _tool("tree", {"path": ".", "depth": 2}),
        _tool("search", {"query": query, "path": "."}),
        _final({"locations": [{"path": rel, "symbol": "Scripted.placeholder"}]}),
    ]
    action_ids = [encode_action(tokenizer, text) for text in actions]
    server = FakeServerManager(
        [TokenOutput(token_ids=list(ids)) for ids in action_ids]
    )

    import hydra
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    configs = OmegaConf.load(str(AGENT_LOOP_CONFIG))
    trainer_config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 32768,
                    "response_length": 8192,
                }
            }
        }
    )
    loop = hydra.utils.instantiate(
        configs[0],
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(OmegaConf.create({})),
        tools=ToolListWrap([]),
        repo_environment=env,
    )
    if not isinstance(loop, RepoExplorationAgentLoop):
        raise SystemExit(f"unexpected loop type {type(loop)!r}")

    output = loop.loop.run_until_complete(
        loop.run({"temperature": 1.0}, raw_prompt=raw_prompt, extra_info=extra)
    )

    segments = output.extra_fields["segments"]
    errors: list[str] = []
    if len(output.response_ids) != len(output.response_mask):
        errors.append("response_ids/mask length mismatch")
    if output.extra_fields.get("termination") != "finish":
        errors.append(f"termination={output.extra_fields.get('termination')!r}")
    submission = output.extra_fields.get("final_submission") or {}
    locations = submission.get("locations") if isinstance(submission, dict) else None
    if not locations or locations[0].get("path") != rel:
        errors.append(f"unexpected submission {submission!r}")
    assistant = [item for item in segments if item["kind"] == "assistant"]
    if [item["token_ids"] for item in assistant] != action_ids:
        errors.append("assistant token ids were not preserved exactly")
    expected_mask = mask_segments(output.response_mask, segments)
    if output.response_mask != expected_mask:
        errors.append("response_mask does not match assistant=1 observation=0")
    if server.calls[0]["prompt_ids"] != list(output.prompt_ids):
        errors.append("turn-1 generate prefix != prompt_ids")
    o1 = segments[1]["token_ids"]
    if server.calls[1]["prompt_ids"] != list(output.prompt_ids) + action_ids[0] + o1:
        errors.append("turn-2 generate prefix dropped or re-encoded turn-1 ids")
    if len(output.prompt_ids) > int(loop.prompt_length):
        errors.append("silent prompt truncation would have been required")

    status = "PASS" if not errors else "FAIL"
    report = {
        "status": status,
        "instance_id": extra.get("instance_id"),
        "repo": extra.get("repo"),
        "base_commit": extra.get("base_commit"),
        "agent_loop_config": str(AGENT_LOOP_CONFIG),
        "prompt_token_count": len(output.prompt_ids),
        "response_token_count": len(output.response_ids),
        "num_turns": output.num_turns,
        "termination": output.extra_fields.get("termination"),
        "final_submission": submission,
        "segment_kinds": [item["kind"] for item in segments],
        "assistant_token_counts": [len(ids) for ids in action_ids],
        "observation_token_counts": [
            len(item["token_ids"]) for item in segments if item["kind"] == "observation"
        ],
        "errors": errors,
        "trace_note": TRACE_NOTE,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    trace_record = {
        "trace_note": TRACE_NOTE,
        "instance_id": extra.get("instance_id"),
        "repo": extra.get("repo"),
        "termination": output.extra_fields.get("termination"),
        "final_submission": submission,
        "events": output.extra_fields.get("events"),
        "prompt_token_count": len(output.prompt_ids),
        "response_token_count": len(output.response_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace_record, ensure_ascii=True) + "\n", encoding="utf-8")

    print("SWE-Gym M2C RepoExplorationAgentLoop smoke")
    print(f"task: {extra.get('instance_id')} {extra.get('repo')} {extra.get('base_commit')}")
    print(f"prompt_tokens={len(output.prompt_ids)} response_tokens={len(output.response_ids)}")
    print(f"termination={output.extra_fields.get('termination')}")
    print(f"report={args.report}")
    print(f"trace={args.output}")
    if errors:
        print("HARD FAIL:", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
