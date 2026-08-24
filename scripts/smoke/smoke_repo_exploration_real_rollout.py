#!/usr/bin/env python
"""M2D smoke: real Qwen3-4B async RepoExplorationAgentLoop on the M0 path.

Path (no standalone model.generate shortcut):

    RLHFDataset row → DataProto
        → ray → ActorRolloutRefWorker → LLMServerManager (vLLM async)
        → AgentLoopManager → AgentLoopWorker → RepoExplorationAgentLoop
        → generate_sequences() → padded DataProto

D1 is greedy infrastructure smoke on pydantic__pydantic-4882.
D2 is sampled behavioral smoke on a few real tasks. No reward / GRPO / gold score.

Usage (compute node, pinned conda env ``verl``):

    python scripts/smoke/smoke_repo_exploration_real_rollout.py --mode d1
    python scripts/smoke/smoke_repo_exploration_real_rollout.py --mode d2
    python scripts/smoke/smoke_repo_exploration_real_rollout.py --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_repo_workspace import PREFERRED_REPOS, load_task_rows  # noqa: E402
from smoke_rlhf_dataset import build_dataset, resolve_tokenizer_path  # noqa: E402

from budget_coder_rl.agent_loop.rollout_verify import (  # noqa: E402
    inspect_turn_boundary,
    segment_decomposition,
    verify_padded_sample,
)
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    bcrl_data_root,
    cache_path_for_repo,
    is_git_dir,
    swe_gym_repos_root,
)
from budget_coder_rl.env import RepoEnvironment, TaskRef  # noqa: E402

AGENT_LOOP_CONFIG = REPO_ROOT / "configs" / "agent_loop" / "repo_exploration_m2d.yaml"
D1_INSTANCE_ID = "pydantic__pydantic-4882"
PROMPT_LENGTH = 16384
RESPONSE_LENGTH = 16384
MAX_MODEL_LEN = 32768
TRACE_NOTE = (
    "Research/debug artifact. AgentLoopOutput / DataProto token arrays are the "
    "training truth. Do not rebuild RL token trajectories from this JSONL."
)
D1_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "do_sample": False}
D2_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "do_sample": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("d1", "d2", "both"), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--repos-root", type=Path, default=None)
    parser.add_argument("--snapshots-root", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--d2-tasks", type=int, default=3)
    parser.add_argument("--d2-rollouts", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke",
    )
    return parser.parse_args(argv)


def resolve_model_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    env_path = os.environ.get("BCRL_MODEL_PATH")
    if env_path:
        return env_path
    data_root = os.environ.get(
        "BCRL_DATA_ROOT", os.path.expanduser("~/my_data/budget-coder-rl")
    )
    return str(Path(data_root) / "models" / "Qwen3-4B-Instruct-2507")


def get_verl_info() -> dict[str, str]:
    import verl

    source_root = Path(verl.__file__).resolve().parents[1]
    info = {
        "version": getattr(verl, "__version__", "unknown"),
        "source_root": str(source_root),
        "commit": "unknown",
    }
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass
    return info


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def find_dataset_row(dataset, instance_id: str) -> dict[str, Any]:
    for index in range(len(dataset)):
        item = dataset[index]
        extra = _as_mapping(item.get("extra_info"))
        if str(extra.get("instance_id") or "") == instance_id:
            return item
    raise SystemExit(f"instance {instance_id} not in RLHFDataset")


def select_d2_tasks(
    rows: list[TaskRef],
    *,
    repos_root: Path,
    n_tasks: int,
) -> list[TaskRef]:
    by_id = {row.instance_id: row for row in rows}
    selected: list[TaskRef] = []
    seen_repos: set[str] = set()
    preferred = by_id.get(D1_INSTANCE_ID)
    if preferred is not None and is_git_dir(
        cache_path_for_repo(preferred.repo, repos_root)
    ):
        selected.append(preferred)
        seen_repos.add(preferred.repo)
    for repo in PREFERRED_REPOS:
        if len(selected) >= n_tasks:
            break
        if repo in seen_repos:
            continue
        store = cache_path_for_repo(repo, repos_root)
        if not is_git_dir(store):
            continue
        match = next((row for row in rows if row.repo == repo), None)
        if match is None:
            continue
        selected.append(match)
        seen_repos.add(repo)
    if len(selected) < 2:
        raise SystemExit(
            f"need at least 2 local-store D2 tasks, got {[t.instance_id for t in selected]}"
        )
    return selected[:n_tasks]


def ensure_snapshot(task: TaskRef, env: RepoEnvironment) -> None:
    store = cache_path_for_repo(task.repo, env.repos_root)
    if not is_git_dir(store):
        raise SystemExit(f"HARD FAIL: local object store missing: {store}")
    workspace = env.prepare(task)
    workspace.validate()


def build_config(
    model_path: str,
    *,
    n_gpus: int,
    tensor_model_parallel_size: int,
) -> Any:
    import verl
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
            ],
        )

    config.data.max_prompt_length = PROMPT_LENGTH
    config.data.max_response_length = RESPONSE_LENGTH
    config.data.filter_overlong_prompts = False
    config.data.truncation = "error"
    config.actor_rollout_ref.model.path = model_path
    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm"
    rollout.mode = "async"
    rollout.enforce_eager = True
    rollout.prompt_length = PROMPT_LENGTH
    rollout.response_length = RESPONSE_LENGTH
    rollout.max_model_len = MAX_MODEL_LEN
    rollout.max_num_batched_tokens = MAX_MODEL_LEN
    rollout.tensor_model_parallel_size = tensor_model_parallel_size
    rollout.n = 1
    rollout.skip_tokenizer_init = True
    rollout.layered_summon = True
    rollout.checkpoint_engine.update_weights_bucket_megabytes = 512
    rollout.temperature = D2_SAMPLING["temperature"]
    rollout.top_p = D2_SAMPLING["top_p"]
    rollout.top_k = D2_SAMPLING["top_k"]
    rollout.do_sample = True
    rollout.val_kwargs.temperature = D1_SAMPLING["temperature"]
    rollout.val_kwargs.top_p = D1_SAMPLING["top_p"]
    rollout.val_kwargs.top_k = D1_SAMPLING["top_k"]
    rollout.val_kwargs.do_sample = False
    rollout.agent.num_workers = 1
    rollout.agent.default_agent_loop = "repo_exploration"
    rollout.agent.agent_loop_config_path = str(AGENT_LOOP_CONFIG)
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = n_gpus
    return config


def init_agent_loop_manager(config):
    """Real veRL runtime bootstrap (public APIs only, reward loop skipped)."""
    import ray

    from verl.checkpoint_engine import CheckpointEngineManager
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import omega_conf_to_dataclass
    from verl.utils.device import get_device_name
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.rollout.llm_server import LLMServerManager

    assert config.actor_rollout_ref.rollout.mode == "async"
    global_pool_id = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        },
        mapping={Role.ActorRollout: global_pool_id},
    )
    resource_pool_manager.create_resource_pool()
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    worker_dict_cls = create_colocated_worker_cls(
        class_dict={"actor_rollout": actor_rollout_cls}
    )
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool_manager.get_resource_pool(Role.ActorRollout),
        ray_cls_with_init=worker_dict_cls,
        device_name=get_device_name(),
    )
    actor_rollout_wg = wg_dict.spawn(prefix_set={"actor_rollout"})["actor_rollout"]
    actor_rollout_wg.init_model()
    llm_server_manager = LLMServerManager.create(
        config=config, worker_group=actor_rollout_wg
    )
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=None,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()
    return agent_loop_manager


def build_batch(items: list[dict[str, Any]], *, validate: bool):
    from verl.protocol import DataProto

    raw_prompts = []
    extras = []
    names = []
    indices = []
    for offset, item in enumerate(items):
        raw_prompts.append(item.get("raw_prompt"))
        extras.append(_as_mapping(item.get("extra_info")))
        names.append("repo_exploration")
        indices.append(offset)
    return DataProto(
        non_tensor_batch={
            "raw_prompt": _object_array(raw_prompts),
            "extra_info": _object_array(extras),
            "agent_name": _object_array(names),
            "index": np.array(indices, dtype=object),
        },
        meta_info={"validate": validate, "global_steps": 0},
    )


def _nt(result, key: str, index: int) -> Any:
    payload = result.non_tensor_batch.get(key)
    if payload is None:
        return None
    return payload[index]


def verify_result_row(
    result,
    index: int,
    *,
    prompt_width: int,
    response_width: int,
) -> dict[str, Any]:
    segments = list(_nt(result, "segments", index) or [])
    prompt_ids = list(_nt(result, "unpadded_prompt_ids", index) or [])
    errors = verify_padded_sample(
        prompt_width=prompt_width,
        response_width=response_width,
        prompts_row=result.batch["prompts"][index],
        responses_row=result.batch["responses"][index],
        response_mask_row=result.batch["response_mask"][index],
        attention_mask_row=result.batch["attention_mask"][index],
        unpadded_prompt_ids=prompt_ids,
        segments=segments,
    )
    events = list(_nt(result, "events", index) or [])
    n_assistant = sum(1 for item in segments if item["kind"] == "assistant")
    n_observation = sum(1 for item in segments if item["kind"] == "observation")
    termination = _nt(result, "termination", index)
    if n_assistant < 1:
        errors.append("no assistant generation")
    if termination not in {"finish", "max_turns", "response_length"}:
        errors.append(f"unexpected termination {termination!r}")
    if termination != "finish" and n_assistant < 2 and n_observation < 1:
        errors.append("did not observe multi-turn tool interaction or finish")
    return {
        "index": index,
        "instance_id": _nt(result, "instance_id", index),
        "repo": _nt(result, "repo", index),
        "base_commit": _nt(result, "base_commit", index),
        "termination": termination,
        "final_submission": _nt(result, "final_submission", index),
        "prompt_token_count": _nt(result, "prompt_token_count", index),
        "n_assistant": n_assistant,
        "n_observation": n_observation,
        "segment_decomposition": segment_decomposition(segments),
        "events": events,
        "segments": segments,
        "unpadded_prompt_ids": prompt_ids,
        "sampling_params": _nt(result, "sampling_params", index),
        "max_new_tokens_per_turn": _nt(result, "max_new_tokens_per_turn", index),
        "errors": errors,
        "ok": not errors,
    }


def turn_table(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        raw = str(event.get("raw_action") or "")
        preview = raw if len(raw) <= 180 else raw[:180] + "…"
        rows.append(
            {
                "turn": event.get("turn"),
                "action": event.get("action_name") or event.get("action_type"),
                "parse_error": event.get("parse_error_code"),
                "tool_status": event.get("tool_status"),
                "gen_tokens": event.get("generated_token_count"),
                "obs_tokens": event.get("observation_token_count"),
                "stop_reason": event.get("stop_reason"),
                "raw_preview": preview.replace("\n", "\\n"),
            }
        )
    return rows


def protocol_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        code = event.get("parse_error_code") or event.get("error_code") or "ok"
        if event.get("error_kind") is None and event.get("parse_error_code") is None:
            code = "ok"
        counts[str(code)] = counts.get(str(code), 0) + 1
    return counts


def episode_record(
    stats: dict[str, Any],
    *,
    mode: str,
    model: str,
    sampling: dict[str, Any],
    tokenizer,
) -> dict[str, Any]:
    boundary = None
    if tokenizer is not None and stats.get("segments"):
        boundary = inspect_turn_boundary(
            tokenizer,
            prompt_ids=stats["unpadded_prompt_ids"],
            segments=stats["segments"],
        )
    compact_events = []
    for event in stats["events"]:
        compact_events.append(
            {
                "turn": event.get("turn"),
                "action_type": event.get("action_type"),
                "action_name": event.get("action_name"),
                "action_arguments": event.get("action_arguments"),
                "raw_action": event.get("raw_action"),
                "generated_token_count": event.get("generated_token_count"),
                "stop_reason": event.get("stop_reason"),
                "max_tokens": event.get("max_tokens"),
                "generate_prefix_n": event.get("generate_prefix_n"),
                "parse_error_code": event.get("parse_error_code"),
                "error_kind": event.get("error_kind"),
                "error_code": event.get("error_code"),
                "tool": event.get("tool"),
                "tool_status": event.get("tool_status"),
                "observation_preview": event.get("observation_preview"),
                "observation_token_count": event.get("observation_token_count"),
                "cumulative_response_tokens": event.get("cumulative_response_tokens"),
                "terminal": event.get("terminal"),
                "termination": event.get("termination"),
                "submission": event.get("submission"),
            }
        )
    return {
        "trace_note": TRACE_NOTE,
        "mode": mode,
        "model": model,
        "sampling": sampling,
        "instance_id": stats["instance_id"],
        "repo": stats["repo"],
        "base_commit": stats["base_commit"],
        "initial_prompt_token_count": stats["prompt_token_count"],
        "termination": stats["termination"],
        "final_submission": stats["final_submission"],
        "segment_decomposition": stats["segment_decomposition"],
        "turn_table": turn_table(stats["events"]),
        "protocol_counts": protocol_summary(stats["events"]),
        "turn_boundary": boundary,
        "events": compact_events,
        "segments": [
            {
                "kind": item["kind"],
                "n_tokens": len(item["token_ids"]),
                "token_ids": list(item["token_ids"]),
            }
            for item in stats["segments"]
        ],
        "unpadded_prompt_ids": list(stats["unpadded_prompt_ids"]),
        "errors": stats["errors"],
        "ok": stats["ok"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=True) + "\n")


def print_turn_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n--- {title} ---")
    print(
        f"{'turn':<5} {'action':<12} {'parse':<18} {'A':>4} {'O':>4} "
        f"{'status':<8} {'stop':<12} raw"
    )
    for row in rows:
        print(
            f"{str(row.get('turn')):<5} {str(row.get('action') or '-'):<12} "
            f"{str(row.get('parse_error') or 'ok'):<18} "
            f"{str(row.get('gen_tokens') if row.get('gen_tokens') is not None else '-'):>4} "
            f"{str(row.get('obs_tokens') if row.get('obs_tokens') is not None else '-'):>4} "
            f"{str(row.get('tool_status') or '-'):<8} "
            f"{str(row.get('stop_reason') or '-'):<12} "
            f"{row.get('raw_preview')}"
        )


def run_mode(
    *,
    mode: str,
    items: list[dict[str, Any]],
    validate: bool,
    sampling: dict[str, Any],
    manager,
    tokenizer,
    model_path: str,
    prompt_width_expected: int,
    response_width_expected: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batch = build_batch(items, validate=validate)
    result = manager.generate_sequences(prompts=batch)
    if len(result) != len(items):
        raise SystemExit(f"{mode}: got {len(result)} outputs for {len(items)} inputs")
    prompt_width = int(result.batch["prompts"].size(1))
    response_width = int(result.batch["responses"].size(1))
    width_errors: list[str] = []
    if prompt_width != prompt_width_expected:
        width_errors.append(
            f"postprocess prompt width {prompt_width} != {prompt_width_expected}"
        )
    if response_width != response_width_expected:
        width_errors.append(
            f"postprocess response width {response_width} != {response_width_expected}"
        )
    stats_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index in range(len(result)):
        stats = verify_result_row(
            result,
            index,
            prompt_width=prompt_width,
            response_width=response_width,
        )
        stats["errors"] = list(width_errors) + list(stats["errors"])
        stats["ok"] = not stats["errors"]
        stats_rows.append(stats)
        records.append(
            episode_record(
                stats,
                mode=mode,
                model=model_path,
                sampling=sampling,
                tokenizer=tokenizer,
            )
        )
    return stats_rows, records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    model_path = resolve_model_path(args.model_path)
    if not Path(model_path).exists():
        print(f"HARD FAIL: model path does not exist: {model_path}", file=sys.stderr)
        return 1
    if not AGENT_LOOP_CONFIG.is_file():
        print(f"HARD FAIL: missing {AGENT_LOOP_CONFIG}", file=sys.stderr)
        return 1

    tokenizer_path = resolve_tokenizer_path(None) or model_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_path = (
        args.train.resolve()
        if args.train is not None
        else repo_root / "data" / "processed" / "swe_gym" / "train.parquet"
    )
    if not train_path.is_file():
        print(f"HARD FAIL: missing M1E train parquet {train_path}", file=sys.stderr)
        return 1

    data_root = args.data_root
    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(data_root)
    )
    env = RepoEnvironment(
        repos_root=repos_root,
        snapshots_root=(
            args.snapshots_root.expanduser() if args.snapshots_root is not None else None
        ),
        data_root=data_root,
    )
    rows = load_task_rows(repo_root, args.train, None)
    cache_dir = repo_root / "outputs" / "smoke" / "rlhf_dataset_cache" / "m2d"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(train_path, tokenizer, cache_dir)

    d1_item = find_dataset_row(dataset, D1_INSTANCE_ID)
    d1_extra = _as_mapping(d1_item.get("extra_info"))
    d1_task = TaskRef.from_extra_info(d1_extra)
    ensure_snapshot(d1_task, env)

    d2_items: list[dict[str, Any]] = []
    d2_tasks: list[TaskRef] = []
    if args.mode in {"d2", "both"}:
        d2_tasks = select_d2_tasks(rows, repos_root=repos_root, n_tasks=args.d2_tasks)
        for task in d2_tasks:
            ensure_snapshot(task, env)
            item = find_dataset_row(dataset, task.instance_id)
            for _repeat in range(args.d2_rollouts):
                d2_items.append(item)

    verl_info = get_verl_info()
    print(f"[m2d] model: {model_path}")
    print(f"[m2d] verl: {verl_info['version']} @ {verl_info['commit']}")
    print(f"[m2d] agent loop config: {AGENT_LOOP_CONFIG}")
    print(
        f"[m2d] envelope prompt={PROMPT_LENGTH} response={RESPONSE_LENGTH} "
        f"max_model_len={MAX_MODEL_LEN} gpus={args.n_gpus}"
    )
    print(f"[m2d] D1 task: {d1_task.instance_id} {d1_task.repo} {d1_task.base_commit}")
    if d2_tasks:
        print(
            "[m2d] D2 tasks: "
            + ", ".join(f"{task.instance_id}({task.repo})" for task in d2_tasks)
        )

    import ray
    import torch  # noqa: F401

    config = build_config(
        model_path,
        n_gpus=args.n_gpus,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
    )
    data_root_env = str(bcrl_data_root(data_root))
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "BCRL_DATA_ROOT": data_root_env,
            }
        }
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    traj_root = bcrl_data_root(data_root) / "trajectories" / "m2d"
    output_dir = args.output_dir
    reports: dict[str, Any] = {
        "trace_note": TRACE_NOTE,
        "date": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "verl": verl_info,
        "length_envelope": {
            "prompt_length": PROMPT_LENGTH,
            "response_length": RESPONSE_LENGTH,
            "max_model_len": MAX_MODEL_LEN,
            "max_turns": 6,
            "max_new_tokens_per_turn": 2048,
        },
        "gpu": {
            "n_gpus": args.n_gpus,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
        },
        "agent_loop_config": str(AGENT_LOOP_CONFIG),
        "modes": {},
    }

    status = "PASS"
    try:
        manager = init_agent_loop_manager(config)
        if args.mode in {"d1", "both"}:
            d1_stats, d1_records = run_mode(
                mode="d1",
                items=[d1_item],
                validate=True,
                sampling=D1_SAMPLING,
                manager=manager,
                tokenizer=tokenizer,
                model_path=model_path,
                prompt_width_expected=PROMPT_LENGTH,
                response_width_expected=RESPONSE_LENGTH,
            )
            d1_ok = all(row["ok"] for row in d1_stats)
            reports["modes"]["d1"] = {
                "status": "PASS" if d1_ok else "FAIL",
                "sampling": D1_SAMPLING,
                "samples": [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"segments", "events", "unpadded_prompt_ids"}
                    }
                    for row in d1_stats
                ],
                "turn_table": turn_table(d1_stats[0]["events"]) if d1_stats else [],
                "turn_boundary": d1_records[0].get("turn_boundary") if d1_records else None,
            }
            write_json(output_dir / "m2d_d1_report.json", reports["modes"]["d1"] | {
                "model": model_path,
                "verl": verl_info,
                "length_envelope": reports["length_envelope"],
                "gpu": reports["gpu"],
            })
            write_jsonl(traj_root / f"m2d_d1_{stamp}.jsonl", d1_records)
            print_turn_table(
                f"D1 {d1_task.instance_id}", turn_table(d1_stats[0]["events"])
            )
            print(f"[m2d] D1 segments: {d1_stats[0]['segment_decomposition']}")
            print(f"[m2d] D1 termination: {d1_stats[0]['termination']}")
            if not d1_ok:
                status = "FAIL"
                print("HARD FAIL D1:", file=sys.stderr)
                for row in d1_stats:
                    for err in row["errors"]:
                        print(f"  - {err}", file=sys.stderr)
                if args.mode == "both":
                    print("[m2d] skipping D2 because D1 failed", file=sys.stderr)

        run_d2 = args.mode == "d2" or (args.mode == "both" and status == "PASS")
        if run_d2:
            d2_stats, d2_records = run_mode(
                mode="d2",
                items=d2_items,
                validate=False,
                sampling=D2_SAMPLING,
                manager=manager,
                tokenizer=tokenizer,
                model_path=model_path,
                prompt_width_expected=PROMPT_LENGTH,
                response_width_expected=RESPONSE_LENGTH,
            )
            d2_ok = all(row["ok"] for row in d2_stats)
            reports["modes"]["d2"] = {
                "status": "PASS" if d2_ok else "FAIL",
                "sampling": D2_SAMPLING,
                "n_tasks": len(d2_tasks),
                "n_rollouts_per_task": args.d2_rollouts,
                "samples": [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"segments", "events", "unpadded_prompt_ids"}
                    }
                    for row in d2_stats
                ],
            }
            write_json(output_dir / "m2d_d2_report.json", reports["modes"]["d2"] | {
                "model": model_path,
                "verl": verl_info,
                "length_envelope": reports["length_envelope"],
                "gpu": reports["gpu"],
            })
            write_jsonl(traj_root / f"m2d_d2_{stamp}.jsonl", d2_records)
            for row, rec in zip(d2_stats, d2_records):
                print_turn_table(
                    f"D2 {row['instance_id']} termination={row['termination']}",
                    rec["turn_table"],
                )
                print(f"[m2d] D2 segments: {row['segment_decomposition']}")
            if not d2_ok:
                status = "FAIL"
                print("HARD FAIL D2:", file=sys.stderr)
                for row in d2_stats:
                    for err in row["errors"]:
                        print(f"  - {row['instance_id']}: {err}", file=sys.stderr)
    except Exception as exc:
        status = "FAIL"
        reports["exception"] = repr(exc)
        traceback.print_exc()
        print(f"HARD FAIL: {exc}", file=sys.stderr)
    finally:
        reports["status"] = status
        write_json(output_dir / "m2d_smoke_report.json", reports)
        ray.shutdown()

    print(f"\nreport: {output_dir / 'm2d_smoke_report.json'}")
    print(f"trajectories: {traj_root}")
    print(f"STATUS: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
