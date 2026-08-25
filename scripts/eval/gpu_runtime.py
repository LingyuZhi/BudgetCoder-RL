"""Shared GPU AgentLoopManager bootstrap for M3B/M3C (M2D/M3A path, no RewardLoop)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from budget_coder_rl.eval.m3b import QWEN3_SAMPLING

PROMPT_LENGTH = 16384
RESPONSE_LENGTH = 16384
MAX_MODEL_LEN = 32768
AGENT_LOOP_CONFIG_RELPATH = "configs/agent_loop/repo_exploration_m3b.yaml"
M3C_AGENT_LOOP_CONFIG_RELPATH = "configs/agent_loop/repo_exploration_m3c.yaml"


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


def get_verl_info() -> dict[str, Any]:
    import verl

    source_root = Path(verl.__file__).resolve().parents[1]
    info: dict[str, Any] = {
        "version": getattr(verl, "__version__", "unknown"),
        "source_root": str(source_root),
        "commit": "unknown",
    }
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        ).strip()
        info["dirty"] = bool(dirty)
    except Exception:
        pass
    return info


def object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def pick_free_gpu() -> dict[str, Any]:
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing not in {None, ""}:
        return {"cuda_visible_devices": existing, "source": "env"}
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        return {"cuda_visible_devices": "0", "source": "fallback", "error": str(exc)}
    free: list[tuple[int, int]] = []
    busy: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        index = int(parts[0])
        used = int(float(parts[1]))
        total = int(float(parts[2]))
        util = int(float(parts[3])) if len(parts) > 3 and parts[3] else 0
        row = {"index": index, "memory_used_mi": used, "memory_total_mi": total, "util": util}
        if used < 512 and util < 5:
            free.append((used, index))
        else:
            busy.append(row)
    if not free:
        raise SystemExit(
            "HARD FAIL: no idle GPU (used<512MiB). "
            f"busy={busy}. Set CUDA_VISIBLE_DEVICES explicitly if this is yours."
        )
    chosen = sorted(free)[0][1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    return {
        "cuda_visible_devices": str(chosen),
        "source": "nvidia-smi",
        "busy": busy,
    }


def build_config(
    model_path: str,
    *,
    n_gpus: int,
    tensor_model_parallel_size: int,
    agent_loop_config: str,
    sampling: Mapping[str, Any] | None = None,
) -> Any:
    import verl
    from hydra import compose, initialize_config_dir

    sampling = dict(sampling or QWEN3_SAMPLING)
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
    rollout.temperature = float(sampling["temperature"])
    rollout.top_p = float(sampling["top_p"])
    rollout.top_k = int(sampling["top_k"])
    rollout.do_sample = True
    # val_kwargs stay greedy so a mistaken validate=True is obvious in traces.
    rollout.val_kwargs.temperature = 0.0
    rollout.val_kwargs.top_p = 1.0
    rollout.val_kwargs.top_k = -1
    rollout.val_kwargs.do_sample = False
    rollout.agent.num_workers = 1
    rollout.agent.default_agent_loop = "repo_exploration"
    rollout.agent.agent_loop_config_path = str(agent_loop_config)
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = n_gpus
    return config


def assert_sampling_config(config: Any) -> dict[str, Any]:
    rollout = config.actor_rollout_ref.rollout
    recorded = {
        "temperature": float(rollout.temperature),
        "top_p": float(rollout.top_p),
        "top_k": int(rollout.top_k),
        "n": int(rollout.n),
        "do_sample": bool(rollout.do_sample),
    }
    if recorded["temperature"] == 0:
        raise SystemExit(
            "HARD FAIL: rollout.temperature==0 would make vLLM greedy; "
            "M3B/M3C must use Qwen3 sampling 0.7/0.8/20"
        )
    if recorded["n"] != 1:
        raise SystemExit(
            "HARD FAIL: AgentLoop measurement forbids vLLM n>1; "
            "grouped rollouts expand tasks with distinct sampling_seed"
        )
    if abs(recorded["temperature"] - QWEN3_SAMPLING["temperature"]) > 1e-6:
        raise SystemExit(f"HARD FAIL: unexpected temperature {recorded['temperature']}")
    if abs(recorded["top_p"] - QWEN3_SAMPLING["top_p"]) > 1e-6:
        raise SystemExit(f"HARD FAIL: unexpected top_p {recorded['top_p']}")
    if recorded["top_k"] != QWEN3_SAMPLING["top_k"]:
        raise SystemExit(f"HARD FAIL: unexpected top_k {recorded['top_k']}")
    return recorded


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
        extras.append(as_mapping(item.get("extra_info")))
        names.append("repo_exploration")
        indices.append(offset)
    return DataProto(
        non_tensor_batch={
            "raw_prompt": object_array(raw_prompts),
            "extra_info": object_array(extras),
            "agent_name": object_array(names),
            "index": np.array(indices, dtype=object),
        },
        meta_info={"validate": validate, "global_steps": 0},
    )


def nt(result, key: str, index: int) -> Any:
    payload = result.non_tensor_batch.get(key)
    if payload is None:
        return None
    return payload[index]
