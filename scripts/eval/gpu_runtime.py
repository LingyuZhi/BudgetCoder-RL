"""Shared GPU AgentLoopManager bootstrap for M3B/M3C/M4A/M4B/M4C.

M3B/M3C default: RewardLoop skipped, ``rollout.n=1``.
M4A: optional RewardLoop handles and trainer-level ``rollout.n=4``.
M4B: LoRA + one-step ``RayPPOTrainer`` config on the same freeze envelope.
M4C: official FSDP save_freq=1 then resume_path reload; same freeze envelope.
M5: main-run / pilot GRPO+LoRA from stage1_m5_main.json knobs.
"""

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


def require_visible_gpus(n: int = 2, *, idle: bool = True) -> dict[str, Any]:
    """Require exactly ``n`` visible GPUs. Never silently fall back to 1 GPU."""
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        raise SystemExit(f"HARD FAIL: nvidia-smi failed while requiring {n} GPUs: {exc}") from exc
    physical: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        physical.append(
            {
                "index": int(float(parts[0])),
                "name": parts[1],
                "memory_used_mi": int(float(parts[2])),
                "memory_total_mi": int(float(parts[3])),
                "util": int(float(parts[4])) if len(parts) > 4 and parts[4] else 0,
            }
        )
    if existing not in {None, ""}:
        wanted = [item.strip() for item in str(existing).split(",") if item.strip()]
        if len(wanted) != int(n):
            raise SystemExit(
                f"HARD FAIL: CUDA_VISIBLE_DEVICES={existing!r} does not list exactly {n} GPUs"
            )
        selected = []
        for token in wanted:
            match = next((row for row in physical if str(row["index"]) == token), None)
            if match is None and token.isdigit() and int(token) < len(physical):
                match = physical[int(token)]
                match = {**match, "visible_index": int(token)}
            if match is None:
                raise SystemExit(
                    f"HARD FAIL: CUDA_VISIBLE_DEVICES token {token!r} is not a visible GPU"
                )
            selected.append(match)
        if idle:
            busy = [
                row
                for row in selected
                if int(row["memory_used_mi"]) >= 512 or int(row["util"]) >= 5
            ]
            if busy:
                raise SystemExit(
                    f"HARD FAIL: required GPUs are busy: {busy}. "
                    "Unset leftover jobs or set CUDA_VISIBLE_DEVICES to idle devices."
                )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(wanted)
        return {
            "cuda_visible_devices": ",".join(wanted),
            "n_gpus": int(n),
            "source": "env",
            "physical": physical,
            "selected": selected,
        }
    if len(physical) < int(n):
        raise SystemExit(
            f"HARD FAIL: need {n} GPUs, nvidia-smi reports {len(physical)}: {physical}"
        )
    selected = physical[: int(n)]
    if idle:
        busy = [
            row
            for row in selected
            if int(row["memory_used_mi"]) >= 512 or int(row["util"]) >= 5
        ]
        if busy:
            raise SystemExit(f"HARD FAIL: first {n} GPUs are busy: {busy}")
    visible = ",".join(str(row["index"]) for row in selected)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return {
        "cuda_visible_devices": visible,
        "n_gpus": int(n),
        "source": "nvidia-smi",
        "physical": physical,
        "selected": selected,
    }


def build_config(
    model_path: str,
    *,
    n_gpus: int,
    tensor_model_parallel_size: int,
    agent_loop_config: str,
    sampling: Mapping[str, Any] | None = None,
    rollout_n: int = 1,
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
    rollout.n = int(rollout_n)
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


def assert_sampling_config(
    config: Any,
    *,
    require_rollout_n: int = 1,
) -> dict[str, Any]:
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
            "M3B/M3C/M4A/M4B/M4C must use Qwen3 sampling 0.7/0.8/20"
        )
    if recorded["n"] != int(require_rollout_n):
        if int(require_rollout_n) == 1:
            raise SystemExit(
                "HARD FAIL: AgentLoop measurement forbids vLLM n>1; "
                "grouped rollouts expand tasks with distinct sampling_seed"
            )
        raise SystemExit(
            f"HARD FAIL: expected actor_rollout_ref.rollout.n="
            f"{require_rollout_n} (GRPO group size), got {recorded['n']}"
        )
    if abs(recorded["temperature"] - QWEN3_SAMPLING["temperature"]) > 1e-6:
        raise SystemExit(f"HARD FAIL: unexpected temperature {recorded['temperature']}")
    if abs(recorded["top_p"] - QWEN3_SAMPLING["top_p"]) > 1e-6:
        raise SystemExit(f"HARD FAIL: unexpected top_p {recorded['top_p']}")
    if recorded["top_k"] != QWEN3_SAMPLING["top_k"]:
        raise SystemExit(f"HARD FAIL: unexpected top_k {recorded['top_k']}")
    return recorded


def apply_reward_loop_config(
    config: Any,
    *,
    reward_fn_path: str,
    reward_fn_name: str = "compute_score",
    num_workers: int = 2,
) -> Any:
    """Attach a rule-based RewardLoop custom function. No reward model GPU."""
    from omegaconf import open_dict

    with open_dict(config):
        config.reward.custom_reward_function.path = str(reward_fn_path)
        config.reward.custom_reward_function.name = str(reward_fn_name)
        config.reward.num_workers = int(num_workers)
        config.reward.reward_model.enable = False
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
    return config


def apply_m4b_train_config(
    config: Any,
    *,
    train_files: str,
    val_files: str,
    n_tasks: int,
    n_gpus: int,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    default_local_dir: str | None = None,
) -> Any:
    """Minimal one-step GRPO+LoRA trainer settings. Does not edit freeze JSON."""
    from omegaconf import open_dict

    with open_dict(config):
        config.actor_rollout_ref.model.lora_rank = int(lora_rank)
        config.actor_rollout_ref.model.lora_alpha = int(lora_alpha)
        config.actor_rollout_ref.model.target_modules = "all-linear"
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.rollout.load_format = "safetensors"
        config.actor_rollout_ref.actor.strategy = "fsdp"
        config.actor_rollout_ref.actor.ppo_mini_batch_size = int(n_tasks)
        config.actor_rollout_ref.actor.ppo_epochs = 1
        config.actor_rollout_ref.actor.entropy_coeff = 0.0
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.actor_rollout_ref.actor.calculate_entropy = False
        config.actor_rollout_ref.actor.use_dynamic_bsz = True
        # After no-padding, E003 seqs were ~3k response + prompt; 32768 packed
        # all 8 trajectories into one backward and OOM'd. 8192 keeps
        # max_token_len >= typical seq while splitting the optimizer microbatch.
        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = 8192
        config.actor_rollout_ref.actor.optim.lr_warmup_steps = 0
        config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz = True
        config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = MAX_MODEL_LEN
        # Same as hydra/M4A default. 0.4 left only ~2.4GiB KV, below 32k need.
        config.data.train_files = str(train_files)
        config.data.val_files = str(val_files)
        config.data.train_batch_size = int(n_tasks)
        config.data.shuffle = False
        config.data.dataloader_num_workers = 0
        config.data.filter_overlong_prompts = False
        config.data.truncation = "error"
        config.data.max_prompt_length = PROMPT_LENGTH
        config.data.max_response_length = RESPONSE_LENGTH
        config.data.return_raw_chat = True
        config.trainer.total_epochs = 1
        config.trainer.total_training_steps = 1
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.save_freq = -1
        config.trainer.logger = ["console"]
        config.trainer.resume_mode = "disable"
        config.trainer.nnodes = 1
        config.trainer.n_gpus_per_node = int(n_gpus)
        config.trainer.project_name = "budget-coder-rl"
        config.trainer.experiment_name = "E003-m4b"
        config.trainer.critic_warmup = 0
        if default_local_dir:
            config.trainer.default_local_dir = str(default_local_dir)
        config.critic.enable = False
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
    return config


def apply_m4c_save_config(
    config: Any,
    *,
    train_files: str,
    val_files: str,
    n_tasks: int,
    n_gpus: int,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    default_local_dir: str | None = None,
) -> Any:
    """M4B one-step envelope plus official ``save_freq=1`` FSDP checkpointing."""
    apply_m4b_train_config(
        config,
        train_files=train_files,
        val_files=val_files,
        n_tasks=n_tasks,
        n_gpus=n_gpus,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        default_local_dir=default_local_dir,
    )
    from omegaconf import open_dict

    with open_dict(config):
        config.trainer.save_freq = 1
        config.trainer.max_actor_ckpt_to_keep = 1
        config.trainer.experiment_name = "E009-m4c"
        config.trainer.del_local_ckpt_after_load = False
    return config


def apply_m4c_reload_config(
    config: Any,
    *,
    train_files: str,
    val_files: str,
    n_tasks: int,
    n_gpus: int,
    resume_from_path: str,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    default_local_dir: str | None = None,
) -> Any:
    """Fresh-process FSDP resume. Does not run a second optimizer step."""
    apply_m4b_train_config(
        config,
        train_files=train_files,
        val_files=val_files,
        n_tasks=n_tasks,
        n_gpus=n_gpus,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        default_local_dir=default_local_dir,
    )
    from omegaconf import open_dict

    with open_dict(config):
        config.actor_rollout_ref.rollout.n = 1
        config.trainer.save_freq = -1
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.resume_mode = "resume_path"
        config.trainer.resume_from_path = str(resume_from_path)
        config.trainer.experiment_name = "E009-m4c-reload"
        config.trainer.del_local_ckpt_after_load = False
        config.data.train_batch_size = int(n_tasks)
    if "global_step_" not in str(resume_from_path):
        raise SystemExit(
            "HARD FAIL: resume_from_path must contain global_step_ "
            f"(got {resume_from_path!r})"
        )
    return config


def apply_m5_train_config(
    config: Any,
    *,
    train_files: str,
    val_files: str,
    n_tasks: int,
    n_gpus: int,
    ppo_max_token_len_per_gpu: int,
    total_training_steps: int,
    experiment_name: str,
    default_local_dir: str,
    save_freq: int,
    max_actor_ckpt_to_keep: int,
    resume_mode: str = "disable",
    lora_rank: int = 16,
    lora_alpha: int = 16,
    actor_lr: float = 1e-6,
    seed: int = 20260826,
    calculate_entropy: bool = True,
    wandb: bool = True,
    wandb_proxy: str | None = None,
) -> Any:
    """M5 GRPO+LoRA trainer settings. Does not edit freeze JSON."""
    from omegaconf import open_dict

    apply_m4b_train_config(
        config,
        train_files=train_files,
        val_files=val_files,
        n_tasks=n_tasks,
        n_gpus=n_gpus,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        default_local_dir=default_local_dir,
    )
    logger = ["console", "wandb"] if wandb else ["console"]
    with open_dict(config):
        config.actor_rollout_ref.actor.ppo_mini_batch_size = int(n_tasks)
        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = int(
            ppo_max_token_len_per_gpu
        )
        config.actor_rollout_ref.actor.optim.lr = float(actor_lr)
        config.actor_rollout_ref.actor.optim.lr_warmup_steps = 0
        config.actor_rollout_ref.actor.calculate_entropy = bool(calculate_entropy)
        config.actor_rollout_ref.actor.entropy_coeff = 0.0
        config.actor_rollout_ref.actor.use_kl_loss = False
        config.actor_rollout_ref.actor.use_dynamic_bsz = True
        config.actor_rollout_ref.actor.data_loader_seed = int(seed)
        config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = MAX_MODEL_LEN
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
        config.data.train_batch_size = int(n_tasks)
        config.data.seed = int(seed)
        config.data.shuffle = False
        config.trainer.total_epochs = 1
        config.trainer.total_training_steps = int(total_training_steps)
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.save_freq = int(save_freq)
        config.trainer.max_actor_ckpt_to_keep = int(max_actor_ckpt_to_keep)
        config.trainer.logger = list(logger)
        config.trainer.resume_mode = str(resume_mode)
        config.trainer.nnodes = 1
        config.trainer.n_gpus_per_node = int(n_gpus)
        config.trainer.project_name = "budget-coder-rl"
        config.trainer.experiment_name = str(experiment_name)
        config.trainer.default_local_dir = str(default_local_dir)
        config.trainer.del_local_ckpt_after_load = False
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
        if wandb_proxy:
            config.trainer.wandb_proxy = str(wandb_proxy)
    return config


def init_agent_loop_manager(config, reward_loop_worker_handles=None):
    """Real veRL runtime bootstrap. RewardLoop handles optional (M4A)."""
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
        reward_loop_worker_handles=reward_loop_worker_handles,
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
    data_sources = []
    reward_models = []
    has_reward_fields = False
    for offset, item in enumerate(items):
        extra = as_mapping(item.get("extra_info"))
        raw_prompts.append(item.get("raw_prompt"))
        extras.append(extra)
        names.append("repo_exploration")
        if extra.get("index") is not None:
            indices.append(extra.get("index"))
        else:
            indices.append(offset)
        if "data_source" in item or "reward_model" in item:
            has_reward_fields = True
        data_sources.append(item.get("data_source"))
        reward_models.append(item.get("reward_model"))
    non_tensor: dict[str, Any] = {
        "raw_prompt": object_array(raw_prompts),
        "extra_info": object_array(extras),
        "agent_name": object_array(names),
        "index": np.array(indices, dtype=object),
    }
    if has_reward_fields:
        non_tensor["data_source"] = object_array(data_sources)
        non_tensor["reward_model"] = object_array(reward_models)
    return DataProto(
        non_tensor_batch=non_tensor,
        meta_info={"validate": validate, "global_steps": 0},
    )


def nt(result, key: str, index: int) -> Any:
    payload = result.non_tensor_batch.get(key)
    if payload is None:
        return None
    return payload[index]
