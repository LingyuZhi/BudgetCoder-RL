"""M5 trainer hooks: BCRL metrics JSONL + W&B extras on the official fit path.

Does not implement an optimizer, LoRA fingerprinting, or a custom vLLM server.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any

import ray

from budget_coder_rl.eval.m4a import GROUP_N
from budget_coder_rl.eval.m4b import write_json
from budget_coder_rl.eval.m5a import (
    OUTPUT_ENV,
    append_jsonl,
    coerce_sequence,
    compact_episode_from_extra,
    compute_bcrl_step_metrics,
)
from budget_coder_rl.train.m4b_trainer import (
    _row_extra_fields,
    m4b_collate_fn,
)

from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.workers.engine_workers import ActorRolloutRefWorker


def evidence_dir() -> Path:
    raw = os.environ.get(OUTPUT_ENV)
    if not raw:
        raise RuntimeError(f"{OUTPUT_ENV} is not set")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nt(batch, key: str, index: int) -> Any:
    payload = batch.non_tensor_batch.get(key)
    if payload is None:
        return None
    return payload[index]


def _as_float_scalar(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "sum") and getattr(value, "ndim", 0) > 0:
        value = value.sum()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def extra_row_from_batch(batch, index: int) -> dict[str, Any]:
    extra = _row_extra_fields(batch, index)
    for key in (
        "file_f1",
        "symbol_f1",
        "parse_ok",
        "localization_score",
        "symbol_status",
        "submission_missing",
        "score",
    ):
        if key not in extra:
            value = _nt(batch, key, index)
            if value is not None:
                extra[key] = value
    return extra


def _scalar_reward(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return _as_float_scalar(value)
    except (TypeError, ValueError):
        return None


def batch_bcrl_metrics(batch) -> dict[str, Any]:
    uids = [str(item) for item in coerce_sequence(batch.non_tensor_batch.get("uid"))]
    n = len(uids)
    rewards: list[float] = []
    token_rewards = None
    if "token_level_scores" in batch.batch.keys():
        token_rewards = batch.batch["token_level_scores"]
    elif "token_level_rewards" in batch.batch.keys():
        token_rewards = batch.batch["token_level_rewards"]
    elif "rm_scores" in batch.batch.keys():
        token_rewards = batch.batch["rm_scores"]
    extras = [extra_row_from_batch(batch, index) for index in range(n)]
    for index in range(n):
        extra = extras[index]
        scalar = _scalar_reward(extra.get("localization_score"))
        if scalar is None:
            scalar = _scalar_reward(extra.get("score"))
        if scalar is None and token_rewards is not None:
            scalar = _scalar_reward(token_rewards[index])
        rewards.append(0.0 if scalar is None else float(scalar))
    return compute_bcrl_step_metrics(
        uids=uids,
        rewards=rewards,
        extra_rows=extras,
        group_n=GROUP_N,
    )


def _trainer_safe_metrics(metrics: MappingABC[str, Any]) -> dict[str, Any]:
    """Drop None so veRL reduce_metrics/np.mean does not crash after the update."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[str(key)] = int(value)
            continue
        out[str(key)] = value
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, MappingABC):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    return str(value)


def _persist_wandb_run(output_dir: Path) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        write_json(
            Path(output_dir) / "wandb_run.json",
            {
                "id": wandb.run.id,
                "name": wandb.run.name,
                "project": wandb.run.project,
                "url": getattr(wandb.run, "url", None),
                "mode": str(getattr(getattr(wandb.run, "settings", None), "mode", None)),
            },
        )
    except Exception:
        pass


def install_metrics_jsonl_logger(output_dir: Path) -> None:
    from verl.utils.tracking import Tracking

    if getattr(Tracking.log, "_bcrl_m5", False):
        return
    original_init = Tracking.__init__
    original_log = Tracking.log

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _persist_wandb_run(output_dir)

    def wrapped_log(self, data, step, backend=None):
        payload = {
            "step": int(step),
            "metrics": _jsonable(data),
        }
        try:
            append_jsonl(Path(output_dir) / "metrics.jsonl", payload)
        except Exception as exc:
            write_json(
                Path(output_dir) / "metrics_jsonl_error.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        _persist_wandb_run(output_dir)
        return original_log(self, data, step, backend=backend)

    wrapped_init._bcrl_m5 = True
    wrapped_log._bcrl_m5 = True
    Tracking.__init__ = wrapped_init
    Tracking.log = wrapped_log


class M5Trainer(RayPPOTrainer):
    """Official RayPPOTrainer plus BCRL group/budget metrics on each update."""

    def _update_actor(self, batch):
        output_dir = evidence_dir()
        bcrl = batch_bcrl_metrics(batch)
        extras = [
            extra_row_from_batch(batch, index)
            for index in range(len(coerce_sequence(batch.non_tensor_batch.get("uid"))))
        ]
        for extra in extras:
            try:
                append_jsonl(output_dir / "episodes.jsonl", compact_episode_from_extra(extra))
            except Exception as exc:
                append_jsonl(
                    output_dir / "episodes.jsonl",
                    {"error": f"{type(exc).__name__}: {exc}", "instance_id": extra.get("instance_id")},
                )
        append_jsonl(
            output_dir / "step_bcrl.jsonl",
            {"global_steps": int(getattr(self, "global_steps", 0) or 0), "metrics": bcrl},
        )
        actor_output = super()._update_actor(batch)
        metrics = dict((actor_output.meta_info.get("metrics") if actor_output is not None else None) or {})
        metrics.update(_trainer_safe_metrics(bcrl))
        actor_output.meta_info["metrics"] = metrics
        return actor_output


class M5TaskRunner(TaskRunner):
    """TaskRunner that uses stock ActorRolloutRefWorker + M5Trainer."""

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"M5 TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        output_dir = evidence_dir()
        install_metrics_jsonl_logger(output_dir)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = M5Trainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=m4b_collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()
