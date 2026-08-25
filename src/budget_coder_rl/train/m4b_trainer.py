"""M4B production-path hooks: snapshot LoRA around RayPPOTrainer._update_actor.

Does not implement an optimizer. `super()._update_actor` is the pinned veRL path.
"""

from __future__ import annotations

import hashlib
import os
import socket
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import ray
import torch

from budget_coder_rl.agent_loop.rollout_verify import verify_padded_sample
from budget_coder_rl.eval.m4a import GROUP_N, leakage_errors, scalar_advantage
from budget_coder_rl.eval.m4b import (
    ADVANTAGE_ABS_EPS,
    OUTPUT_ENV,
    PROMPT_LENGTH,
    RESPONSE_LENGTH,
    assemble_groups_from_members,
    assemble_loss_mask_evidence,
    attention_response_slice,
    compare_param_snapshots,
    count_mask_tokens,
    is_lora_param_name,
    m4b_gate,
    mask_correctness_errors,
    metric_finite_nonzero,
    step_learning_signal,
    write_json,
)

from verl.single_controller.base.decorator import Dispatch, register
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


def m4b_collate_fn(data_list: list[dict]) -> dict:
    from verl.utils.dataset.rl_dataset import collate_fn as verl_collate

    batch = verl_collate(data_list)
    if "multi_modal_inputs" not in batch:
        batch["multi_modal_inputs"] = np.array([{} for _ in data_list], dtype=object)
    return batch


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [int(item) for item in list(values)]


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [float(item) for item in list(values)]


def _nt(batch, key: str, index: int) -> Any:
    payload = batch.non_tensor_batch.get(key)
    if payload is None:
        return None
    return payload[index]


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    return {}


def _unwrap_snapshot(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        if not result:
            raise RuntimeError("snapshot_trainable_params returned empty list")
        return dict(result[0])
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError(f"unexpected snapshot type {type(result)!r}")


def _tensor_fingerprint(tensor: torch.Tensor, *, full: bool) -> dict[str, Any]:
    data = tensor.detach()
    if hasattr(data, "full_tensor"):
        try:
            data = data.full_tensor()
        except Exception:
            if hasattr(data, "to_local"):
                data = data.to_local()
    elif hasattr(data, "to_local"):
        data = data.to_local()
    flat = data.reshape(-1)
    numel = int(flat.numel())
    if numel == 0:
        return {
            "sha256": "empty",
            "numel": 0,
            "mean": 0.0,
            "max_abs": 0.0,
            "full_hash": full,
        }
    if full:
        sample = flat.float().cpu().contiguous()
    else:
        step = max(1, numel // 4096)
        sample = flat[::step].float().cpu().contiguous()
    mean = float(sample.mean().item())
    max_abs = float(sample.abs().max().item())
    return {
        "sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
        "numel": numel,
        "mean": mean,
        "max_abs": max_abs,
        "full_hash": bool(full),
    }


class M4BActorWorker(ActorRolloutRefWorker):
    """ActorRolloutRefWorker plus FSDP-safe LoRA/base fingerprints."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def snapshot_trainable_params(self) -> dict[str, Any]:
        engine = self.actor.engine
        module = engine.module
        lora: dict[str, Any] = {}
        frozen: dict[str, Any] = {}
        unexpected: list[str] = []
        n_trainable = 0
        n_frozen = 0
        try:
            for name, param in module.named_parameters():
                trainable = bool(param.requires_grad)
                lora_name = is_lora_param_name(name)
                numel = int(param.numel())
                if trainable:
                    n_trainable += 1
                    if not lora_name:
                        unexpected.append(name)
                    lora[name] = _tensor_fingerprint(
                        param, full=bool(lora_name or numel < 2_000_000)
                    )
                else:
                    n_frozen += 1
                    frozen[name] = _tensor_fingerprint(param, full=False)
                    if lora_name:
                        unexpected.append(name)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "rank": int(getattr(self, "rank", 0) or 0),
            "n_trainable": n_trainable,
            "n_frozen": n_frozen,
            "unexpected_trainable": unexpected,
            "lora": lora,
            "frozen": frozen,
        }


class M4BTrainer(RayPPOTrainer):
    """RayPPOTrainer that records one optimizer-step correctness bundle."""

    def _update_actor(self, batch):
        output_dir = evidence_dir()
        try:
            members, mask_rows, tito_errors, leak_errors, mask_errors = (
                self._collect_batch_evidence(batch)
            )
            groups = assemble_groups_from_members(members, group_n=GROUP_N)
            learning = step_learning_signal(groups)
            loss_mask = assemble_loss_mask_evidence(
                mask_rows, tito_errors=tito_errors, mask_errors=mask_errors
            )
            write_json(output_dir / "group_evidence.json", {"groups": groups, "learning": learning})
            write_json(output_dir / "loss_mask_evidence.json", loss_mask)
            write_json(
                output_dir / "pre_update_status.json",
                {
                    "learning": learning,
                    "loss_mask_ok": loss_mask.get("ok"),
                    "tito_errors": tito_errors,
                    "leak_errors": leak_errors,
                    "mask_errors": mask_errors,
                },
            )
            if not learning.get("ok"):
                raise RuntimeError(
                    "HARD FAIL: optimizer batch has no non-zero GRPO advantage"
                )
            if not loss_mask.get("ok"):
                raise RuntimeError(
                    "HARD FAIL: response_mask / advantage contract failed: "
                    + "; ".join(list(mask_errors)[:8] + list(tito_errors)[:8])
                )
            before = _unwrap_snapshot(self.actor_rollout_wg.snapshot_trainable_params())
            write_json(output_dir / "lora_before.json", _snapshot_public(before))
            actor_output = None
            update_error: Exception | None = None
            try:
                actor_output = super()._update_actor(batch)
            except Exception as exc:
                update_error = exc
            after = _unwrap_snapshot(self.actor_rollout_wg.snapshot_trainable_params())
            write_json(output_dir / "lora_after.json", _snapshot_public(after))
            metrics = dict((actor_output.meta_info.get("metrics") if actor_output is not None else None) or {})
            grad = metric_finite_nonzero(
                _first_metric(metrics, "actor/grad_norm", "grad_norm")
            )
            pg_loss = metric_finite_nonzero(
                _first_metric(metrics, "actor/pg_loss", "actor/loss", "loss", "actor/actor/pg_loss")
            )
            lora = compare_param_snapshots(before, after)
            gate = m4b_gate(
                learning=learning,
                loss_mask=loss_mask,
                grad=grad,
                pg_loss=pg_loss,
                lora=lora,
            )
            payload = {
                "metrics": {str(key): _jsonable_metric(val) for key, val in metrics.items()},
                "grad": grad,
                "pg_loss": pg_loss,
                "lora": lora,
                "gate": gate,
                "leak_errors": leak_errors,
                "update_error": None if update_error is None else f"{type(update_error).__name__}: {update_error}",
            }
            write_json(output_dir / "lora_delta.json", payload)
            if update_error is not None:
                raise update_error
            if not gate.get("pass"):
                raise RuntimeError(
                    "HARD FAIL: M4B optimizer gate failed: " + "; ".join(gate.get("reasons") or [])
                )
            return actor_output
        except Exception:
            write_json(
                output_dir / "update_actor_error.json",
                {"traceback": traceback.format_exc()},
            )
            raise

    def _collect_batch_evidence(self, batch):
        if "response_mask" not in batch.batch.keys():
            raise RuntimeError("HARD FAIL: response_mask missing before actor.update_actor")
        response_mask = batch.batch["response_mask"]
        attention_mask = batch.batch["attention_mask"]
        advantages = batch.batch["advantages"]
        rewards = batch.batch.get("token_level_rewards", batch.batch.get("rm_scores"))
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        uids = [str(item) for item in list(batch.non_tensor_batch["uid"])]
        response_width = int(responses.size(1))
        prompt_width = int(prompts.size(1))
        mask_lists = [_as_int_list(row) for row in response_mask]
        attn_lists = attention_response_slice(
            [_as_int_list(row) for row in attention_mask],
            response_width,
        )
        adv_lists = [_as_float_list(row) for row in advantages]
        members: list[dict[str, Any]] = []
        mask_rows: list[dict[str, Any]] = []
        tito_errors: list[str] = []
        leak_errors: list[str] = []
        n_obs_tokens: list[int] = []
        for index in range(len(uids)):
            extra_info = _mapping(_nt(batch, "extra_info", index))
            extra_fields = _row_extra_fields(batch, index)
            leak_keys = [
                key
                for key in list(extra_fields.keys())
                if key not in extra_info
            ]
            instance_id = str(
                extra_fields.get("instance_id")
                or extra_info.get("instance_id")
                or ""
            )
            segments = list(extra_fields.get("segments") or [])
            n_obs = sum(
                len(item.get("token_ids") or [])
                for item in segments
                if item.get("kind") == "observation"
            )
            n_obs_tokens.append(n_obs)
            counts = count_mask_tokens(mask_lists[index], n_obs)
            adv_zero = True
            for bit, adv in zip(mask_lists[index], adv_lists[index]):
                if int(bit) == 0 and abs(float(adv)) > ADVANTAGE_ABS_EPS:
                    adv_zero = False
                    break
            equals_attn = mask_lists[index] == attn_lists[index]
            mask_rows.append(
                {
                    "index": index,
                    "instance_id": instance_id,
                    "uid": uids[index],
                    "response_mask_present": True,
                    "n_policy": counts["n_policy"],
                    "n_observation": counts["n_observation"],
                    "n_pad": counts["n_pad"],
                    "obs_and_mask_equals_attention": bool(n_obs > 0 and equals_attn),
                    "advantages_zero_on_mask0": adv_zero,
                    "prompt_width": prompt_width,
                    "response_width": response_width,
                }
            )
            unpadded = extra_fields.get("unpadded_prompt_ids") or []
            tito = verify_padded_sample(
                prompt_width=prompt_width or PROMPT_LENGTH,
                response_width=response_width or RESPONSE_LENGTH,
                prompts_row=prompts[index],
                responses_row=responses[index],
                response_mask_row=response_mask[index],
                attention_mask_row=attention_mask[index],
                unpadded_prompt_ids=unpadded,
                segments=segments,
            )
            if tito:
                tito_errors.extend(f"{instance_id}[{index}]: {err}" for err in tito)
            prompt_text = self.tokenizer.decode(list(unpadded), skip_special_tokens=True)
            decoded_obs = []
            for item in segments:
                if item.get("kind") != "observation":
                    continue
                decoded_obs.append(
                    self.tokenizer.decode(
                        list(item.get("token_ids") or []), skip_special_tokens=True
                    )
                )
            leaks = leakage_errors(
                decoded_prompt=prompt_text,
                decoded_observations=decoded_obs,
                extra_field_keys=leak_keys,
            )
            if leaks:
                leak_errors.extend(f"{instance_id}[{index}]: {err}" for err in leaks)
            reward_extra = extra_fields.get("reward_extra_info") or {}
            if not isinstance(reward_extra, Mapping):
                reward_extra = {}
            loc_score = reward_extra.get("score")
            if loc_score is None:
                loc_score = _nt(batch, "score", index)
            loc_score_f = float(loc_score) if loc_score is not None else None
            rm_score = float(rewards[index].sum().item())
            members.append(
                {
                    "instance_id": instance_id,
                    "uid": uids[index],
                    "rm_score": rm_score,
                    "localization_score": loc_score_f if loc_score_f is not None else rm_score,
                    "advantage_scalar": scalar_advantage(
                        adv_lists[index], mask_lists[index]
                    ),
                    "termination": extra_fields.get("termination"),
                    "rollout_n": index,
                }
            )
        mask_errors = mask_correctness_errors(
            response_mask=mask_lists,
            attention_response=attn_lists,
            advantages=adv_lists,
            n_observation_tokens=n_obs_tokens,
        )
        if leak_errors:
            tito_errors = list(tito_errors) + [f"leakage: {item}" for item in leak_errors]
        return members, mask_rows, tito_errors, leak_errors, mask_errors


class M4BTaskRunner(TaskRunner):
    """TaskRunner that uses M4BActorWorker + M4BTrainer on the official fit path."""

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        actor_rollout_cls = M4BActorWorker
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

        print(f"M4B TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

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
        trainer = M4BTrainer(
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


def _row_extra_fields(batch, index: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    skip = {"uid", "raw_prompt", "data_source", "reward_model", "multi_modal_inputs"}
    for key, payload in batch.non_tensor_batch.items():
        if key in skip or payload is None:
            continue
        try:
            out[str(key)] = payload[index]
        except Exception:
            continue
    extra_info = _mapping(out.get("extra_info"))
    for key, value in extra_info.items():
        out.setdefault(key, value)
    return out


def _first_metric(metrics: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    for key in keys:
        suffix = str(key).rsplit("/", 1)[-1]
        for metric_key, value in metrics.items():
            if str(metric_key).rsplit("/", 1)[-1] == suffix:
                return value
    return None


def _jsonable_metric(value: Any) -> Any:
    from budget_coder_rl.eval.m4b import unwrap_metric_value

    try:
        return unwrap_metric_value(value)
    except Exception:
        return str(value)


def _snapshot_public(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    lora = dict(snapshot.get("lora") or {})
    frozen = dict(snapshot.get("frozen") or {})
    return {
        "rank": snapshot.get("rank"),
        "n_trainable": snapshot.get("n_trainable"),
        "n_frozen": snapshot.get("n_frozen"),
        "unexpected_trainable": list(snapshot.get("unexpected_trainable") or []),
        "n_lora_tensors": len(lora),
        "lora_names": sorted(lora)[:64],
        "lora_sample": {name: lora[name] for name in sorted(lora)[:4]},
        "frozen_sample": {name: frozen[name] for name in sorted(frozen)[:4]},
    }
