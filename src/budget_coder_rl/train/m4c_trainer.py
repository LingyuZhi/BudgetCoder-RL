"""M4C production-path hooks: official FSDP save/load and vLLM adapter sync evidence.

Does not implement an optimizer or rewrite the rollout backend.
"""

from __future__ import annotations

import json
import os
import socket
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import ray

from budget_coder_rl.agent_loop.rollout_verify import verify_padded_sample
from budget_coder_rl.eval.episode import build_episode_record
from budget_coder_rl.eval.m3b import QWEN3_SAMPLING
from budget_coder_rl.eval.m4a import leakage_errors
from budget_coder_rl.eval.m4b import (
    PROMPT_LENGTH,
    RESPONSE_LENGTH,
    is_lora_param_name,
    write_json,
)
from budget_coder_rl.eval.m4c import (
    VLLM_LORA_INT_ID,
    adapter_payload_summary,
    append_jsonl,
    compare_lora_fingerprints,
    current_phase,
    evidence_dir,
    fingerprint_digest,
    persist_lora_fingerprint,
)
from budget_coder_rl.train.m4b_trainer import (
    M4BActorWorker,
    M4BTaskRunner,
    M4BTrainer,
    _row_extra_fields,
    _tensor_fingerprint,
    _unwrap_snapshot,
    m4b_collate_fn,
)

from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config


def _peft_config_public(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        return {"repr": str(config)}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if hasattr(value, "value"):
            out[str(key)] = value.value
        elif isinstance(value, (set, tuple)):
            out[str(key)] = list(value)
        else:
            out[str(key)] = value
    return out


def _load_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


class M4CActorWorker(M4BActorWorker):
    """Actor worker that records the LoRA payload sent on the official update_weights path."""

    def _fingerprint_lora_payload(self) -> dict[str, Any]:
        """Hash LoRA tensors the same way as M4B snapshots.

        Do not call ``collect_lora_params`` / ``layered_summon`` here. That
        FSDP summon mutates module state and breaks the official
        ``ActorRolloutRefWorker.update_weights`` path (E009 attempt 1).
        """
        import torch

        engine = self.actor.engine
        module = engine.module
        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        peft_config = None
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default")
        tensors: dict[str, Any] = {}
        lora_b_max_abs = 0.0
        try:
            for name, param in module.named_parameters():
                if not is_lora_param_name(name):
                    continue
                fingerprint = _tensor_fingerprint(param, full=True)
                tensors[str(name)] = fingerprint
                if "lora_b" in str(name).lower():
                    lora_b_max_abs = max(
                        lora_b_max_abs, float(fingerprint.get("max_abs") or 0.0)
                    )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sha_map = {name: str(info.get("sha256") or "") for name, info in tensors.items()}
        public_config = _peft_config_public(peft_config)
        return {
            "phase": current_phase(),
            "peft_config_present": public_config is not None,
            "peft_config": public_config,
            "n_adapter_tensors": len(tensors),
            "tensors": tensors,
            "digest": fingerprint_digest(sha_map),
            "lora_b_max_abs": lora_b_max_abs,
            "adapter_nonzero": lora_b_max_abs > 0.0,
            "layered_summon": bool(getattr(self, "layered_summon", False)),
            "load_format": str(self.config.rollout.get("load_format")),
            "hash_source": "named_parameters",
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        output_dir = evidence_dir()
        before_flag = bool(getattr(self, "base_sync_done", False))
        result = await super().update_weights(global_steps=global_steps, mode=mode)
        after_flag = bool(getattr(self, "base_sync_done", False))
        try:
            payload = self._fingerprint_lora_payload()
        except Exception:
            payload = {
                "phase": current_phase(),
                "peft_config_present": False,
                "n_adapter_tensors": 0,
                "tensors": {},
                "digest": "",
                "lora_b_max_abs": 0.0,
                "adapter_nonzero": False,
                "error": traceback.format_exc(),
            }
        payload.update(
            {
                "global_steps": global_steps,
                "mode": mode,
                "base_sync_done_before": before_flag,
                "base_sync_done_after": after_flag,
                "did_base_weight_sync": (not before_flag) and after_flag,
                "peft_merge": bool(getattr(self, "peft_merge", False)),
                "vllm_lora_int_id": int(VLLM_LORA_INT_ID),
            }
        )
        write_json(output_dir / f"vllm_sync_payload_{current_phase()}.json", payload)
        write_json(output_dir / "vllm_sync_payload.json", payload)
        return result


def register_m4c_vllm_replica() -> None:
    """Swap the vLLM replica factory for a thin evidence subclass. Not a backend rewrite."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    from budget_coder_rl.train.m4c_vllm_server import M4CvLLMReplica

    RolloutReplicaRegistry.register("vllm", lambda: M4CvLLMReplica)


class M4CTrainer(M4BTrainer):
    """Save full LoRA fingerprints, official checkpoint, and reload-only fit()."""

    def _update_actor(self, batch):
        output_dir = evidence_dir()
        before = _unwrap_snapshot(self.actor_rollout_wg.snapshot_trainable_params())
        write_json(output_dir / "lora_theta0.json", persist_lora_fingerprint(before))
        actor_output = super()._update_actor(batch)
        after = _unwrap_snapshot(self.actor_rollout_wg.snapshot_trainable_params())
        write_json(output_dir / "lora_theta1.json", persist_lora_fingerprint(after))
        write_json(
            output_dir / "lora_theta_compare_save.json",
            compare_lora_fingerprints(before, after),
        )
        return actor_output

    def _save_checkpoint(self):
        super()._save_checkpoint()
        output_dir = evidence_dir()
        ckpt_root = Path(self.config.trainer.default_local_dir)
        write_json(
            output_dir / "save_checkpoint_evidence.json",
            {
                "global_steps": int(self.global_steps),
                "default_local_dir": str(ckpt_root),
                "global_step_dir": str(ckpt_root / f"global_step_{int(self.global_steps)}"),
                "resume_from_path": str(ckpt_root / f"global_step_{int(self.global_steps)}"),
                "path": (
                    "RayPPOTrainer._save_checkpoint -> actor_rollout_wg.save_checkpoint "
                    "-> FSDPCheckpointManager.save_checkpoint"
                ),
            },
        )

    def fit(self):
        if current_phase() != "reload":
            return super().fit()
        return self._fit_reload()

    def _fit_reload(self):
        from verl.protocol import DataProto

        output_dir = evidence_dir()
        self.global_steps = 0
        self._load_checkpoint()
        reloaded = _unwrap_snapshot(self.actor_rollout_wg.snapshot_trainable_params())
        write_json(output_dir / "lora_theta_reloaded.json", persist_lora_fingerprint(reloaded))
        theta0 = _load_fingerprint(output_dir / "lora_theta0.json")
        theta1 = _load_fingerprint(output_dir / "lora_theta1.json")
        compare = {
            "reloaded_vs_theta1": compare_lora_fingerprints(theta1, reloaded),
            "reloaded_vs_theta0": compare_lora_fingerprints(theta0, reloaded),
            "theta1_vs_theta0": compare_lora_fingerprints(theta0, theta1),
        }
        write_json(output_dir / "lora_fingerprint_compare.json", compare)
        if not compare["reloaded_vs_theta1"].get("equal"):
            raise RuntimeError(
                "HARD FAIL: reloaded LoRA fingerprint != saved θ1: "
                + str(compare["reloaded_vs_theta1"].get("mismatched_names"))
            )
        if compare["reloaded_vs_theta0"].get("equal"):
            raise RuntimeError("HARD FAIL: reloaded LoRA fingerprint == θ0")
        self.checkpoint_manager.update_weights(self.global_steps)

        batch_dict = next(iter(self.train_dataloader))
        batch = DataProto.from_single_dict(batch_dict)
        if "multi_modal_inputs" not in batch.non_tensor_batch:
            n_rows = len(batch)
            batch.non_tensor_batch["multi_modal_inputs"] = np.array(
                [{} for _ in range(n_rows)], dtype=object
            )
        if len(batch) > 1:
            batch = batch.slice(0, 1)
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4())], dtype=object)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.meta_info["validate"] = False
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["validate"] = False
        gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
        episode, tito_errors, leak_errors = self._record_reload_episode(gen_output)
        write_json(
            output_dir / "reload_rollout_evidence.json",
            {
                "n_episodes": 1 if episode else 0,
                "episode": episode,
                "tito_errors": tito_errors,
                "leak_errors": leak_errors,
                "sampling": dict(QWEN3_SAMPLING),
                "rollout_n": int(self.config.actor_rollout_ref.rollout.n),
                "validate": False,
            },
        )
        if tito_errors:
            raise RuntimeError(
                "HARD FAIL: post-reload AgentLoop TITO/mask errors: " + "; ".join(tito_errors[:6])
            )
        if leak_errors:
            raise RuntimeError(
                "HARD FAIL: post-reload leakage: " + "; ".join(leak_errors[:6])
            )
        payload = _load_fingerprint(output_dir / "vllm_sync_payload_reload.json")
        if payload and not adapter_payload_summary(payload).get("adapter_nonzero"):
            raise RuntimeError("HARD FAIL: reload adapter payload looks empty/zero")
        return gen_output

    def _record_reload_episode(self, gen_output):
        output_dir = evidence_dir()
        extra_fields = _row_extra_fields(gen_output, 0)
        sampling = extra_fields.get("sampling_params") or {}
        episode = build_episode_record(
            extra_fields,
            sampling=sampling if isinstance(sampling, Mapping) else {},
            provenance={"milestone": "M4C", "phase": "reload"},
        )
        episodes_path = output_dir / "episodes.jsonl"
        if episodes_path.is_file():
            episodes_path.unlink()
        append_jsonl(episodes_path, episode)
        tito_errors: list[str] = []
        leak_errors: list[str] = []
        if "response_mask" not in gen_output.batch.keys():
            tito_errors.append("response_mask missing on post-reload generate")
            return episode, tito_errors, leak_errors
        segments = list(extra_fields.get("segments") or [])
        unpadded = extra_fields.get("unpadded_prompt_ids") or []
        tito = verify_padded_sample(
            prompt_width=PROMPT_LENGTH,
            response_width=RESPONSE_LENGTH,
            prompts_row=gen_output.batch["prompts"][0],
            responses_row=gen_output.batch["responses"][0],
            response_mask_row=gen_output.batch["response_mask"][0],
            attention_mask_row=gen_output.batch["attention_mask"][0],
            unpadded_prompt_ids=unpadded,
            segments=segments,
        )
        instance_id = str(extra_fields.get("instance_id") or "")
        tito_errors.extend(f"{instance_id}: {err}" for err in tito)
        prompt_text = self.tokenizer.decode(list(unpadded), skip_special_tokens=True)
        decoded_obs = [
            self.tokenizer.decode(list(item.get("token_ids") or []), skip_special_tokens=True)
            for item in segments
            if item.get("kind") == "observation"
        ]
        leak_errors.extend(
            leakage_errors(
                decoded_prompt=prompt_text,
                decoded_observations=decoded_obs,
                extra_field_keys=list(extra_fields.keys()),
            )
        )
        return episode, tito_errors, leak_errors


class M4CTaskRunner(M4BTaskRunner):
    """TaskRunner that uses M4CActorWorker + M4CTrainer on the official fit/reload path."""

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        actor_rollout_cls = M4CActorWorker
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

        register_m4c_vllm_replica()
        print(
            f"M4C TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}, "
            f"phase={current_phase()}"
        )
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
        trainer = M4CTrainer(
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
