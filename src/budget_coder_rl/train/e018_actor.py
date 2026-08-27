"""E018 eval actor worker: FSDP LoRA snapshot + official update_weights payload.

Does not train. Used only on the M_scaled AgentLoopManager path.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

from budget_coder_rl.eval.e018 import OUTPUT_ENV
from budget_coder_rl.eval.m4b import is_lora_param_name, write_json
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID, fingerprint_digest
from budget_coder_rl.train.m4b_trainer import M4BActorWorker, _tensor_fingerprint
from verl.single_controller.base.decorator import Dispatch, register


def e018_evidence_dir() -> Path:
    raw = os.environ.get(OUTPUT_ENV)
    if not raw:
        raise RuntimeError(f"{OUTPUT_ENV} is not set")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


class E018ActorWorker(M4BActorWorker):
    """ActorRolloutRefWorker plus eval-only LoRA payload fingerprinting."""

    def _fingerprint_lora_payload(self) -> dict[str, Any]:
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
        public_config = None
        if peft_config is not None:
            if hasattr(peft_config, "to_dict"):
                public_config = peft_config.to_dict()
            elif isinstance(peft_config, dict):
                public_config = dict(peft_config)
            else:
                public_config = {"repr": str(peft_config)}
        return {
            "peft_config_present": public_config is not None,
            "peft_config": public_config,
            "n_adapter_tensors": len(tensors),
            "tensors": tensors,
            "digest": fingerprint_digest(sha_map),
            "lora_b_max_abs": lora_b_max_abs,
            "adapter_nonzero": lora_b_max_abs > 0.0,
            "hash_source": "named_parameters",
            "vllm_lora_int_id": int(VLLM_LORA_INT_ID),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        result = await super().update_weights(global_steps=global_steps, mode=mode)
        try:
            payload = self._fingerprint_lora_payload()
        except Exception:
            payload = {
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
                "vllm_lora_int_id": int(VLLM_LORA_INT_ID),
            }
        )
        output_dir = e018_evidence_dir()
        rank = int(getattr(self, "rank", 0) or 0)
        write_json(output_dir / f"vllm_sync_payload_rank{rank}.json", payload)
        if rank == 0:
            write_json(output_dir / "vllm_sync_payload.json", payload)
        return result
