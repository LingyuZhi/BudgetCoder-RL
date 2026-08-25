"""Thin vLLM HTTP-server subclass for M4C adapter-load evidence.

Imported only on the GPU rollout path. Does not change weight sync or sampling.
"""

from __future__ import annotations

from typing import Any

from budget_coder_rl.eval.m4c import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    append_jsonl,
    current_phase,
    evidence_dir,
)
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID as PINNED_VLLM_LORA_INT_ID,
    VLLM_LORA_NAME as PINNED_VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


class M4CvLLMHttpServer(vLLMHttpServer):
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        priority: int = 0,
    ):
        listed: list[int] = []
        lora_loaded = False
        attached = False
        lora_as_adapter = bool(getattr(self, "lora_as_adapter", False))
        list_error = None
        try:
            if lora_as_adapter and getattr(self, "engine", None) is not None:
                listed = [int(item) for item in list(await self.engine.list_loras())]
                lora_loaded = int(VLLM_LORA_INT_ID) in listed
                attached = lora_loaded
        except Exception as exc:
            listed = []
            lora_loaded = False
            attached = False
            list_error = f"{type(exc).__name__}: {exc}"
        evidence = {
            "phase": current_phase(),
            "lora_as_adapter": lora_as_adapter,
            "listed_lora_ids": listed,
            "lora_int_id": int(VLLM_LORA_INT_ID) if attached else None,
            "lora_name": VLLM_LORA_NAME if attached else None,
            "lora_path": VLLM_LORA_PATH if attached else None,
            "lora_loaded": lora_loaded,
            "lora_request_attached": attached,
            "n_prompt_tokens": len(list(prompt_ids)),
            "request_id": str(request_id),
            "list_error": list_error,
            "pinned_lora_int_id": int(PINNED_VLLM_LORA_INT_ID),
            "pinned_lora_name": PINNED_VLLM_LORA_NAME,
        }
        try:
            append_jsonl(evidence_dir() / "vllm_generate_evidence.jsonl", evidence)
        except Exception:
            pass
        result = await super().generate(
            prompt_ids,
            sampling_params,
            request_id,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
        )
        extra = dict(getattr(result, "extra_fields", None) or {})
        extra["vllm_lora_int_id"] = evidence["lora_int_id"]
        extra["vllm_lora_request_attached"] = attached
        extra["vllm_listed_lora_ids"] = listed
        try:
            result.extra_fields = extra
        except Exception:
            pass
        return result


class M4CvLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import ray

        self.server_class = ray.remote(M4CvLLMHttpServer)
