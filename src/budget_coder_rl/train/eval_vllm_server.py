"""Eval-only vLLM HTTP-server subclass for E018 LoRARequest evidence.

Does not change sampling, weight sync, or the official generate path.
Wraps ``engine.generate`` so the LoRARequest actually passed is recorded.
Imported only on the GPU rollout path.
"""

from __future__ import annotations

from typing import Any

from budget_coder_rl.eval.e018 import OUTPUT_ENV, evidence_dir
from budget_coder_rl.eval.m4c import VLLM_LORA_INT_ID, VLLM_LORA_NAME, append_jsonl
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID as PINNED_VLLM_LORA_INT_ID,
    VLLM_LORA_NAME as PINNED_VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


def _lora_int_id(request: Any) -> int | None:
    if request is None:
        return None
    value = getattr(request, "lora_int_id", None)
    if value is None:
        value = getattr(request, "lora_id", None)
    if value is None:
        return None
    return int(value)


class E018vLLMHttpServer(vLLMHttpServer):
    async def e018_engine_lora_state(self) -> dict[str, Any]:
        listed: list[int] = []
        list_error = None
        try:
            if getattr(self, "engine", None) is not None:
                listed = [int(item) for item in list(await self.engine.list_loras())]
        except Exception as exc:
            list_error = f"{type(exc).__name__}: {exc}"
        return {
            "lora_as_adapter": bool(getattr(self, "lora_as_adapter", False)),
            "listed_lora_ids": listed,
            "list_error": list_error,
            "pinned_lora_int_id": int(PINNED_VLLM_LORA_INT_ID),
        }

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
        captured: dict[str, Any] = {
            "lora_request_attached": False,
            "lora_int_id": None,
            "listed_lora_ids": [],
            "list_error": None,
        }
        engine = getattr(self, "engine", None)
        original = getattr(engine, "generate", None) if engine is not None else None

        async def wrapped(*args, **kwargs):
            request = kwargs.get("lora_request")
            if request is None and len(args) >= 5:
                request = args[4]
            captured["lora_request_attached"] = request is not None
            captured["lora_int_id"] = _lora_int_id(request)
            captured["lora_request_type"] = type(request).__name__ if request is not None else None
            try:
                captured["listed_lora_ids"] = [
                    int(item) for item in list(await engine.list_loras())
                ]
            except Exception as exc:
                captured["list_error"] = f"{type(exc).__name__}: {exc}"
            gen = original(*args, **kwargs)
            if not hasattr(gen, "__aiter__"):
                gen = await gen
            async for item in gen:
                yield item

        if engine is not None and original is not None:
            engine.generate = wrapped
        try:
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
        finally:
            if engine is not None and original is not None:
                engine.generate = original

        evidence = {
            "lora_as_adapter": bool(getattr(self, "lora_as_adapter", False)),
            "listed_lora_ids": captured.get("listed_lora_ids") or [],
            "lora_int_id": captured.get("lora_int_id"),
            "lora_name": VLLM_LORA_NAME if captured.get("lora_request_attached") else None,
            "lora_path": VLLM_LORA_PATH if captured.get("lora_request_attached") else None,
            "lora_request_attached": bool(captured.get("lora_request_attached")),
            "lora_request_type": captured.get("lora_request_type"),
            "n_prompt_tokens": len(list(prompt_ids)),
            "request_id": str(request_id),
            "list_error": captured.get("list_error"),
            "pinned_lora_int_id": int(PINNED_VLLM_LORA_INT_ID),
            "pinned_lora_name": PINNED_VLLM_LORA_NAME,
            "expected_lora_int_id": int(VLLM_LORA_INT_ID),
            "output_env": OUTPUT_ENV,
        }
        try:
            append_jsonl(evidence_dir() / "vllm_generate_evidence.jsonl", evidence)
        except Exception:
            pass
        extra = dict(getattr(result, "extra_fields", None) or {})
        extra["vllm_lora_int_id"] = evidence["lora_int_id"]
        extra["vllm_lora_request_attached"] = evidence["lora_request_attached"]
        extra["vllm_listed_lora_ids"] = evidence["listed_lora_ids"]
        try:
            result.extra_fields = extra
        except Exception:
            pass
        return result


class E018vLLMReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import ray

        self.server_class = ray.remote(E018vLLMHttpServer)


def register_e018_vllm_replica() -> None:
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    RolloutReplicaRegistry.register("vllm", lambda: E018vLLMReplica)
