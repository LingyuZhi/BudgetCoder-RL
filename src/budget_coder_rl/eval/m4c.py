"""M4C CPU helpers: LoRA persist/reload equality, FSDP checkpoint, vLLM evidence.

Does not run GPU, the optimizer, or vLLM. Gold lists stay in the evaluator sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.eval.m4a import (
    BUDGET_VISIBLE,
    GROUP_N,
    OBS_TOKENS_LIMIT,
    artifact_hashes,
    default_candidate_path,
    default_e007_groups_path,
    default_freeze_path,
    freeze_contract_errors,
    load_candidate_ordered_ids,
    load_e007_groups,
    load_json,
    select_smoke_instance_ids,
)
from budget_coder_rl.eval.m4b import (
    LORA_ALPHA,
    LORA_RANK,
    N_TASKS,
    OUTPUT_ENV as M4B_OUTPUT_ENV,
    PINNED_VERL_COMMIT,
    PINNED_VERL_VERSION,
    REWARD_FN_RELPATH,
    compare_param_snapshots,
    is_lora_param_name,
    write_json,
    write_smoke_parquet,
)
from budget_coder_rl.eval.provenance import sha256_file

EXPERIMENT_ID = "E009"
MILESTONE = "M4C"
PHASE_ENV = "BCRL_M4C_PHASE"
OUTPUT_ENV = "BCRL_M4C_OUTPUT_DIR"
RELOAD_N = 1
# Pinned `verl.workers.rollout.vllm_rollout.utils.VLLM_LORA_INT_ID`.
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
ADAPTER_ABS_EPS = 1e-12

VERL_PATH_TEXT = """# M4C pinned veRL save / load / vLLM sync path

Checkout: `{commit}` (`0.8.0.dev0`, fork LingyuZhi/rtrl-verl).

## Official FSDP+LoRA persist (not a project-local torch.save)

- Trigger: `RayPPOTrainer.fit` with `trainer.save_freq > 0` after `actor.update_actor`.
- Entry: `RayPPOTrainer._save_checkpoint`
  → `actor_rollout_wg.save_checkpoint`
  → `ActorRolloutRefWorker.save_checkpoint`
  → `FSDPEngine.save_checkpoint`
  → `FSDPCheckpointManager.save_checkpoint`.
- Default `actor.checkpoint.save_contents = ['model', 'optimizer', 'extra']`
  (sharded FSDP `.pt`, not `hf_model` gather, not PEFT `adapter_model.safetensors`).
- Layout: `default_local_dir/global_step_{{N}}/actor/` plus
  `latest_checkpointed_iteration.txt`. Rank-0 also writes `huggingface/`
  (config/tokenizer only) and `fsdp_config.json`.

## Official reload

- `trainer.resume_mode=resume_path` and path contains `global_step_`.
- `RayPPOTrainer.fit` / reload hook: `_load_checkpoint`
  → `actor_rollout_wg.load_checkpoint(actor/)`.
- `lora_adapter_path` + `PeftModel.from_pretrained` is the SFT/continue-train
  entry, not the PPO resume path used here.

## Official actor → vLLM weight sync

- `CheckpointEngineManager.update_weights` with `checkpoint_engine.backend=naive`
  → `ActorRolloutRefWorker.update_weights`.
- `model.lora.merge=False`: `get_per_tensor_param(base_sync_done=True)` yields
  LoRA tensors + `peft_config`. With `rollout.load_format=safetensors`,
  `base_sync_done` starts True (base already in vLLM); sync is adapter-only.
- vLLM worker: `remove_lora(123)` + `add_lora(TensorLoRARequest(lora_int_id=123, ...))`.
- Generate: `vLLMHttpServer.generate` attaches `LoRARequest(123)` only if
  `lora_as_adapter` and `123 in engine.list_loras()`.

M4C does not treat rollout text change or reward improvement as a pass gate.
"""


def default_output_dir(repo_root: Path, experiment_id: str = EXPERIMENT_ID) -> Path:
    return Path(repo_root) / "outputs" / "experiments" / experiment_id


def evidence_dir() -> Path:
    raw = os.environ.get(OUTPUT_ENV) or os.environ.get(M4B_OUTPUT_ENV)
    if not raw:
        raise RuntimeError(f"{OUTPUT_ENV} is not set")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_phase() -> str:
    return str(os.environ.get(PHASE_ENV) or "save")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True, default=str) + "\n")


def default_checkpoint_root(output_dir: Path) -> Path:
    return Path(output_dir) / "checkpoints"


def global_step_dir(checkpoint_root: Path, step: int) -> Path:
    return Path(checkpoint_root) / f"global_step_{int(step)}"


def lora_sha256_map(snapshot: Mapping[str, Any]) -> dict[str, str]:
    payload = dict(snapshot.get("lora") or {})
    return {str(name): str((info or {}).get("sha256") or "") for name, info in payload.items()}


def fingerprint_digest(sha_map: Mapping[str, str]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(sha_map):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(sha_map[name]).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def persist_lora_fingerprint(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Keep every LoRA tensor hash so reload equality can be checked across processes."""
    lora = dict(snapshot.get("lora") or {})
    frozen = dict(snapshot.get("frozen") or {})
    sha_map = lora_sha256_map({"lora": lora})
    return {
        "rank": snapshot.get("rank"),
        "n_trainable": snapshot.get("n_trainable"),
        "n_frozen": snapshot.get("n_frozen"),
        "unexpected_trainable": list(snapshot.get("unexpected_trainable") or []),
        "n_lora_tensors": len(lora),
        "digest": fingerprint_digest(sha_map),
        "lora": lora,
        "frozen_sample": {name: frozen[name] for name in sorted(frozen)[:4]},
        "n_frozen_hashed": len(frozen),
    }


def compare_lora_fingerprints(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_map = lora_sha256_map(left)
    right_map = lora_sha256_map(right)
    names = sorted(set(left_map) | set(right_map))
    mismatched = [name for name in names if left_map.get(name) != right_map.get(name)]
    left_digest = fingerprint_digest(left_map)
    right_digest = fingerprint_digest(right_map)
    return {
        "n_left": len(left_map),
        "n_right": len(right_map),
        "n_mismatched": len(mismatched),
        "mismatched_names": mismatched[:32],
        "equal": bool(left_map) and left_map == right_map,
        "left_digest": left_digest,
        "right_digest": right_digest,
    }


def required_actor_shard_names(world_size: int = 1, rank: int = 0) -> list[str]:
    size = int(world_size)
    index = int(rank)
    return [
        f"model_world_size_{size}_rank_{index}.pt",
        f"optim_world_size_{size}_rank_{index}.pt",
        f"extra_state_world_size_{size}_rank_{index}.pt",
    ]


def checkpoint_integrity_errors(
    checkpoint_root: Path,
    *,
    expected_step: int = 1,
    world_size: int = 1,
) -> list[str]:
    """Assert the official FSDP checkpoint layout exists and is non-empty."""
    root = Path(checkpoint_root)
    errors: list[str] = []
    if not root.is_dir():
        return [f"missing checkpoint root {root}"]
    latest_path = root / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        errors.append("missing latest_checkpointed_iteration.txt")
        latest = None
    else:
        latest = latest_path.read_text(encoding="utf-8").strip()
        if latest != str(int(expected_step)):
            errors.append(
                f"latest_checkpointed_iteration={latest!r} != expected {expected_step}"
            )
    step = int(expected_step)
    step_dir = global_step_dir(root, step)
    if not step_dir.is_dir():
        errors.append(f"missing {step_dir.name}")
        return errors
    actor = step_dir / "actor"
    if not actor.is_dir():
        errors.append("missing actor/ directory")
        return errors
    for name in required_actor_shard_names(world_size=world_size, rank=0):
        path = actor / name
        if not path.is_file():
            errors.append(f"missing actor/{name}")
        elif path.stat().st_size <= 0:
            errors.append(f"empty actor/{name}")
    fsdp_config = actor / "fsdp_config.json"
    if not fsdp_config.is_file():
        errors.append("missing actor/fsdp_config.json")
    else:
        try:
            payload = json.loads(fsdp_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"actor/fsdp_config.json is not JSON: {exc}")
            payload = {}
        if int(payload.get("world_size") or 0) != int(world_size):
            errors.append(
                f"fsdp_config.world_size={payload.get('world_size')!r} != {world_size}"
            )
    hf_config = actor / "huggingface" / "config.json"
    if not hf_config.is_file():
        errors.append("missing actor/huggingface/config.json")
    return errors


def iter_checkpoint_files(checkpoint_root: Path) -> list[Path]:
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def build_checkpoint_manifest(
    checkpoint_root: Path,
    *,
    expected_step: int = 1,
    project_commit: str | None = None,
    verl_commit: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_root)
    files = []
    for path in iter_checkpoint_files(root):
        files.append(
            {
                "relpath": str(path.relative_to(root)),
                "nbytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "checkpoint_root": str(root),
        "expected_step": int(expected_step),
        "global_step_dir": str(global_step_dir(root, expected_step)),
        "n_files": len(files),
        "nbytes_total": sum(int(item["nbytes"]) for item in files),
        "files": files,
        "integrity_errors": checkpoint_integrity_errors(
            root, expected_step=expected_step
        ),
        "project_commit": project_commit,
        "verl_commit": verl_commit,
        "save_path": (
            "RayPPOTrainer._save_checkpoint -> ActorRolloutRefWorker.save_checkpoint "
            "-> FSDPEngine.save_checkpoint -> FSDPCheckpointManager.save_checkpoint"
        ),
        "extra": dict(extra or {}),
    }


def adapter_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    tensors = dict(payload.get("tensors") or {})
    sha_map = {str(name): str((info or {}).get("sha256") or "") for name, info in tensors.items()}
    lora_b_max_abs = float(payload.get("lora_b_max_abs") or 0.0)
    if not lora_b_max_abs:
        for name, info in tensors.items():
            if "lora_b" in str(name).lower():
                lora_b_max_abs = max(lora_b_max_abs, float((info or {}).get("max_abs") or 0.0))
    digest = str(payload.get("digest") or fingerprint_digest(sha_map))
    return {
        "peft_config_present": bool(payload.get("peft_config_present")),
        "n_adapter_tensors": int(payload.get("n_adapter_tensors") or len(tensors)),
        "digest": digest,
        "lora_b_max_abs": lora_b_max_abs,
        "adapter_nonzero": bool(
            payload.get("adapter_nonzero")
            if "adapter_nonzero" in payload
            else lora_b_max_abs > ADAPTER_ABS_EPS
        ),
    }


def vllm_sync_errors(
    *,
    payload: Mapping[str, Any] | None,
    generate_rows: Sequence[Mapping[str, Any]] | None = None,
    saved_payload: Mapping[str, Any] | None = None,
    lora_int_id: int = VLLM_LORA_INT_ID,
) -> list[str]:
    """Fail unless vLLM actually loaded/used the updated adapter id."""
    errors: list[str] = []
    summary = adapter_payload_summary(payload or {})
    if not summary["peft_config_present"]:
        errors.append("adapter sync peft_config missing")
    if summary["n_adapter_tensors"] <= 0:
        errors.append("adapter sync payload has no LoRA tensors")
    if not summary["adapter_nonzero"]:
        errors.append("adapter payload looks like an empty/zero LoRA")
    if saved_payload:
        saved = adapter_payload_summary(saved_payload)
        if saved["digest"] and summary["digest"] and saved["digest"] != summary["digest"]:
            errors.append(
                "reload adapter payload digest != saved θ1 payload digest"
            )
    rows = [dict(item) for item in list(generate_rows or [])]
    if not rows:
        errors.append("no vLLM generate evidence rows")
        return errors
    listed_ok = False
    attached_ok = False
    missing_request = 0
    for row in rows:
        listed = [int(item) for item in list(row.get("listed_lora_ids") or [])]
        if int(lora_int_id) in listed:
            listed_ok = True
        attached = bool(row.get("lora_request_attached"))
        if attached:
            attached_ok = True
        if row.get("lora_as_adapter") and not attached:
            missing_request += 1
        attached_id = row.get("lora_int_id")
        if attached and attached_id is not None and int(attached_id) != int(lora_int_id):
            errors.append(f"generate attached lora_int_id={attached_id} != {lora_int_id}")
    if not listed_ok:
        errors.append(f"list_loras missing adapter id {lora_int_id}")
    if not attached_ok:
        errors.append("generate did not attach LoRARequest")
    if missing_request:
        errors.append(
            f"{missing_request} generate call(s) had lora_as_adapter but no LoRARequest"
        )
    return errors


def m4c_gate(
    *,
    optimizer_gate: Mapping[str, Any],
    theta0: Mapping[str, Any],
    theta1: Mapping[str, Any],
    reloaded: Mapping[str, Any],
    checkpoint_errors: Sequence[str],
    vllm_errors: Sequence[str],
    n_reload_episodes: int,
    reload_tito_errors: Sequence[str] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not optimizer_gate.get("pass"):
        reasons.append(
            "phase-A optimizer gate failed: "
            + "; ".join(str(item) for item in (optimizer_gate.get("reasons") or ["missing"]))
        )
    vs_theta0 = compare_lora_fingerprints(theta0, theta1)
    vs_reload = compare_lora_fingerprints(theta1, reloaded)
    vs_reload_theta0 = compare_lora_fingerprints(reloaded, theta0)
    if vs_theta0.get("equal"):
        reasons.append("θ1 == θ0; LoRA did not change before save")
    if not vs_theta0.get("n_left"):
        reasons.append("θ0 LoRA fingerprint missing")
    if not vs_reload.get("equal"):
        reasons.append("reloaded LoRA fingerprint != saved θ1")
    if vs_reload_theta0.get("equal"):
        reasons.append("reloaded LoRA fingerprint == θ0; empty adapter was reloaded")
    if checkpoint_errors:
        reasons.extend(f"checkpoint: {item}" for item in list(checkpoint_errors)[:8])
    if vllm_errors:
        reasons.extend(f"vLLM: {item}" for item in list(vllm_errors)[:8])
    if int(n_reload_episodes) < 1:
        reasons.append("no post-reload AgentLoop episode")
    tito = [str(item) for item in list(reload_tito_errors or [])]
    if tito:
        reasons.append("post-reload TITO/mask errors: " + "; ".join(tito[:4]))
    return {
        "pass": not reasons,
        "reasons": reasons,
        "theta1_ne_theta0": not vs_theta0.get("equal"),
        "reloaded_eq_theta1": bool(vs_reload.get("equal")),
        "reloaded_ne_theta0": not vs_reload_theta0.get("equal"),
        "theta0_digest": vs_theta0.get("left_digest"),
        "theta1_digest": vs_theta0.get("right_digest"),
        "reloaded_digest": vs_reload.get("right_digest"),
    }


def load_generate_evidence(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, MappingABC):
            rows.append(dict(payload))
    return rows


def write_summary(
    path: Path,
    *,
    status: str,
    gate: Mapping[str, Any],
    verl_commit: str,
    instance_ids: Sequence[str],
    elapsed_s: float,
    checkpoint_root: str,
    n_reload_episodes: int,
    optimizer: Mapping[str, Any] | None = None,
    vllm: Mapping[str, Any] | None = None,
) -> None:
    optimizer = dict(optimizer or {})
    vllm = dict(vllm or {})
    lines = [
        "# M4C / E009 Adapter Persistence and Updated-Policy Rollout",
        "",
        f"- status: **{status}**",
        f"- elapsed_s: {float(elapsed_s):.1f}",
        f"- veRL: `{verl_commit}`",
        f"- instance_ids: {', '.join(instance_ids)}",
        f"- θ0 digest: `{gate.get('theta0_digest')}`",
        f"- θ1 digest: `{gate.get('theta1_digest')}`",
        f"- reloaded digest: `{gate.get('reloaded_digest')}`",
        f"- θ1 != θ0: {gate.get('theta1_ne_theta0')}",
        f"- reloaded == θ1: {gate.get('reloaded_eq_theta1')}",
        f"- reloaded != θ0: {gate.get('reloaded_ne_theta0')}",
        f"- checkpoint: `{checkpoint_root}`",
        f"- vLLM lora_int_id: {vllm.get('lora_int_id', VLLM_LORA_INT_ID)}",
        f"- vLLM lora_request_attached: {vllm.get('lora_request_attached')}",
        f"- post-reload episodes: {int(n_reload_episodes)}",
        f"- optimizer LoRA changed: {optimizer.get('n_lora_changed')}/"
        f"{optimizer.get('n_lora_tensors')} base_frozen={optimizer.get('base_frozen')}",
        f"- gate reasons: {gate.get('reasons') or ['(none)']}",
        "",
        "PASS requires θ0 → GRPO → θ1 → official FSDP save → fresh reload(θ1)",
        "→ naive actor→vLLM TensorLoRARequest(123) → real AgentLoop generate.",
        "Rollout text change and reward improvement are not pass criteria.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Re-export smoke selection helpers so the GPU driver can import from one module.
__all__ = [
    "ADAPTER_ABS_EPS",
    "BUDGET_VISIBLE",
    "EXPERIMENT_ID",
    "GROUP_N",
    "LORA_ALPHA",
    "LORA_RANK",
    "MILESTONE",
    "N_TASKS",
    "OBS_TOKENS_LIMIT",
    "OUTPUT_ENV",
    "PHASE_ENV",
    "PINNED_VERL_COMMIT",
    "PINNED_VERL_VERSION",
    "RELOAD_N",
    "REWARD_FN_RELPATH",
    "VERL_PATH_TEXT",
    "VLLM_LORA_INT_ID",
    "VLLM_LORA_NAME",
    "adapter_payload_summary",
    "append_jsonl",
    "artifact_hashes",
    "build_checkpoint_manifest",
    "checkpoint_integrity_errors",
    "compare_lora_fingerprints",
    "compare_param_snapshots",
    "current_phase",
    "default_candidate_path",
    "default_checkpoint_root",
    "default_e007_groups_path",
    "default_freeze_path",
    "default_output_dir",
    "evidence_dir",
    "fingerprint_digest",
    "freeze_contract_errors",
    "global_step_dir",
    "is_lora_param_name",
    "load_candidate_ordered_ids",
    "load_e007_groups",
    "load_generate_evidence",
    "load_json",
    "lora_sha256_map",
    "m4c_gate",
    "persist_lora_fingerprint",
    "select_smoke_instance_ids",
    "vllm_sync_errors",
    "write_json",
    "write_smoke_parquet",
    "write_summary",
]
