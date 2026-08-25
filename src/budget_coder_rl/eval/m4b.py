"""M4B CPU helpers: one-step GRPO smoke selection, mask/LoRA evidence.

Does not run the optimizer. Gold lists stay in the evaluator sidecar.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.eval.m4a import (
    ADVANTAGE_ABS_EPS,
    BUDGET_VISIBLE,
    GROUP_N,
    OBS_TOKENS_LIMIT,
    artifact_hashes,
    assemble_group_evidence,
    default_candidate_path,
    default_e007_groups_path,
    default_freeze_path,
    freeze_contract_errors,
    leakage_errors,
    load_candidate_ordered_ids,
    load_e007_groups,
    load_json,
    scalar_advantage,
    select_smoke_instance_ids,
)
from budget_coder_rl.eval.provenance import git_info, sha256_file

EXPERIMENT_ID = "E003"
MILESTONE = "M4B"
N_TASKS = 2
PROMPT_LENGTH = 16384
RESPONSE_LENGTH = 16384
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET = "all-linear"
GRAD_ABS_EPS = 1e-8
PARAM_DELTA_EPS = 1e-8
PINNED_VERL_COMMIT = "8481f9f9880d0f46a75b3db0329d3de8abad3d81"
PINNED_VERL_VERSION = "0.8.0.dev0"
OUTPUT_ENV = "BCRL_M4B_OUTPUT_DIR"
REWARD_FN_RELPATH = "src/budget_coder_rl/reward/localization_score.py"

VERL_PATH_TEXT = """# M4B pinned veRL optimizer path

Checkout: `{commit}` (`0.8.0.dev0`, fork LingyuZhi/rtrl-verl).

## Production update path

- Entry: `verl.trainer.main_ppo.run_ppo` → `TaskRunner.run` → `RayPPOTrainer.fit`
- Grouping: `uid = uuid4()` per logical prompt, then `DataProto.repeat(n=4, interleave=True)`
  (`verl/trainer/ppo/ray_trainer.py` fit).
- Rollout: `async_rollout_manager.generate_sequences` (AgentLoop; vLLM per-request n=1).
- Reward: RewardLoop / NaiveRewardManager `rm_scores`; `use_kl_in_reward=false`.
- Advantage: `compute_grpo_outcome_advantage(..., index=uid)`.
- Actor update: `RayPPOTrainer._update_actor` → `actor_rollout_wg.update_actor`
  → `ActorRolloutRefWorker.update_actor` → `TrainingWorker.train_mini_batch`
  → `engine.train_batch` (`optimizer_zero_grad` / `forward_backward` `ppo_loss` /
  `optimizer_step`).
- Loss mask: `ppo_loss` uses AgentLoop `response_mask`. Trainer only synthesizes
  `attention_mask[:, -R:]` when `response_mask` is missing.
- LoRA: PEFT via `actor_rollout_ref.model.lora_rank`; FSDP1 (`strategy=fsdp`).
- M4B does not verify vLLM consumed the post-step adapter (M4C).
"""


def default_output_dir(repo_root: Path, experiment_id: str = EXPERIMENT_ID) -> Path:
    return Path(repo_root) / "outputs" / "experiments" / experiment_id


def is_lora_param_name(name: str) -> bool:
    lowered = str(name).lower()
    return (
        "lora_a" in lowered
        or "lora_b" in lowered
        or "lora_embedding" in lowered
        or ".lora_" in lowered
    )


def attention_response_slice(
    attention_mask: Sequence[Sequence[int]],
    response_width: int,
) -> list[list[int]]:
    width = int(response_width)
    out: list[list[int]] = []
    for row in attention_mask:
        values = [int(item) for item in list(row)]
        out.append(values[-width:] if width else [])
    return out


def mask_correctness_errors(
    *,
    response_mask: Sequence[Sequence[int]],
    attention_response: Sequence[Sequence[int]],
    advantages: Sequence[Sequence[float]],
    n_observation_tokens: Sequence[int],
) -> list[str]:
    """Assert AgentLoop mask semantics without reconstructing token IDs."""
    errors: list[str] = []
    if not response_mask:
        return ["response_mask is empty"]
    if len(response_mask) != len(attention_response):
        errors.append("response_mask / attention_response row count mismatch")
    if len(response_mask) != len(advantages):
        errors.append("response_mask / advantages row count mismatch")
    for index, mask_row in enumerate(response_mask):
        attn_row = list(attention_response[index]) if index < len(attention_response) else []
        adv_row = list(advantages[index]) if index < len(advantages) else []
        if len(mask_row) != len(attn_row):
            errors.append(f"row {index}: mask width {len(mask_row)} != attention width {len(attn_row)}")
            continue
        n_obs = int(n_observation_tokens[index]) if index < len(n_observation_tokens) else 0
        if n_obs > 0 and mask_row == attn_row:
            errors.append(
                f"row {index}: response_mask equals attention[:, -R:] despite "
                f"{n_obs} observation tokens; trainer likely used the fallback mask"
            )
        n_policy = sum(1 for bit in mask_row if int(bit) == 1)
        if n_policy <= 0:
            errors.append(f"row {index}: no policy tokens in response_mask")
        for pos, (bit, attn) in enumerate(zip(mask_row, attn_row)):
            if int(attn) == 0 and int(bit) != 0:
                errors.append(f"row {index}: padding position {pos} has response_mask=1")
                break
        if len(adv_row) == len(mask_row):
            for pos, (bit, adv) in enumerate(zip(mask_row, adv_row)):
                if int(bit) == 0 and abs(float(adv)) > ADVANTAGE_ABS_EPS:
                    errors.append(
                        f"row {index}: advantage {adv} on mask=0 position {pos}"
                    )
                    break
    return errors


def count_mask_tokens(response_mask: Sequence[int], n_obs: int) -> dict[str, int]:
    bits = [int(item) for item in list(response_mask)]
    n_policy = sum(1 for bit in bits if bit == 1)
    n_zero = sum(1 for bit in bits if bit == 0)
    n_obs_i = int(n_obs)
    n_pad = max(0, n_zero - n_obs_i)
    return {
        "n_policy": n_policy,
        "n_observation": n_obs_i,
        "n_pad": n_pad,
        "n_mask0": n_zero,
        "width": len(bits),
    }


def fingerprint_numeric(
    *,
    sha256: str,
    numel: int,
    mean: float,
    max_abs: float,
    full_hash: bool = True,
) -> dict[str, Any]:
    return {
        "sha256": str(sha256),
        "numel": int(numel),
        "mean": float(mean),
        "max_abs": float(max_abs),
        "full_hash": bool(full_hash),
    }


def compare_param_snapshots(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    delta_eps: float = PARAM_DELTA_EPS,
) -> dict[str, Any]:
    before_lora = dict(before.get("lora") or {})
    after_lora = dict(after.get("lora") or {})
    before_frozen = dict(before.get("frozen") or {})
    after_frozen = dict(after.get("frozen") or {})
    lora_changed: list[str] = []
    lora_unchanged: list[str] = []
    max_abs_delta = 0.0
    for name, prior in before_lora.items():
        later = after_lora.get(name)
        if later is None:
            lora_changed.append(name)
            continue
        delta = abs(float(later.get("mean") or 0.0) - float(prior.get("mean") or 0.0))
        abs_delta = abs(float(later.get("max_abs") or 0.0) - float(prior.get("max_abs") or 0.0))
        max_abs_delta = max(max_abs_delta, delta, abs_delta)
        if str(later.get("sha256")) != str(prior.get("sha256")):
            lora_changed.append(name)
        else:
            lora_unchanged.append(name)
    frozen_changed = [
        name
        for name, prior in before_frozen.items()
        if str((after_frozen.get(name) or {}).get("sha256")) != str(prior.get("sha256"))
    ]
    unexpected_trainable = [
        str(item) for item in list(after.get("unexpected_trainable") or [])
    ]
    n_trainable = int(after.get("n_trainable") or 0)
    n_lora = len(after_lora)
    return {
        "n_trainable": n_trainable,
        "n_frozen": int(after.get("n_frozen") or 0),
        "n_lora_tensors": n_lora,
        "n_lora_changed": len(lora_changed),
        "n_lora_unchanged": len(lora_unchanged),
        "lora_changed_names": lora_changed[:32],
        "frozen_changed_names": frozen_changed[:32],
        "n_frozen_changed": len(frozen_changed),
        "max_abs_delta": max_abs_delta,
        "unexpected_trainable": unexpected_trainable,
        "lora_changed": bool(lora_changed),
        "base_frozen": not frozen_changed,
        "trainable_are_lora": n_trainable > 0 and n_lora == n_trainable,
    }


def unwrap_metric_value(value: Any) -> Any:
    """Reduce veRL Metric / allgather list / tensor wrappers to a scalar."""
    current = value
    for _ in range(8):
        if current is None:
            return None
        if isinstance(current, (list, tuple)):
            current = current[0] if current else None
            continue
        if isinstance(current, MappingABC):
            if "value" in current:
                current = current["value"]
                continue
            break
        aggregate = getattr(current, "aggregate", None)
        if callable(aggregate) and not isinstance(current, (str, bytes)):
            try:
                current = aggregate()
                continue
            except Exception:
                pass
        values = getattr(current, "values", None)
        if isinstance(values, list) and values and not isinstance(current, (str, bytes)):
            current = values[0]
            continue
        item = getattr(current, "item", None)
        if callable(item) and not isinstance(current, (str, bytes)):
            try:
                current = item()
                continue
            except Exception:
                break
        break
    return current


def metric_finite_nonzero(value: Any, *, eps: float = GRAD_ABS_EPS) -> dict[str, Any]:
    try:
        number = float(unwrap_metric_value(value))
    except (TypeError, ValueError):
        return {"value": None, "finite": False, "nonzero": False}
    finite = number == number and abs(number) != float("inf")
    return {
        "value": number,
        "finite": bool(finite),
        "nonzero": bool(finite and abs(number) > float(eps)),
    }


def assemble_loss_mask_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    tito_errors: Sequence[str],
    mask_errors: Sequence[str],
) -> dict[str, Any]:
    n_policy = sum(int(item.get("n_policy") or 0) for item in rows)
    n_obs = sum(int(item.get("n_observation") or 0) for item in rows)
    n_pad = sum(int(item.get("n_pad") or 0) for item in rows)
    fallback_conflicts = [
        item for item in rows if item.get("obs_and_mask_equals_attention")
    ]
    return {
        "n_rows": len(rows),
        "n_policy_tokens": n_policy,
        "n_observation_tokens": n_obs,
        "n_pad_tokens": n_pad,
        "response_mask_present": all(item.get("response_mask_present") for item in rows),
        "advantages_zero_on_mask0": all(
            item.get("advantages_zero_on_mask0") for item in rows
        ),
        "n_obs_rows_matching_attention_fallback": len(fallback_conflicts),
        "tito_errors": list(tito_errors),
        "mask_errors": list(mask_errors),
        "rows": [dict(item) for item in rows],
        "ok": not tito_errors and not mask_errors and n_policy > 0,
    }


def assemble_groups_from_members(
    members: Sequence[Mapping[str, Any]],
    *,
    group_n: int = GROUP_N,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for member in members:
        uid = str(member.get("uid") or "")
        if uid not in grouped:
            order.append(uid)
        grouped[uid].append(dict(member))
    groups: list[dict[str, Any]] = []
    for uid in order:
        evidence = assemble_group_evidence(grouped[uid])
        evidence["group_n_expected"] = int(group_n)
        groups.append(evidence)
    return groups


def step_learning_signal(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nonzero = any(item.get("nonzero_advantage") for item in groups)
    mixed = any(item.get("mixed") for item in groups)
    return {
        "n_groups": len(groups),
        "n_mixed": sum(1 for item in groups if item.get("mixed")),
        "n_nonzero_advantage": sum(1 for item in groups if item.get("nonzero_advantage")),
        "any_nonzero_advantage": bool(nonzero),
        "any_mixed": bool(mixed),
        "ok": bool(nonzero),
    }


def m4b_gate(
    *,
    learning: Mapping[str, Any],
    loss_mask: Mapping[str, Any],
    grad: Mapping[str, Any],
    pg_loss: Mapping[str, Any],
    lora: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not learning.get("ok"):
        reasons.append("batch has no non-zero GRPO advantage")
    if not loss_mask.get("ok"):
        reasons.append("masked actor-loss contract failed")
    if not grad.get("finite") or not grad.get("nonzero"):
        reasons.append("grad_norm is not finite and non-zero")
    if not pg_loss.get("finite"):
        reasons.append("actor/pg_loss is not finite")
    if not lora.get("lora_changed"):
        reasons.append("no LoRA parameter hash changed")
    if not lora.get("base_frozen"):
        reasons.append("frozen/base parameters changed")
    if not lora.get("trainable_are_lora"):
        reasons.append("trainable parameters are not the expected LoRA set")
    passed = not reasons
    return {"pass": passed, "reasons": reasons}


def write_smoke_parquet(
    source_parquet: Path,
    dest_parquet: Path,
    instance_ids: Sequence[str],
) -> dict[str, Any]:
    import pandas as pd

    ordered = [str(item) for item in instance_ids]
    wanted = set(ordered)
    frame = pd.read_parquet(source_parquet)
    if "extra_info" not in frame.columns:
        raise ValueError(f"{source_parquet} missing extra_info")
    index_by_id: dict[str, int] = {}
    extras: list[dict[str, Any]] = []
    for row_i, extra in enumerate(frame["extra_info"].tolist()):
        payload = dict(extra) if isinstance(extra, MappingABC) else {}
        instance_id = str(payload.get("instance_id") or "").strip()
        extras.append(payload)
        if instance_id in wanted and instance_id not in index_by_id:
            index_by_id[instance_id] = row_i
    missing = [item for item in ordered if item not in index_by_id]
    if missing:
        raise ValueError(f"instance_ids not in parquet: {missing}")
    rows = []
    for instance_id in ordered:
        row = frame.iloc[index_by_id[instance_id]].to_dict()
        extra = dict(extras[index_by_id[instance_id]])
        extra["obs_tokens_limit"] = OBS_TOKENS_LIMIT
        extra["budget_visible"] = BUDGET_VISIBLE
        extra.pop("sampling_seed", None)
        row["extra_info"] = extra
        rows.append(row)
    dest_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(dest_parquet, index=False)
    return {
        "path": str(dest_parquet),
        "n_rows": len(rows),
        "instance_ids": ordered,
        "sha256": sha256_file(dest_parquet),
    }


def audit_verl_checkout(
    *,
    output_dir: Path | None = None,
    require_pin: bool = True,
) -> dict[str, Any]:
    import verl

    source = Path(verl.__file__).resolve().parents[1]
    info = git_info(source)
    info["version"] = getattr(verl, "__version__", None)
    info["pinned_commit"] = PINNED_VERL_COMMIT
    info["pinned_version"] = PINNED_VERL_VERSION
    info["matches_pin"] = str(info.get("commit") or "") == PINNED_VERL_COMMIT
    patch_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if info.get("dirty"):
            patch_path = output_dir / "verl_local.patch"
            diff = _git_output(["diff"], cwd=source) or ""
            staged = _git_output(["diff", "--cached"], cwd=source) or ""
            patch_path.write_text(diff + staged, encoding="utf-8")
            info["local_patch"] = str(patch_path)
            info["local_patch_sha256"] = sha256_file(patch_path)
        (output_dir / "verl_checkout.md").write_text(
            _verl_checkout_markdown(info),
            encoding="utf-8",
        )
    if require_pin and not info["matches_pin"]:
        raise SystemExit(
            "HARD FAIL: imported veRL commit "
            f"{info.get('commit')!r} != pinned {PINNED_VERL_COMMIT}. "
            "Do not silent-checkout or follow latest main."
        )
    return info


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    *,
    status: str,
    groups: Sequence[Mapping[str, Any]],
    learning: Mapping[str, Any],
    loss_mask: Mapping[str, Any],
    grad: Mapping[str, Any],
    pg_loss: Mapping[str, Any],
    lora: Mapping[str, Any],
    gate: Mapping[str, Any],
    verl_commit: str,
    instance_ids: Sequence[str],
    elapsed_s: float,
) -> None:
    lines = [
        "# M4B / E003 One-Step GRPO Optimizer",
        "",
        f"- status: **{status}**",
        f"- elapsed_s: {float(elapsed_s):.1f}",
        f"- veRL: `{verl_commit}`",
        f"- instance_ids: {', '.join(instance_ids)}",
        f"- groups: {learning.get('n_groups')} mixed={learning.get('n_mixed')} "
        f"nonzero_adv={learning.get('n_nonzero_advantage')}",
        f"- pg_loss: {pg_loss.get('value')} finite={pg_loss.get('finite')}",
        f"- grad_norm: {grad.get('value')} finite={grad.get('finite')} "
        f"nonzero={grad.get('nonzero')}",
        f"- LoRA changed: {lora.get('n_lora_changed')}/{lora.get('n_lora_tensors')} "
        f"base_frozen={lora.get('base_frozen')}",
        f"- mask ok: {loss_mask.get('ok')} policy_tokens={loss_mask.get('n_policy_tokens')} "
        f"obs={loss_mask.get('n_observation_tokens')} pad={loss_mask.get('n_pad_tokens')}",
        f"- gate reasons: {gate.get('reasons') or ['(none)']}",
        "",
        "PASS requires non-zero GRPO advantage, AgentLoop `response_mask` loss,",
        "finite non-zero grad, one real optimizer step, LoRA change, and frozen base.",
        "Adapter reload / vLLM using the new adapter is M4C.",
        "",
    ]
    for group in groups:
        lines.append(f"## {group.get('instance_id')}")
        lines.append("")
        lines.append(f"- uid: `{group.get('uid')}`")
        lines.append(f"- rewards: {group.get('rewards')}")
        lines.append(f"- advantages: {group.get('advantages')}")
        lines.append(f"- mixed: {group.get('mixed')}")
        lines.append(f"- nonzero_advantage: {group.get('nonzero_advantage')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verl_checkout_markdown(info: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# veRL checkout audit (M4B)",
            "",
            f"- path: `{info.get('path')}`",
            f"- version: `{info.get('version')}`",
            f"- HEAD: `{info.get('commit')}`",
            f"- pinned: `{info.get('pinned_commit')}`",
            f"- matches_pin: {info.get('matches_pin')}",
            f"- dirty: {info.get('dirty')}",
            f"- n_dirty_files: {info.get('n_dirty_files')}",
            f"- local_patch: `{info.get('local_patch')}`",
            "",
            "Dirty files (first 50):",
            "",
            "```text",
            "\n".join(str(item) for item in (info.get("dirty_files") or ["(clean)"])),
            "```",
            "",
        ]
    )


def _git_output(args: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
