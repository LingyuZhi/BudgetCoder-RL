#!/usr/bin/env python
"""M3A smoke: hard obs-token budget + post-rollout localization evaluator.

CPU mode: real M1 row + fake TITO AgentLoop (no Ray/vLLM).
GPU mode: real Qwen3-4B async AgentLoopManager path (compute node).

Does not run GRPO, LoRA, or veRL RewardLoop.

Usage (pinned RL conda env):

    python scripts/smoke/smoke_budget_eval_contract.py --mode cpu
    python scripts/smoke/smoke_budget_eval_contract.py --mode gpu --n-gpus 1
    python scripts/smoke/smoke_budget_eval_contract.py --mode both --n-gpus 1
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_exploration_scaffold import first_tree_file, select_smoke_task  # noqa: E402
from smoke_repo_exploration_agent_loop import (  # noqa: E402
    FakeServerManager,
    encode_action,
    find_dataset_row,
    mask_segments,
)
from smoke_repo_workspace import load_task_rows  # noqa: E402
from smoke_rlhf_dataset import build_dataset, resolve_tokenizer_path  # noqa: E402

from budget_coder_rl.agent_loop.repo_exploration import (  # noqa: E402
    RepoExplorationAgentLoop,
)
from budget_coder_rl.budget.state import BUDGET_OBS_VERSION  # noqa: E402
from budget_coder_rl.data.swe_gym_materialize import oracle_parquet_path  # noqa: E402
from budget_coder_rl.data.swe_gym_repos import (  # noqa: E402
    bcrl_data_root,
    cache_path_for_repo,
    is_git_dir,
    swe_gym_repos_root,
)
from budget_coder_rl.env import ExplorationSession, RepoEnvironment, TaskRef  # noqa: E402
from budget_coder_rl.eval.episode import build_episode_record, summarize_episodes  # noqa: E402
from budget_coder_rl.eval.localization import evaluate_episode  # noqa: E402
from budget_coder_rl.eval.oracle import load_evaluator_oracle  # noqa: E402
from budget_coder_rl.eval.provenance import collect_run_provenance  # noqa: E402
from budget_coder_rl.protocol.observation import OBS_VERSION  # noqa: E402
from budget_coder_rl.protocol.prompt import runtime_prompt_audit  # noqa: E402

AGENT_LOOP_CONFIG = REPO_ROOT / "configs" / "agent_loop" / "repo_exploration_m3a.yaml"
D1_INSTANCE_ID = "pydantic__pydantic-4882"
SMOKE_OBS_TOKENS_LIMIT = 8192
PROMPT_LENGTH = 16384
RESPONSE_LENGTH = 16384
MAX_MODEL_LEN = 32768
D1_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "do_sample": False}
TRACE_NOTE = (
    "Research/debug artifact. AgentLoopOutput / DataProto token arrays are the "
    "training truth. Do not rebuild RL token trajectories from this JSONL."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu", "both"), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--repos-root", type=Path, default=None)
    parser.add_argument("--snapshots-root", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--raw-parquet", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke",
    )
    return parser.parse_args(argv)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if hasattr(value, "items"):
        return {str(key): val for key, val in value.items()}
    raise TypeError(f"expected mapping, got {type(value)!r}")


def _tool(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>\n{payload}\n</tool_call>"


def _final(payload: dict[str, Any]) -> str:
    return "<final>\n" + json.dumps(payload, separators=(",", ":")) + "\n</final>"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=True) + "\n")


def with_budget(extra: Mapping[str, Any], *, visible: bool, limit: int | None) -> dict[str, Any]:
    out = dict(extra)
    out["budget_visible"] = visible
    out["obs_tokens_limit"] = limit
    return out


def instantiate_loop(env, tokenizer, server, *, prompt_length: int, response_length: int):
    import hydra
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset

    configs = OmegaConf.load(str(AGENT_LOOP_CONFIG))
    trainer_config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                }
            }
        }
    )
    loop = hydra.utils.instantiate(
        configs[0],
        trainer_config=DictConfigWrap(trainer_config),
        server_manager=server,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(OmegaConf.create({})),
        tools=ToolListWrap([]),
        repo_environment=env,
    )
    if not isinstance(loop, RepoExplorationAgentLoop):
        raise SystemExit(f"unexpected loop type {type(loop)!r}")
    return loop


def verify_loop_output(output, *, expect_termination: str | None = None) -> list[str]:
    errors: list[str] = []
    extra = output.extra_fields
    segments = extra.get("segments") or []
    if len(output.response_ids) != len(output.response_mask):
        errors.append("response_ids/mask length mismatch")
    expected_mask = mask_segments(output.response_mask, segments)
    if output.response_mask != expected_mask:
        errors.append("response_mask does not match assistant=1 observation=0")
    obs_total = sum(
        len(item["token_ids"]) for item in segments if item["kind"] == "observation"
    )
    repo_obs = extra.get("tool_observation_token_count")
    if extra.get("budget_accounting_version") != "bcrl-bobs-v2":
        errors.append(
            f"budget_accounting_version={extra.get('budget_accounting_version')!r}"
        )
    if extra.get("obs_tokens_used") != repo_obs:
        errors.append(
            f"obs_tokens_used {extra.get('obs_tokens_used')} != repo obs {repo_obs}"
        )
    if extra.get("observation_token_count") != obs_total:
        errors.append("observation_token_count != inserted observation ids")
    if extra.get("total_env_tokens") != obs_total:
        errors.append("total_env_tokens != inserted observation ids")
    if extra.get("repo_observation_tokens") != repo_obs:
        errors.append("repo_observation_tokens != tool_observation_token_count")
    metadata = extra.get("budget_metadata_tokens")
    if metadata is None:
        errors.append("budget_metadata_tokens missing")
    elif int(metadata) != int(obs_total) - int(repo_obs or 0):
        errors.append("budget_metadata_tokens != total_env - repo_obs")
    if extra.get("budget_visible") and obs_total > 0 and int(metadata or 0) <= 0:
        errors.append("visible inserted obs missing envelope metadata tokens")
    if extra.get("budget_visible") is False and int(metadata or 0) != 0:
        errors.append("hidden episode has nonzero budget_metadata_tokens")
    if extra.get("policy_token_count") != output.response_mask.count(1):
        errors.append("policy_token_count != mask==1 count")
    if expect_termination is not None and extra.get("termination") != expect_termination:
        errors.append(
            f"termination={extra.get('termination')!r} expected {expect_termination!r}"
        )
    if extra.get("termination") == "budget_exhausted" and extra.get("final_submission"):
        errors.append("budget_exhausted must not carry a valid final submission")
    keys = set(extra)
    for forbidden in ("oracle_symbols", "base_changed_files", "patch", "hints_text"):
        if forbidden in keys:
            errors.append(f"oracle/privileged key leaked into extra_fields: {forbidden}")
    return errors


def score_output(output, oracle_index, *, provenance: dict[str, Any]) -> dict[str, Any]:
    extra = output.extra_fields
    instance_id = str(extra.get("instance_id") or "")
    loc = None
    if instance_id and instance_id in oracle_index:
        loc = evaluate_episode(
            termination=extra.get("termination"),
            submission=extra.get("final_submission"),
            oracle=oracle_index.get(instance_id),
        ).as_dict()
    record = build_episode_record(extra, localization=loc, provenance=provenance)
    record["trace_note"] = TRACE_NOTE
    return record


def run_cpu(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from verl.workers.rollout.replica import TokenOutput

    repo_root = args.repo_root.resolve()
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    if tokenizer_path is None:
        raise SystemExit("HARD FAIL: no local Qwen tokenizer; set BCRL_TOKENIZER_PATH")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_path = (
        args.train.resolve()
        if args.train is not None
        else repo_root / "data" / "processed" / "swe_gym" / "train.parquet"
    )
    if not train_path.is_file():
        raise SystemExit(f"HARD FAIL: missing M1E train parquet {train_path}")
    oracle_path = (
        args.oracle.resolve()
        if args.oracle is not None
        else oracle_parquet_path(repo_root)
    )
    if not oracle_path.is_file():
        raise SystemExit(f"HARD FAIL: missing evaluator oracle {oracle_path}")
    oracle_index = load_evaluator_oracle(oracle_path)

    cache_dir = repo_root / "outputs" / "smoke" / "rlhf_dataset_cache" / "m3a"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(train_path, tokenizer, cache_dir)
    rows = load_task_rows(repo_root, args.train, args.raw_parquet)
    task = select_smoke_task(rows)
    item = find_dataset_row(dataset, task.instance_id)
    extra = _as_mapping(item.get("extra_info"))
    raw_prompt = item.get("raw_prompt")

    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(args.data_root)
    )
    store = cache_path_for_repo(task.repo, repos_root)
    if not is_git_dir(store):
        raise SystemExit(f"HARD FAIL: local object store missing: {store}")
    env = RepoEnvironment(
        repos_root=repos_root,
        snapshots_root=(
            args.snapshots_root.expanduser() if args.snapshots_root is not None else None
        ),
        data_root=args.data_root,
    )
    workspace = env.prepare(task)
    workspace.validate()
    probe = ExplorationSession(workspace)
    tree_obs = probe.step(_tool("tree", {"path": ".", "depth": 2}))
    if tree_obs.error_kind is not None:
        raise SystemExit(f"HARD FAIL: probe tree failed:\n{tree_obs.observation}")
    rel = first_tree_file(tree_obs.observation)
    filename = rel.rsplit("/", 1)[-1]
    query = filename[:-3] if filename.endswith(".py") else filename[:32]
    actions = [
        _tool("tree", {"path": ".", "depth": 2}),
        _tool("search", {"query": query, "path": "."}),
        _final({"locations": [{"path": rel, "symbol": "Scripted.placeholder"}]}),
    ]
    action_ids = [encode_action(tokenizer, text) for text in actions]
    provenance = collect_run_provenance(
        repo_root,
        agent_loop_config=AGENT_LOOP_CONFIG,
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
    )
    episodes: list[dict[str, Any]] = []
    errors: list[str] = []

    def run_scripted(extra_info: dict[str, Any], ids: list[list[int]]):
        server = FakeServerManager(
            [TokenOutput(token_ids=list(token_ids)) for token_ids in ids]
        )
        loop = instantiate_loop(
            env,
            tokenizer,
            server,
            prompt_length=32768,
            response_length=8192,
        )
        return loop.loop.run_until_complete(
            loop.run({"temperature": 1.0}, raw_prompt=raw_prompt, extra_info=extra_info)
        )

    hidden_extra = with_budget(extra, visible=False, limit=SMOKE_OBS_TOKENS_LIMIT)
    hidden_out = run_scripted(hidden_extra, action_ids)
    errors.extend(verify_loop_output(hidden_out, expect_termination="finish"))
    hidden_decode = tokenizer.decode(
        hidden_out.extra_fields["segments"][1]["token_ids"],
        skip_special_tokens=True,
    )
    if OBS_VERSION not in hidden_decode:
        errors.append("hidden observation missing bcrl-obs-v1")
    if BUDGET_OBS_VERSION in hidden_decode:
        errors.append("hidden observation unexpectedly contains bcrl-budget-v1")
    episodes.append(score_output(hidden_out, oracle_index, provenance=provenance))

    visible_extra = with_budget(extra, visible=True, limit=SMOKE_OBS_TOKENS_LIMIT)
    visible_out = run_scripted(visible_extra, action_ids)
    errors.extend(verify_loop_output(visible_out, expect_termination="finish"))
    vis_decode = tokenizer.decode(
        visible_out.extra_fields["segments"][1]["token_ids"],
        skip_special_tokens=True,
    )
    if BUDGET_OBS_VERSION not in vis_decode:
        errors.append("visible observation missing bcrl-budget-v1")
    if OBS_VERSION not in vis_decode:
        errors.append("visible observation missing bcrl-obs-v1")
    if visible_out.extra_fields["events"][0]["observation"] != hidden_out.extra_fields[
        "events"
    ][0]["observation"]:
        errors.append("hidden/visible v1 tool observation bodies drifted")
    episodes.append(score_output(visible_out, oracle_index, provenance=provenance))

    exhaust_ids = [action_ids[0], action_ids[1]]
    exhaust_extra = with_budget(extra, visible=False, limit=1)
    exhaust_out = run_scripted(exhaust_extra, exhaust_ids)
    errors.extend(verify_loop_output(exhaust_out, expect_termination="budget_exhausted"))
    if exhaust_out.extra_fields.get("obs_tokens_used") != 0:
        errors.append("exhausted episode inserted observation tokens")
    episodes.append(score_output(exhaust_out, oracle_index, provenance=provenance))
    if episodes[-1].get("localization", {}).get("localization_score") != 0:
        errors.append("budget_exhausted localization_score must be 0")

    audit = runtime_prompt_audit()
    if not audit["search_is_case_sensitive_literal_substring"]:
        errors.append("runtime prompt no longer states literal substring search")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "instance_id": extra.get("instance_id"),
        "repo": extra.get("repo"),
        "prompt_audit": audit,
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
        "hidden": {
            "termination": hidden_out.extra_fields.get("termination"),
            "obs_tokens_used": hidden_out.extra_fields.get("obs_tokens_used"),
            "file_f1": episodes[0].get("localization", {}).get("file_f1"),
        },
        "visible": {
            "termination": visible_out.extra_fields.get("termination"),
            "obs_tokens_used": visible_out.extra_fields.get("obs_tokens_used"),
            "file_f1": episodes[1].get("localization", {}).get("file_f1"),
        },
        "exhaust": {
            "termination": exhaust_out.extra_fields.get("termination"),
            "obs_tokens_used": exhaust_out.extra_fields.get("obs_tokens_used"),
        },
        "provenance": provenance,
    }


def run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    from smoke_repo_exploration_real_rollout import (
        build_batch,
        build_config,
        ensure_snapshot,
        get_verl_info,
        init_agent_loop_manager,
        resolve_model_path,
    )
    from budget_coder_rl.agent_loop.rollout_verify import (
        inspect_turn_boundary,
        segment_decomposition,
        verify_padded_sample,
    )
    from transformers import AutoTokenizer

    repo_root = args.repo_root.resolve()
    model_path = resolve_model_path(args.model_path)
    if not Path(model_path).exists():
        raise SystemExit(f"HARD FAIL: model path does not exist: {model_path}")
    tokenizer_path = resolve_tokenizer_path(None) or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_path = (
        args.train.resolve()
        if args.train is not None
        else repo_root / "data" / "processed" / "swe_gym" / "train.parquet"
    )
    if not train_path.is_file():
        raise SystemExit(f"HARD FAIL: missing M1E train parquet {train_path}")
    oracle_path = (
        args.oracle.resolve()
        if args.oracle is not None
        else oracle_parquet_path(repo_root)
    )
    oracle_index = load_evaluator_oracle(oracle_path)

    data_root = args.data_root
    repos_root = (
        args.repos_root.expanduser()
        if args.repos_root is not None
        else swe_gym_repos_root(data_root)
    )
    env = RepoEnvironment(
        repos_root=repos_root,
        snapshots_root=(
            args.snapshots_root.expanduser() if args.snapshots_root is not None else None
        ),
        data_root=data_root,
    )
    cache_dir = repo_root / "outputs" / "smoke" / "rlhf_dataset_cache" / "m3a_gpu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(train_path, tokenizer, cache_dir)
    d1_item = find_dataset_row(dataset, D1_INSTANCE_ID)
    d1_extra = _as_mapping(d1_item.get("extra_info"))
    d1_task = TaskRef.from_extra_info(d1_extra)
    ensure_snapshot(d1_task, env)

    hidden_item = dict(d1_item)
    hidden_item["extra_info"] = with_budget(
        d1_extra, visible=False, limit=SMOKE_OBS_TOKENS_LIMIT
    )
    visible_item = dict(d1_item)
    visible_item["extra_info"] = with_budget(
        d1_extra, visible=True, limit=SMOKE_OBS_TOKENS_LIMIT
    )

    import ray
    import torch  # noqa: F401

    config = build_config(
        model_path,
        n_gpus=args.n_gpus,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
    )
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(AGENT_LOOP_CONFIG)
    data_root_env = str(bcrl_data_root(data_root))
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "BCRL_DATA_ROOT": data_root_env,
            }
        }
    )
    provenance = collect_run_provenance(
        repo_root,
        agent_loop_config=AGENT_LOOP_CONFIG,
        model_path=model_path,
        tokenizer_name_or_path=getattr(tokenizer, "name_or_path", None),
    )
    provenance["verl_runtime"] = get_verl_info()
    errors: list[str] = []
    episodes: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    try:
        manager = init_agent_loop_manager(config)
        batch = build_batch([hidden_item, visible_item], validate=True)
        result = manager.generate_sequences(prompts=batch)
        if len(result) != 2:
            raise SystemExit(f"gpu: got {len(result)} outputs, expected 2")
        labels = ("hidden", "visible")
        for index, label in enumerate(labels):
            extra_keys = result.non_tensor_batch
            segments = list(extra_keys.get("segments", [None] * 2)[index] or [])
            prompt_ids = list(extra_keys.get("unpadded_prompt_ids", [None] * 2)[index] or [])
            row_errors = verify_padded_sample(
                prompt_width=PROMPT_LENGTH,
                response_width=RESPONSE_LENGTH,
                prompts_row=result.batch["prompts"][index],
                responses_row=result.batch["responses"][index],
                response_mask_row=result.batch["response_mask"][index],
                attention_mask_row=result.batch["attention_mask"][index],
                unpadded_prompt_ids=prompt_ids,
                segments=segments,
            )
            termination = extra_keys.get("termination", [None] * 2)[index]
            if termination not in {"finish", "max_turns", "response_length", "budget_exhausted"}:
                row_errors.append(f"{label}: unexpected termination {termination!r}")
            obs_used = extra_keys.get("obs_tokens_used", [None] * 2)[index]
            repo_obs = extra_keys.get("tool_observation_token_count", [None] * 2)[index]
            obs_total = sum(
                len(item["token_ids"]) for item in segments if item["kind"] == "observation"
            )
            if obs_used != repo_obs:
                row_errors.append(
                    f"{label}: obs_tokens_used {obs_used} != repo obs {repo_obs}"
                )
            if extra_keys.get("total_env_tokens", [None] * 2)[index] not in {None, obs_total}:
                if extra_keys.get("total_env_tokens", [None] * 2)[index] != obs_total:
                    row_errors.append(
                        f"{label}: total_env_tokens != inserted {obs_total}"
                    )
            visible_flag = extra_keys.get("budget_visible", [None] * 2)[index]
            obs_segments = [item for item in segments if item["kind"] == "observation"]
            if obs_segments:
                decoded = tokenizer.decode(
                    obs_segments[0]["token_ids"], skip_special_tokens=True
                )
                if label == "hidden" and BUDGET_OBS_VERSION in decoded:
                    row_errors.append("hidden gpu obs contains budget envelope")
                if label == "visible" and BUDGET_OBS_VERSION not in decoded:
                    row_errors.append("visible gpu obs missing budget envelope")
            fake_extra = {
                key: extra_keys[key][index]
                for key in extra_keys
                if key
                not in {
                    "raw_prompt",
                    "extra_info",
                    "agent_name",
                    "index",
                }
            }
            # Flatten may not include every extra_fields key if missing on a sibling;
            # reconstruct from known names.
            for key in (
                "instance_id",
                "repo",
                "base_commit",
                "split",
                "final_submission",
                "termination",
                "segments",
                "events",
                "prompt_token_count",
                "policy_token_count",
                "observation_token_count",
                "tool_observation_token_count",
                "repo_observation_tokens",
                "budget_metadata_tokens",
                "total_env_tokens",
                "obs_tokens_used",
                "obs_tokens_limit",
                "obs_tokens_remaining",
                "budget_accounting_version",
                "budget_visible",
                "budget_exhausted",
                "sampling_params",
                "sampling_seed",
                "max_turns",
                "max_new_tokens_per_turn",
                "model_name_or_path",
                "trace_role",
                "unpadded_prompt_ids",
            ):
                payload = extra_keys.get(key)
                if payload is not None:
                    fake_extra[key] = payload[index]
            record = score_output_from_extra(
                fake_extra, oracle_index, provenance=provenance
            )
            record["condition_label"] = label
            if tokenizer is not None and segments:
                record["turn_boundary"] = inspect_turn_boundary(
                    tokenizer, prompt_ids=prompt_ids, segments=segments
                )
                record["segment_decomposition"] = segment_decomposition(segments)
            episodes.append(record)
            samples.append(
                {
                    "label": label,
                    "termination": termination,
                    "obs_tokens_used": obs_used,
                    "budget_visible": visible_flag,
                    "file_f1": (record.get("localization") or {}).get("file_f1"),
                    "symbol_status": (record.get("localization") or {}).get("symbol_status"),
                    "errors": row_errors,
                    "ok": not row_errors,
                    "segment_decomposition": record.get("segment_decomposition"),
                }
            )
            errors.extend(row_errors)
    finally:
        import ray as _ray

        _ray.shutdown()

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "model": model_path,
        "envelope": {
            "prompt_length": PROMPT_LENGTH,
            "response_length": RESPONSE_LENGTH,
            "max_model_len": MAX_MODEL_LEN,
            "obs_tokens_limit": SMOKE_OBS_TOKENS_LIMIT,
        },
        "samples": samples,
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
        "provenance": provenance,
    }


def score_output_from_extra(
    extra: Mapping[str, Any],
    oracle_index,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    instance_id = str(extra.get("instance_id") or "")
    loc = None
    if instance_id and instance_id in oracle_index:
        loc = evaluate_episode(
            termination=extra.get("termination"),
            submission=extra.get("final_submission"),
            oracle=oracle_index.get(instance_id),
        ).as_dict()
    record = build_episode_record(extra, localization=loc, provenance=provenance)
    record["trace_note"] = TRACE_NOTE
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "trace_note": TRACE_NOTE,
        "date": datetime.now(timezone.utc).isoformat(),
        "agent_loop_config": str(AGENT_LOOP_CONFIG),
        "smoke_obs_tokens_limit": SMOKE_OBS_TOKENS_LIMIT,
        "modes": {},
    }
    status = "PASS"
    all_episodes: list[dict[str, Any]] = []
    try:
        if args.mode in {"cpu", "both"}:
            cpu = run_cpu(args)
            report["modes"]["cpu"] = {
                k: v for k, v in cpu.items() if k != "episodes"
            }
            write_jsonl(output_dir / "m3a_cpu_episodes.jsonl", cpu["episodes"])
            all_episodes.extend(cpu["episodes"])
            if cpu["status"] != "PASS":
                status = "FAIL"
                print("HARD FAIL CPU:", file=sys.stderr)
                for err in cpu["errors"]:
                    print(f"  - {err}", file=sys.stderr)
        if args.mode in {"gpu", "both"} and status == "PASS":
            gpu = run_gpu(args)
            report["modes"]["gpu"] = {
                k: v for k, v in gpu.items() if k != "episodes"
            }
            traj_root = bcrl_data_root(args.data_root) / "trajectories" / "m3a"
            write_jsonl(traj_root / f"m3a_gpu_{stamp}.jsonl", gpu["episodes"])
            write_jsonl(output_dir / "m3a_gpu_episodes.jsonl", gpu["episodes"])
            all_episodes.extend(gpu["episodes"])
            if gpu["status"] != "PASS":
                status = "FAIL"
                print("HARD FAIL GPU:", file=sys.stderr)
                for err in gpu["errors"]:
                    print(f"  - {err}", file=sys.stderr)
        elif args.mode == "both" and status != "PASS":
            print("[m3a] skipping GPU because CPU failed", file=sys.stderr)
    except Exception as exc:
        status = "FAIL"
        report["exception"] = repr(exc)
        traceback.print_exc()
        print(f"HARD FAIL: {exc}", file=sys.stderr)

    report["status"] = status
    if all_episodes:
        report["summary"] = summarize_episodes(all_episodes)
        write_jsonl(output_dir / "m3a_episodes.jsonl", all_episodes)
        write_json(output_dir / "m3a_eval_summary.json", report["summary"])
    write_json(output_dir / "m3a_smoke_report.json", report)
    print(f"\nreport: {output_dir / 'm3a_smoke_report.json'}")
    print(f"STATUS: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
