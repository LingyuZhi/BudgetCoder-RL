#!/usr/bin/env python
"""M0 smoke: run DummyTwoTurnAgentLoop through the real veRL runtime.

Path exercised (no shortcuts, no hand-instantiated agent loop):

    ray -> ActorRolloutRefWorker (FSDP) -> LLMServerManager (vLLM async server)
        -> AgentLoopManager -> AgentLoopWorker -> DummyTwoTurnAgentLoop
        -> generate_sequences() -> padded trainer batch

Verifies (M0-E):
- exact rollout token ids are preserved end-to-end (full per-token equality
  between the trainer-side `responses` tensor and the per-segment ids recorded
  inside the agent loop);
- response_mask segmentation: assistant spans = 1, environment span = 0;
- len(response_ids) == len(response_mask).

The runtime bootstrap below is replicated from veRL's own agent loop test
harness (tests/experimental/agent_loop/agent_utils.py) using only public veRL
APIs; the reward loop is intentionally skipped (M0 has no reward).

Usage (inside the pinned RL conda env, on a GPU node):

    python scripts/smoke/smoke_agent_loop.py [--model-path PATH]

Model path resolution: --model-path > $BCRL_MODEL_PATH >
$BCRL_DATA_ROOT/models/Qwen3-4B-Instruct-2507.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_LOOP_CONFIG = REPO_ROOT / "configs" / "agent" / "dummy_two_turn.yaml"
OUTPUT_DIR = REPO_ROOT / "outputs" / "smoke"

RAW_PROMPTS = [
    [{"role": "user", "content": "What is 17 * 23? Think briefly, then answer."}],
    [{"role": "user", "content": "Name the capital of Australia and one fact about it."}],
]


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


def get_verl_info() -> dict:
    import verl

    source_root = Path(verl.__file__).resolve().parents[1]
    info = {
        "version": getattr(verl, "__version__", "unknown"),
        "source_root": str(source_root),
        "commit": "unknown",
    }
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass
    return info


def build_config(model_path: str):
    import verl
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(verl.__file__).resolve().parent / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                # bf16 actor: M0 runs no training step; fp32 4B params (16GB) plus
                # a full gathered copy at weight sync OOMs a single A100-40GB.
                "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
            ],
        )

    config.actor_rollout_ref.model.path = model_path
    rollout = config.actor_rollout_ref.rollout
    rollout.name = "vllm"
    rollout.mode = "async"
    rollout.enforce_eager = True
    rollout.prompt_length = 1024
    rollout.response_length = 1024
    # M0 smoke-only cap, NOT a Stage 1 training default: without an explicit
    # cap, vLLM sizes the KV cache for the model's native max context (262k for
    # Qwen3-4B-Instruct-2507) and cannot fit next to the colocated FSDP actor
    # on one 40GB GPU. Re-derive from the real context budget for training.
    rollout.max_model_len = 2048
    rollout.tensor_model_parallel_size = 1
    rollout.n = 1
    rollout.skip_tokenizer_init = True
    # Gather weights layer-by-layer during rollout weight sync instead of
    # materializing a second full copy of the model on GPU.
    rollout.layered_summon = True
    rollout.checkpoint_engine.update_weights_bucket_megabytes = 512
    rollout.agent.num_workers = 1
    rollout.agent.agent_loop_config_path = str(AGENT_LOOP_CONFIG)
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = 1
    return config


def init_agent_loop_manager(config):
    """Real veRL runtime bootstrap (public APIs only, reward loop skipped)."""
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

    assert config.actor_rollout_ref.rollout.mode == "async", "M0 smoke requires async rollout"

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

    llm_server_manager = LLMServerManager.create(config=config, worker_group=actor_rollout_wg)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=None,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()
    return agent_loop_manager


def verify_sample(i: int, result, prompt_width: int, response_width: int) -> dict:
    """Assert M0-E contract for one sample; return its stats."""
    responses = result.batch["responses"][i]
    response_mask = result.batch["response_mask"][i]
    attention_mask = result.batch["attention_mask"][i]

    segments = result.non_tensor_batch["dummy_segments"][i]
    turn1_ids = list(segments["turn1_ids"])
    obs_ids = list(segments["obs_ids"])
    turn2_ids = list(segments["turn2_ids"])
    n1, n_obs, n2 = len(turn1_ids), len(obs_ids), len(turn2_ids)
    valid_len = n1 + n_obs + n2

    # length contract
    assert responses.shape == response_mask.shape, (
        f"sample {i}: responses {tuple(responses.shape)} != response_mask "
        f"{tuple(response_mask.shape)}"
    )
    attn_response_len = int(attention_mask[prompt_width:].sum().item())
    assert attn_response_len == valid_len, (
        f"sample {i}: attention-mask response length {attn_response_len} != "
        f"segment total {valid_len}"
    )

    # exact token ids: full per-token equality, no sampling
    got = responses[:valid_len].tolist()
    expected = turn1_ids + obs_ids + turn2_ids
    assert got == expected, f"sample {i}: trainer-side response ids != exact rollout ids"

    # mask segmentation
    mask = response_mask.tolist()
    assert mask[:n1] == [1] * n1, f"sample {i}: turn-1 span must be all 1"
    assert mask[n1 : n1 + n_obs] == [0] * n_obs, f"sample {i}: env span must be all 0"
    assert mask[n1 + n_obs : valid_len] == [1] * n2, f"sample {i}: turn-2 span must be all 1"
    assert mask[valid_len:] == [0] * (response_width - valid_len), (
        f"sample {i}: padding span must be all 0"
    )

    prompt_token_count = int(attention_mask[:prompt_width].sum().item())
    return {
        "sample": i,
        "prompt_token_count": prompt_token_count,
        "turn1_token_count": n1,
        "env_observation_token_count": n_obs,
        "turn2_token_count": n2,
        "total_response_token_count": valid_len,
        "response_mask_length": len(mask),
        "num_mask_1": int(sum(mask)),
        "num_mask_0": int(len(mask) - sum(mask)),
        "num_turns": int(result.non_tensor_batch["__num_turns__"][i]),
        "segments": {"turn1_ids": turn1_ids, "obs_ids": obs_ids, "turn2_ids": turn2_ids},
    }


def render_trace(tokenizer, raw_prompt, stats) -> str:
    """Human-readable trace for debugging ONLY. Never used to rebuild token ids."""
    seg = stats["segments"]
    lines = [
        "### DEBUG TRACE (decoded for humans; token ids above are the source of truth)",
        "",
        "USER",
        raw_prompt[-1]["content"],
        "",
        "ASSISTANT (turn 1)",
        tokenizer.decode(seg["turn1_ids"]),
        "",
        "ENV (observation, mask=0)",
        tokenizer.decode(seg["obs_ids"]),
        "",
        "ASSISTANT (turn 2)",
        tokenizer.decode(seg["turn2_ids"]),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    assert Path(model_path).exists(), f"model path does not exist: {model_path}"

    import ray
    import torch  # noqa: F401  (fail fast if env broken)

    from verl.protocol import DataProto

    verl_info = get_verl_info()
    print(f"[smoke] model: {model_path}")
    print(f"[smoke] verl: {verl_info['version']} @ {verl_info['commit']}")
    print(f"[smoke] agent loop config: {AGENT_LOOP_CONFIG}")

    config = build_config(model_path)

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    agent_loop_manager = init_agent_loop_manager(config)

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(p) for p in RAW_PROMPTS], dtype=object),
            "agent_name": np.array(["dummy_two_turn"] * len(RAW_PROMPTS)),
        },
    )
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(RAW_PROMPTS)

    prompt_width = result.batch["prompts"].size(1)
    response_width = result.batch["responses"].size(1)

    all_stats = []
    failures = []
    for i in range(len(result)):
        try:
            all_stats.append(verify_sample(i, result, prompt_width, response_width))
        except AssertionError as e:
            failures.append(str(e))
            print(f"[smoke] FAIL: {e}")

    status = "PASS" if not failures else "FAIL"

    report = {
        "status": status,
        "date": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "verl": verl_info,
        "rollout": {
            "name": "vllm",
            "mode": "async",
            "prompt_length": int(config.actor_rollout_ref.rollout.prompt_length),
            "response_length": int(config.actor_rollout_ref.rollout.response_length),
        },
        "samples": [
            {k: v for k, v in s.items() if k != "segments"} for s in all_stats
        ],
        "failures": failures,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "m0_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n========== M0 SMOKE REPORT ==========")
    print(f"model:            {model_path}")
    print(f"verl:             {verl_info['version']} @ {verl_info['commit']}")
    for s in report["samples"]:
        print(
            f"sample {s['sample']}: prompt={s['prompt_token_count']} "
            f"turn1={s['turn1_token_count']} env_obs={s['env_observation_token_count']} "
            f"turn2={s['turn2_token_count']} total_response={s['total_response_token_count']} "
            f"mask_len={s['response_mask_length']} "
            f"mask1={s['num_mask_1']} mask0={s['num_mask_0']} turns={s['num_turns']}"
        )
    print(f"report:           {report_path}")
    print(f"STATUS:           {status}")

    # human-readable trace, for debugging only
    if all_stats:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        trace_text = "\n\n".join(
            render_trace(tokenizer, RAW_PROMPTS[s["sample"]], s) for s in all_stats
        )
        trace_path = OUTPUT_DIR / "m0_smoke_trace.txt"
        trace_path.write_text(trace_text)
        print(f"trace (debug):    {trace_path}")

    ray.shutdown()
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
