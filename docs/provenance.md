# Provenance pins

Scientific knobs are frozen. This file records where those contracts live.
It is not a results narrative and does not retrain or re-evaluate.

## Split

- Version: `swe-gym-group-repo-v1`
- Manifest: [`data/manifests/swe_gym_m1d_split.json`](../data/manifests/swe_gym_m1d_split.json)
- `verify-split` checks this file; it does not re-cut the split.

## Runtime

Pinned environment: [`configs/versions/environment.md`](../configs/versions/environment.md)

- veRL `0.8.0.dev0` / fork `LingyuZhi/rtrl-verl` @ `8481f9f9880d0f46a75b3db0329d3de8abad3d81`
- upstream core `verl-project/verl` @ `60546ef2`
- model: `Qwen/Qwen3-4B-Instruct-2507`

## Training contract

Canonical user config: `configs/training/grpo_qwen3_4b.json` (paths use `$BCRL_DATA_ROOT`).

Frozen scaled-training contract (byte-identical historical snapshot):

- `configs/historical/stage1_m5_scaled.json`
- sha256 `672f064399a1d42062dd4360b4bd22b30f101988f3325e29338781e934e9ae8a`
- overlay lock: `configs/historical/stage1_m5_scaled_e017.lock.json`

Related historical files that must remain byte-identical:

- `configs/historical/stage1_m3c_freeze.json`
- `configs/historical/stage1_m5_main.json`
- `configs/historical/stage1_m5_e014_runtime.json`
- `configs/historical/stage1_canonical_execution_envelope.json`
- `data/manifests/m5_scaled_train_candidates.json`
- `data/manifests/m3c_train_candidates.json`

Internal JSON still contains original `configs/experiments/...` path strings. Those strings are snapshots, not live lookup paths.

## Evaluation contract

Canonical user config: `configs/evaluation/localization.json`.

- parent freeze: `configs/historical/stage1_m6_eval.json`
  sha256 `bdaabd34520d86a7514fde485dd2037d5920e6ebd6945fe31bc90a1c701b7c76`
- overlay: `configs/historical/stage1_m6_e018.json`
  sha256 `8749e58fbe88ce2560e37f4a32861e4a1c8ffc739136c2389dc582c055bb15f9`
- 244 held-out tasks, budgets `{2048, 4096, 8192}`

## Checkpoints

Historical artifact directory (already on disk; do not rename):

```text
$BCRL_DATA_ROOT/checkpoints/stage1_m5_scaled_e017/global_step_275
```

New training writes to `$BCRL_DATA_ROOT/checkpoints/grpo_qwen3_4b` unless the config is overridden. Eval prefers an existing canonical directory, then the historical path above.

## Known limitation

Sibling expansion under veRL `DataProto.repeat` / `np.repeat` aliases object-array `extra_info` rows. Locked by `tests/test_rollout_grouping.py`. A prior forensic replay did not reproduce the historical training pathology; that conclusion is not a license to change grouping semantics.
