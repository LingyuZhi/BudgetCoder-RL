# BudgetCoder-RL

BudgetCoder-RL is an Agentic RL research prototype for **budget-aware repository exploration**. A coding agent receives a real software issue, a repository snapshot, a small set of exploration tools, and a **hard context budget**. It must search and read the repo over multiple turns, then submit the code locations it believes are relevant to the issue.

Stage 1 isolates **localization** from patch generation and test execution. The question is whether reinforcement learning can improve how a coding agent spends a limited exploration budget, not whether it can fully repair the bug.

## Status

`prototype / experiment`

This repository is an early Stage 1 slice. It is **not**:

- a complete SWE-bench repair agent
- a patch / test / debug loop
- an SFT or multi-agent scaffold
- a paper-ready or SOTA codebase

Training, evaluation, and installable package metadata are not locked yet.

## Stage 1 stack

Pinned starting point:

- Framework: [veRL](https://github.com/verl-project/verl) `release/v0.8.0` (sibling clone, not vendored)
- Algorithm: GRPO
- Parameter update: LoRA
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Data: SWE-Gym-derived localization tasks
- Budget: hard cumulative tool-observation token limit, plus a max-turn safety cap
- Reward: deterministic file/symbol F1 against hidden labels (no learned reward model)

Tool and environment tokens are observations. They must not contribute to the policy loss.

## Repository layout

```text
src/budget_coder_rl/   # core Python package
configs/               # Stage 1 and experiment configs
scripts/               # setup / data / smoke / train / eval entrypoints
data/                  # small manifests and fixtures only
tests/
```

Raw datasets, repository snapshots, model weights, checkpoints, and full trajectories do **not** belong in Git. Runtime data should live outside the repository, under an external root pointed to by `BCRL_DATA_ROOT`.

veRL is a pinned upstream sibling dependency, not part of this tree:

```text
<WORKSPACE_ROOT>/
├── budget-coder-rl/
└── deps/
    └── verl/
```

## Setup

The Stage 1 environment is not locked yet. After the first working smoke, this README will record the exact veRL commit and install steps.

Until then:

1. Clone this repository.
2. Clone veRL `release/v0.8.0` as a sibling under `../deps/verl`.
3. Do not vendor veRL into this tree or silently upgrade veRL / vLLM / PyTorch / Transformers.

## License

MIT. See [LICENSE](LICENSE).
