# BudgetCoder-RL
<div align="center">
  <img src="assets/logo.png" alt="logo" width="35%">
</div>

BudgetCoder-RL 是一个面向 **budget-aware repository exploration** 的 Agentic RL 研究原型。Coding agent 会收到真实软件 issue、仓库快照、少量探索工具，以及一个 **硬性 context budget**。它需要在多轮交互中搜索并阅读仓库，最终提交其认为与该 issue 相关的代码位置。

Stage 1 将 **localization** 从 patch 生成与测试执行中独立出来。核心问题是：强化学习能否改善 coding agent 在有限探索预算下的花费方式，而不是它能否完整修复这个 bug。

## Stage 1 技术栈

当前固定起点：

- 框架：[veRL](https://github.com/verl-project/verl) `main` @ `60546ef2a7464a158cd170f58f852a62a4e552ba`（`0.8.0.dev0`，早于 `v0.8.0`；作为 sibling clone，不 vendor）。精确 pinned runtime 见 `configs/versions/environment.md`。
- 算法：GRPO
- 参数更新：LoRA
- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- 数据：由 SWE-Gym 派生的 localization 任务
- Budget：硬性累计 tool-observation token 上限，外加 max-turn 安全上限
- Reward：相对隐藏标签的确定性 file/symbol F1（不使用 learned reward model）

Tool 与 environment token 属于 observation，不得计入 policy loss。

## Quickstart

```bash
# 1. 准备 SWE-Gym localization 数据集（frozen split 仅用于 verify）
python scripts/data/prepare_swe_gym.py all

# 2. 训练 GRPO + LoRA（计算节点；pinned conda 环境）
python scripts/train/train_grpo.py
python scripts/train/train_grpo.py --dry-run

# 3. Held-out localization 评估（base 和/或 LoRA checkpoint × budgets）
python scripts/eval/evaluate_localization.py --phase all
python scripts/eval/evaluate_localization.py --dry-run
```

配置：

- 训练：`configs/training/grpo_qwen3_4b.json`
- 评估：`configs/evaluation/localization.json`
- Agent：`configs/agent/repo_exploration.yaml`

已冻结的科学 contract 与 hash：`docs/provenance.md`。

## 仓库结构

```text
src/budget_coder_rl/   # 核心 Python package
configs/               # training / evaluation / agent / 历史 provenance
scripts/               # data / train / eval / smoke 入口
data/                  # 数据集生命周期：manifests/fixtures 进 Git；raw/processed 被 gitignore
docs/                  # provenance pin（不是结果叙事）
tests/
```

表格类数据集（例如官方 SWE-Gym parquet）放在 `data/` 下，并被 gitignore。仓库快照、Docker / 可执行镜像、模型权重、checkpoint 以及完整 trajectory 不进入 Git 树，而是放在 `$BCRL_DATA_ROOT`。`data/raw/` 与 `data/processed/` 中的内容都不应提交。

veRL 是 pinned 的上游 sibling 依赖，不属于本仓库源码：

```text
<WORKSPACE_ROOT>/
├── budget-coder-rl/
└── deps/
    └── verl/
```

## 环境搭建

Runtime config 固定在 `configs/versions/environment.md`（veRL core =
upstream `main` @ `60546ef2`，Python 3.12 / torch 2.8.0+cu128 / vLLM 0.11.0 /
Ray 2.55.1 / Transformers 4.57.6）。

1. Clone 本仓库。
2. 将 pinned commit 的 veRL checkout 作为 sibling 放在 `../deps/verl`
   （以 editable 方式安装进 RL conda 环境）。
3. 安装本 package，且不要改动 RL 环境：

```bash
pip install --no-deps -e .
```

4. 不要把 veRL vendor 进本仓库，也不要静默升级 veRL / vLLM / PyTorch / Transformers。

Smoke 检查（GPU 节点，pinned conda 环境）：

```bash
python scripts/smoke/smoke_agent_loop.py --model-path <local-model-dir>
pytest tests/
```

## License

MIT。见 [LICENSE](LICENSE)。
