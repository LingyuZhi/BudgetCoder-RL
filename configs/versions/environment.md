# Stage 1 validated runtime pin

| Component | Version |
| --------- | ------- |
| Python | 3.12.0 |
| PyTorch | 2.8.0+cu128 |
| vLLM | 0.11.0 |
| Ray | 2.55.1 |
| Transformers | 4.57.6 |
| veRL | 0.8.0.dev0 |
| veRL upstream core revision | `verl-project/verl@60546ef2a7464a158cd170f58f852a62a4e552ba` |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |

Framework upgrades should rerun the M0 AgentLoop smoke before being used for
Stage 1 experiments.
