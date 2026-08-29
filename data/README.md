# Data

This directory is the **dataset lifecycle** root: tabular sources, processed
labels, audits, and version pins. Acquire and transform data with the
canonical orchestrator; do not copy files by hand.

```bash
python scripts/data/prepare_swe_gym.py download
python scripts/data/prepare_swe_gym.py inspect
python scripts/data/prepare_swe_gym.py audit
python scripts/data/prepare_swe_gym.py extract-oracles
python scripts/data/prepare_swe_gym.py prepare-repos
python scripts/data/prepare_swe_gym.py extract-symbol-oracle
python scripts/data/prepare_swe_gym.py extract-features
python scripts/data/prepare_swe_gym.py verify-split   # default: check frozen split, do not re-cut
python scripts/data/prepare_swe_gym.py materialize
python scripts/data/prepare_swe_gym.py all            # same order; split is verify-only
python scripts/smoke/smoke_rlhf_dataset.py
```

`verify-split` / `all` do **not** rewrite
`manifests/swe_gym_m1d_split.json`. Split version `swe-gym-group-repo-v1`.
See `docs/provenance.md`.

Git tracks only small, versionable files:

- `manifests/`: source / revision / sha256 / split / filter metadata
- `fixtures/`: tiny samples for tests and smoke runs
- `stats/`: counts and summaries (no full gold patches)

Gitignored working-tree data (see `.gitignore`):

- `raw/`: official tables such as SWE-Gym parquet
- `interim/`: regenerable per-instance audits / oracles / features
- `processed/`: veRL parquet (train/dev + evaluator oracle)

Repository clones / worktrees, Docker / executable images, model weights,
checkpoints, full trajectories, and Hub caches live **outside** `data/`,
under `$BCRL_DATA_ROOT`. Bare object stores:
`$BCRL_DATA_ROOT/repos/swe_gym/`. Read-only snapshots:
`$BCRL_DATA_ROOT/repos/swe_gym_snapshots/`.

## Official SWE-Gym parquet

The Hugging Face `SWE-Gym/SWE-Gym` train split (~42MB) is stored repo-local at
`data/raw/swe_gym/` and is gitignored. Do **not** use `SWE-Gym/SWE-Gym-Lite`
or `SWE-Gym/SWE-Gym-Raw`.

Download talks to Hugging Face (or `HF_ENDPOINT` if set). It does not install
packages. Schema / row-count / repo-count / parquet-identity mismatches fail
inspect and audit.

| Item | Location |
| --- | --- |
| Raw parquet + `SOURCE.json` + `profile.json` | `raw/swe_gym/` (not in Git) |
| Pinned source / revision / sha256 | `manifests/swe_gym_raw.json` |
| Field visibility / leakage contract | `manifests/swe_gym_field_policy.json` |
| Audit / oracle / feature summaries | `manifests/swe_gym_m1*.json` |
| Frozen train/dev split | `manifests/swe_gym_m1d_split.json` |
| veRL schema + dataset manifest | `manifests/swe_gym_m1e_schema.json`, `swe_gym_m1e_dataset_manifest.json` |
| Policy train/dev parquet | `processed/swe_gym/train.parquet`, `dev.parquet` (not in Git) |
| Evaluator oracle sidecar | `processed/swe_gym/evaluator_oracle.parquet` (not in Git) |
| Tiny fixtures | `fixtures/swe_gym_*.json` |

## Policy-visible vs evaluator-only

The 11 official columns are partitioned in `manifests/swe_gym_field_policy.json`:

- **Agent task input:** `problem_statement`
- **Runtime / identity metadata:** `instance_id`, `repo`, `base_commit`, `version`, `created_at`
- **Privileged / policy-hidden:** `hints_text`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`

Gold patch, test patch, F2P, P2P, and oracle symbols must not appear in
policy-visible prompts, tool output, or `extra_info`.

Policy parquet (`processed/swe_gym/train.parquet`, `dev.parquet`) contains
`data_source`, `prompt`, `reward_model`, and `extra_info` only.
`reward_model.ground_truth` is the opaque `instance_id`. The evaluator
sidecar is physically separate.

Pinned veRL `RLHFDataset` smoke must set `filter_overlong_prompts=false`.

## Repository snapshots (runtime, not dataset)

`RepoEnvironment` materializes a read-only exact-tree snapshot of
`(repo, 40-char SHA)` from the bare store via `ls-tree` + `cat-file`.
Snapshots contain no `.git` and never apply the gold patch.

```bash
python scripts/smoke/smoke_repo_workspace.py
python scripts/smoke/smoke_repo_exploration_agent_loop.py
```

## Appendix: frozen pipeline notes

Historical stage names for the files above: download/inspect, field-visibility
audit, gold-patch file oracle, base-commit symbol oracle, eligibility features,
keep-all group-repo split (`swe-gym-group-repo-v1`, seed 42), then veRL parquet
materialize. Those stages are now subcommands of `prepare_swe_gym.py`. Do not
re-cut the frozen split. Full scientific hashes: `docs/provenance.md`.
