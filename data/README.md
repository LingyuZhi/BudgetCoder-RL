# Data

This directory is the **dataset lifecycle** root: tabular sources, processed
labels, audits, and version pins. Acquire and transform data with
`scripts/data/`; do not copy files by hand.

Git tracks only small, versionable files:

- `manifests/`: source / revision / sha256 / split / filter metadata
- `fixtures/`: tiny samples for tests and smoke runs
- `stats/`: counts and summaries (no full gold patches)

Gitignored working-tree data (see `.gitignore`):

- `raw/`: official tables such as SWE-Gym parquet
- `interim/`: regenerable per-instance audits / oracles / features (M1B / M1C / M1D-A JSONL)
- `processed/`: Stage 1 veRL parquet (M1E train/dev + evaluator oracle; gitignored)

Repository clones / worktrees, Docker / executable images, model weights,
checkpoints, full trajectories, and Hub caches live **outside** `data/`, under
`$BCRL_DATA_ROOT` (on this server: `~/my_data/budget-coder-rl`). M1 bare
object stores are `$BCRL_DATA_ROOT/repos/swe_gym/`. M2A read-only snapshots
are `$BCRL_DATA_ROOT/repos/swe_gym_snapshots/` (not under `data/`).

## Official SWE-Gym parquet

The Hugging Face `SWE-Gym/SWE-Gym` train split (~42MB) is stored repo-local at
`data/raw/swe_gym/` and is gitignored. This is the Stage 1 convention, not a
one-off exception.

| Item | Location |
| --- | --- |
| Raw parquet + `SOURCE.json` + `profile.json` | `data/raw/swe_gym/` (not in Git) |
| Pinned source / revision / sha256 | `manifests/swe_gym_raw.json` (in Git) |
| Field visibility / leakage contract | `manifests/swe_gym_field_policy.json` (in Git) |
| M1B audit summary | `manifests/swe_gym_m1b_audit_summary.json` (in Git) |
| M1B per-instance flags | `interim/swe_gym/m1b_audit.jsonl` (not in Git; regenerable) |
| M1C-A oracle summary | `manifests/swe_gym_m1c_oracle_summary.json` (in Git) |
| M1C-A per-instance oracles | `interim/swe_gym/m1c_oracle.jsonl` (not in Git; regenerable) |
| Tiny inspect fixture | `fixtures/swe_gym_tiny.json` |
| Tiny M1B audit fixture | `fixtures/swe_gym_m1b_audit.json` |
| Tiny M1C-A oracle fixture | `fixtures/swe_gym_m1c_oracle.json` |
| M1C-B symbol oracle summary | `manifests/swe_gym_m1c_symbol_summary.json` (in Git) |
| M1C-B per-instance symbol oracles | `interim/swe_gym/m1c_symbol_oracle.jsonl` (not in Git; regenerable) |
| M1C-B repo source pins | `manifests/swe_gym_repo_sources.json` (in Git; no host-absolute paths) |
| Tiny M1C-B symbol fixture | `fixtures/swe_gym_m1c_symbol_oracle.json` |
| M1D-A feature summary | `manifests/swe_gym_m1d_feature_summary.json` (in Git) |
| M1D-A per-instance features | `interim/swe_gym/m1d_features.jsonl` (not in Git; regenerable) |
| Tiny M1D-A feature fixture | `fixtures/swe_gym_m1d_features.json` |
| M1D-B eligibility policy | `manifests/swe_gym_m1d_policy.json` (in Git) |
| M1D-B train/dev split | `manifests/swe_gym_m1d_split.json` (in Git; assignments, no patches) |
| M1D-B split audit summary | `manifests/swe_gym_m1d_split_summary.json` (in Git) |
| M1E schema contract | `manifests/swe_gym_m1e_schema.json` (in Git) |
| M1E dataset manifest | `manifests/swe_gym_m1e_dataset_manifest.json` (in Git; checksums, no host paths) |
| M1E policy train/dev parquet | `processed/swe_gym/train.parquet`, `dev.parquet` (not in Git) |
| M1E evaluator oracle sidecar | `processed/swe_gym/evaluator_oracle.parquet` (not in Git) |
| M2C runtime prompt-length summary | `stats/swe_gym_m2c_prompt_length.json` (in Git; no issue text) |

Do **not** use `SWE-Gym/SWE-Gym-Lite` or `SWE-Gym/SWE-Gym-Raw`.

```bash
python scripts/data/download_swe_gym.py
python scripts/data/inspect_swe_gym.py
python scripts/data/audit_swe_gym.py
python scripts/data/extract_swe_gym_oracle.py
python scripts/data/prepare_swe_gym_repos.py
python scripts/data/extract_swe_gym_symbol_oracle.py
python scripts/data/extract_swe_gym_m1d_features.py
python scripts/data/split_swe_gym_m1d.py
python scripts/data/materialize_swe_gym_m1e.py
python scripts/smoke/smoke_rlhf_dataset.py
```

Download talks to Hugging Face (or `HF_ENDPOINT` if set). It does not install
packages. Schema / row-count / repo-count / parquet-identity mismatches fail
the inspect and audit scripts.

### M1B field visibility (Stage 1)

The 11 official columns are partitioned in `manifests/swe_gym_field_policy.json`:

- **Agent task input:** `problem_statement`
- **Runtime / identity metadata:** `instance_id`, `repo`, `base_commit`, `version`, `created_at`
- **Privileged / policy-hidden:** `hints_text`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`

Stage 1 does **not** expose `hints_text` to the agent. Gold patch, test patch,
F2P, and P2P are privileged evaluator information. Future rollout should forbid
external web/GitHub lookup; M1B records that contract and does not implement a
network sandbox.

M1B audits all 2438 rows in place. It does **not** drop instances, rewrite the
raw parquet, extract oracle localization labels, or create train/val/test
splits. Identity / schema / cardinality errors are hard fails.

Audit labels are three disjoint classes:

- **Heuristic structural suspicion** (e.g. unbalanced `[]` / `()`): count-mismatch
  heuristics only. `n_heuristic_suspicion_rows` is **not** a count of confirmed
  malformed rows. Pytest parametrized IDs may contain brackets, parentheses,
  and special characters.
- **Dataset / correlation property** (e.g. empty `PASS_TO_PASS`, exact duplicate
  `problem_statement`): statistics, not a drop filter.
- **Observational signal** (e.g. nonempty `hints_text`, F2P text appearing in
  the issue): not leakage verdicts.

### M1C-A gold-patch oracle (diff-level only)

M1C-A parses every gold `patch` with `unidiff.PatchSet` into file / hunk /
line-coordinate oracles. It does **not**:

- read `test_patch`, `hints_text`, `FAIL_TO_PASS`, or `PASS_TO_PASS` into
  `gold_edit_files`
- extract functions / classes / AST symbols (that is M1C-B)
- drop instances, rewrite the raw parquet, or create train/val/test splits
- emit a single ambiguous `oracle_lines` field

Two derived file views:

- **`gold_edit_files`**: gold patch structure. May include added target
  paths that do not exist at `base_commit`. This is **not** the Stage-1
  retrieval reward target.
- **`base_changed_files`**: source-side paths present at `base_commit`
  (`source_path` is not `/dev/null`). Pure added files are excluded.
  This is the candidate oracle for future base-repository localization
  reward.

Whether to filter zero-base-visible or added-heavy instances is left to
M1D. Path differences are labeled `path_changed` (normalized
`source_path != target_path`), not a confirmed Git rename.

### M1C-B base-repository symbol oracle

M1C-B maps gold-patch change sites onto function/class symbols that
exist at `base_commit`. It does **not**:

- create one checkout per instance (one bare Git object store per unique repo)
- download files from GitHub HTTP APIs
- parse non-Python files with tree-sitter / Cython
- drop instances, rewrite the raw parquet, or create train/val/test splits
- put symbol oracles into `agent_task_view`

Repository object stores live at `$BCRL_DATA_ROOT/repos/swe_gym/`, not
under `data/raw/`. Preparation may use the network and fetches
parquet-derived `base_commit` SHAs into one bare repo per unique
`repo`. Extraction is offline (`git rev-parse` / `git cat-file` only).
Missing commits or blobs are reported and the instance is retained.

Coordinates are split on purpose:

- `removed_source_lines` = base-commit (source) line numbers
- `added_target_lines` = gold-patched (target) line numbers

Test-like gold paths (`tests/`, `test/`, `test_*.py`, `*_test.py`) are
observational statistics only and are not used as a drop filter. Parse
failures are reported with `instance_id` + parser error; there is no regex
fallback.

### M1D-A eligibility / structural-difficulty audit

M1D-A joins the frozen parquet, M1C-A/B JSONL, and local bare mirrors into
a 2438-row feature table. It does **not**:

- drop or filter instances, or write `keep` / `drop`
- create train/dev/test splits
- synthesize a composite easy/medium/hard score
- re-parse gold patches or re-run AST
- checkout worktrees, use the network, or depend on CodeScout
- put derived features into `agent_task_view`

Technical validity reasons are recorded in place. Stage-1 localization
counts use `base_changed_files`, not added gold targets. Issue mentions of
gold paths/symbols are difficulty/hint features, not leakage. Correlation
groups are connected components over the same `(repo, base_commit)` or the
same normalized `problem_statement`; they are not a split. Filtering and
split decisions are **M1D-B**.

### M1D-B eligibility policy and train/dev split

M1D-B freezes Stage-1 eligibility as **keep-all**: 2438/2438 instances are
eligible. Zero-symbol, non-Python, non-code-only, large-patch, hint, M1B
audit-flag, and structural-difficulty attributes are **not** drop criteria.
Difficulty features remain analysis/sampling metadata. Curriculum is not
enabled at the dataset stage. Reward is not implemented here.

SWE-Gym is split into **train/dev only**. There is no internal `test` split;
the external final test does not come from SWE-Gym. The atomic unit is the
frozen M1D-A `correlation_group_id`. A group is never split across train and
dev. Cross-repo correlation groups are a hard fail (expected 0).

Repo-level ~10% dev seats use integer largest-remainder / Hamilton
allocation. Within a repo, whole groups are selected with deterministic 0/1
subset-sum DP. SHA256(`split_version|seed|correlation_group_id`) breaks ties.
`split_version = swe-gym-group-repo-v1`, `seed = 42`; there is no seed search.

The split manifest stores one assignment per instance (`instance_id`, `repo`,
`correlation_group_id`, `split`) and does **not** copy `patch`, `test_patch`,
oracle symbol details, or `problem_statement`. Secondary feature distributions
are audited only. M1D-B does **not** materialize veRL parquet (that is M1E).

### M1E veRL-ready parquet and evaluator sidecar

M1E consumes the frozen M1D split and M1C oracles. It does **not** re-run
split, AST, or unidiff, and does not filter instances.

Policy/runtime files `processed/swe_gym/train.parquet` and `dev.parquet`
contain only `data_source`, `prompt`, `reward_model`, and `extra_info`.
`prompt` is a single user message whose content is the raw
`problem_statement`. `reward_model.ground_truth` is the opaque
`instance_id`. Privileged gold labels are not stored, including inside
`extra_info`. `agent_name` is omitted so AgentLoop can use
`default_agent_loop`.

Evaluator-only `processed/swe_gym/evaluator_oracle.parquet` is physically
separate and holds `base_changed_files`, canonical `oracle_symbols`
(`path` + `qualname`), and `symbol_applicable`. Zero-symbol rows are
kept (`symbol_applicable=False`). Future reward is not implemented here.

Pinned veRL `RLHFDataset` smoke must set `filter_overlong_prompts=false`;
the constructor default would drop long issues. Training config should
do the same. Do not upgrade veRL for M1E.

### M2A repository snapshots (runtime, not dataset)

M2A does **not** add tables under `data/`. Given M1E `extra_info`
(`instance_id`, `repo`, `base_commit`), `RepoEnvironment` materializes a
read-only exact-tree snapshot of that commit from the M1 bare store via
`ls-tree` + `cat-file` (not `git archive`: several SWE-Gym repos use
`export-ignore` / `export-subst`). Cache key is `(repo, 40-char SHA)`,
not `instance_id`. Snapshots contain no `.git` and never apply the gold
patch. Prepare is offline: missing repos/commits fail; there is no
fallback to `HEAD`.

```bash
python scripts/smoke/smoke_repo_workspace.py
```

### M2C runtime initial-prompt length

M1E audited issue-only tokens. M2C audits
`system/tool/protocol + issue` with the real Qwen tokenizer and the same
runtime prompt builder used by `RepoExplorationAgentLoop`. Frozen train/dev
are not re-split. Overlong prompts are reported, not truncated.

```bash
python scripts/data/audit_runtime_prompt_length.py
python scripts/smoke/smoke_repo_exploration_agent_loop.py
```
