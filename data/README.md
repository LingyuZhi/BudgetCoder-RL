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
- `interim/`: regenerable per-instance audits / oracles (M1B JSONL, M1C-A JSONL)
- `processed/`: Stage 1 JSONL (when it exists)

Repository clones / worktrees, Docker / executable images, model weights,
checkpoints, full trajectories, and Hub caches live **outside** `data/`, under
`$BCRL_DATA_ROOT` (on this server: `~/my_data/budget-coder-rl`).

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

Do **not** use `SWE-Gym/SWE-Gym-Lite` or `SWE-Gym/SWE-Gym-Raw`.

```bash
python scripts/data/download_swe_gym.py
python scripts/data/inspect_swe_gym.py
python scripts/data/audit_swe_gym.py
python scripts/data/extract_swe_gym_oracle.py
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

Coordinates are split on purpose:

- `removed_source_lines` = base-commit (source) line numbers
- `added_target_lines` = gold-patched (target) line numbers

Test-like gold paths (`tests/`, `test/`, `test_*.py`, `*_test.py`) are
observational statistics only and are not used as a drop filter. Parse
failures are reported with `instance_id` + parser error; there is no regex
fallback.
