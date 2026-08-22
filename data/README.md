# Data

This directory is the **dataset lifecycle** root: tabular sources, processed
labels, audits, and version pins. Acquire and transform data with
`scripts/data/`; do not copy files by hand.

Git tracks only small, versionable files:

- `manifests/`: source / revision / sha256 / split / filter metadata
- `fixtures/`: tiny samples for tests and smoke runs
- `stats/`: counts and summaries (no full gold patches)

Gitignored working-tree data (see `.gitignore` → `data/raw/`):

- `raw/`: official tables such as SWE-Gym parquet
- `processed/`: Stage 1 JSONL (when it exists)
- `audits/`: large audit dumps, if any

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
| Tiny inspect fixture | `fixtures/swe_gym_tiny.json` |

Do **not** use `SWE-Gym/SWE-Gym-Lite` or `SWE-Gym/SWE-Gym-Raw`.

```bash
python scripts/data/download_swe_gym.py
python scripts/data/inspect_swe_gym.py
```

Download talks to Hugging Face (or `HF_ENDPOINT` if set). It does not install
packages. Schema / row-count / repo-count mismatches fail the inspect script.

Stage 1 training data is derived later from these raw rows. Filtering, splits,
and gold-patch oracle extraction are **not** done in M1A.
