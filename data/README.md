# Data

Large datasets and runtime artifacts (repo snapshots, trajectories, model
weights, caches) live outside this Git repository. The runtime data root is
configured via the `BCRL_DATA_ROOT` environment variable.

This directory only holds small, versionable files:

- `manifests/`: dataset version / split metadata
- `fixtures/`: tiny samples for tests and smoke runs

Stage 1 data is derived from SWE-Gym localization tasks. Detailed data design
lives in the private design docs.
