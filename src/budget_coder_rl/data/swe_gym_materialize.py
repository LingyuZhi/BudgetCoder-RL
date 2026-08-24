"""SWE-Gym M1E: freeze veRL-ready policy parquet and evaluator oracle sidecar.

Consumes frozen M1D split membership and M1C oracles. Does not re-run split,
AST, or unidiff. Does not implement reward, Agent scaffold, or GRPO.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_N_REPOS,
    EXPECTED_N_ROWS,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    HF_REPO_ID,
    HF_REVISION,
    is_null,
    parquet_path,
    sha256_file,
    validate_schema_and_cardinality,
    verify_parquet_file,
    write_json,
)
from budget_coder_rl.data.swe_gym_fields import (
    POLICY_FORBIDDEN_DERIVED_FIELDS,
    committed_field_policy_errors,
    validate_policy_rows_leakage,
)
from budget_coder_rl.data.swe_gym_features import (
    FEATURE_SUMMARY_RELPATH,
    read_jsonl,
)
from budget_coder_rl.data.swe_gym_oracle import oracle_jsonl_path, oracle_summary_path
from budget_coder_rl.data.swe_gym_split import (
    POLICY_RELPATH,
    SPLIT_RELPATH,
)
from budget_coder_rl.data.swe_gym_symbol_oracle import (
    symbol_oracle_jsonl_path,
    symbol_oracle_summary_path,
)

SCHEMA_VERSION = "swe-gym-m1e-verl-v1"
DATA_SOURCE = "budget_coder_swe_gym_localization"
REWARD_STYLE = "rule"
ROW_ORDERING = "instance_id lexicographic (Python str sort)"
EXPECTED_TRAIN_ROWS = 2194
EXPECTED_DEV_ROWS = 244
PROCESSED_RELDIR = "data/processed/swe_gym"
TRAIN_PARQUET_RELPATH = f"{PROCESSED_RELDIR}/train.parquet"
DEV_PARQUET_RELPATH = f"{PROCESSED_RELDIR}/dev.parquet"
ORACLE_PARQUET_RELPATH = f"{PROCESSED_RELDIR}/evaluator_oracle.parquet"
SCHEMA_RELPATH = "data/manifests/swe_gym_m1e_schema.json"
MANIFEST_RELPATH = "data/manifests/swe_gym_m1e_dataset_manifest.json"
POLICY_COLUMNS: tuple[str, ...] = (
    "data_source",
    "prompt",
    "reward_model",
    "extra_info",
)
EXTRA_INFO_KEYS: tuple[str, ...] = (
    "index",
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "split",
)
SIDECAR_COLUMNS: tuple[str, ...] = (
    "instance_id",
    "split",
    "base_changed_files",
    "oracle_symbols",
    "symbol_applicable",
)
SYMBOL_IDENTITY_KEYS: tuple[str, ...] = ("path", "qualname")
ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "dev"})


class MaterializeInputError(ValueError):
    """Hard-fail: missing/duplicate/unexpected IDs or contract violation."""


def processed_dir(repo_root: Path) -> Path:
    return Path(repo_root) / PROCESSED_RELDIR


def train_parquet_path(repo_root: Path) -> Path:
    return Path(repo_root) / TRAIN_PARQUET_RELPATH


def dev_parquet_path(repo_root: Path) -> Path:
    return Path(repo_root) / DEV_PARQUET_RELPATH


def oracle_parquet_path(repo_root: Path) -> Path:
    return Path(repo_root) / ORACLE_PARQUET_RELPATH


def schema_path(repo_root: Path) -> Path:
    return Path(repo_root) / SCHEMA_RELPATH


def dataset_manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / MANIFEST_RELPATH


def manifest_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def canonical_row_json(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(to_jsonable(row), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def logical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_row_json(row))
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, MappingABC):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            converted = value.tolist()
        except (TypeError, ValueError):
            converted = None
        if converted is not None:
            return to_jsonable(converted)
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_jsonable(value.item())
        except (ValueError, TypeError, AttributeError):
            pass
    return str(value)


def index_records_by_id(
    records: Sequence[Mapping[str, Any]],
    *,
    source: str,
    id_key: str = "instance_id",
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in records:
        instance_id = str(item.get(id_key) or "").strip()
        if not instance_id:
            raise MaterializeInputError(f"{source}: empty {id_key}")
        if instance_id in indexed:
            raise MaterializeInputError(
                f"{source}: duplicate {id_key} {instance_id}"
            )
        indexed[instance_id] = item
    return indexed


def require_exact_id_set(
    actual: set[str],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return
    parts: list[str] = []
    if missing:
        preview = missing[:5]
        parts.append(f"missing {len(missing)} {preview}")
    if unexpected:
        preview = unexpected[:5]
        parts.append(f"unexpected {len(unexpected)} {preview}")
    raise MaterializeInputError(f"{source}: {'; '.join(parts)}")


def split_assignments(
    split_manifest: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, Mapping[str, Any]]]:
    assignments = split_manifest.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise MaterializeInputError("split manifest has no assignments")
    by_id = index_records_by_id(assignments, source="m1d_split.assignments")
    train_ids = sorted(
        instance_id
        for instance_id, item in by_id.items()
        if str(item.get("split") or "") == "train"
    )
    dev_ids = sorted(
        instance_id
        for instance_id, item in by_id.items()
        if str(item.get("split") or "") == "dev"
    )
    other = sorted(
        instance_id
        for instance_id, item in by_id.items()
        if str(item.get("split") or "") not in ALLOWED_SPLITS
    )
    if other:
        raise MaterializeInputError(
            f"split assignments contain non train/dev values: {other[:8]}"
        )
    return train_ids, dev_ids, by_id


def validate_split_membership(
    split_manifest: Mapping[str, Any],
    *,
    expected_n_rows: int = EXPECTED_N_ROWS,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_dev_rows: int = EXPECTED_DEV_ROWS,
) -> tuple[list[str], list[str], dict[str, Mapping[str, Any]]]:
    train_ids, dev_ids, by_id = split_assignments(split_manifest)
    train_set = set(train_ids)
    dev_set = set(dev_ids)
    if train_set & dev_set:
        leaked = sorted(train_set & dev_set)
        raise MaterializeInputError(
            f"train/dev intersection is not empty: {leaked[:8]}"
        )
    union = train_set | dev_set
    if len(union) != expected_n_rows:
        raise MaterializeInputError(
            f"train∪dev size {len(union)} != expected {expected_n_rows}"
        )
    if len(train_ids) != expected_train_rows:
        raise MaterializeInputError(
            f"train rows {len(train_ids)} != expected {expected_train_rows}"
        )
    if len(dev_ids) != expected_dev_rows:
        raise MaterializeInputError(
            f"dev rows {len(dev_ids)} != expected {expected_dev_rows}"
        )
    header_train = split_manifest.get("actual_train_rows")
    header_dev = split_manifest.get("actual_dev_rows")
    if header_train is not None and int(header_train) != len(train_ids):
        raise MaterializeInputError(
            f"split header actual_train_rows {header_train} != assignment train {len(train_ids)}"
        )
    if header_dev is not None and int(header_dev) != len(dev_ids):
        raise MaterializeInputError(
            f"split header actual_dev_rows {header_dev} != assignment dev {len(dev_ids)}"
        )
    return train_ids, dev_ids, by_id


def parquet_source_records(frame: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw in frame.to_dict(orient="records"):
        instance_id = "" if is_null(raw.get("instance_id")) else str(raw["instance_id"])
        records.append(
            {
                "instance_id": instance_id,
                "problem_statement": (
                    ""
                    if is_null(raw.get("problem_statement"))
                    else str(raw["problem_statement"])
                ),
                "repo": "" if is_null(raw.get("repo")) else str(raw["repo"]),
                "base_commit": (
                    ""
                    if is_null(raw.get("base_commit"))
                    else str(raw["base_commit"]).strip()
                ),
                "version": "" if is_null(raw.get("version")) else str(raw["version"]),
            }
        )
    return records


def _string_list(value: Any) -> list[str]:
    if value is None or is_null(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None and not is_null(item)]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _string_list(value.tolist())
        except (TypeError, ValueError):
            pass
    return [str(value)]


def project_oracle_symbols(raw_symbols: Any) -> list[dict[str, str]]:
    """Project frozen M1C-B symbols to canonical (path, qualname) identity.

    Does not re-run AST. Dedup + sort by (path, qualname).
    """
    if raw_symbols is None or is_null(raw_symbols):
        items: list[Any] = []
    elif isinstance(raw_symbols, (list, tuple)):
        items = list(raw_symbols)
    elif hasattr(raw_symbols, "tolist"):
        items = list(raw_symbols.tolist())
    else:
        raise MaterializeInputError(
            f"oracle_symbols is not a list: {type(raw_symbols)!r}"
        )
    seen: set[tuple[str, str]] = set()
    projected: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, MappingABC):
            raise MaterializeInputError("oracle_symbols entry is not a mapping")
        path = str(item.get("path") or "").strip()
        qualname = str(item.get("qualname") or "").strip()
        if not path or not qualname:
            raise MaterializeInputError(
                "oracle_symbols entry missing path or qualname "
                f"(path={path!r}, qualname={qualname!r})"
            )
        key = (path, qualname)
        if key in seen:
            continue
        seen.add(key)
        projected.append({"path": path, "qualname": qualname})
    projected.sort(key=lambda item: (item["path"], item["qualname"]))
    return projected


def build_policy_row(
    source: Mapping[str, Any],
    *,
    split: str,
    index: int,
    data_source: str = DATA_SOURCE,
) -> dict[str, Any]:
    instance_id = str(source["instance_id"])
    problem_statement = str(source.get("problem_statement") or "")
    if split not in ALLOWED_SPLITS:
        raise MaterializeInputError(f"{instance_id}: invalid split {split!r}")
    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": problem_statement,
            }
        ],
        "reward_model": {
            "style": REWARD_STYLE,
            "ground_truth": instance_id,
        },
        "extra_info": {
            "index": int(index),
            "instance_id": instance_id,
            "repo": str(source.get("repo") or ""),
            "base_commit": str(source.get("base_commit") or ""),
            "version": str(source.get("version") or ""),
            "split": split,
        },
    }


def build_sidecar_row(
    *,
    instance_id: str,
    split: str,
    oracle: Mapping[str, Any],
    symbol: Mapping[str, Any],
) -> dict[str, Any]:
    symbols = project_oracle_symbols(symbol.get("oracle_symbols"))
    return {
        "instance_id": instance_id,
        "split": split,
        "base_changed_files": _string_list(oracle.get("base_changed_files")),
        "oracle_symbols": symbols,
        "symbol_applicable": bool(symbols),
    }


def build_policy_rows_for_split(
    instance_ids: Sequence[str],
    *,
    split: str,
    sources: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(instance_ids)
    rows: list[dict[str, Any]] = []
    for index, instance_id in enumerate(ordered):
        if instance_id not in sources:
            raise MaterializeInputError(
                f"{split} source missing instance_id {instance_id}"
            )
        rows.append(
            build_policy_row(sources[instance_id], split=split, index=index)
        )
    return rows


def build_sidecar_rows(
    instance_ids: Sequence[str],
    *,
    assignments: Mapping[str, Mapping[str, Any]],
    oracles: Mapping[str, Mapping[str, Any]],
    symbols: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(instance_ids):
        if instance_id not in assignments:
            raise MaterializeInputError(f"sidecar missing split for {instance_id}")
        if instance_id not in oracles:
            raise MaterializeInputError(f"sidecar missing M1C-A oracle for {instance_id}")
        if instance_id not in symbols:
            raise MaterializeInputError(
                f"sidecar missing M1C-B symbol oracle for {instance_id}"
            )
        split = str(assignments[instance_id]["split"])
        rows.append(
            build_sidecar_row(
                instance_id=instance_id,
                split=split,
                oracle=oracles[instance_id],
                symbol=symbols[instance_id],
            )
        )
    return rows


def _policy_features():
    from datasets import Features, Value

    return Features(
        {
            "data_source": Value("string"),
            "prompt": [{"role": Value("string"), "content": Value("string")}],
            "reward_model": {
                "style": Value("string"),
                "ground_truth": Value("string"),
            },
            "extra_info": {
                "index": Value("int64"),
                "instance_id": Value("string"),
                "repo": Value("string"),
                "base_commit": Value("string"),
                "version": Value("string"),
                "split": Value("string"),
            },
        }
    )


def _sidecar_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "instance_id": Value("string"),
            "split": Value("string"),
            "base_changed_files": Sequence(Value("string")),
            "oracle_symbols": [
                {"path": Value("string"), "qualname": Value("string")}
            ],
            "symbol_applicable": Value("bool"),
        }
    )


def write_parquet_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    features: Any,
) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise MaterializeInputError(
            "HuggingFace datasets is required to write M1E parquet. "
            "Use the pinned RL conda env. Do not pip-install packages from this module."
        ) from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list([to_jsonable(row) for row in rows], features=features)
    dataset.to_parquet(str(path))


def file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "schema_version": SCHEMA_VERSION,
        "datasets": None,
        "pyarrow": None,
        "pandas": None,
    }
    try:
        import datasets as hf_datasets

        versions["datasets"] = getattr(hf_datasets, "__version__", None)
    except ImportError:
        pass
    try:
        import pyarrow as pa

        versions["pyarrow"] = getattr(pa, "__version__", None)
    except ImportError:
        pass
    try:
        import pandas as pd

        versions["pandas"] = getattr(pd, "__version__", None)
    except ImportError:
        pass
    return versions


def schema_record() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": HF_REVISION,
        "data_source": DATA_SOURCE,
        "row_ordering": ROW_ORDERING,
        "agent_name": {
            "present_in_dataset": False,
            "reason": (
                "Pinned AgentLoopWorker fills config.agent.default_agent_loop "
                "when agent_name is missing from the batch. Dataset must not "
                "bake a specific Agent implementation name."
            ),
        },
        "prompt_contract": {
            "prompt_key": "prompt",
            "messages": [{"role": "user", "content": "raw problem_statement only"}],
            "not_included": [
                "hints_text",
                "oracle labels",
                "system prompt",
                "tool protocol",
                "Agent scaffold",
            ],
            "note": (
                "Future system/tool/scaffold prompts belong in AgentLoop/runtime, "
                "not in this dataset."
            ),
        },
        "oracle_separation": {
            "policy_runtime": [TRAIN_PARQUET_RELPATH, DEV_PARQUET_RELPATH],
            "evaluator_only": ORACLE_PARQUET_RELPATH,
            "reward_model.ground_truth": "instance_id (opaque; lookup sidecar later)",
            "compute_score": "not implemented in M1E",
        },
        "verl": {
            "prompt_key": "prompt",
            "return_raw_chat": True,
            "filter_overlong_prompts": {
                "required_for_full_corpus": False,
                "constructor_default": True,
                "note": (
                    "Pinned RLHFDataset defaults filter_overlong_prompts=True and "
                    "max_prompt_length=1024. Training/smoke must set "
                    "filter_overlong_prompts=false or long SWE-Gym issues are dropped. "
                    "Do not change veRL."
                ),
            },
        },
        "policy_runtime": {
            "columns": list(POLICY_COLUMNS),
            "semantics": {
                "data_source": DATA_SOURCE,
                "prompt": "list of one user message; content = raw problem_statement",
                "reward_model.style": REWARD_STYLE,
                "reward_model.ground_truth": "instance_id only",
                "extra_info.index": (
                    "0-based rank within this split after instance_id lex sort"
                ),
                "extra_info.instance_id": "SWE-Gym instance_id",
                "extra_info.repo": "owner/name",
                "extra_info.base_commit": "base commit SHA",
                "extra_info.version": "SWE-Gym version string",
                "extra_info.split": "train | dev",
            },
            "extra_info_keys": list(EXTRA_INFO_KEYS),
            "forbidden_keys": list(POLICY_FORBIDDEN_DERIVED_FIELDS),
        },
        "evaluator_sidecar": {
            "columns": list(SIDECAR_COLUMNS),
            "semantics": {
                "instance_id": "join key",
                "split": "frozen M1D assignment",
                "base_changed_files": "frozen M1C-A source-side paths at base_commit",
                "oracle_symbols": (
                    "frozen M1C-B conservative base-existing symbols, "
                    "identity = path + qualname"
                ),
                "symbol_applicable": (
                    "True iff at least one canonical oracle symbol; False does not drop"
                ),
            },
            "symbol_identity": list(SYMBOL_IDENTITY_KEYS),
            "future_reward_note": (
                "symbol applicable: 0.5 * F1_file + 0.5 * F1_symbol; "
                "symbol N/A: masked normalization -> F1_file. Not implemented in M1E."
            ),
        },
    }


def _checksum_if_exists(path: Path) -> str | None:
    if Path(path).is_file():
        return sha256_file(path)
    return None


def dataset_manifest_record(
    *,
    repo_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    train_path: Path,
    dev_path: Path,
    oracle_path: Path,
    raw_parquet: Path,
    oracle_jsonl: Path,
    symbol_jsonl: Path,
    split_path: Path,
    policy_path: Path,
    n_symbol_applicable: int,
    n_symbol_na: int,
) -> dict[str, Any]:
    def _artifact(
        path: Path,
        rows: Sequence[Mapping[str, Any]],
        relpath: str,
    ) -> dict[str, Any]:
        identity = file_identity(path)
        return {
            "path": relpath,
            "n_rows": len(rows),
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
            "logical_sha256": logical_rows_sha256(rows),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": HF_REVISION,
        "raw_parquet": {
            "path": "data/raw/swe_gym/data/train-00000-of-00001.parquet",
            "sha256": (
                sha256_file(raw_parquet) if raw_parquet.is_file() else EXPECTED_SHA256
            ),
            "size_bytes": (
                int(raw_parquet.stat().st_size)
                if raw_parquet.is_file()
                else EXPECTED_SIZE_BYTES
            ),
            "n_rows": EXPECTED_N_ROWS,
        },
        "inputs": {
            "m1c_oracle_jsonl": {
                "path": "data/interim/swe_gym/m1c_oracle.jsonl",
                "sha256": _checksum_if_exists(oracle_jsonl),
            },
            "m1c_symbol_jsonl": {
                "path": "data/interim/swe_gym/m1c_symbol_oracle.jsonl",
                "sha256": _checksum_if_exists(symbol_jsonl),
            },
            "m1c_oracle_summary": {
                "path": "data/manifests/swe_gym_m1c_oracle_summary.json",
                "sha256": _checksum_if_exists(oracle_summary_path(repo_root)),
            },
            "m1c_symbol_summary": {
                "path": "data/manifests/swe_gym_m1c_symbol_summary.json",
                "sha256": _checksum_if_exists(symbol_oracle_summary_path(repo_root)),
            },
            "m1d_policy": {
                "path": POLICY_RELPATH,
                "sha256": _checksum_if_exists(policy_path),
            },
            "m1d_split": {
                "path": SPLIT_RELPATH,
                "sha256": _checksum_if_exists(split_path),
            },
            "m1d_feature_summary": {
                "path": FEATURE_SUMMARY_RELPATH,
                "sha256": _checksum_if_exists(Path(repo_root) / FEATURE_SUMMARY_RELPATH),
                "consumed_feature_values": False,
                "note": (
                    "Checksum of the git-tracked M1D-A summary only. "
                    "Feature values are not joined into policy or sidecar rows."
                ),
            },
        },
        "artifacts": {
            "train": _artifact(train_path, train_rows, TRAIN_PARQUET_RELPATH),
            "dev": _artifact(dev_path, dev_rows, DEV_PARQUET_RELPATH),
            "evaluator_oracle": _artifact(
                oracle_path, oracle_rows, ORACLE_PARQUET_RELPATH
            ),
        },
        "split_checks": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "oracle_rows": len(oracle_rows),
            "intersection": 0,
            "union": len(train_rows) + len(dev_rows),
            "exact_m1d_match": True,
        },
        "symbol_applicable": {
            "true": int(n_symbol_applicable),
            "false": int(n_symbol_na),
            "denominator": len(oracle_rows),
        },
        "libraries": library_versions(),
        "row_ordering": ROW_ORDERING,
        "not_implemented": [
            "reward compute_score",
            "Agent scaffold",
            "tool environment",
            "GRPO",
        ],
    }


def join_materialize_tables(
    *,
    source_records: Sequence[Mapping[str, Any]],
    oracle_records: Sequence[Mapping[str, Any]],
    symbol_records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    expected_n_rows: int = EXPECTED_N_ROWS,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_dev_rows: int = EXPECTED_DEV_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids, dev_ids, assignments = validate_split_membership(
        split_manifest,
        expected_n_rows=expected_n_rows,
        expected_train_rows=expected_train_rows,
        expected_dev_rows=expected_dev_rows,
    )
    expected_ids = set(train_ids) | set(dev_ids)
    sources = index_records_by_id(source_records, source="raw_parquet")
    oracles = index_records_by_id(oracle_records, source="m1c_oracle.jsonl")
    symbols = index_records_by_id(symbol_records, source="m1c_symbol_oracle.jsonl")
    require_exact_id_set(set(sources), expected_ids, source="raw_parquet")
    require_exact_id_set(set(oracles), expected_ids, source="m1c_oracle.jsonl")
    require_exact_id_set(set(symbols), expected_ids, source="m1c_symbol_oracle.jsonl")

    train_rows = build_policy_rows_for_split(
        train_ids, split="train", sources=sources
    )
    dev_rows = build_policy_rows_for_split(dev_ids, split="dev", sources=sources)
    sidecar_rows = build_sidecar_rows(
        sorted(expected_ids),
        assignments=assignments,
        oracles=oracles,
        symbols=symbols,
    )
    problem_statements = {
        instance_id: str(item.get("problem_statement") or "")
        for instance_id, item in sources.items()
    }
    leakage = validate_policy_rows_leakage(
        train_rows + dev_rows,
        problem_statements=problem_statements,
    )
    if leakage:
        raise MaterializeInputError("policy leakage:\n  - " + "\n  - ".join(leakage[:20]))
    _assert_disjoint_policy_splits(train_rows, dev_rows)
    return train_rows, dev_rows, sidecar_rows


def _assert_disjoint_policy_splits(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
) -> None:
    train_ids = {str(row["extra_info"]["instance_id"]) for row in train_rows}
    dev_ids = {str(row["extra_info"]["instance_id"]) for row in dev_rows}
    if train_ids & dev_ids:
        raise MaterializeInputError(
            f"materialized train/dev intersection: {sorted(train_ids & dev_ids)[:8]}"
        )


def format_materialize_report(manifest: Mapping[str, Any]) -> str:
    split_checks = manifest["split_checks"]
    applicable = manifest["symbol_applicable"]
    artifacts = manifest["artifacts"]
    lines = [
        "SWE-Gym M1E veRL-ready dataset materialize",
        f"schema_version: {manifest['schema_version']}",
        f"revision: {manifest['revision']}",
        f"raw parquet sha256: {manifest['raw_parquet']['sha256']}",
        (
            f"rows: train={split_checks['train_rows']} "
            f"dev={split_checks['dev_rows']} "
            f"oracle={split_checks['oracle_rows']} "
            f"union={split_checks['union']} "
            f"intersection={split_checks['intersection']}"
        ),
        (
            f"symbol_applicable: true={applicable['true']} "
            f"false={applicable['false']} "
            f"denominator={applicable['denominator']}"
        ),
        f"exact M1D split match: {split_checks['exact_m1d_match']}",
        "",
        "artifacts:",
    ]
    for name in ("train", "dev", "evaluator_oracle"):
        item = artifacts[name]
        lines.append(
            f"  {name}: n={item['n_rows']} size={item['size_bytes']} "
            f"sha256={item['sha256']} logical={item['logical_sha256']}"
        )
    lines.append("")
    lines.append("not implemented: reward / Agent scaffold / GRPO")
    lines.append("")
    return "\n".join(lines)


def materialize(
    *,
    repo_root: Path,
    raw_parquet: Path | None = None,
    split_json: Path | None = None,
    policy_json: Path | None = None,
    oracle_jsonl: Path | None = None,
    symbol_jsonl: Path | None = None,
    train_out: Path | None = None,
    dev_out: Path | None = None,
    oracle_out: Path | None = None,
    schema_out: Path | None = None,
    manifest_out: Path | None = None,
    expected_n_rows: int = EXPECTED_N_ROWS,
    expected_n_repos: int = EXPECTED_N_REPOS,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_dev_rows: int = EXPECTED_DEV_ROWS,
    verify_raw_identity: bool = True,
    check_field_policy: bool = True,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    raw_parquet = Path(raw_parquet) if raw_parquet is not None else parquet_path(repo_root)
    split_json = (
        Path(split_json)
        if split_json is not None
        else repo_root / SPLIT_RELPATH
    )
    policy_json = (
        Path(policy_json)
        if policy_json is not None
        else repo_root / POLICY_RELPATH
    )
    oracle_jsonl = (
        Path(oracle_jsonl)
        if oracle_jsonl is not None
        else oracle_jsonl_path(repo_root)
    )
    symbol_jsonl = (
        Path(symbol_jsonl)
        if symbol_jsonl is not None
        else symbol_oracle_jsonl_path(repo_root)
    )
    train_out = Path(train_out) if train_out is not None else train_parquet_path(repo_root)
    dev_out = Path(dev_out) if dev_out is not None else dev_parquet_path(repo_root)
    oracle_out = (
        Path(oracle_out) if oracle_out is not None else oracle_parquet_path(repo_root)
    )
    schema_out = Path(schema_out) if schema_out is not None else schema_path(repo_root)
    manifest_out = (
        Path(manifest_out)
        if manifest_out is not None
        else dataset_manifest_path(repo_root)
    )

    errors: list[str] = []
    if check_field_policy:
        errors.extend(committed_field_policy_errors(repo_root))
    if not raw_parquet.is_file():
        errors.append(f"raw parquet not found: {raw_parquet}")
    elif verify_raw_identity:
        errors.extend(verify_parquet_file(raw_parquet))
    for label, path in (
        ("M1D split", split_json),
        ("M1C-A oracle JSONL", oracle_jsonl),
        ("M1C-B symbol JSONL", symbol_jsonl),
    ):
        if not path.is_file():
            errors.append(f"{label} not found: {path}")
    if errors:
        raise MaterializeInputError("\n".join(errors))

    try:
        import pandas as pd
    except ImportError as exc:
        raise MaterializeInputError(
            "pandas is required to read the official parquet. "
            "Use the pinned RL conda env."
        ) from exc
    try:
        frame = pd.read_parquet(raw_parquet)
    except ImportError as exc:
        raise MaterializeInputError(
            "reading parquet requires pyarrow in the pinned env."
        ) from exc

    schema_errors = validate_schema_and_cardinality(
        frame,
        expected_n_rows=expected_n_rows,
        expected_n_repos=expected_n_repos,
    )
    if schema_errors:
        raise MaterializeInputError(
            "raw parquet schema/cardinality:\n  - " + "\n  - ".join(schema_errors)
        )

    split_manifest = json.loads(split_json.read_text(encoding="utf-8"))
    train_rows, dev_rows, sidecar_rows = join_materialize_tables(
        source_records=parquet_source_records(frame),
        oracle_records=read_jsonl(oracle_jsonl),
        symbol_records=read_jsonl(symbol_jsonl),
        split_manifest=split_manifest,
        expected_n_rows=expected_n_rows,
        expected_train_rows=expected_train_rows,
        expected_dev_rows=expected_dev_rows,
    )

    # In-memory rebuild must be byte-identical before writing.
    again_train, again_dev, again_oracle = join_materialize_tables(
        source_records=parquet_source_records(frame),
        oracle_records=read_jsonl(oracle_jsonl),
        symbol_records=read_jsonl(symbol_jsonl),
        split_manifest=split_manifest,
        expected_n_rows=expected_n_rows,
        expected_train_rows=expected_train_rows,
        expected_dev_rows=expected_dev_rows,
    )
    if logical_rows_sha256(train_rows) != logical_rows_sha256(again_train):
        raise MaterializeInputError("train logical checksum is not deterministic")
    if logical_rows_sha256(dev_rows) != logical_rows_sha256(again_dev):
        raise MaterializeInputError("dev logical checksum is not deterministic")
    if logical_rows_sha256(sidecar_rows) != logical_rows_sha256(again_oracle):
        raise MaterializeInputError("oracle logical checksum is not deterministic")

    write_parquet_rows(train_out, train_rows, features=_policy_features())
    write_parquet_rows(dev_out, dev_rows, features=_policy_features())
    write_parquet_rows(oracle_out, sidecar_rows, features=_sidecar_features())

    n_applicable = sum(1 for row in sidecar_rows if row["symbol_applicable"])
    n_na = len(sidecar_rows) - n_applicable
    schema = schema_record()
    if manifest_json_bytes(schema) != manifest_json_bytes(schema_record()):
        raise MaterializeInputError("schema serialization is not deterministic")

    manifest = dataset_manifest_record(
        repo_root=repo_root,
        train_rows=train_rows,
        dev_rows=dev_rows,
        oracle_rows=sidecar_rows,
        train_path=train_out,
        dev_path=dev_out,
        oracle_path=oracle_out,
        raw_parquet=raw_parquet,
        oracle_jsonl=oracle_jsonl,
        symbol_jsonl=symbol_jsonl,
        split_path=split_json,
        policy_path=policy_json,
        n_symbol_applicable=n_applicable,
        n_symbol_na=n_na,
    )
    write_json(schema_out, schema)
    write_json(manifest_out, manifest)
    return {
        "schema": schema,
        "manifest": manifest,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "oracle_rows": sidecar_rows,
        "train_path": train_out,
        "dev_path": dev_out,
        "oracle_path": oracle_out,
        "schema_path": schema_out,
        "manifest_path": manifest_out,
    }
