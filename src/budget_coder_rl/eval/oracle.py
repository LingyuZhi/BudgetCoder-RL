"""Frozen M1E evaluator-oracle sidecar. Never imported by AgentLoop."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym_materialize import (
    SIDECAR_COLUMNS,
    oracle_parquet_path,
)

SIDECAR_COLUMN_TUPLE = SIDECAR_COLUMNS


class OracleLookupError(KeyError):
    """Requested instance_id is missing from the evaluator sidecar."""


@dataclass(frozen=True)
class OracleRow:
    instance_id: str
    split: str
    base_changed_files: tuple[str, ...]
    oracle_symbols: tuple[tuple[str, str], ...]
    symbol_applicable: bool


class EvaluatorOracleIndex:
    """instance_id -> privileged oracle row. Policy code must not receive this."""

    def __init__(self, rows: Mapping[str, OracleRow]) -> None:
        self._rows = dict(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def get(self, instance_id: str) -> OracleRow:
        key = str(instance_id)
        if key not in self._rows:
            raise OracleLookupError(key)
        return self._rows[key]

    def __contains__(self, instance_id: object) -> bool:
        return str(instance_id) in self._rows


def load_evaluator_oracle(path: str | Path) -> EvaluatorOracleIndex:
    """Load the frozen M1E sidecar parquet. Does not touch policy parquet."""
    parquet = Path(path)
    if not parquet.is_file():
        raise FileNotFoundError(f"evaluator oracle parquet missing: {parquet}")
    import pandas as pd

    frame = pd.read_parquet(parquet)
    missing = [name for name in SIDECAR_COLUMN_TUPLE if name not in frame.columns]
    if missing:
        raise ValueError(f"{parquet} missing sidecar columns: {missing}")
    indexed: dict[str, OracleRow] = {}
    for raw in frame.to_dict(orient="records"):
        row = _parse_sidecar_record(raw)
        if row.instance_id in indexed:
            raise ValueError(f"duplicate oracle instance_id {row.instance_id}")
        indexed[row.instance_id] = row
    return EvaluatorOracleIndex(indexed)


def default_oracle_path(repo_root: str | Path) -> Path:
    return oracle_parquet_path(Path(repo_root))


def _parse_sidecar_record(raw: Mapping[str, Any]) -> OracleRow:
    instance_id = str(raw.get("instance_id") or "").strip()
    if not instance_id:
        raise ValueError("oracle row missing instance_id")
    files = _string_tuple(raw.get("base_changed_files"))
    symbols = _symbol_tuple(raw.get("oracle_symbols"))
    applicable = bool(raw.get("symbol_applicable"))
    return OracleRow(
        instance_id=instance_id,
        split=str(raw.get("split") or ""),
        base_changed_files=files,
        oracle_symbols=symbols,
        symbol_applicable=applicable,
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    items = _sequence(value)
    return tuple(str(item) for item in items if str(item))


def _symbol_tuple(value: Any) -> tuple[tuple[str, str], ...]:
    items = _sequence(value)
    out: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, MappingABC):
            path = str(item.get("path") or "").strip()
            qualname = str(item.get("qualname") or "").strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            path = str(item[0]).strip()
            qualname = str(item[1]).strip()
        else:
            raise ValueError(f"oracle symbol entry is not a mapping: {type(item)!r}")
        if not path or not qualname:
            raise ValueError("oracle symbol missing path or qualname")
        out.append((path, qualname))
    return tuple(out)


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
        except (TypeError, ValueError):
            converted = None
        if converted is None:
            return []
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise ValueError(f"expected a sequence, got {type(value)!r}")
