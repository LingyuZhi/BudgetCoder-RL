"""Deterministic post-rollout localization metrics. Not a GRPO reward path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from budget_coder_rl.eval.oracle import OracleRow

FILE_WEIGHT = 0.5
SYMBOL_WEIGHT = 0.5
VALID_FINAL_TERMINATION = "finish"


@dataclass(frozen=True)
class LocalizationMetrics:
    parse_ok: bool
    submission_missing: bool
    symbol_status: str
    file_precision: float
    file_recall: float
    file_f1: float
    symbol_precision: float | None
    symbol_recall: float | None
    symbol_f1: float | None
    localization_score: float
    n_pred_files: int
    n_pred_symbols: int
    n_gold_files: int
    n_gold_symbols: int
    n_true_positive_files: int
    n_true_positive_symbols: int | None
    n_duplicate_file_preds: int
    n_duplicate_symbol_preds: int
    n_pred_locations_raw: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "parse_ok": self.parse_ok,
            "submission_missing": self.submission_missing,
            "symbol_status": self.symbol_status,
            "file_precision": self.file_precision,
            "file_recall": self.file_recall,
            "file_f1": self.file_f1,
            "symbol_precision": self.symbol_precision,
            "symbol_recall": self.symbol_recall,
            "symbol_f1": self.symbol_f1,
            "localization_score": self.localization_score,
            "n_pred_files": self.n_pred_files,
            "n_pred_symbols": self.n_pred_symbols,
            "n_gold_files": self.n_gold_files,
            "n_gold_symbols": self.n_gold_symbols,
            "n_true_positive_files": self.n_true_positive_files,
            "n_true_positive_symbols": self.n_true_positive_symbols,
            "n_duplicate_file_preds": self.n_duplicate_file_preds,
            "n_duplicate_symbol_preds": self.n_duplicate_symbol_preds,
            "n_pred_locations_raw": self.n_pred_locations_raw,
            "file_weight": FILE_WEIGHT,
            "symbol_weight": SYMBOL_WEIGHT,
        }


def set_precision_recall_f1(
    pred: set[Any],
    gold: set[Any],
) -> tuple[float, float, float, int]:
    """Set P/R/F1 with documented empty-set conventions."""
    tp = len(pred & gold)
    if not pred and not gold:
        return 1.0, 1.0, 1.0, tp
    if not pred and gold:
        return 0.0, 0.0, 0.0, tp
    if pred and not gold:
        return 0.0, 1.0, 0.0, tp
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1, tp


def evaluate_episode(
    *,
    termination: str | None,
    submission: Mapping[str, Any] | None,
    oracle: OracleRow,
) -> LocalizationMetrics:
    """Score one rollout against a frozen oracle row.

    Gold path/symbol lists are not copied into the returned metrics.
    """
    valid = _valid_final(termination, submission)
    gold_files = set(oracle.base_changed_files)
    gold_symbols = set(oracle.oracle_symbols)
    if not valid:
        return LocalizationMetrics(
            parse_ok=False,
            submission_missing=True,
            symbol_status="not_scored",
            file_precision=0.0,
            file_recall=0.0,
            file_f1=0.0,
            symbol_precision=None,
            symbol_recall=None,
            symbol_f1=None,
            localization_score=0.0,
            n_pred_files=0,
            n_pred_symbols=0,
            n_gold_files=len(gold_files),
            n_gold_symbols=len(gold_symbols),
            n_true_positive_files=0,
            n_true_positive_symbols=None,
            n_duplicate_file_preds=0,
            n_duplicate_symbol_preds=0,
            n_pred_locations_raw=0,
        )
    return evaluate_submission(submission, oracle)


def evaluate_submission(
    submission: Mapping[str, Any] | None,
    oracle: OracleRow,
) -> LocalizationMetrics:
    locations = _locations(submission)
    file_preds, n_dup_files = _dedup([item[0] for item in locations])
    symbol_items = [item for item in locations if item[1] is not None]
    symbol_preds, n_dup_symbols = _dedup(
        [(path, symbol) for path, symbol in symbol_items if symbol is not None]
    )
    gold_files = set(oracle.base_changed_files)
    gold_symbols = set(oracle.oracle_symbols)
    file_p, file_r, file_f1, file_tp = set_precision_recall_f1(
        set(file_preds), gold_files
    )
    if not oracle.symbol_applicable:
        return LocalizationMetrics(
            parse_ok=True,
            submission_missing=False,
            symbol_status="unavailable",
            file_precision=file_p,
            file_recall=file_r,
            file_f1=file_f1,
            symbol_precision=None,
            symbol_recall=None,
            symbol_f1=None,
            localization_score=file_f1,
            n_pred_files=len(file_preds),
            n_pred_symbols=len(symbol_preds),
            n_gold_files=len(gold_files),
            n_gold_symbols=len(gold_symbols),
            n_true_positive_files=file_tp,
            n_true_positive_symbols=None,
            n_duplicate_file_preds=n_dup_files,
            n_duplicate_symbol_preds=n_dup_symbols,
            n_pred_locations_raw=len(locations),
        )
    symbol_p, symbol_r, symbol_f1, symbol_tp = set_precision_recall_f1(
        set(symbol_preds), gold_symbols
    )
    score = FILE_WEIGHT * file_f1 + SYMBOL_WEIGHT * symbol_f1
    return LocalizationMetrics(
        parse_ok=True,
        submission_missing=False,
        symbol_status="scored",
        file_precision=file_p,
        file_recall=file_r,
        file_f1=file_f1,
        symbol_precision=symbol_p,
        symbol_recall=symbol_r,
        symbol_f1=symbol_f1,
        localization_score=score,
        n_pred_files=len(file_preds),
        n_pred_symbols=len(symbol_preds),
        n_gold_files=len(gold_files),
        n_gold_symbols=len(gold_symbols),
        n_true_positive_files=file_tp,
        n_true_positive_symbols=symbol_tp,
        n_duplicate_file_preds=n_dup_files,
        n_duplicate_symbol_preds=n_dup_symbols,
        n_pred_locations_raw=len(locations),
    )


def _valid_final(termination: str | None, submission: Mapping[str, Any] | None) -> bool:
    if termination != VALID_FINAL_TERMINATION:
        return False
    if not isinstance(submission, Mapping):
        return False
    locations = submission.get("locations")
    return isinstance(locations, list)


def _locations(submission: Mapping[str, Any] | None) -> list[tuple[str, str | None]]:
    if not isinstance(submission, Mapping):
        return []
    raw = submission.get("locations")
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, str | None]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        symbol: str | None = None
        if "symbol" in item and item.get("symbol") is not None:
            text = str(item.get("symbol") or "").strip()
            symbol = text or None
        out.append((path, symbol))
    return out


def _dedup(items: Sequence[Any]) -> tuple[list[Any], int]:
    seen: set[Any] = set()
    out: list[Any] = []
    n_dup = 0
    for item in items:
        if item in seen:
            n_dup += 1
            continue
        seen.add(item)
        out.append(item)
    return out, n_dup
