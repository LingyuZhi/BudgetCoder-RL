"""Deterministic localization evaluator. No AgentLoop / oracle leakage."""

from __future__ import annotations

from budget_coder_rl.eval.localization import (
    evaluate_episode,
    evaluate_submission,
    set_precision_recall_f1,
)
from budget_coder_rl.eval.oracle import OracleRow


def _oracle(
    *,
    files: list[str],
    symbols: list[tuple[str, str]] | None = None,
    applicable: bool | None = None,
) -> OracleRow:
    symbols = list(symbols or [])
    if applicable is None:
        applicable = bool(symbols)
    return OracleRow(
        instance_id="owner__repo-1",
        split="train",
        base_changed_files=tuple(files),
        oracle_symbols=tuple(symbols),
        symbol_applicable=applicable,
    )


def test_set_prf_empty_conventions():
    assert set_precision_recall_f1(set(), set()) == (1.0, 1.0, 1.0, 0)
    assert set_precision_recall_f1(set(), {"a"}) == (0.0, 0.0, 0.0, 0)
    assert set_precision_recall_f1({"a"}, set()) == (0.0, 1.0, 0.0, 0)
    p, r, f1, tp = set_precision_recall_f1({"a", "b"}, {"b", "c"})
    assert tp == 1
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_perfect_file_and_symbol():
    oracle = _oracle(
        files=["src/foo.py"],
        symbols=[("src/foo.py", "Foo.bar")],
    )
    metrics = evaluate_submission(
        {"locations": [{"path": "src/foo.py", "symbol": "Foo.bar"}]},
        oracle,
    )
    assert metrics.parse_ok is True
    assert metrics.symbol_status == "scored"
    assert metrics.file_f1 == 1.0
    assert metrics.symbol_f1 == 1.0
    assert metrics.localization_score == 1.0


def test_duplicate_predictions_deduped_and_logged():
    oracle = _oracle(files=["a.py", "b.py"], symbols=[("a.py", "A")])
    metrics = evaluate_submission(
        {
            "locations": [
                {"path": "a.py", "symbol": "A"},
                {"path": "a.py", "symbol": "A"},
                {"path": "b.py"},
            ]
        },
        oracle,
    )
    assert metrics.n_pred_files == 2
    assert metrics.n_pred_symbols == 1
    assert metrics.n_duplicate_file_preds == 1
    assert metrics.n_duplicate_symbol_preds == 1
    assert metrics.file_f1 == 1.0
    assert metrics.symbol_f1 == 1.0


def test_empty_valid_final_against_nonempty_gold():
    oracle = _oracle(files=["a.py"], symbols=[("a.py", "A")])
    metrics = evaluate_episode(
        termination="finish",
        submission={"locations": []},
        oracle=oracle,
    )
    assert metrics.parse_ok is True
    assert metrics.file_f1 == 0.0
    assert metrics.symbol_status == "scored"
    assert metrics.symbol_f1 == 0.0
    assert metrics.localization_score == 0.0


def test_symbol_unavailable_does_not_fabricate_symbol_score():
    oracle = _oracle(files=["a.py"], symbols=[], applicable=False)
    metrics = evaluate_submission(
        {"locations": [{"path": "a.py", "symbol": "Whatever"}]},
        oracle,
    )
    assert metrics.symbol_status == "unavailable"
    assert metrics.symbol_precision is None
    assert metrics.symbol_recall is None
    assert metrics.symbol_f1 is None
    assert metrics.file_f1 == 1.0
    assert metrics.localization_score == 1.0
    assert "a.py" not in metrics.as_dict().values()


def test_path_only_prediction_ignored_for_symbol_set():
    oracle = _oracle(files=["a.py"], symbols=[("a.py", "Foo.bar")])
    metrics = evaluate_submission({"locations": [{"path": "a.py"}]}, oracle)
    assert metrics.file_f1 == 1.0
    assert metrics.n_pred_symbols == 0
    assert metrics.symbol_f1 == 0.0


def test_missing_final_and_budget_exhausted_are_zero():
    oracle = _oracle(files=["a.py"], symbols=[("a.py", "A")])
    for termination in ("budget_exhausted", "max_turns", "response_length"):
        metrics = evaluate_episode(
            termination=termination,
            submission=None,
            oracle=oracle,
        )
        assert metrics.parse_ok is False
        assert metrics.submission_missing is True
        assert metrics.symbol_status == "not_scored"
        assert metrics.symbol_f1 is None
        assert metrics.localization_score == 0.0
        assert metrics.file_f1 == 0.0


def test_evaluator_is_deterministic():
    oracle = _oracle(files=["a.py", "b.py"], symbols=[("a.py", "A"), ("b.py", "B")])
    submission = {
        "locations": [
            {"path": "b.py", "symbol": "B"},
            {"path": "miss.py", "symbol": "Nope"},
        ]
    }
    first = evaluate_submission(submission, oracle).as_dict()
    second = evaluate_submission(submission, oracle).as_dict()
    assert first == second
    assert first["file_precision"] == 0.5
    assert first["file_recall"] == 0.5
    assert "base_changed_files" not in first
    assert "oracle_symbols" not in first


def test_load_real_m1e_sidecar_if_present():
    from pathlib import Path

    import pytest

    from budget_coder_rl.eval.oracle import load_evaluator_oracle

    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "swe_gym"
        / "evaluator_oracle.parquet"
    )
    if not path.is_file():
        pytest.skip(f"missing sidecar {path}")
    index = load_evaluator_oracle(path)
    assert len(index) == 2438
    row = index.get("pydantic__pydantic-4882")
    assert row.base_changed_files
    assert row.split in {"train", "dev"}
