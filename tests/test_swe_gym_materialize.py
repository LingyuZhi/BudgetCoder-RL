"""Unit tests for M1E veRL-ready SWE-Gym materialize.

Uses synthetic records and tmp parquet only. Does not read the official
SWE-Gym parquet, Git mirrors, or start GPU / Ray / vLLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from budget_coder_rl.data.swe_gym import REQUIRED_COLUMNS, write_json
from budget_coder_rl.data.swe_gym_audit import write_jsonl
from budget_coder_rl.data.swe_gym_fields import (
    POLICY_FORBIDDEN_DERIVED_FIELDS,
    POLICY_FORBIDDEN_DERIVED_FIELD_SET,
    PRIVILEGED_FIELDS,
    collect_forbidden_policy_keys,
    validate_policy_row_leakage,
    validate_policy_rows_leakage,
)
from budget_coder_rl.data.swe_gym_materialize import (
    DATA_SOURCE,
    SCHEMA_VERSION,
    MaterializeInputError,
    build_policy_row,
    build_sidecar_row,
    join_materialize_tables,
    logical_rows_sha256,
    manifest_json_bytes,
    materialize,
    project_oracle_symbols,
    schema_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(
    instance_id: str,
    *,
    problem: str = "fix it",
    repo: str = "owner/repo",
    hints: str = "do not use this hint",
    patch: str = "diff --git a/x b/x\n",
) -> dict:
    return {
        "instance_id": instance_id,
        "problem_statement": problem,
        "repo": repo,
        "base_commit": "abc123",
        "version": "1.0",
        "hints_text": hints,
        "patch": patch,
    }


def _oracle(instance_id: str, files: list[str] | None = None) -> dict:
    return {
        "instance_id": instance_id,
        "base_changed_files": list(files or ["src/foo.py"]),
        "gold_edit_files": [{"path": "src/foo.py", "operation": "modified"}],
    }


def _symbol(
    instance_id: str,
    symbols: list[dict] | None = None,
) -> dict:
    return {
        "instance_id": instance_id,
        "oracle_symbols": list(symbols or []),
        "oracle_symbol_count": len(symbols or []),
    }


def _assignment(instance_id: str, split: str, repo: str = "owner/repo") -> dict:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "correlation_group_id": f"cg:{instance_id}",
        "split": split,
    }


def _split_manifest(assignments: list[dict]) -> dict:
    train = sum(1 for item in assignments if item["split"] == "train")
    dev = sum(1 for item in assignments if item["split"] == "dev")
    return {
        "actual_train_rows": train,
        "actual_dev_rows": dev,
        "assignments": assignments,
    }


def _join_ok(
    ids_train: list[str],
    ids_dev: list[str],
    *,
    symbols: dict[str, list[dict]] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    all_ids = ids_train + ids_dev
    sources = [
        _source(instance_id, problem=f"issue {instance_id}", hints=f"hint-{instance_id}")
        for instance_id in all_ids
    ]
    oracles = [_oracle(instance_id) for instance_id in all_ids]
    symbol_records = [
        _symbol(instance_id, (symbols or {}).get(instance_id, []))
        for instance_id in all_ids
    ]
    assignments = [_assignment(instance_id, "train") for instance_id in ids_train] + [
        _assignment(instance_id, "dev") for instance_id in ids_dev
    ]
    return join_materialize_tables(
        source_records=sources,
        oracle_records=oracles,
        symbol_records=symbol_records,
        split_manifest=_split_manifest(assignments),
        expected_n_rows=len(all_ids),
        expected_train_rows=len(ids_train),
        expected_dev_rows=len(ids_dev),
    )


def test_exact_split_join_counts_and_disjoint():
    train_rows, dev_rows, oracle_rows = _join_ok(
        ["owner__z", "owner__a"],
        ["owner__m"],
        symbols={
            "owner__a": [{"path": "src/foo.py", "qualname": "foo"}],
            "owner__z": [],
            "owner__m": [{"path": "src/bar.py", "qualname": "Bar.baz"}],
        },
    )
    assert len(train_rows) == 2
    assert len(dev_rows) == 1
    assert len(oracle_rows) == 3
    train_ids = {row["extra_info"]["instance_id"] for row in train_rows}
    dev_ids = {row["extra_info"]["instance_id"] for row in dev_rows}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids | dev_ids == {"owner__z", "owner__a", "owner__m"}


def test_duplicate_instance_fails():
    sources = [_source("owner__1"), _source("owner__1")]
    with pytest.raises(MaterializeInputError, match="duplicate"):
        join_materialize_tables(
            source_records=sources,
            oracle_records=[_oracle("owner__1")],
            symbol_records=[_symbol("owner__1")],
            split_manifest=_split_manifest([_assignment("owner__1", "train")]),
            expected_n_rows=1,
            expected_train_rows=1,
            expected_dev_rows=0,
        )


def test_missing_instance_fails():
    with pytest.raises(MaterializeInputError, match="missing"):
        join_materialize_tables(
            source_records=[_source("owner__1")],
            oracle_records=[_oracle("owner__1")],
            symbol_records=[_symbol("owner__1")],
            split_manifest=_split_manifest(
                [_assignment("owner__1", "train"), _assignment("owner__2", "dev")]
            ),
            expected_n_rows=2,
            expected_train_rows=1,
            expected_dev_rows=1,
        )


def test_unexpected_instance_fails():
    with pytest.raises(MaterializeInputError, match="unexpected"):
        join_materialize_tables(
            source_records=[_source("owner__1"), _source("owner__extra")],
            oracle_records=[_oracle("owner__1"), _oracle("owner__extra")],
            symbol_records=[_symbol("owner__1"), _symbol("owner__extra")],
            split_manifest=_split_manifest([_assignment("owner__1", "train")]),
            expected_n_rows=1,
            expected_train_rows=1,
            expected_dev_rows=0,
        )


def test_train_dev_disjoint_assignment_fail():
    with pytest.raises(MaterializeInputError, match="duplicate"):
        join_materialize_tables(
            source_records=[_source("owner__1")],
            oracle_records=[_oracle("owner__1")],
            symbol_records=[_symbol("owner__1")],
            split_manifest={
                "actual_train_rows": 1,
                "actual_dev_rows": 1,
                "assignments": [
                    _assignment("owner__1", "train"),
                    {
                        "instance_id": "owner__1",
                        "repo": "owner/repo",
                        "correlation_group_id": "cg:dup",
                        "split": "dev",
                    },
                ],
            },
            expected_n_rows=1,
            expected_train_rows=1,
            expected_dev_rows=0,
        )


def test_policy_prompt_is_exact_problem_statement():
    train_rows, _dev_rows, _oracle_rows = _join_ok(
        ["owner__a"],
        ["owner__b"],
    )
    row = train_rows[0]
    assert row["prompt"] == [{"role": "user", "content": "issue owner__a"}]
    assert "hint-" not in row["prompt"][0]["content"]
    assert row["data_source"] == DATA_SOURCE


def test_hints_do_not_enter_prompt_or_policy_keys():
    source = _source("owner__1", problem="the issue", hints="SECRET_HINT")
    row = build_policy_row(source, split="train", index=0)
    dumped = json.dumps(row)
    assert "SECRET_HINT" not in dumped
    assert "hints_text" not in dumped
    assert row["prompt"][0]["content"] == "the issue"
    assert validate_policy_row_leakage(row, problem_statement="the issue") == []


def test_privileged_raw_fields_absent_from_policy_row():
    source = _source("owner__1")
    row = build_policy_row(source, split="dev", index=0)
    blob = json.dumps(row)
    for name in PRIVILEGED_FIELDS:
        assert name not in blob
    assert collect_forbidden_policy_keys(row) == []


def test_derived_oracle_and_difficulty_fields_absent_from_policy():
    train_rows, _dev, _oracle = _join_ok(["owner__a"], ["owner__b"])
    blob = json.dumps(train_rows)
    for name in (
        "base_changed_files",
        "gold_edit_files",
        "oracle_symbols",
        "oracle_symbol_count",
        "base_changed_file_count",
        "file_target_density",
        "gold_full_path_mentioned",
    ):
        assert name not in blob
        assert name in POLICY_FORBIDDEN_DERIVED_FIELD_SET


def test_sidecar_contains_file_oracle_and_symbol_flags():
    _train, _dev, oracle_rows = _join_ok(
        ["owner__a"],
        ["owner__b"],
        symbols={
            "owner__a": [
                {
                    "path": "src/foo.py",
                    "qualname": "Foo.bar",
                    "kind": "function",
                    "start_line": 1,
                    "end_line": 4,
                }
            ],
            "owner__b": [],
        },
    )
    by_id = {item["instance_id"]: item for item in oracle_rows}
    assert by_id["owner__a"]["base_changed_files"] == ["src/foo.py"]
    assert by_id["owner__a"]["oracle_symbols"] == [
        {"path": "src/foo.py", "qualname": "Foo.bar"}
    ]
    assert by_id["owner__a"]["symbol_applicable"] is True
    assert by_id["owner__b"]["oracle_symbols"] == []
    assert by_id["owner__b"]["symbol_applicable"] is False
    assert "kind" not in by_id["owner__a"]["oracle_symbols"][0]
    assert "gold_edit_files" not in by_id["owner__a"]


def test_symbol_identity_dedup_and_sort():
    projected = project_oracle_symbols(
        [
            {"path": "b.py", "qualname": "z", "kind": "function", "start_line": 10},
            {"path": "a.py", "qualname": "a", "kind": "class", "start_line": 1},
            {"path": "b.py", "qualname": "z", "kind": "function", "start_line": 99},
        ]
    )
    assert projected == [
        {"path": "a.py", "qualname": "a"},
        {"path": "b.py", "qualname": "z"},
    ]


def test_opaque_ground_truth_is_instance_id():
    row = build_policy_row(_source("owner__99"), split="train", index=3)
    assert row["reward_model"] == {"style": "rule", "ground_truth": "owner__99"}
    sidecar = build_sidecar_row(
        instance_id="owner__99",
        split="train",
        oracle=_oracle("owner__99", ["src/a.py", "src/b.py"]),
        symbol=_symbol("owner__99"),
    )
    assert "reward_model" not in sidecar
    assert sidecar["instance_id"] == "owner__99"


def test_deterministic_row_ordering_and_index():
    train_rows, dev_rows, oracle_rows = _join_ok(
        ["owner__z", "owner__a", "owner__m"],
        ["owner__b"],
    )
    assert [row["extra_info"]["instance_id"] for row in train_rows] == [
        "owner__a",
        "owner__m",
        "owner__z",
    ]
    assert [row["extra_info"]["index"] for row in train_rows] == [0, 1, 2]
    assert dev_rows[0]["extra_info"]["instance_id"] == "owner__b"
    assert dev_rows[0]["extra_info"]["index"] == 0
    assert [row["instance_id"] for row in oracle_rows] == [
        "owner__a",
        "owner__b",
        "owner__m",
        "owner__z",
    ]


def test_schema_and_manifest_json_are_deterministic():
    first = manifest_json_bytes(schema_record())
    second = manifest_json_bytes(schema_record())
    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["agent_name"]["present_in_dataset"] is False
    train_rows, _dev, _oracle = _join_ok(["owner__a"], ["owner__b"])
    assert logical_rows_sha256(train_rows) == logical_rows_sha256(train_rows)


def test_leakage_validator_catches_nested_forbidden_field():
    row = build_policy_row(_source("owner__1", problem="ps"), split="train", index=0)
    assert validate_policy_row_leakage(row, problem_statement="ps") == []
    leaked = json.loads(json.dumps(row))
    leaked["extra_info"]["oracle_symbols"] = [{"path": "x.py", "qualname": "x"}]
    errors = collect_forbidden_policy_keys(leaked)
    assert any("oracle_symbols" in item for item in errors)
    prompt_wrong = json.loads(json.dumps(row))
    prompt_wrong["prompt"][0]["content"] = "ps\nhint"
    prompt_errors = validate_policy_row_leakage(
        prompt_wrong, problem_statement="ps", instance_id="owner__1"
    )
    assert any("problem_statement" in item for item in prompt_errors)


def test_leakage_validator_catches_nested_list_of_dicts():
    payload = {
        "extra_info": {
            "nested": [
                {"ok": 1},
                {"hints_text": "secret", "patch": "diff"},
            ]
        }
    }
    errors = collect_forbidden_policy_keys(payload)
    assert any("hints_text" in item for item in errors)
    assert any("patch" in item for item in errors)


def test_wrong_prompt_fails_row_validator():
    row = build_policy_row(_source("owner__1", problem="alpha"), split="train", index=0)
    errors = validate_policy_rows_leakage(
        [row], problem_statements={"owner__1": "beta"}
    )
    assert errors


def _write_raw_parquet(path: Path, rows: list[dict]) -> None:
    records = []
    for row in rows:
        record = {name: "" for name in REQUIRED_COLUMNS}
        record.update(row)
        record.setdefault("FAIL_TO_PASS", [])
        record.setdefault("PASS_TO_PASS", [])
        records.append(record)
    pd.DataFrame(records).to_parquet(path, index=False)


def test_materialize_end_to_end_tmp(tmp_path: Path):
    raw = tmp_path / "raw.parquet"
    _write_raw_parquet(
        raw,
        [
            {
                "instance_id": "owner__b",
                "problem_statement": "issue b",
                "repo": "owner/repo",
                "base_commit": "bbb",
                "version": "1",
                "hints_text": "HINT_B",
                "patch": "PATCH_B",
                "created_at": "2020-01-01",
            },
            {
                "instance_id": "owner__a",
                "problem_statement": "issue a",
                "repo": "owner/repo",
                "base_commit": "aaa",
                "version": "1",
                "hints_text": "HINT_A",
                "patch": "PATCH_A",
                "created_at": "2020-01-02",
            },
            {
                "instance_id": "owner__c",
                "problem_statement": "issue c",
                "repo": "owner/other",
                "base_commit": "ccc",
                "version": "1",
                "hints_text": "",
                "patch": "PATCH_C",
                "created_at": "2020-01-03",
            },
        ],
    )
    split = tmp_path / "split.json"
    write_json(
        split,
        _split_manifest(
            [
                _assignment("owner__a", "train"),
                _assignment("owner__b", "dev"),
                _assignment("owner__c", "train", repo="owner/other"),
            ]
        ),
    )
    oracle_jsonl = tmp_path / "oracle.jsonl"
    write_jsonl(
        oracle_jsonl,
        [
            _oracle("owner__a", ["src/a.py"]),
            _oracle("owner__b", ["src/b.py"]),
            _oracle("owner__c", ["src/c.py"]),
        ],
    )
    symbol_jsonl = tmp_path / "symbol.jsonl"
    write_jsonl(
        symbol_jsonl,
        [
            _symbol(
                "owner__a",
                [{"path": "src/a.py", "qualname": "a_fn", "kind": "function"}],
            ),
            _symbol("owner__b", []),
            _symbol(
                "owner__c",
                [{"path": "src/c.py", "qualname": "C.meth", "kind": "method"}],
            ),
        ],
    )
    result = materialize(
        repo_root=REPO_ROOT,
        raw_parquet=raw,
        split_json=split,
        oracle_jsonl=oracle_jsonl,
        symbol_jsonl=symbol_jsonl,
        train_out=tmp_path / "train.parquet",
        dev_out=tmp_path / "dev.parquet",
        oracle_out=tmp_path / "oracle.parquet",
        schema_out=tmp_path / "schema.json",
        manifest_out=tmp_path / "manifest.json",
        expected_n_rows=3,
        expected_n_repos=2,
        expected_train_rows=2,
        expected_dev_rows=1,
        verify_raw_identity=False,
        check_field_policy=True,
    )
    manifest = result["manifest"]
    assert manifest["split_checks"]["train_rows"] == 2
    assert manifest["split_checks"]["dev_rows"] == 1
    assert manifest["symbol_applicable"]["true"] == 2
    assert manifest["symbol_applicable"]["false"] == 1
    train = pd.read_parquet(result["train_path"])
    assert list(train.columns) == [
        "data_source",
        "prompt",
        "reward_model",
        "extra_info",
    ]
    blob = train.to_json()
    assert "HINT_A" not in blob
    assert "PATCH_A" not in blob
    assert "base_changed_files" not in blob
    extra0 = train.iloc[0]["extra_info"]
    if isinstance(extra0, str):
        extra0 = json.loads(extra0)
    assert extra0["instance_id"] == "owner__a"
    sidecar = pd.read_parquet(result["oracle_path"])
    assert len(sidecar) == 3
    assert set(sidecar.columns) == {
        "instance_id",
        "split",
        "base_changed_files",
        "oracle_symbols",
        "symbol_applicable",
    }


def test_forbidden_constant_covers_required_names():
    required = {
        "hints_text",
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "base_changed_files",
        "gold_edit_files",
        "oracle_symbols",
        "oracle_symbol_count",
        "base_changed_file_count",
    }
    assert required <= set(POLICY_FORBIDDEN_DERIVED_FIELDS)


def _resolve_tokenizer_path() -> str | None:
    import os

    env_path = os.environ.get("BCRL_TOKENIZER_PATH")
    if env_path:
        return env_path
    data_root = Path(
        os.environ.get("BCRL_DATA_ROOT", os.path.expanduser("~/my_data/budget-coder-rl"))
    )
    candidates: list[Path] = []
    if (data_root / "models").is_dir():
        candidates.extend(sorted((data_root / "models").glob("*")))
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    for repo_dir in sorted(hub.glob("models--Qwen--*")):
        candidates.extend(sorted((repo_dir / "snapshots").glob("*")))
    for cand in candidates:
        if (cand / "tokenizer_config.json").exists():
            return str(cand)
    return None


def test_tiny_rlhfdataset_optional(tmp_path: Path):
    pytest.importorskip("verl.utils.dataset.rl_dataset")
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset

    raw = tmp_path / "raw.parquet"
    _write_raw_parquet(
        raw,
        [
            {
                "instance_id": "owner__a",
                "problem_statement": "issue a",
                "repo": "owner/repo",
                "base_commit": "aaa",
                "version": "1",
                "hints_text": "HINT",
                "patch": "PATCH",
                "created_at": "2020-01-01",
            },
            {
                "instance_id": "owner__b",
                "problem_statement": "issue b",
                "repo": "owner/repo",
                "base_commit": "bbb",
                "version": "1",
                "hints_text": "",
                "patch": "PATCH",
                "created_at": "2020-01-02",
            },
        ],
    )
    write_json(
        tmp_path / "split.json",
        _split_manifest(
            [_assignment("owner__a", "train"), _assignment("owner__b", "dev")]
        ),
    )
    write_jsonl(tmp_path / "oracle.jsonl", [_oracle("owner__a"), _oracle("owner__b")])
    write_jsonl(tmp_path / "symbol.jsonl", [_symbol("owner__a"), _symbol("owner__b")])
    result = materialize(
        repo_root=REPO_ROOT,
        raw_parquet=raw,
        split_json=tmp_path / "split.json",
        oracle_jsonl=tmp_path / "oracle.jsonl",
        symbol_jsonl=tmp_path / "symbol.jsonl",
        train_out=tmp_path / "train.parquet",
        dev_out=tmp_path / "dev.parquet",
        oracle_out=tmp_path / "oracle.parquet",
        schema_out=tmp_path / "schema.json",
        manifest_out=tmp_path / "manifest.json",
        expected_n_rows=2,
        expected_n_repos=1,
        expected_train_rows=1,
        expected_dev_rows=1,
        verify_raw_identity=False,
        check_field_policy=True,
    )
    tokenizer_path = _resolve_tokenizer_path()
    if tokenizer_path is None:
        tokenizer = type("StubTokenizer", (), {})()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "return_raw_chat": True,
            "filter_overlong_prompts": False,
            "cache_dir": str(tmp_path / "cache"),
        }
    )
    dataset = RLHFDataset(
        data_files=str(result["train_path"]),
        tokenizer=tokenizer,
        config=config,
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item["raw_prompt"] == [{"role": "user", "content": "issue a"}]
    extra = item["extra_info"]
    assert extra["instance_id"] == "owner__a"
    assert extra["repo"] == "owner/repo"
    assert extra["base_commit"] == "aaa"
    assert collect_forbidden_policy_keys(item["extra_info"]) == []
    assert collect_forbidden_policy_keys({"prompt": item["prompt"]}) == []
