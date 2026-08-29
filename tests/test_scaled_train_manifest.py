"""CPU tests for scaled-M5 contract freeze (hash padding, not prefix-repeat).

Does not start GPU / Ray / vLLM. Does not edit frozen M3C/M5/E014 files.
"""

from __future__ import annotations

from pathlib import Path

from budget_coder_rl.eval.m3b import repo_round_robin_ids, sha256_ids
from budget_coder_rl.eval.m3c import OVERLONG_INSTANCE_IDS, default_candidate_path
from budget_coder_rl.eval.m4a import GROUP_N, load_json
from budget_coder_rl.eval.m5_scale_audit import (
    EXPECTED_CANDIDATE_FILE_SHA256,
    EXPECTED_M3C_FREEZE_SHA256,
    EXPECTED_M5_MAIN_SHA256,
    M1D_SPLIT_RELPATH,
    load_m1d_train_rows,
    pad_ordered_ids,
)
from budget_coder_rl.eval.m5_scaled import (
    CANDIDATE_SCHEMA,
    CONTRACT_SCHEMA,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    EXPECTED_PAD_IDS,
    EXPECTED_PADDED_IDS_SHA256,
    EXPECTED_UNIQUE_IDS_SHA256,
    GROUP_N as SCALED_GROUP_N,
    MAIN_STEPS,
    N_PAD,
    N_ROWS,
    N_UNIQUE,
    PADDING_SALT,
    PREFIX_REPEAT_PAD_IDS,
    PREFLIGHT_MAX_STEPS,
    PREFLIGHT_STEPS,
    SAVE_FREQ,
    SYMBOL_UNAVAILABLE_IN_FIRST_BATCH,
    TRAIN_BATCH_SIZE,
    build_preflight_overlay,
    build_scaled_contract,
    build_scaled_train_manifest,
    checkpoint_audit,
    consume_scaled_errors,
    default_candidate_path as scaled_candidate_path,
    default_contract_path,
    historical_untouched_errors,
    load_padded_ids,
    load_unique_ids,
    manifest_errors,
    pad_unique_ids,
    padding_key,
    preflight_overlay_errors,
    save_event_steps,
    scaled_contract_errors,
    select_hash_padding_ids,
)
from budget_coder_rl.eval.m5b import EXPECTED_M3C_SHA256
from budget_coder_rl.eval.provenance import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _m1d_unique_ordered() -> list[str]:
    rows = load_m1d_train_rows(REPO_ROOT / M1D_SPLIT_RELPATH)
    blocked = frozenset(str(item) for item in OVERLONG_INSTANCE_IDS)
    return [
        iid
        for iid in repo_round_robin_ids(rows)
        if iid not in blocked
    ]


def test_historical_freeze_files_untouched():
    assert historical_untouched_errors(REPO_ROOT) == []
    assert sha256_file(REPO_ROOT / "configs/historical/stage1_m3c_freeze.json") == EXPECTED_M3C_FREEZE_SHA256
    assert sha256_file(REPO_ROOT / "configs/historical/stage1_m5_main.json") == EXPECTED_M5_MAIN_SHA256
    assert sha256_file(default_candidate_path(REPO_ROOT)) == EXPECTED_CANDIDATE_FILE_SHA256
    assert EXPECTED_M3C_SHA256 == EXPECTED_M3C_FREEZE_SHA256


def test_hash_padding_is_not_prefix_repeat():
    unique = _m1d_unique_ordered()
    assert len(unique) == N_UNIQUE
    assert sha256_ids(unique) == EXPECTED_UNIQUE_IDS_SHA256
    pad = pad_unique_ids(unique)
    assert pad["n_pad"] == N_PAD
    assert pad["n_rows_padded"] == N_ROWS
    assert pad["optimizer_steps"] == MAIN_STEPS
    assert pad["remainder_if_unpadded"] == 1
    assert pad["silent_drop_if_unpadded_drop_last"] == 1
    assert tuple(pad["pad_ids"]) == EXPECTED_PAD_IDS
    assert pad["pad_ids"] != list(PREFIX_REPEAT_PAD_IDS)
    prefix = pad_ordered_ids(unique, batch_size=TRAIN_BATCH_SIZE)
    assert prefix["pad_ids"] == list(PREFIX_REPEAT_PAD_IDS)
    assert pad["padded_ids_sha256"] != prefix["padded_ids_sha256"]
    assert pad["padded_ids_sha256"] == EXPECTED_PADDED_IDS_SHA256
    assert N_ROWS % TRAIN_BATCH_SIZE == 0
    assert N_ROWS // TRAIN_BATCH_SIZE == MAIN_STEPS
    replay = select_hash_padding_ids(unique)
    assert replay == pad["pad_ids"]
    ranked = sorted(unique, key=lambda item: (padding_key(item, salt=PADDING_SALT), item))
    assert ranked[:7] == pad["pad_ids"]


def test_build_scaled_manifest_from_m1d():
    rows = load_m1d_train_rows(REPO_ROOT / M1D_SPLIT_RELPATH)
    payload = build_scaled_train_manifest(
        rows,
        repo_root=REPO_ROOT,
        oracle=None,
        identity_source="m1d_split",
    )
    assert payload["schema_version"] == CANDIDATE_SCHEMA
    assert manifest_errors(payload) == []
    assert payload["n_unique"] == 2193
    assert payload["n_rows"] == 2200
    assert payload["symbol_applicable_false"] == 145
    assert payload["symbol_applicable_true"] == 2048
    assert payload["old_256_coverage_n"] == 256
    assert payload["old_256_is_prefix"] is False
    assert payload["oracle_used_as_drop"] is False
    assert payload["gold_used_for_cherry_pick"] is False
    assert payload["reward_used_for_selection"] is False
    assert payload["excluded_from_m1e_train"] == ["Project-MONAI__MONAI-6344"]
    old = load_json(default_candidate_path(REPO_ROOT))["ordered_ids"]
    assert set(old).issubset(set(payload["ordered_ids"]))
    assert SYMBOL_UNAVAILABLE_IN_FIRST_BATCH in payload["ordered_ids"][:8]
    assert payload["group_n"] == GROUP_N == SCALED_GROUP_N == 4
    assert payload["expected_trajectories"] == 8800


def test_written_scaled_manifest_and_contract():
    path = scaled_candidate_path(REPO_ROOT)
    assert path.is_file()
    payload = load_json(path)
    assert manifest_errors(payload) == []
    assert load_unique_ids(path) == payload["ordered_ids"]
    assert load_padded_ids(path) == payload["padded_ids"]
    assert len(load_padded_ids(path)) == 2200
    contract_path = default_contract_path(REPO_ROOT)
    contract = load_json(contract_path)
    assert contract["schema_version"] == CONTRACT_SCHEMA
    assert scaled_contract_errors(contract) == []
    newly = contract["newly_frozen"]
    assert newly["algorithm"]["rollout_n"] == 4
    assert newly["trainer"]["total_training_steps"] == 275
    assert newly["actor"]["ppo_max_token_len_per_gpu"] == 20480
    assert newly["gpu"]["n_gpus"] == 2
    assert newly["trainer"]["save_freq"] == SAVE_FREQ == 32
    freeze = load_json(REPO_ROOT / "configs/historical/stage1_m3c_freeze.json")
    rebuilt = build_scaled_contract(
        freeze=freeze,
        freeze_path=REPO_ROOT / "configs/historical/stage1_m3c_freeze.json",
        candidate=payload,
        candidate_path=path,
        envelope_path=REPO_ROOT
        / "configs/historical/stage1_canonical_execution_envelope.json",
        project_commit=None,
    )
    assert scaled_contract_errors(rebuilt) == []
    overlay = build_preflight_overlay(
        output_dir=REPO_ROOT / "outputs/experiments/E016",
        checkpoint_dir=Path("/tmp/stage1_m5_scaled_e016"),
        n_steps=PREFLIGHT_STEPS,
    )
    assert preflight_overlay_errors(overlay, contract=contract) == []
    assert overlay["n_preflight_steps"] == PREFLIGHT_STEPS
    assert overlay["overrides"]["trainer"]["total_training_steps"] == 2
    assert PREFLIGHT_STEPS <= PREFLIGHT_MAX_STEPS
    assert sha256_file(path) == EXPECTED_MANIFEST_FILE_SHA256
    assert sha256_file(contract_path) == EXPECTED_CONTRACT_SHA256
    assert consume_scaled_errors(REPO_ROOT) == []


def test_checkpoint_cadence_and_preflight_cap():
    audit = checkpoint_audit()
    assert audit["save_freq"] == 32
    assert audit["n_save_events"] == 9
    assert audit["save_events"][-1] == 275
    assert 275 in audit["save_events"]
    assert audit["does_not_delete_parent_global_step_dir"] is True
    assert audit["storage_bound_unpruned_gib"] == 82.8
    assert save_event_steps(275, 32) == [
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        275,
    ]
    overlay = build_preflight_overlay(
        output_dir=REPO_ROOT / "outputs/experiments/E016",
        checkpoint_dir=Path("/tmp/stage1_m5_scaled_e016"),
        n_steps=2,
    )
    assert overlay["do_not_start_275"] is True
    try:
        build_preflight_overlay(
            output_dir=REPO_ROOT / "outputs/experiments/E016",
            checkpoint_dir=Path("/tmp/x"),
            n_steps=3,
        )
        raise AssertionError("steps=3 must fail")
    except ValueError:
        pass
