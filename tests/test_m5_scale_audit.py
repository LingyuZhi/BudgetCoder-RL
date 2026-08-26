"""CPU tests for the M5 scale-correction audit.

Uses frozen git-tracked manifests. Does not start GPU / Ray / vLLM.
Live oracle/parquet replay is skipped when those artifacts are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

from budget_coder_rl.data.swe_gym import sha256_file
from budget_coder_rl.eval.m3b import repo_round_robin_ids, sha256_ids
from budget_coder_rl.eval.m3c import (
    OVERLONG_INSTANCE_IDS,
    TRAIN_CANDIDATE_TARGET_N,
    default_candidate_path,
)
from budget_coder_rl.eval.m4a import load_json
from budget_coder_rl.eval.m5_scale_audit import (
    EXPECTED_CANDIDATE_FILE_SHA256,
    EXPECTED_ELIGIBLE_IDS_SHA256,
    EXPECTED_M1D_SPLIT_SHA256,
    EXPECTED_M1E_MANIFEST_SHA256,
    EXPECTED_M3C_FREEZE_SHA256,
    EXPECTED_M5_MAIN_SHA256,
    EXPECTED_ORDERED_IDS_SHA256,
    FREEZE_RELPATH,
    M1D_SPLIT_RELPATH,
    M1E_MANIFEST_RELPATH,
    M5_MAIN_RELPATH,
    PPO_MAX_TOKEN_LEN,
    PROMPT_LENGTH,
    TRAIN_BATCH_SIZE,
    classify_historical_exclusions,
    dataloader_semantics,
    pad_ordered_ids,
    reconstruct_eligible_from_skipped,
    run_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_candidate_arithmetic_and_hashes():
    path = default_candidate_path(REPO_ROOT)
    payload = load_json(path)
    skipped = payload["skipped"]
    n_overlong = len(skipped["overlong_prompt"])
    n_symbol = len(skipped["symbol_unavailable"])
    n_missing = len(skipped["missing_oracle"])
    assert payload["n_universe"] == 2194
    assert payload["n_eligible"] == 2048
    assert payload["n_selected"] == TRAIN_CANDIDATE_TARGET_N == 256
    assert n_overlong == 1
    assert n_symbol == 145
    assert n_missing == 0
    assert 2194 - 1 - 145 - 0 == 2048
    assert 2048 - 256 == 1792
    assert skipped["overlong_prompt"] == ["Project-MONAI__MONAI-6344"]
    assert OVERLONG_INSTANCE_IDS == frozenset({"Project-MONAI__MONAI-6344"})
    assert payload["ordered_ids_sha256"] == EXPECTED_ORDERED_IDS_SHA256
    assert payload["eligible_ids_sha256"] == EXPECTED_ELIGIBLE_IDS_SHA256
    assert sha256_ids(payload["ordered_ids"]) == EXPECTED_ORDERED_IDS_SHA256
    assert sha256_file(path) == EXPECTED_CANDIDATE_FILE_SHA256
    selected = set(payload["ordered_ids"])
    assert selected.isdisjoint(set(skipped["symbol_unavailable"]))
    assert "Project-MONAI__MONAI-6344" not in selected


def test_frozen_contract_files_not_rewritten():
    assert sha256_file(REPO_ROOT / FREEZE_RELPATH) == EXPECTED_M3C_FREEZE_SHA256
    assert sha256_file(REPO_ROOT / M5_MAIN_RELPATH) == EXPECTED_M5_MAIN_SHA256
    assert sha256_file(REPO_ROOT / M1E_MANIFEST_RELPATH) == EXPECTED_M1E_MANIFEST_SHA256
    assert sha256_file(REPO_ROOT / M1D_SPLIT_RELPATH) == EXPECTED_M1D_SPLIT_SHA256


def test_pad_ordered_ids_prefix_repeat():
    even = [f"id{i}" for i in range(16)]
    even_pad = pad_ordered_ids(even, batch_size=8)
    assert even_pad["n_pad"] == 0
    assert even_pad["n_rows_padded"] == 16
    assert even_pad["optimizer_steps"] == 2
    assert even_pad["padded_ids"] == even

    n2193 = [f"id{i:04d}" for i in range(2193)]
    pad = pad_ordered_ids(n2193, batch_size=8)
    assert 2193 % 8 == 1
    assert pad["n_pad"] == 7
    assert pad["n_rows_padded"] == 2200
    assert pad["optimizer_steps"] == 275
    assert pad["pad_ids"] == n2193[:7]
    assert pad["padded_ids"][-7:] == n2193[:7]
    assert pad["silent_drop_if_unpadded_drop_last"] == 1
    assert pad["padded_ids_sha256"] != pad["unique_ids_sha256"]


def test_classify_historical_matches_frozen_counts():
    candidate = load_json(default_candidate_path(REPO_ROOT))
    split = load_json(REPO_ROOT / M1D_SPLIT_RELPATH)
    train_ids = [
        str(item["instance_id"])
        for item in split["assignments"]
        if item.get("split") == "train"
    ]
    rows = [
        {"instance_id": str(item["instance_id"]), "repo": str(item["repo"])}
        for item in split["assignments"]
        if item.get("split") == "train"
    ]
    skipped = candidate["skipped"]
    eligible = reconstruct_eligible_from_skipped(rows, skipped)
    eligible_ordered = [
        iid for iid in repo_round_robin_ids(rows) if iid in set(eligible)
    ]
    classified = classify_historical_exclusions(
        train_ids=train_ids,
        skipped=skipped,
        selected=candidate["ordered_ids"],
        eligible=eligible_ordered,
        hard_unusable=skipped["overlong_prompt"],
    )
    assert classified["A_genuine_hard_unusable"]["n"] == 1
    assert classified["B_old_systems_limitation"]["n"] == 0
    assert classified["C_valid_symbol_unavailable"]["n"] == 145
    assert classified["D_prototype_compute_exclusion"]["n"] == 1792
    assert classified["unclassified_leftover"] == []
    assert classified["n_historical_excluded_from_256"] == 2194 - 256
    assert sha256_ids(eligible_ordered) == EXPECTED_ELIGIBLE_IDS_SHA256


def test_dataloader_semantics_forbids_silent_drop():
    payload = dataloader_semantics()
    assert payload["train_dataloader"]["drop_last"] is True
    assert payload["recommendation"]["do_not_patch_verl"] is True
    assert payload["recommendation"]["do_not_silent_drop"] is True
    assert payload["recommendation"]["partial_final_batch"].startswith("rejected")
    assert payload["sampler"]["shuffle_false"] == "torch.utils.data.SequentialSampler"


def test_run_audit_cpu_from_frozen_manifests():
    payload = run_audit(REPO_ROOT, prompt_tokens_by_id=None, require_live_oracle=False)
    assert payload["errors"] == []
    assert payload["READY_FOR_SCALED_M5_DESIGN"] is True
    primary = payload["primary_pool"]
    assert primary["n_unique"] == 2193
    assert primary["excluded_from_m1e_train"] == ["Project-MONAI__MONAI-6344"]
    assert 2193 % TRAIN_BATCH_SIZE == 1
    assert primary["pad"]["n_pad"] == 7
    assert primary["pad"]["n_rows_padded"] == 2200
    assert primary["pad"]["optimizer_steps"] == 275
    assert primary["stats"]["old_256_coverage_n"] == 256
    assert primary["stats"]["old_256_coverage_ratio"] == 1.0
    assert payload["fallback_pool"]["n_unique"] == 2048
    assert payload["symbol_applicable_resolution"]["n_selected"] == 256
    assert payload["symbol_applicable_resolution"]["n_selected_symbol_applicable_false"] == 0
    assert payload["m3c_replay"]["n_symbol_unavailable"] == 145
    assert PROMPT_LENGTH == 16384
    assert PPO_MAX_TOKEN_LEN == 20480
    freeze = json.loads((REPO_ROOT / FREEZE_RELPATH).read_text(encoding="utf-8"))
    assert freeze["localization_reward"]["symbol_unavailable"].startswith("file-only")
    assert freeze["envelope"]["prompt_length"] == PROMPT_LENGTH
