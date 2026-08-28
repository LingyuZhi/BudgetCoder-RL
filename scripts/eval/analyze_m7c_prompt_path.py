#!/usr/bin/env python
"""CPU M7C prompt/data-path audit. No GPU, no parser/prompt/reward change."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "smoke") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke"))

from budget_coder_rl.data.swe_gym_materialize import (  # noqa: E402
    TRAIN_PARQUET_RELPATH,
    DEV_PARQUET_RELPATH,
)
from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m7c import (  # noqa: E402
    EXPERIMENT_ID,
    N_AUDIT,
    SCHEMA_VERSION,
    audit_prompt_path,
    build_execution_contract,
    default_m7c_output_dir,
    forbidden_output_dir_errors,
    render_summary,
    scan_source_split_branches,
    subset_tasks,
    tokenizer_facts,
)
from smoke_rlhf_dataset import resolve_tokenizer_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--n-audit", type=int, default=N_AUDIT)
    parser.add_argument("--n-subset", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_m7c_output_dir(repo_root)
    blocked = forbidden_output_dir_errors(output_dir, repo_root)
    if blocked:
        print(f"HARD FAIL: {blocked}", file=sys.stderr)
        return 1
    train_parquet = repo_root / TRAIN_PARQUET_RELPATH
    dev_parquet = repo_root / DEV_PARQUET_RELPATH
    if not train_parquet.is_file() or not dev_parquet.is_file():
        print(
            f"HARD FAIL: missing M1E parquet {train_parquet} / {dev_parquet}",
            file=sys.stderr,
        )
        return 1
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    if not tokenizer_path:
        print("HARD FAIL: no local Qwen tokenizer; set BCRL_TOKENIZER_PATH", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = subset_tasks(repo_root=repo_root, n=int(args.n_subset))
    source_scan = scan_source_split_branches(repo_root)
    facts = tokenizer_facts(tokenizer, tokenizer_path)
    contract = build_execution_contract(
        tokenizer_facts=facts,
        source_scan=source_scan,
        subset={
            "n": subset["n"],
            "train_ids_sha256": subset["train_ids_sha256"],
            "dev_ids_sha256": subset["dev_ids_sha256"],
            "train_repo_counts": subset["train_repo_counts"],
            "dev_repo_counts": subset["dev_repo_counts"],
            "seed_formula": subset["seed_formula"],
        },
    )
    audit = audit_prompt_path(
        repo_root=repo_root,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        n_audit=int(args.n_audit),
    )
    write_json(output_dir / "execution_contract.json", contract)
    write_json(output_dir / "prompt_path_audit.json", audit)
    write_json(output_dir / "subset_manifest.json", subset)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_path_audit": audit,
        "execution_contract": contract,
        "train": {},
        "dev": {},
        "comparison": {},
        "decision": {
            "verdict": (
                "execution_path_confound_found" if not audit.get("allow_replay") else None
            ),
            "ambiguous_band": False,
            "primary_delta": None,
            "reasons": list(audit.get("confound_reasons") or []),
        },
        "q1": (
            "No. Unexplained model-visible train/dev difference: "
            f"{audit.get('confound_reasons')}"
            if not audit.get("allow_replay")
            else (
                "Yes on the inspected M7C path: same RLHFDataset / AgentLoop / "
                "build_stage1_messages / apply_chat_template kwargs. Synthetic "
                "same-issue train vs dev extra_info.split yields identical "
                "messages and prompt token IDs. extra_info.split is not model-visible."
            )
        ),
        "q2": (
            "Not run. GPU matched replay gated off."
            if not audit.get("allow_replay")
            else "Pending GPU matched Base replay (same execution, 64+64)."
        ),
    }
    payload["summary_markdown"] = render_summary(payload)
    (output_dir / "SUMMARY.md").write_text(payload["summary_markdown"], encoding="utf-8")
    write_json(output_dir / "audit_gate.json", {
        "experiment_id": EXPERIMENT_ID,
        "allow_replay": audit.get("allow_replay"),
        "confound_reasons": audit.get("confound_reasons"),
        "tokenizer_path": tokenizer_path,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
    })
    print(
        json_line(
            {
                "allow_replay": audit.get("allow_replay"),
                "n_leak_errors": len(audit.get("leak_errors") or []),
                "synthetic_equal": (audit.get("synthetic_equivalence") or {}).get(
                    "prompt_ids_equal"
                ),
                "output_dir": str(output_dir),
            }
        )
    )
    return 0 if audit.get("allow_replay") else 2


def json_line(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True)


if __name__ == "__main__":
    raise SystemExit(main())
