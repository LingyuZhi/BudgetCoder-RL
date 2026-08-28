#!/usr/bin/env python
"""CPU M7D first-request / sibling-expansion audit. No GPU, no training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "smoke") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "smoke"))

from budget_coder_rl.eval.m4b import write_json  # noqa: E402
from budget_coder_rl.eval.m7d import (  # noqa: E402
    EXPERIMENT_ID,
    N_SUBSET,
    SCHEMA_VERSION,
    audit_first_requests,
    build_execution_cells,
    default_m7d_output_dir,
    forbidden_output_dir_errors,
    render_summary,
    subset_tasks,
)
from smoke_rlhf_dataset import resolve_tokenizer_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--n-subset", type=int, default=N_SUBSET)
    return parser.parse_args(argv)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir) if args.output_dir else default_m7d_output_dir(repo_root)
    blocked = forbidden_output_dir_errors(output_dir, repo_root)
    if blocked:
        print(f"HARD FAIL: {blocked}", file=sys.stderr)
        return 1
    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    if not tokenizer_path:
        print("HARD FAIL: no local Qwen tokenizer; set BCRL_TOKENIZER_PATH", file=sys.stderr)
        return 1
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = subset_tasks(repo_root=repo_root, n=int(args.n_subset))
    cells = build_execution_cells()
    audit = audit_first_requests(
        repo_root=repo_root,
        tokenizer=tokenizer,
        n=int(args.n_subset),
    )
    records = list(audit.pop("records"))
    append_jsonl(output_dir / "first_request_audit.jsonl", records)
    write_json(output_dir / "subset_manifest.json", subset)
    write_json(output_dir / "execution_cells.json", cells)
    compact_audit = dict(audit)
    compact_audit["n_first_request_records"] = len(records)
    write_json(output_dir / "first_request_cpu.json", compact_audit)
    if audit.get("divergences"):
        write_json(output_dir / "first_divergence.json", audit["divergences"][0])
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "decision": {
            "verdict": (
                "first_request_divergence_found"
                if audit.get("divergences")
                else None
            ),
            "reasons": (
                [
                    f"{item.get('field')}: {item.get('why')}"
                    for item in (audit.get("divergences") or [])
                ]
                or [
                    "CPU first-request prompt_ids and sampling knobs are identical across A/B/C/D.",
                    "DataProto.repeat(np.repeat) aliases extra_info and raw_prompt among G=4 siblings; reported, not fixed.",
                    "GPU cells A/B/C then D are still required for the first-generation verdict.",
                ]
            ),
        },
        "prompt_identity": audit.get("prompt_identity"),
        "sampling_identity": audit.get("sampling_identity"),
        "aliasing": audit.get("aliasing"),
        "n_first_requests": len(records),
        "cells": {},
        "q_gpu": (
            "GPU cells gated off."
            if not audit.get("allow_gpu")
            else "Pending GPU cells A/B/C then D."
        ),
    }
    (output_dir / "SUMMARY.md").write_text(render_summary(summary_payload), encoding="utf-8")
    write_json(
        output_dir / "audit_gate.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "allow_gpu": audit.get("allow_gpu"),
            "n_divergences": len(audit.get("divergences") or []),
            "expansion_errors": audit.get("expansion_errors"),
            "aliasing": {
                "extra_info": (audit.get("aliasing") or {}).get("any_extra_info_aliased"),
                "raw_prompt": (audit.get("aliasing") or {}).get("any_raw_prompt_aliased"),
            },
            "tokenizer_path": tokenizer_path,
            "host": os.uname().nodename if hasattr(os, "uname") else "",
        },
    )
    print(
        json.dumps(
            {
                "allow_gpu": audit.get("allow_gpu"),
                "n_records": len(records),
                "prompt_identical": (audit.get("prompt_identity") or {}).get("identical"),
                "sampling_identical": (audit.get("sampling_identity") or {}).get("identical"),
                "extra_info_aliased": (audit.get("aliasing") or {}).get("any_extra_info_aliased"),
                "output_dir": str(output_dir),
            },
            ensure_ascii=True,
        )
    )
    return 0 if audit.get("allow_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
