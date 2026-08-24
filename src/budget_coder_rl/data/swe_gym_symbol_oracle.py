"""SWE-Gym base-repository function/class symbol oracles (M1C-B).

Consumes M1C-A file/hunk semantics without changing them. Symbol oracles
are privileged evaluator metadata and must not enter the agent task view.
Every input row is retained. Missing commits/blobs are reported, not dropped.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from budget_coder_rl.data.swe_gym import (
    EXPECTED_SHA256,
    HF_REPO_ID,
    HF_REVISION,
    is_null,
    length_stats,
)
from budget_coder_rl.data.swe_gym_oracle import (
    SPOTLIGHT_INSTANCE_ID,
    extract_oracle_from_patch,
    file_extension,
)
from budget_coder_rl.data.swe_gym_repos import (
    CACHE_RELPATH,
    BlobStore,
    directory_size_bytes,
    is_safe_repo_path,
)
from budget_coder_rl.data.swe_gym_symbols import (
    ELIGIBLE_EXTENSIONS,
    FILE_STATUS_OK,
    REASON_AMBIGUOUS,
    REASON_AST,
    REASON_DECODE,
    REASON_INVALID_PATH,
    REASON_MISSING_BLOB,
    REASON_MISSING_COMMIT,
    REASON_MISSING_REPO,
    REASON_MODULE_LEVEL,
    REASON_NO_ENCLOSING,
    REASON_UNSUPPORTED,
    addition_runs_by_source_path,
    is_eligible_path,
    map_file_change_sites,
    parser_version_record,
    symbols_from_blob,
)
UNMAPPED_REASONS: tuple[str, ...] = (
    REASON_MODULE_LEVEL,
    REASON_NO_ENCLOSING,
    REASON_AMBIGUOUS,
    REASON_UNSUPPORTED,
    REASON_DECODE,
    REASON_AST,
    REASON_MISSING_BLOB,
    REASON_MISSING_COMMIT,
    REASON_MISSING_REPO,
    REASON_INVALID_PATH,
)


def symbol_oracle_jsonl_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "interim" / "swe_gym" / "m1c_symbol_oracle.jsonl"


def symbol_oracle_summary_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "manifests" / "swe_gym_m1c_symbol_summary.json"


def _empty_counts() -> dict[str, int]:
    return {
        "n_removed_line_sites": 0,
        "n_removed_line_mapped": 0,
        "n_addition_run_sites": 0,
        "n_addition_anchor_same_symbol": 0,
        "n_addition_anchor_one_sided": 0,
        "n_module_level": 0,
        "n_no_enclosing_symbol": 0,
        "n_ambiguous_addition_anchor": 0,
    }


def _add_counts(left: dict[str, int], right: Mapping[str, int]) -> None:
    for key, value in right.items():
        left[key] = left.get(key, 0) + int(value)


def _file_status_record(
    *,
    path: str,
    status: str,
    error: str | None,
    encoding: str | None,
    n_symbols_extracted: int,
    n_oracle_symbols: int,
    n_span_missing: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "status": status,
        "error": error,
        "extension": file_extension(path),
        "eligible": is_eligible_path(path),
        "encoding": encoding,
        "n_symbols_extracted": n_symbols_extracted,
        "n_oracle_symbols": n_oracle_symbols,
        "n_span_missing": n_span_missing,
    }


def extract_symbol_record(
    row: Mapping[str, Any],
    store: BlobStore | None,
    *,
    sources: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Build one symbol-oracle record. Never drops the instance."""
    instance_id = "" if is_null(row.get("instance_id")) else str(row["instance_id"])
    repo = "" if is_null(row.get("repo")) else str(row["repo"])
    base_commit = (
        "" if is_null(row.get("base_commit")) else str(row["base_commit"]).strip()
    )
    patch = "" if is_null(row.get("patch")) else str(row["patch"])
    parsed = extract_oracle_from_patch(patch)

    repo_ok = True
    commit_ok = True
    if store is not None:
        repo_ok = store.repo_path(repo) is not None
        commit_ok = bool(base_commit) and repo_ok and store.commit_ok(repo, base_commit)

    runs_by_path = addition_runs_by_source_path(patch) if parsed.parse_ok else {}
    by_source = {
        item["source_path"]: item for item in parsed.files if item.get("source_path")
    }
    oracle: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    file_results: list[dict[str, Any]] = []
    totals = _empty_counts()
    n_blob_ok = 0
    n_blob_missing = 0

    for path in parsed.base_changed_files:
        file_rec = by_source.get(path, {})
        removed = [
            line
            for hunk in file_rec.get("hunks", [])
            for line in hunk.get("removed_source_lines", [])
        ]
        runs = runs_by_path.get(path, [])
        file_reason: str | None = None
        file_symbols = None
        error = None
        encoding = None
        n_extracted = 0
        n_span_missing = 0

        if not repo_ok:
            file_reason = REASON_MISSING_REPO
            n_blob_missing += 1
        elif not commit_ok:
            file_reason = REASON_MISSING_COMMIT
            n_blob_missing += 1
        elif not is_safe_repo_path(path):
            file_reason = REASON_INVALID_PATH
            n_blob_missing += 1
        elif store is not None:
            if not store.exists(repo, base_commit, path):
                file_reason = REASON_MISSING_BLOB
                n_blob_missing += 1
            else:
                n_blob_ok += 1
                if is_eligible_path(path):
                    blob = store.read(repo, base_commit, path)
                    file_symbols = symbols_from_blob(path, blob)
                else:
                    file_reason = REASON_UNSUPPORTED
        elif sources is not None:
            blob = sources.get(path)
            if blob is None:
                file_reason = REASON_MISSING_BLOB
                n_blob_missing += 1
            else:
                n_blob_ok += 1
                file_symbols = symbols_from_blob(path, blob)
        else:
            file_reason = REASON_MISSING_BLOB
            n_blob_missing += 1

        mapped, file_unmapped, counts = map_file_change_sites(
            path=path,
            file_symbols=file_symbols,
            removed_lines=removed,
            addition_runs=runs,
            file_reason=file_reason,
        )
        oracle.extend(mapped)
        unmapped.extend(file_unmapped)
        _add_counts(totals, counts)
        if file_symbols is not None:
            error = file_symbols.error
            encoding = file_symbols.encoding
            n_extracted = len(file_symbols.symbols)
            n_span_missing = file_symbols.n_span_missing
            status = file_reason or file_symbols.status
        else:
            status = file_reason or REASON_MISSING_BLOB
        file_results.append(
            _file_status_record(
                path=path,
                status=status,
                error=error,
                encoding=encoding,
                n_symbols_extracted=n_extracted,
                n_oracle_symbols=len(mapped),
                n_span_missing=n_span_missing,
            )
        )

    oracle.sort(
        key=lambda item: (
            str(item["path"]),
            int(item["start_line"]),
            int(item["end_line"]),
            str(item["qualname"]),
            str(item["kind"]),
        )
    )
    file_results.sort(key=lambda item: str(item["path"]))
    unmapped.sort(
        key=lambda item: (
            str(item["path"]),
            str(item["site_kind"]),
            item.get("line") is None,
            item.get("line") if item.get("line") is not None else 0,
            item.get("prev_line") is None,
            item.get("prev_line") if item.get("prev_line") is not None else 0,
        )
    )
    n_skipped_added = sum(1 for item in parsed.files if item["operation"] == "added")
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "parse_ok": parsed.parse_ok,
        "parse_error": parsed.parse_error,
        "repo_ok": repo_ok,
        "commit_ok": commit_ok,
        "base_changed_files": list(parsed.base_changed_files),
        "n_gold_edit_files": parsed.n_gold_edit_files,
        "n_base_changed_files": len(parsed.base_changed_files),
        "n_skipped_added_files": n_skipped_added,
        "n_blob_ok": n_blob_ok,
        "n_blob_missing": n_blob_missing,
        "oracle_symbols": oracle,
        "n_oracle_symbols": len(oracle),
        "file_results": file_results,
        "unmapped_sites": unmapped,
        "n_unmapped_sites": len(unmapped),
        "n_removed_line_sites": totals["n_removed_line_sites"],
        "n_removed_line_mapped": totals["n_removed_line_mapped"],
        "n_addition_run_sites": totals["n_addition_run_sites"],
        "n_addition_anchor_same_symbol": totals["n_addition_anchor_same_symbol"],
        "n_addition_anchor_one_sided": totals["n_addition_anchor_one_sided"],
        "n_module_level": totals["n_module_level"],
        "n_no_enclosing_symbol": totals["n_no_enclosing_symbol"],
        "n_ambiguous_addition_anchor": totals["n_ambiguous_addition_anchor"],
    }


def _row_mapping(frame: Any, index: Any) -> dict[str, Any]:
    row = frame.loc[index]
    return {str(column): row[column] for column in frame.columns}


def extract_symbol_frame(
    frame: Any,
    store: BlobStore | None,
    *,
    sources_by_instance: Mapping[str, Mapping[str, bytes]] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for index in frame.index:
        row = _row_mapping(frame, index)
        instance_id = (
            "" if is_null(row.get("instance_id")) else str(row["instance_id"])
        )
        sources = None
        if sources_by_instance is not None:
            sources = sources_by_instance.get(instance_id)
        records.append(extract_symbol_record(row, store, sources=sources))
    records.sort(key=lambda item: item["instance_id"])
    return records


def _argmax_instance(
    records: Sequence[Mapping[str, Any]], field_name: str
) -> dict[str, Any] | None:
    if not records:
        return None
    best = max(records, key=lambda item: (int(item[field_name]), item["instance_id"]))
    return {
        "instance_id": best["instance_id"],
        field_name: int(best[field_name]),
    }


def _reason_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for site in record.get("unmapped_sites", []):
            counts[str(site["reason"])] += 1
    return {name: int(counts.get(name, 0)) for name in UNMAPPED_REASONS}


def _file_status_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for item in record.get("file_results", []):
            counts[str(item["status"])] += 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _extension_counts(
    records: Sequence[Mapping[str, Any]], *, eligible_only: bool | None = None
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for item in record.get("file_results", []):
            if eligible_only is True and not item["eligible"]:
                continue
            if eligible_only is False and item["eligible"]:
                continue
            counts[str(item["extension"])] += 1
    return {
        key: int(counts[key])
        for key in sorted(counts, key=lambda name: (-counts[name], name))
    }


def _failure_examples(
    records: Sequence[Mapping[str, Any]], status: str, *, limit: int = 8
) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        for item in record.get("file_results", []):
            if item["status"] != status:
                continue
            examples.append(
                {
                    "instance_id": record["instance_id"],
                    "path": item["path"],
                    "error": item.get("error"),
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _representative_symbols(
    records: Sequence[Mapping[str, Any]], *, limit: int = 6
) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        if not record.get("oracle_symbols"):
            continue
        first = record["oracle_symbols"][0]
        examples.append(
            {
                "instance_id": record["instance_id"],
                "path": first["path"],
                "qualname": first["qualname"],
                "kind": first["kind"],
                "start_line": first["start_line"],
                "end_line": first["end_line"],
                "evidence": list(first.get("evidence") or []),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _zero_symbol_examples(
    records: Sequence[Mapping[str, Any]], *, limit: int = 8
) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        if int(record["n_oracle_symbols"]) != 0:
            continue
        reasons = Counter(str(site["reason"]) for site in record.get("unmapped_sites", []))
        examples.append(
            {
                "instance_id": record["instance_id"],
                "n_base_changed_files": int(record["n_base_changed_files"]),
                "file_statuses": [item["status"] for item in record.get("file_results", [])],
                "unmapped_reason_counts": dict(reasons),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _spotlight(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {item["instance_id"]: item for item in records}
    record = by_id.get(SPOTLIGHT_INSTANCE_ID)
    if record is None:
        return {
            "instance_id": SPOTLIGHT_INSTANCE_ID,
            "present_in_input": False,
        }
    files = []
    for item in record.get("file_results", []):
        path = item["path"]
        files.append(
            {
                "path": path,
                "status": item["status"],
                "error": item.get("error"),
                "n_symbols_extracted": item["n_symbols_extracted"],
                "n_oracle_symbols": item["n_oracle_symbols"],
                "oracle_symbols": [
                    symbol
                    for symbol in record["oracle_symbols"]
                    if symbol["path"] == path
                ],
                "unmapped_sites": [
                    site
                    for site in record.get("unmapped_sites", [])
                    if site["path"] == path
                ],
            }
        )
    return {
        "instance_id": SPOTLIGHT_INSTANCE_ID,
        "present_in_input": True,
        "parse_ok": record["parse_ok"],
        "repo_ok": record["repo_ok"],
        "commit_ok": record["commit_ok"],
        "n_gold_edit_files": int(record["n_gold_edit_files"]),
        "n_base_changed_files": int(record["n_base_changed_files"]),
        "n_skipped_added_files": int(record["n_skipped_added_files"]),
        "n_oracle_symbols": int(record["n_oracle_symbols"]),
        "n_removed_line_sites": int(record["n_removed_line_sites"]),
        "n_removed_line_mapped": int(record["n_removed_line_mapped"]),
        "n_addition_run_sites": int(record["n_addition_run_sites"]),
        "n_addition_anchor_same_symbol": int(record["n_addition_anchor_same_symbol"]),
        "n_addition_anchor_one_sided": int(record["n_addition_anchor_one_sided"]),
        "files": files,
    }


def _has_eligible_file(record: Mapping[str, Any]) -> bool:
    return any(item.get("eligible") for item in record.get("file_results", []))


def build_symbol_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    repos_root: Path | None = None,
    prepare: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    n_rows = len(records)
    n_with_symbol = sum(1 for item in records if int(item["n_oracle_symbols"]) >= 1)
    n_zero_symbol = sum(1 for item in records if int(item["n_oracle_symbols"]) == 0)
    eligible_records = [item for item in records if _has_eligible_file(item)]
    commit_flags: dict[tuple[str, str], bool] = {}
    repo_flags: dict[str, bool] = {}
    for item in records:
        repo = str(item.get("repo") or "")
        commit = str(item.get("base_commit") or "")
        if repo:
            repo_flags[repo] = repo_flags.get(repo, True) and bool(item.get("repo_ok"))
        if repo and commit:
            key = (repo, commit)
            commit_flags[key] = commit_flags.get(key, True) and bool(item.get("commit_ok"))

    n_repos = len(repo_flags)
    n_repos_ok = sum(1 for ok in repo_flags.values() if ok)
    n_unique_commits = len(commit_flags)
    n_commits_resolved = sum(1 for ok in commit_flags.values() if ok)
    n_commits_missing = n_unique_commits - n_commits_resolved

    n_blob_ok = sum(int(item["n_blob_ok"]) for item in records)
    n_blob_missing = sum(int(item["n_blob_missing"]) for item in records)
    n_base_files = sum(int(item["n_base_changed_files"]) for item in records)

    file_items = [file_rec for item in records for file_rec in item.get("file_results", [])]
    n_eligible_files = sum(1 for item in file_items if item["eligible"])
    n_unsupported_files = sum(1 for item in file_items if item["status"] == REASON_UNSUPPORTED)
    n_decode_ok = sum(
        1
        for item in file_items
        if item["eligible"] and item["status"] in {FILE_STATUS_OK, REASON_AST}
    )
    n_decode_fail = sum(1 for item in file_items if item["status"] == REASON_DECODE)
    n_ast_ok = sum(
        1 for item in file_items if item["eligible"] and item["status"] == FILE_STATUS_OK
    )
    n_ast_fail = sum(1 for item in file_items if item["status"] == REASON_AST)

    disk_bytes = directory_size_bytes(repos_root) if repos_root is not None else None

    return {
        "dataset": "SWE-Gym",
        "hf_repo": HF_REPO_ID,
        "revision": HF_REVISION,
        "sha256": EXPECTED_SHA256,
        "oracle_source": "base_repository_ast",
        "parser": parser_version_record(),
        "n_rows": n_rows,
        "n_rows_written": n_rows,
        "rows_dropped": 0,
        "repo_preparation": {
            "n_repos": n_repos,
            "n_repos_available": n_repos_ok,
            "n_repos_failed": n_repos - n_repos_ok,
            "cache_root_env": "BCRL_DATA_ROOT",
            "cache_relpath": CACHE_RELPATH,
            "cache_disk_bytes": disk_bytes,
            **(dict(prepare) if prepare else {}),
        },
        "commits": {
            "n_unique_repo_base_commit": n_unique_commits,
            "n_resolved": n_commits_resolved,
            "n_missing": n_commits_missing,
        },
        "blobs": {
            "n_base_changed_file_occurrences": n_base_files,
            "n_ok": n_blob_ok,
            "n_missing": n_blob_missing,
        },
        "files": {
            "n_base_changed_file_occurrences": n_base_files,
            "n_eligible_py_pyi": n_eligible_files,
            "n_unsupported_extension": n_unsupported_files,
            "eligible_extensions": sorted(ELIGIBLE_EXTENSIONS),
            "extension_counts_all": _extension_counts(records),
            "extension_counts_eligible": _extension_counts(records, eligible_only=True),
            "extension_counts_unsupported": _extension_counts(
                records, eligible_only=False
            ),
            "file_status_counts": _file_status_counts(records),
            "decode_success": n_decode_ok,
            "decode_failure": n_decode_fail,
            "ast_parse_success": n_ast_ok,
            "ast_parse_failure": n_ast_fail,
        },
        "instances": {
            "denominator": n_rows,
            "n_with_ge1_oracle_symbol": n_with_symbol,
            "n_zero_oracle_symbol": n_zero_symbol,
            "n_with_ast_eligible_file": len(eligible_records),
        },
        "symbols_per_instance_all": length_stats(
            item["n_oracle_symbols"] for item in records
        ),
        "symbols_per_instance_ast_eligible": length_stats(
            item["n_oracle_symbols"] for item in eligible_records
        ),
        "change_sites": {
            "n_removed_line_sites": sum(int(item["n_removed_line_sites"]) for item in records),
            "n_removed_line_mapped": sum(
                int(item["n_removed_line_mapped"]) for item in records
            ),
            "n_addition_run_sites": sum(
                int(item["n_addition_run_sites"]) for item in records
            ),
            "n_addition_anchor_same_symbol": sum(
                int(item["n_addition_anchor_same_symbol"]) for item in records
            ),
            "n_addition_anchor_one_sided_weaker": sum(
                int(item["n_addition_anchor_one_sided"]) for item in records
            ),
            "unmapped_reason_counts": _reason_counts(records),
            "note": (
                "addition_anchor_same_symbol is conservative mapped evidence. "
                "addition_anchor_one_sided is weaker and is not included in "
                "the conservative mapped count. Reasons are observational; "
                "they are not drop filters."
            ),
        },
        "examples": {
            "decode_error": _failure_examples(records, REASON_DECODE),
            "ast_parse_error": _failure_examples(records, REASON_AST),
            "representative_oracle_symbols": _representative_symbols(records),
            "zero_symbol_instances": _zero_symbol_examples(records),
        },
        "outliers": {
            "max_oracle_symbols": _argmax_instance(records, "n_oracle_symbols"),
            "max_removed_line_sites": _argmax_instance(records, "n_removed_line_sites"),
            "max_addition_run_sites": _argmax_instance(records, "n_addition_run_sites"),
            "max_unmapped_sites": _argmax_instance(records, "n_unmapped_sites"),
        },
        "spotlight": _spotlight(records),
        "notes": {
            "source_path_only": True,
            "no_fake_module_symbol": True,
            "no_target_only_symbols": True,
            "added_files_excluded_from_base_oracle": True,
            "conservative_addition_gap_is_unmapped": (
                "Insertions between two different innermost base symbols, "
                "including end-of-function before the next sibling, stay "
                "unmapped. Indentation / target AST is not used."
            ),
            "one_sided_addition_is_weaker_evidence": True,
            "ast_extraction": True,
            "test_patch_used_for_oracle": False,
            "zero_base_visible_filter": "deferred to M1D",
            "not_m1d": True,
        },
    }


def format_symbol_report(summary: dict[str, Any]) -> str:
    prep = summary["repo_preparation"]
    commits = summary["commits"]
    blobs = summary["blobs"]
    files = summary["files"]
    instances = summary["instances"]
    sites = summary["change_sites"]
    spotlight = summary["spotlight"]

    def _stats(name: str, stats: Mapping[str, Any]) -> str:
        return (
            f"  {name}: n={stats['n']} min={stats['min']} mean={stats['mean']} "
            f"p50={stats['p50']} p90={stats['p90']} p95={stats['p95']} "
            f"p99={stats['p99']} max={stats['max']}"
        )

    lines = [
        "SWE-Gym M1C-B base-repository symbol oracle",
        f"revision: {summary['revision']}",
        f"sha256: {summary['sha256']}",
        f"parser: {summary['parser']['python_version']} "
        f"{summary['parser']['implementation']}",
        f"rows: {summary['n_rows']} (dropped: {summary['rows_dropped']})",
        "",
        "repository cache:",
        f"  available: {prep['n_repos_available']}/{prep['n_repos']}",
        f"  failed: {prep['n_repos_failed']}",
        f"  disk bytes: {prep.get('cache_disk_bytes')}",
        f"  layout: ${prep['cache_root_env']}/{prep['cache_relpath']}",
        "",
        "commits:",
        f"  unique (repo, base_commit): {commits['n_unique_repo_base_commit']}",
        f"  resolved: {commits['n_resolved']}",
        f"  missing: {commits['n_missing']}",
        "",
        "blobs (base_changed_files occurrences):",
        f"  ok: {blobs['n_ok']}/{blobs['n_base_changed_file_occurrences']}",
        f"  missing: {blobs['n_missing']}",
        "",
        "files:",
        f"  eligible .py/.pyi: {files['n_eligible_py_pyi']}",
        f"  unsupported extension: {files['n_unsupported_extension']}",
        f"  decode ok/fail: {files['decode_success']}/{files['decode_failure']}",
        f"  AST ok/fail: {files['ast_parse_success']}/{files['ast_parse_failure']}",
        "",
        "instances (denominator=all rows):",
        f"  >=1 oracle symbol: {instances['n_with_ge1_oracle_symbol']}/{instances['denominator']}",
        f"  zero oracle symbol: {instances['n_zero_oracle_symbol']}/{instances['denominator']}",
        f"  with AST-eligible file: {instances['n_with_ast_eligible_file']}",
        "",
        "symbols per instance:",
        _stats("all instances", summary["symbols_per_instance_all"]),
        _stats("AST-eligible instances", summary["symbols_per_instance_ast_eligible"]),
        "",
        "change sites:",
        f"  removed-line mapped: {sites['n_removed_line_mapped']}/{sites['n_removed_line_sites']}",
        f"  addition-anchor same-symbol: {sites['n_addition_anchor_same_symbol']}/{sites['n_addition_run_sites']}",
        f"  addition-anchor one-sided (weaker): {sites['n_addition_anchor_one_sided_weaker']}",
        "  unmapped reasons:",
    ]
    for name, count in sites["unmapped_reason_counts"].items():
        lines.append(f"    {name}: {count}")

    lines.append("")
    lines.append("file extensions (base_changed):")
    for name, count in list(files["extension_counts_all"].items())[:15]:
        lines.append(f"  {name}: {count}")

    lines.append("")
    lines.append(
        f"spotlight {spotlight['instance_id']}: "
        f"present={spotlight.get('present_in_input')} "
        f"gold_edit={spotlight.get('n_gold_edit_files')} "
        f"base_changed={spotlight.get('n_base_changed_files')} "
        f"skipped_added={spotlight.get('n_skipped_added_files')} "
        f"symbols={spotlight.get('n_oracle_symbols')}"
    )
    for item in spotlight.get("files") or []:
        names = [symbol["qualname"] for symbol in item.get("oracle_symbols", [])]
        lines.append(
            f"  {item['path']}: status={item['status']} "
            f"oracle={item['n_oracle_symbols']} extracted={item['n_symbols_extracted']} "
            f"qualnames={names}"
        )

    lines.append("")
    lines.append("outliers:")
    for key, value in summary["outliers"].items():
        lines.append(f"  {key}: {value}")

    examples = summary["examples"]
    if examples["decode_error"]:
        lines.append("")
        lines.append("decode_error examples:")
        for item in examples["decode_error"]:
            lines.append(f"  {item['instance_id']} {item['path']}: {item['error']}")
    if examples["ast_parse_error"]:
        lines.append("")
        lines.append("ast_parse_error examples:")
        for item in examples["ast_parse_error"]:
            lines.append(f"  {item['instance_id']} {item['path']}: {item['error']}")
    return "\n".join(lines) + "\n"
