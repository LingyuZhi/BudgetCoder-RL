#!/usr/bin/env python
"""Download the official SWE-Gym train parquet (M1A).

Uses huggingface_hub only. Does not install packages, filter rows, or
download Docker images / SWE-Gym-Lite / SWE-Gym-Raw.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from budget_coder_rl.data.swe_gym import (  # noqa: E402
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    HF_FILENAME,
    HF_REPO_ID,
    HF_REVISION,
    HF_URL,
    manifest_path,
    manifest_record,
    raw_dir,
    sha256_file,
    source_json_path,
    verify_parquet_file,
    write_json,
)


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required to download SWE-Gym but is not importable. "
            "Use the pinned RL conda env. Do not pip-install packages into that env "
            "from this script."
        ) from exc
    return hf_hub_download


def _write_records(repo_root: Path, dest: Path, *, skipped: bool) -> None:
    verified_at = datetime.now(timezone.utc).isoformat()
    try:
        local_path = str(dest.relative_to(repo_root))
    except ValueError:
        local_path = str(dest)
    source = {
        "hf_repo": HF_REPO_ID,
        "hf_url": HF_URL,
        "hf_revision": HF_REVISION,
        "hf_filename": HF_FILENAME,
        "local_path": local_path,
        "sha256": EXPECTED_SHA256,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "skipped": skipped,
        "downloaded_at": verified_at,
    }
    write_json(source_json_path(repo_root), source)
    record = manifest_record(verified_at=verified_at)
    record["skipped"] = skipped
    write_json(manifest_path(repo_root), record)


def download(repo_root: Path, output_dir: Path, revision: str) -> Path:
    dest = output_dir / HF_FILENAME
    if dest.is_file() and not verify_parquet_file(dest):
        print(f"already present and checksum matches: {dest}")
        _write_records(repo_root, dest, skipped=True)
        return dest
    if dest.is_file():
        print(f"existing file failed checksum; removing: {dest}")
        dest.unlink()

    hf_hub_download = _require_huggingface_hub()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"downloading {HF_REPO_ID}@{revision} {HF_FILENAME} -> {output_dir}"
    )
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            revision=revision,
            local_dir=str(output_dir),
        )
    except Exception as exc:
        raise SystemExit(
            f"download failed: {exc}\n"
            "If this is a compute node, set HF_ENDPOINT / http_proxy as in the "
            "pinned environment notes. This script will not install packages."
        ) from exc

    dest = Path(downloaded)
    errors = verify_parquet_file(dest)
    if errors:
        dest.unlink(missing_ok=True)
        raise SystemExit("checksum/size mismatch after download:\n" + "\n".join(errors))

    digest = sha256_file(dest)
    print(f"ok: {dest}")
    print(f"size_bytes={dest.stat().st_size} sha256={digest}")
    _write_records(repo_root, dest, skipped=False)
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory that will contain the HF data/ layout (default: data/raw/swe_gym)",
    )
    parser.add_argument(
        "--revision",
        default=HF_REVISION,
        help=f"Hugging Face dataset revision (default: {HF_REVISION})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else raw_dir(repo_root)
    )
    if args.revision != HF_REVISION:
        print(
            f"warning: revision {args.revision} != pinned {HF_REVISION}; "
            "checksum gate still uses the official parquet pin",
            file=sys.stderr,
        )
    download(repo_root, output_dir, args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
