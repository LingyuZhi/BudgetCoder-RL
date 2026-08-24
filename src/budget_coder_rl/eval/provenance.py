"""Run provenance for formal M3A / later experiment artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from budget_coder_rl.data.swe_gym_materialize import dataset_manifest_path


def collect_run_provenance(
    repo_root: str | Path,
    *,
    verl_source: str | Path | None = None,
    model_path: str | None = None,
    agent_loop_config: str | Path | None = None,
    tokenizer_name_or_path: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "budget_coder_rl": git_info(root),
        "model_path": model_path,
        "tokenizer_name_or_path": tokenizer_name_or_path,
    }
    if verl_source is not None:
        payload["verl"] = git_info(Path(verl_source))
    else:
        payload["verl"] = _infer_verl()
    if agent_loop_config is not None:
        config_path = Path(agent_loop_config)
        payload["agent_loop_config"] = {
            "path": str(config_path),
            "sha256": sha256_file(config_path) if config_path.is_file() else None,
        }
    manifest = dataset_manifest_path(root)
    try:
        manifest_rel = str(manifest.relative_to(root))
    except ValueError:
        manifest_rel = str(manifest)
    payload["m1e_dataset_manifest"] = {
        "path": manifest_rel,
        "sha256": sha256_file(manifest) if manifest.is_file() else None,
    }
    return payload


def git_info(path: Path) -> dict[str, Any]:
    root = _git_root(path)
    if root is None:
        return {
            "path": str(path),
            "commit": None,
            "dirty": None,
            "dirty_files": [],
        }
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    porcelain = _git(["status", "--porcelain"], cwd=root) or ""
    dirty_files = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "path": str(root),
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:50],
        "n_dirty_files": len(dirty_files),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_verl() -> dict[str, Any]:
    try:
        import verl
    except Exception:
        return {"path": None, "commit": None, "dirty": None, "version": None}
    source = Path(verl.__file__).resolve().parents[1]
    info = git_info(source)
    info["version"] = getattr(verl, "__version__", None)
    return info


def _git_root(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    try:
        out = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out.strip())


def _git(args: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
