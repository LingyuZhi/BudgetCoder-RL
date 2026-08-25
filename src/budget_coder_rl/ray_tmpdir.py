"""Short-path Ray / TMPDIR root for compute nodes with a full ``/`` disk.

Unix-domain sockets must stay under 107 bytes. A long ``RAY_TMPDIR`` under
``$BCRL_DATA_ROOT`` fails AF_UNIX. Prefer ``/dev/shm/u<uid>/r``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

UNIX_SOCKET_MAX_BYTES = 107
OBJECT_STORE_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
SESSION_SUFFIX_BUDGET = 64


class RayTmpdirError(RuntimeError):
    """No usable short temp root for Ray sockets."""


def uid_tag(uid: int | None = None) -> str:
    return f"u{os.getuid() if uid is None else int(uid)}"


def candidate_roots(uid: int | None = None) -> list[Path]:
    tag = uid_tag(uid)
    return [
        Path(f"/dev/shm/{tag}/r"),
        Path(f"/tmp/{tag}/r"),
    ]


def short_temp_root(uid: int | None = None) -> Path:
    """Return a writable directory whose Ray socket paths stay under 107 bytes."""
    errors: list[str] = []
    for root in candidate_roots(uid):
        try:
            _validate_root(root)
            return root
        except (OSError, RayTmpdirError) as exc:
            errors.append(f"{root}: {exc}")
    raise RayTmpdirError(
        "no short Ray temp root available (need AF_UNIX path <= "
        f"{UNIX_SOCKET_MAX_BYTES} bytes): " + "; ".join(errors)
    )


def apply_process_tmpdir(root: Path | None = None) -> Path:
    resolved = Path(root) if root is not None else short_temp_root()
    _validate_root(resolved)
    os.environ["TMPDIR"] = str(resolved)
    os.environ["RAY_TMPDIR"] = str(resolved)
    os.environ["TMP"] = str(resolved)
    os.environ["TEMP"] = str(resolved)
    return resolved


def ray_init_kwargs(root: Path | None = None) -> dict[str, Any]:
    resolved = apply_process_tmpdir(root)
    return {
        "_temp_dir": str(resolved),
        "object_store_memory": OBJECT_STORE_MEMORY_BYTES,
    }


def cleanup_our_tmp_ray(*, dry_run: bool = False) -> dict[str, Any]:
    """Remove this uid's stale ``/tmp/ray`` and ``/tmp/torchinductor_*`` only."""
    uid = os.getuid()
    removed: list[str] = []
    skipped: list[str] = []
    targets = [Path("/tmp/ray"), Path(f"/tmp/torchinductor_{_username()}")]
    for path in targets:
        if not path.exists():
            skipped.append(f"missing:{path}")
            continue
        if not _owned_by(path, uid):
            skipped.append(f"not_ours:{path}")
            continue
        if dry_run:
            removed.append(f"dry_run:{path}")
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    return {"uid": uid, "removed": removed, "skipped": skipped}


def _validate_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK | os.X_OK):
        raise RayTmpdirError(f"not writable: {root}")
    probe = root / "plasma_store"
    encoded = str(probe).encode("utf-8")
    if len(encoded) + SESSION_SUFFIX_BUDGET > UNIX_SOCKET_MAX_BYTES:
        raise RayTmpdirError(
            f"path too long for AF_UNIX ({len(encoded)}+{SESSION_SUFFIX_BUDGET} > "
            f"{UNIX_SOCKET_MAX_BYTES}): {root}"
        )


def _owned_by(path: Path, uid: int) -> bool:
    try:
        return path.stat().st_uid == uid
    except OSError:
        return False


def _username() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or uid_tag()


def socket_path_budget_ok(root: Path) -> bool:
    try:
        _validate_root(root)
        return True
    except RayTmpdirError:
        return False
