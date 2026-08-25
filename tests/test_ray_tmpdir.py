"""Short Ray temp root path budget."""

from __future__ import annotations

import os
from pathlib import Path

from budget_coder_rl.ray_tmpdir import (
    UNIX_SOCKET_MAX_BYTES,
    candidate_roots,
    ray_init_kwargs,
    short_temp_root,
    socket_path_budget_ok,
)


def test_short_temp_root_fits_unix_socket_budget():
    root = short_temp_root()
    assert socket_path_budget_ok(root)
    encoded = str(root / "plasma_store").encode("utf-8")
    assert len(encoded) + 64 <= UNIX_SOCKET_MAX_BYTES
    saved = {key: os.environ.get(key) for key in ("TMPDIR", "RAY_TMPDIR", "TMP", "TEMP")}
    try:
        kwargs = ray_init_kwargs(root)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert kwargs["_temp_dir"] == str(root)
    assert kwargs["object_store_memory"] == 4 * 1024 * 1024 * 1024
    assert Path(kwargs["_temp_dir"]).is_dir()


def test_candidate_roots_are_short():
    for root in candidate_roots(uid=10158):
        assert "10158" in str(root)
        assert len(str(root).encode("utf-8")) < 40
