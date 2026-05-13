"""Deterministic seed derivation for per-component RNGs.

Python's built-in `hash()` is PYTHONHASHSEED-salted across processes
and would silently destroy reproducibility — different runs of the
same scenario would seed BCU's fault RNG to different values.
blake2b is in the stdlib, fast, and deterministic everywhere.

Contract: `derive_seed(parent, component_id)` returns the same int
forever — across processes, machines, Python versions.
"""

from __future__ import annotations

import hashlib


def derive_seed(parent_seed: int, component_id: str) -> int:
    h = hashlib.blake2b(
        f"{parent_seed}:{component_id}".encode("utf-8"),
        digest_size=8,
    )
    return int.from_bytes(h.digest(), "big", signed=False)
