"""Shared scaffold for the sweep-spec guard tests (test_train_mix_*_spec).

The sweep scripts are plain modules two dirs above py_pkg, not a package --
import them the way run_sweep imports its own siblings, done once here so
every campaign spec test shares one repo walk, one sys.path entry, and one
import-failure policy instead of a per-file copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
try:
    import lhs_sample
    import run_sweep

    _SCRIPTS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover — apt python3 ships numpy/scipy
    lhs_sample = run_sweep = None
    _SCRIPTS_IMPORT_ERROR = exc

needs_scripts = pytest.mark.skipif(
    _SCRIPTS_IMPORT_ERROR is not None,
    reason=f"scripts import failed (numpy/scipy missing?): {_SCRIPTS_IMPORT_ERROR}",
)


def load_sweep_spec(name: str) -> tuple[Path, dict]:
    """Path + parsed YAML for scripts/sweeps/<name>; skips the whole module
    (repo-layout guard) when the file is absent."""
    path = SCRIPTS_DIR / "sweeps" / name
    if not path.is_file():  # pragma: no cover — repo-layout guard
        pytest.skip(f"sweep spec not found at {path}", allow_module_level=True)
    return path, yaml.safe_load(path.read_text())
