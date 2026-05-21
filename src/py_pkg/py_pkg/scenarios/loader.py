"""Load a scenario YAML into a typed `Scenario` model tree.

One entry point: `load_scenario(path) -> Scenario`. Callers pick the
half they care about (`.control` for controllers, `.rig` for HAL
bridges). Pydantic validates the YAML against the spec — unknown keys
and bad types fail before any run launches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .spec.scenario import Scenario


def _read_yaml(path: Path | str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(
            f"Scenario YAML root must be a mapping, got {type(raw).__name__}"
        )
    return raw


def load_scenario(path: Path | str) -> Scenario:
    return Scenario.model_validate(_read_yaml(path))
