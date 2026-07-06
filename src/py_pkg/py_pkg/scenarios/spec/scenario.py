"""Top-level scenario type.

One YAML on disk drives both halves of a run. `Scenario` is the single
type returned by `load_scenario`; callers reach into `.control` for
controller-side fields and `.rig` for HAL/sim-side fields.
"""

from __future__ import annotations

from pydantic import Field

from ._shared import StrictModel
from .control import ControlScenario
from .rig import RigScenario


class Scenario(StrictModel):
    seed: int = 0
    control: ControlScenario = Field(default_factory=ControlScenario)
    rig: RigScenario = Field(default_factory=RigScenario)
