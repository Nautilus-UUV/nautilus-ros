"""Scenario authoring surface for the Nautilus control + sim stack.

One YAML on disk drives both the control-side parameters (gains,
mission, estimator) and the rig-side parameters (plant, faults,
bridge publish rates, noise) for an experiment or production run.
`load_scenario(path)` returns a single typed `Scenario`; callers
pick `.control` or `.rig`.
"""

from .loader import load_scenario
from .seed import derive_seed

__all__ = ["load_scenario", "derive_seed"]
