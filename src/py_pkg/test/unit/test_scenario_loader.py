"""Tier 1 sanity for the scenarios package.

- Strict mode: unknown keys raise ValidationError (ValueError).
- load_scenario returns a single Scenario with both halves populated.
- derive_seed is deterministic across processes (no PYTHONHASHSEED salt).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from py_pkg.scenarios import derive_seed, load_scenario
from py_pkg.scenarios.spec.control import ControlScenario
from py_pkg.scenarios.spec.rig import RigScenario


def _write_yaml(text: str) -> Path:
    p = Path(tempfile.mkstemp(suffix=".yaml")[1])
    p.write_text(text)
    return p


def test_empty_yaml_returns_defaults():
    p = _write_yaml("")
    s = load_scenario(p)
    assert s.seed == 0
    assert s.control == ControlScenario()
    assert s.rig == RigScenario()


def test_unknown_top_level_key_raises():
    p = _write_yaml("control: {}\nbogus: 42\n")
    with pytest.raises(ValueError, match="bogus"):
        load_scenario(p)


def test_unknown_nested_key_raises():
    p = _write_yaml("control:\n  controllers:\n    depth:\n      typo_field: 1\n")
    with pytest.raises(ValueError, match="typo_field"):
        load_scenario(p)


def test_derive_seed_is_deterministic_and_varies():
    a = derive_seed(0, "bcu_rpm_fault")
    b = derive_seed(0, "bcu_rpm_fault")
    c = derive_seed(1, "bcu_rpm_fault")
    d = derive_seed(0, "acu_pitch_fault")
    assert a == b
    assert a != c
    assert a != d


def test_derive_seed_fits_in_ros_int64():
    # rclpy INTEGER params are signed int64. If derive_seed ever returns a
    # value > INT64_MAX, the launch passes it through as DOUBLE and any
    # bridge that declared the seed param as INTEGER dies at startup with
    # InvalidParameterTypeException — which on the BCU bridge silently
    # disables the dive path (see git history for the diagnosis).
    int64_max = (1 << 63) - 1
    # Half of unmasked 64-bit unsigned digests are > INT64_MAX, so sampling
    # a few hundred parent seeds reliably catches a regression.
    for i in range(512):
        s = derive_seed(i, "bcu_rpm_fault")
        assert 0 <= s <= int64_max, f"derive_seed({i}, ...) = {s} > INT64_MAX"


def test_bcu_bridge_params_carry_pump_transient_fields():
    from py_pkg.scenarios.compile import params_for_bcu_bridge
    from py_pkg.scenarios.spec.rig import RigScenario

    # Defaults ARE the 2026-06-24 lake fit (pump_transient_fit.json).
    params = params_for_bcu_bridge(RigScenario())
    assert params["pump_response_delay_s"] == 1.057
    assert params["pump_slew_rpm_per_s"] == 512.1
    assert params["tank_map_shape"] == "gaslaw"
    assert params["tank_air_volume_m3"] == 3.041025e-3

    # Transients-off override (pre-calibration behavior); tank_air_volume_m3
    # <= 0 means the pinned cushion, same convention as the pump knobs.
    scen = RigScenario.model_validate(
        {
            "plant": {
                "pump_response_delay_s": 0.0,
                "pump_slew_rpm_per_s": 0.0,
                "tank_map_shape": "linear",
                "tank_air_volume_m3": 0.0,
            }
        }
    )
    params = params_for_bcu_bridge(scen)
    assert params["pump_response_delay_s"] == 0.0
    assert params["pump_slew_rpm_per_s"] == 0.0
    assert params["tank_map_shape"] == "linear"
    assert params["tank_air_volume_m3"] == 0.0


def test_tank_map_shape_rejects_unknown_value():
    from py_pkg.scenarios.spec.rig import PlantSpec

    with pytest.raises(ValueError):
        PlantSpec(tank_map_shape="cubic")
