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
from py_pkg.scenarios.compile import params_for_bcu_node
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


def test_bcu_command_gate_disarm_above_arm_raises():
    # The disarm band is the inner edge of the arm hysteresis, so a scenario
    # that sets disarm > arm is incoherent and must fail at load, not run.
    p = _write_yaml(
        "control:\n  controllers:\n    depth:\n"
        "      error_disarm_pa: 5000.0\n      error_arm_pa: 4000.0\n"
    )
    with pytest.raises(ValueError, match="error_disarm_pa"):
        load_scenario(p)


def test_bcu_command_gate_negative_dwell_raises():
    p = _write_yaml(
        "control:\n  controllers:\n    depth:\n      min_valve_dwell_s: -0.1\n"
    )
    with pytest.raises(ValueError, match="min_valve_dwell_s"):
        load_scenario(p)


def test_bcu_command_gate_fields_round_trip_through_params():
    # Forward half of the param mapping: the gate knobs reach the BCU node's
    # parameter dict (the inverse half, bcu_spec_from_node, is exercised by
    # the Tier 2 node construction test).
    params = params_for_bcu_node(ControlScenario())
    d = ControlScenario().controllers.depth
    assert params["error_arm_pa"] == d.error_arm_pa
    assert params["error_disarm_pa"] == d.error_disarm_pa
    assert params["min_valve_dwell_s"] == d.min_valve_dwell_s


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
