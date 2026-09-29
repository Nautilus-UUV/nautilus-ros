"""Drift guard: nominal scenario's plant block must mirror robot_specs.

If someone edits ``library/nominal.yaml`` without updating
``robot_specs.py`` (or vice versa), this test fails loudly. The two
files are the dual sources of truth for the *nominal* hardware plant —
this asserts they don't silently diverge. `nominal.yaml` is the launch
default; `baseline.yaml` is the same plant + faults turned on.

MC perturbations live in non-nominal scenarios; they're not in scope
here. The control side has no plant model any more (the bang-bang loop
commands RPM directly), but its one hardware-derived value —
``DepthSpec.pump_rpm`` — is checked the same way.
"""

from __future__ import annotations

from py_pkg import robot_specs
from py_pkg.scenarios.loader import load_scenario


def test_rig_plant_mirrors_robot_specs(nominal_scenario_path):
    rig_plant = load_scenario(nominal_scenario_path).rig.plant
    assert rig_plant.volume_per_rev_m3 == robot_specs.VOLUME_PER_REV_M3
    # bladder_min_m3 / bladder_max_m3 are YAML-only operating-range knobs
    # (no robot_specs counterpart) — same shape as acu_pitch.output_limits.
    assert rig_plant.bladder_nominal_m3 == robot_specs.BLADDER_VOLUME_M3
    assert rig_plant.bcu_motor_min_rpm == robot_specs.BCU_MOTOR_MIN_RPM
    assert rig_plant.bcu_motor_max_rpm == robot_specs.BCU_MOTOR_MAX_RPM


def test_control_pump_rpm_mirrors_robot_specs(nominal_scenario_path):
    depth = load_scenario(nominal_scenario_path).control.controllers.depth
    # The bang-bang command magnitude at nominal IS the hardware ceiling:
    # full authority is the point of the law. A sweep may lower it to model
    # a weaker pump, which is why it lives on the spec and not in
    # robot_specs — but nominal must not drift off the hardware value.
    assert depth.pump_rpm == robot_specs.BCU_MOTOR_MAX_RPM
    # deadband_pa / tank_stop_band are control-tuning knobs with no
    # robot_specs counterpart, so they're not asserted here.


def test_acu_roll_output_limits_mirror_robot_specs(nominal_scenario_path):
    a = load_scenario(nominal_scenario_path).control.controllers.acu_roll
    assert a.output_limits == (
        -robot_specs.ACU_ROLL_MAX_ANGLE_DEG,
        robot_specs.ACU_ROLL_MAX_ANGLE_DEG,
    )
