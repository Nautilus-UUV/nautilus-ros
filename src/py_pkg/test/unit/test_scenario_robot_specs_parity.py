"""Drift guard: nominal scenario's plant block must mirror robot_specs.

If someone edits ``library/nominal.yaml`` without updating
``robot_specs.py`` (or vice versa), this test fails loudly. The two
files are the dual sources of truth for the *nominal* hardware plant —
this asserts they don't silently diverge. `nominal.yaml` is the launch
default; `baseline.yaml` is the same plant + faults turned on.

MC perturbations live in non-nominal scenarios; they're not in scope
here. The control-side controller plant model
(``DepthSpec.plant_model``) is checked too, since at nominal it must
also mirror robot_specs.
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


def test_control_plant_model_mirrors_robot_specs(nominal_scenario_path):
    pm = load_scenario(nominal_scenario_path).control.controllers.depth.plant_model
    assert pm.bladder_nominal_m3 == robot_specs.BLADDER_VOLUME_M3
    assert pm.max_rpm == robot_specs.BCU_MOTOR_MAX_RPM
    # min_rpm / min_operating_rpm are control-tuning knobs (the pump deadband
    # the depth loop snaps RPM commands through), not hardware mirrors — they
    # intentionally diverge from BCU_MOTOR_MIN_RPM, so they're not asserted here.


def test_acu_roll_output_limits_mirror_robot_specs(nominal_scenario_path):
    a = load_scenario(nominal_scenario_path).control.controllers.acu_roll
    assert a.output_limits == (
        -robot_specs.ACU_ROLL_MAX_ANGLE_DEG,
        robot_specs.ACU_ROLL_MAX_ANGLE_DEG,
    )
