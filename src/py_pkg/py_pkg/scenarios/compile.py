"""Compile a typed scenario into per-node ROS parameter dicts (forward),
and reconstruct typed sub-scenarios from a running node's parameters
(inverse).

Forward direction: `params_for_*` returns a flat dict in the shape
`Node(parameters=[...])` accepts. Used at launch time.

Inverse direction: `*_spec_from_node` reads the parameters back at node
startup, returning a typed dataclass. The two halves are wire-format
mirrors of each other — keeping them in one file means a parameter
rename is a single-file edit.

The dataclass fields are the authoring surface; this file is the only
place that knows the ROS parameter wire names.
"""

from __future__ import annotations

from typing import Any

from rclpy.node import Node

from .seed import derive_seed
from .spec.control import (
    AcuPitchSpec,
    AcuRollSpec,
    ControlScenario,
    DepthPlantModel,
    DepthSpec,
    PIDPressureSpec,
)
from .spec.rig import RigScenario

# ---------------------------------------------------------------------------
# Control-side: forward (params_for_*) and inverse (*_spec_from_node)
# ---------------------------------------------------------------------------


def params_for_depth_node(scen: ControlScenario) -> dict[str, Any]:
    d = scen.controllers.depth
    return {
        "frequency_hz": d.frequency_hz,
        "pid_pressure.kp": d.pid_pressure.kp,
        "pid_pressure.ki": d.pid_pressure.ki,
        "pid_pressure.kd": d.pid_pressure.kd,
        "pid_pressure.integral_limit_low": d.pid_pressure.integral_limits[0],
        "pid_pressure.integral_limit_high": d.pid_pressure.integral_limits[1],
        "pid_pressure.output_limit_low": d.pid_pressure.output_limits[0],
        "pid_pressure.output_limit_high": d.pid_pressure.output_limits[1],
        "pid_pressure.derivative_filter": d.pid_pressure.derivative_filter,
        "plant_model.bladder_nominal_m3": d.plant_model.bladder_nominal_m3,
        "plant_model.initial_proportion_full": d.plant_model.initial_proportion_full,
        "plant_model.min_rpm": d.plant_model.min_rpm,
        "plant_model.min_operating_rpm": d.plant_model.min_operating_rpm,
        "plant_model.max_rpm": d.plant_model.max_rpm,
        "plant_model.pump_efficiency": d.plant_model.pump_efficiency,
    }


def depth_spec_from_node(node: Node) -> DepthSpec:
    """Read depth-controller params off `node` into a typed DepthSpec.

    Defaults come from the dataclass — absent any scenario override at
    launch time, behaviour matches the literal values that used to live
    in `init_control` / `init_buoyancy_engine` / `init_motor`.
    """

    default = DepthSpec()
    pp = default.pid_pressure
    pm = default.plant_model

    node.declare_parameter("frequency_hz", default.frequency_hz)

    node.declare_parameter("pid_pressure.kp", pp.kp)
    node.declare_parameter("pid_pressure.ki", pp.ki)
    node.declare_parameter("pid_pressure.kd", pp.kd)
    node.declare_parameter("pid_pressure.integral_limit_low", pp.integral_limits[0])
    node.declare_parameter("pid_pressure.integral_limit_high", pp.integral_limits[1])
    node.declare_parameter("pid_pressure.output_limit_low", pp.output_limits[0])
    node.declare_parameter("pid_pressure.output_limit_high", pp.output_limits[1])
    node.declare_parameter("pid_pressure.derivative_filter", pp.derivative_filter)

    node.declare_parameter("plant_model.bladder_nominal_m3", pm.bladder_nominal_m3)
    node.declare_parameter(
        "plant_model.initial_proportion_full", pm.initial_proportion_full
    )
    node.declare_parameter("plant_model.min_rpm", pm.min_rpm)
    node.declare_parameter("plant_model.min_operating_rpm", pm.min_operating_rpm)
    node.declare_parameter("plant_model.max_rpm", pm.max_rpm)
    node.declare_parameter("plant_model.pump_efficiency", pm.pump_efficiency)

    g = node.get_parameter
    return DepthSpec(
        frequency_hz=g("frequency_hz").value,
        pid_pressure=PIDPressureSpec(
            kp=g("pid_pressure.kp").value,
            ki=g("pid_pressure.ki").value,
            kd=g("pid_pressure.kd").value,
            integral_limits=(
                g("pid_pressure.integral_limit_low").value,
                g("pid_pressure.integral_limit_high").value,
            ),
            output_limits=(
                g("pid_pressure.output_limit_low").value,
                g("pid_pressure.output_limit_high").value,
            ),
            derivative_filter=g("pid_pressure.derivative_filter").value,
        ),
        plant_model=DepthPlantModel(
            bladder_nominal_m3=g("plant_model.bladder_nominal_m3").value,
            initial_proportion_full=g("plant_model.initial_proportion_full").value,
            min_rpm=g("plant_model.min_rpm").value,
            min_operating_rpm=g("plant_model.min_operating_rpm").value,
            max_rpm=g("plant_model.max_rpm").value,
            pump_efficiency=g("plant_model.pump_efficiency").value,
        ),
    )


def params_for_acu_node(scen: ControlScenario) -> dict[str, Any]:
    p = scen.controllers.acu_pitch
    r = scen.controllers.acu_roll
    return {
        "acu_pitch.name": p.name,
        "acu_pitch.output_limit_low": p.output_limits[0],
        "acu_pitch.output_limit_high": p.output_limits[1],
        "acu_roll.frequency_hz": r.frequency_hz,
        "acu_roll.name": r.name,
        "acu_roll.kp": r.kp,
        "acu_roll.ki": r.ki,
        "acu_roll.kd": r.kd,
        "acu_roll.command_tolerance": r.command_tolerance,
        "acu_roll.integral_limit_low": r.integral_limits[0],
        "acu_roll.integral_limit_high": r.integral_limits[1],
        "acu_roll.output_limit_low": r.output_limits[0],
        "acu_roll.output_limit_high": r.output_limits[1],
        "acu_roll.derivative_filter": r.derivative_filter,
    }


def acu_pitch_spec_from_node(node: Node) -> AcuPitchSpec:
    """Read ACU pitch-axis params off `node` into a typed AcuPitchSpec.

    The pitch axis is bang-bang, so the only knobs are the two
    output_limits (front, back) in metres.
    """

    default = AcuPitchSpec()

    node.declare_parameter("acu_pitch.name", default.name)
    node.declare_parameter("acu_pitch.output_limit_low", default.output_limits[0])
    node.declare_parameter("acu_pitch.output_limit_high", default.output_limits[1])

    g = node.get_parameter
    return AcuPitchSpec(
        name=g("acu_pitch.name").value,
        output_limits=(
            g("acu_pitch.output_limit_low").value,
            g("acu_pitch.output_limit_high").value,
        ),
    )


def acu_roll_spec_from_node(node: Node) -> AcuRollSpec:
    """Read ACU roll-controller params off `node` into a typed AcuRollSpec.

    Defaults match the literal values that used to live in the
    `init_acu_roll` dict.
    """

    default = AcuRollSpec()

    node.declare_parameter("acu_roll.frequency_hz", default.frequency_hz)
    node.declare_parameter("acu_roll.name", default.name)
    node.declare_parameter("acu_roll.kp", default.kp)
    node.declare_parameter("acu_roll.ki", default.ki)
    node.declare_parameter("acu_roll.kd", default.kd)
    node.declare_parameter("acu_roll.command_tolerance", default.command_tolerance)
    node.declare_parameter("acu_roll.integral_limit_low", default.integral_limits[0])
    node.declare_parameter("acu_roll.integral_limit_high", default.integral_limits[1])
    node.declare_parameter("acu_roll.output_limit_low", default.output_limits[0])
    node.declare_parameter("acu_roll.output_limit_high", default.output_limits[1])
    node.declare_parameter("acu_roll.derivative_filter", default.derivative_filter)

    g = node.get_parameter
    return AcuRollSpec(
        frequency_hz=g("acu_roll.frequency_hz").value,
        name=g("acu_roll.name").value,
        kp=g("acu_roll.kp").value,
        ki=g("acu_roll.ki").value,
        kd=g("acu_roll.kd").value,
        command_tolerance=g("acu_roll.command_tolerance").value,
        integral_limits=(
            g("acu_roll.integral_limit_low").value,
            g("acu_roll.integral_limit_high").value,
        ),
        output_limits=(
            g("acu_roll.output_limit_low").value,
            g("acu_roll.output_limit_high").value,
        ),
        derivative_filter=g("acu_roll.derivative_filter").value,
    )


# ---------------------------------------------------------------------------
# Rig-side: HAL bridges
# ---------------------------------------------------------------------------


def params_for_bcu_bridge(scen: RigScenario, parent_seed: int = 0) -> dict[str, Any]:
    p = scen.plant
    f = scen.faults.bcu_rpm
    b = scen.bridges.bcu
    return {
        "model_name": scen.sim.model_name,
        "volume_per_rev_m3": p.volume_per_rev_m3,
        "bladder_min_m3": p.bladder_min_m3,
        "bladder_max_m3": p.bladder_max_m3,
        "fault_probability_per_sec": f.probability_per_sec,
        "fault_duration_sec": f.duration_sec,
        "fault_degraded_factor": f.degraded_factor,
        "fault_severe_factor": f.severe_factor,
        "rng_seed": derive_seed(parent_seed, "bcu_rpm_fault"),
        "publish_rate_hz": b.publish_rate_hz,
        "tank_pressure_empty_pa": p.tank_pressure_empty_pa,
        "tank_pressure_full_pa": p.tank_pressure_full_pa,
        "tank_pressure_vacuum_offset_pa": p.tank_pressure_vacuum_offset_pa,
    }


def params_for_acu_bridge(scen: RigScenario) -> dict[str, Any]:
    return {
        "model_name": scen.sim.model_name,
        "world_name": scen.sim.world_name,
    }


def params_for_imu_bridge(scen: RigScenario) -> dict[str, Any]:
    return {"model_name": scen.sim.model_name}


def params_for_external_sensor_bridge(scen: RigScenario) -> dict[str, Any]:
    return {
        "model_name": scen.sim.model_name,
        "publish_rate_hz": scen.bridges.external_sensor.publish_rate_hz,
    }


def params_for_gt_pose_bridge(scen: RigScenario) -> dict[str, Any]:
    return {"model_name": scen.sim.model_name}
