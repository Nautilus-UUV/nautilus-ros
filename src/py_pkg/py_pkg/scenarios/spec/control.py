"""Control-side scenario schema.

Authors what controllers and the mission look like for one run.
Knows nothing about faults, sim plant, or world identifiers — those
live in `rig.py`. Defaults reproduce today's literal values in
`pid/depth_config.py` and `pid/acu_roll_config.py`, with one deliberate
exception: the BCU pump deadband (`DepthPlantModel.min_rpm` /
`min_operating_rpm`) defaults to active (500/1000), because the pump's
minimum reliable speed is a hardware floor that applies on every run.

DepthPlantModel is deliberately separate from RigScenario.plant.
At nominal both mirror robot_specs; MC perturbs them independently
to study controller-model-vs-actual-plant mismatch.
"""

from __future__ import annotations

from pydantic import Field

from py_pkg.robot_specs import (
    ACU_ROLL_MAX_ANGLE_DEG,
    BCU_MOTOR_MAX_RPM,
    BLADDER_VOLUME_M3,
)

from ._shared import StrictModel


class PIDPressureSpec(StrictModel):
    """Inner-loop pressure PID gains for the depth controller.

    kp drives the main pump speed — tuned so a 2.5 m error runs the
    pump at ~4000 RPM. kd damps the approach against quantised
    pressure-sensor noise; `derivative_filter` is the smoothing
    coefficient that makes kd usable at all. ki is kept tiny on
    purpose — long deep dives accumulate huge error, and a larger ki
    would memorise that error and overshoot the target.
    """

    kp: float = 4.0e-7
    ki: float = 1.0e-8
    kd: float = 1.5e-5
    integral_limits: tuple[float, float] = (-0.0025, 0.0025)
    output_limits: tuple[float, float] = (-0.010, 0.010)
    derivative_filter: float = 0.3


class DepthPlantModel(StrictModel):
    """The controller's *model* of the buoyancy plant.

    Distinct from RigScenario.plant: that is what the simulator
    actually does. This is what the controller thinks the plant is.
    Identical at nominal; MC sweeps can perturb either independently.

    pump_efficiency is the volumetric efficiency the controller assumes
    when inverting q (flow ratio) → motor RPM. Lives here, not in
    robot_specs, because it's a tunable plant-model parameter — the
    real pump's efficiency drifts with wear and operating point, and
    MC sweeps want to study controller-vs-actual mismatch on it.
    """

    bladder_nominal_m3: float = BLADDER_VOLUME_M3
    # Matches the sim spawn (SDF default_volume 2.2e-3 of the 2.5e-3
    # bladder) and the real pre-dive float state, so a bare `ros2 run`
    # controller starts from the same belief as the plant.
    initial_proportion_full: float = 0.88
    # Pump deadband, applied to the RPM command in the control loop. The
    # pump can't run reliably at low speed, so the loop snaps the command
    # into three regions: |rpm| < min_rpm -> 0,
    # min_rpm <= |rpm| < min_operating_rpm -> +/-min_operating_rpm, and
    # everything above passes through saturated to +/-max_rpm. On by
    # default — the floor is a hardware fact, not a per-scenario choice —
    # though MC sweeps may override it. min_rpm / min_operating_rpm are
    # control knobs, not robot_specs mirrors; only max_rpm mirrors the
    # hardware ceiling.
    min_rpm: int = 500
    min_operating_rpm: int = 1000
    max_rpm: int = BCU_MOTOR_MAX_RPM
    # Nominal pump volumetric efficiency between 1000 and 3000 RPM.
    pump_efficiency: float = 0.93


class DepthSpec(StrictModel):
    frequency_hz: int = 10
    pid_pressure: PIDPressureSpec = Field(default_factory=PIDPressureSpec)
    plant_model: DepthPlantModel = Field(default_factory=DepthPlantModel)


class AcuPitchSpec(StrictModel):
    """ACU pitch axis — soft-saturation limits for the bang-bang loop.

    DEPRECATED / NOT IMPLEMENTED IN SIM: the simulated ACU actuator was
    removed (the glider_nautilus model is now static, symmetric, BCU-only).
    These limits are retained for the real-hardware ACU path and are not
    exercised in simulation.

    The pitch loop has no PID; the node throws the mass-shifter to one
    of two extremes pulled from `output_limits` (front, back) in metres.
    Wire format is Int16 mm; the node multiplies by 1000.

    Lives here, not in robot_specs, because the saturation is a control
    tuning knob — the *mechanical* travel limit (ACU_PITCH_MAX_TRAVEL_M)
    is the hardware fact and stays in robot_specs. The invariant
    `output_limits[i] in [-ACU_PITCH_MAX_TRAVEL_M, 0]` is the caller's
    responsibility; nothing enforces it at load time.

    `frequency_hz` is not on this dataclass — pitch and roll share the
    ACU node's timer, set by AcuRollSpec.
    """

    output_limits: tuple[float, float] = (-0.11, -0.01)


class AcuRollSpec(StrictModel):
    """ACU roll axis PID.

    DEPRECATED / NOT IMPLEMENTED IN SIM: the simulated ACU actuator was
    removed (the glider_nautilus model is now static, symmetric, BCU-only).
    These gains are retained for the real-hardware ACU path and are not
    exercised in simulation.

    Conservative gains so the roll loop doesn't induce pitch
    coupling. command_tolerance sits just above estimator roll noise
    (~0.5 deg) so steady-state noise alone doesn't republish to the
    EPOS bus.

    `frequency_hz` ticks the outer ACU control loop (pitch bang-bang +
    roll PID). The pitch axis lives in the same node so they share a
    timer; one frequency field covers both.
    """

    frequency_hz: int = 10
    kp: float = 0.5
    ki: float = 0.005
    kd: float = 0.05
    command_tolerance: float = 0.5
    integral_limits: tuple[float, float] = (-30.0, 30.0)
    output_limits: tuple[float, float] = (
        -ACU_ROLL_MAX_ANGLE_DEG,
        ACU_ROLL_MAX_ANGLE_DEG,
    )
    derivative_filter: float = 0.2


class ControllersSpec(StrictModel):
    depth: DepthSpec = Field(default_factory=DepthSpec)
    acu_pitch: AcuPitchSpec = Field(default_factory=AcuPitchSpec)
    acu_roll: AcuRollSpec = Field(default_factory=AcuRollSpec)


class ImuPrefilterSpec(StrictModel):
    pass  # no tunables yet; placeholder so future params have a home


class AttitudeSpec(StrictModel):
    pass  # same


class EstimatorSpec(StrictModel):
    prefilter: ImuPrefilterSpec = Field(default_factory=ImuPrefilterSpec)
    attitude: AttitudeSpec = Field(default_factory=AttitudeSpec)


class ControlScenario(StrictModel):
    controllers: ControllersSpec = Field(default_factory=ControllersSpec)
    estimator: EstimatorSpec = Field(default_factory=EstimatorSpec)
