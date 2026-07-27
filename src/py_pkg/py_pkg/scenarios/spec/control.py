"""Control-side scenario schema.

Authors what controllers and the mission look like for one run.
Knows nothing about faults, sim plant, or world identifiers — those
live in `rig.py`.

The BCU depth loop is bang-bang, so it has no gains and no model of the
plant: it compares the depth error against one deadband and commands the
pump at one speed in one direction. That is why `DepthSpec` is four
fields where it used to be twenty — everything else was either PID tuning
or machinery to stop a modulated command from chattering.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from py_pkg.robot_specs import (
    ACU_ROLL_MAX_ANGLE_DEG,
    BCU_MOTOR_MAX_RPM,
)

from ._shared import StrictModel


class DepthSpec(StrictModel):
    """BCU depth loop — bang-bang on the sign of the depth error.

    The pump runs at `pump_rpm` in whichever direction closes the error,
    and keeps running until either the tank reaches `tank_stop_band` of an
    endpoint or the mission advances to a leg whose target flips the sign.
    Nothing modulates, so there is no chatter to suppress.
    """

    frequency_hz: int = 10

    # The single bang-bang command magnitude. Defaults to the hardware
    # ceiling: full authority is the point of the law, and a leg that ends
    # sooner is a leg with less pump-on time. A sweep can lower it to study
    # a weaker pump without touching robot_specs.
    pump_rpm: int = BCU_MOTOR_MAX_RPM

    # Below this |error| the pump is off and both valves are shut. NOT
    # hysteresis and NOT a latch -- it exists because SURFACE targets gauge
    # 0 Pa, where the reading bobs across zero on sensor noise; without it
    # the error's sign would flip on that noise and the pump would start
    # filling the tank at the surface.
    #
    # Sizing: 2000 Pa is ~0.2 m, comfortably above the depth sensor's 100 Pa
    # quantisation and surface bob, and comfortably below every mission's
    # arrival tolerance (TRIM's NEAR_GOAL_PA is 4903 Pa; the sawtooth /
    # staircase / surface bands are 7845 Pa), so it can never truncate a leg
    # -- the mission always turns before the deadband is reached.
    deadband_pa: float = 2000.0

    # Latching tank-endpoint cutoff (TankLimitGuard), as a fraction of the
    # empty->full span. This is the stop condition every bang-bang leg runs
    # into: pump one way until the tank rails here, then coast until the
    # mission's next leg reverses the command. It also fixes the bench-test
    # bug -- with the vehicle held on the bench it can't dive, so the loop
    # drives the tank onto the guard and parks there, and a bare threshold
    # sitting on tank-sensor noise chatters the valves. The latch holds
    # through that noise; only a command reversal or a reset releases it.
    tank_stop_band: float = 0.05

    @model_validator(mode="after")
    def _check_depth_params(self) -> DepthSpec:
        if self.pump_rpm <= 0:
            raise ValueError(f"pump_rpm must be > 0 (got {self.pump_rpm})")
        if self.deadband_pa < 0.0:
            raise ValueError(f"deadband_pa must be >= 0 (got {self.deadband_pa})")
        if not 0.0 <= self.tank_stop_band < 0.5:
            raise ValueError(
                f"tank_stop_band must be in [0, 0.5) (got {self.tank_stop_band})"
            )
        return self


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
    """Raw-IMU EMA prefilter — the first stage of the estimation front-end.

    `alpha` is the EMA coefficient (0 = very smooth, 1 = no filtering) applied
    to the six accel/gyro components. It sets the bandwidth of the signal the
    attitude estimator sees, since `attitude_node` adds no filter of its own,
    and it is also what band-limits the stream the MQTT egress decimates to
    10 Hz by sample-dropping.

    0.15 puts the -3 dB cutoff at ~5 Hz for a 200 Hz IMU
    (alpha = 1 - exp(-2*pi*fc/fs)) with ~28 ms of group delay
    (~(1-alpha)/alpha samples).
    """

    alpha: float = 0.15

    @model_validator(mode="after")
    def _check_prefilter_params(self) -> ImuPrefilterSpec:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                f"prefilter alpha must be in (0, 1] (got {self.alpha}); "
                "0 would freeze the filter on its first sample"
            )
        return self


class AttitudeSpec(StrictModel):
    pass  # same


class EstimatorSpec(StrictModel):
    prefilter: ImuPrefilterSpec = Field(default_factory=ImuPrefilterSpec)
    attitude: AttitudeSpec = Field(default_factory=AttitudeSpec)


class ControlScenario(StrictModel):
    controllers: ControllersSpec = Field(default_factory=ControllersSpec)
    estimator: EstimatorSpec = Field(default_factory=EstimatorSpec)
