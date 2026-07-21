"""Rig-side scenario schema.

Authors the simulator plant, fault injection, and world identifiers.
Knows nothing about PID gains or mission setpoints — those live in
`control.py`. Defaults at nominal mirror robot_specs and today's
nautilus_params.yaml, so a RigScenario() with no overrides matches
current behaviour. Phase 7 adds a CI parity test that locks the
nominal mirror to robot_specs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from py_pkg.robot_specs import (
    BCU_MOTOR_MAX_RPM,
    BCU_MOTOR_MIN_RPM,
    BLADDER_VOLUME_M3,
    VOLUME_PER_REV_M3,
)

from ._shared import StrictModel


class SimSpec(StrictModel):
    """Sim-only identifiers — the Gazebo model and world."""

    model_name: str = "glider_nautilus"
    # The SDF <world name="..."> string, not the world *file* basename.
    # See dave_worlds/worlds/dave_ocean_waves.world line 20.
    world_name: str = "oceans_waves"


class PlantSpec(StrictModel):
    """Ground-truth plant parameters used by HAL bridges + the sim.

    Mirrors robot_specs.py at nominal. MC perturbs these to model
    uncertainty in the *actual* hardware plant. To perturb the
    controller's *model* of the plant independently, see
    ControlScenario.controllers.depth.plant_model.
    """

    volume_per_rev_m3: float = VOLUME_PER_REV_M3
    # Operating-range clamps the bridge enforces on the simulated bladder.
    # Lake-calibrated: the fresh-water neutral point sits at 2.097e-3 m^3
    # (heave_calibration_fit.json), so the ceiling leaves 2% end-stop
    # guard below the 2.5 L mechanical max and the floor matches the
    # deepest real excursion (dive 2 reached ~1.03e-3 m^3). MC sweeps can
    # widen or shrink either end.
    bladder_min_m3: float = 0.001
    bladder_max_m3: float = 0.00245
    bladder_nominal_m3: float = BLADDER_VOLUME_M3
    bcu_motor_min_rpm: int = BCU_MOTOR_MIN_RPM
    bcu_motor_max_rpm: int = BCU_MOTOR_MAX_RPM

    # Internal tank pressure sensor model. The tank feeds the
    # bladder, so tank pressure runs INVERSE to bladder fill: oil pushed out
    # into the bladder drains the tank. The BCU bridge maps the tank's oil
    # level linearly between these endpoints (full bladder = drained tank =
    # empty endpoint; empty bladder = full tank = full endpoint). Pure fill —
    # depth does not enter. Endpoints mirror the 2026-06-24 lake-test `init`
    # calibration (tank_empty_pa=97800, tank_full_pa=190000, the values the
    # real oil-circuit sensor reports on the wire). YAML-only knobs with
    # no robot_specs counterpart, same as bladder_min_m3 / bladder_max_m3.
    tank_pressure_empty_pa: float = 97_800.0  # tank drained (bladder full)
    tank_pressure_full_pa: float = 190_000.0  # tank full of oil (bladder empty)

    # Pump/tank transient dynamics (plant_dynamics.py). Fitted to the
    # 2026-06-24 lake test — provenance:
    # UG-anomaly_detection/analysis/lake_test_jun24/pump_transient_fit.json.
    # The real pump holds still for a dead time after a command, then ramps
    # at a constant slew to the target (no exponential tail); <= 0 disables
    # either term (exact pre-calibration passthrough). YAML-only knobs with
    # no robot_specs counterpart, same as the tank endpoints above.
    # 2026-06-24 lake fit: delay 1.057 s (raw 1.308 minus 0.251 s feedback
    # -reporting latency), slew 512.1 rpm/s (per-event 450-547). Stage-A
    # acceptance PASSED (pooled RMS 83 rpm, held-out 1.91x fit RMS).
    pump_response_delay_s: float = 1.057
    pump_slew_rpm_per_s: float = 512.1
    # EPOS4 velocity-loop crest: the real shaft overshoots a step command
    # by this fraction of the step before settling (lake feedback peaks at
    # 3113 rpm on a 3000 command -> ~0.038). 0.0 disables (exact
    # delay+slew model, the pre-v2 behavior); sweeps sample it so sim
    # feedback can exceed the command rail the way hardware does.
    pump_overshoot_frac: float = 0.0
    # Tank sensor curve shape: "linear" is the legacy straight-line oil
    # map; "gaslaw" is the isothermal air-cushion hyperbola the lake
    # traces show (flat near tank-empty, steep near tank-full). Default
    # is the fit's pre-registered adoption (gas_free, RMS 4.9 kPa on the
    # entry windows vs 7.6 kPa linear). NOTE: Stage B FAILED its 2 kPa
    # gate — the residual is pump volumetric slip (see
    # pump_transient_findings.md), flagged for a follow-up model.
    tank_map_shape: Literal["linear", "gaslaw"] = "gaslaw"
    # Air-cushion volume at the tank-empty endpoint for the gaslaw map.
    # <= 0 pins the cushion so the curve hits both calibrated endpoints
    # exactly (zero extra dof — same "<= 0 disables" convention as the
    # pump knobs above); a positive value frees the curvature and must
    # exceed bladder_max_m3 - bladder_min_m3. Default = fitted free
    # cushion (its bladder-min-rail reading lands at ~187 kPa, within
    # ~3 kPa of the nominal full endpoint).
    tank_air_volume_m3: float = 3.041025e-3


SensorFaultKind = Literal["none", "bias", "drift", "stuck", "dropout"]

# Wire spelling of sensor_faults.FAULT_SCHEDULE_SHAPES (kept a literal so
# it stays statically checkable; test_fault_schedule locks the two
# together, and anomaly.ONSET_SHAPES derives from this one).
FaultScheduleShape = Literal["step", "ramp", "intermittent"]


class FaultScheduleSpec(StrictModel):
    """Onset + progression envelope for one fault (sensor_faults.FaultSchedule).

    ``m(t)`` in [0, 1] scales the fault's one drawn severity:

      step         -> 0 before `onset_s` (seconds from bridge start),
                      1 after. The default (`onset_s == 0`) is the
                      original whole-run behavior.
      ramp         -> 0 before onset, linear to 1 over `ramp_s`, then 1.
      intermittent -> after onset, on for the first `duty * period_s`
                      of each period, off otherwise.

    Severity itself never changes mid-run — the envelope gates when the
    one drawn magnitude is felt.
    """

    onset_s: float = 0.0
    shape: FaultScheduleShape = "step"
    ramp_s: float = 0.0
    period_s: float = 0.0
    duty: float = 0.5

    @model_validator(mode="after")
    def _check_shape_fields(self) -> "FaultScheduleSpec":
        if self.onset_s < 0.0:
            raise ValueError(f"onset_s must be >= 0, got {self.onset_s}")
        if self.shape == "step":
            if self.ramp_s != 0.0 or self.period_s != 0.0:
                raise ValueError("shape='step' uses neither ramp_s nor period_s")
        elif self.shape == "ramp":
            if self.ramp_s <= 0.0:
                raise ValueError("shape='ramp' requires ramp_s > 0")
            if self.period_s != 0.0:
                raise ValueError("shape='ramp' does not use period_s")
        else:  # intermittent
            if self.period_s <= 0.0:
                raise ValueError("shape='intermittent' requires period_s > 0")
            if not (0.0 < self.duty < 1.0):
                raise ValueError(
                    f"shape='intermittent' requires 0 < duty < 1, got {self.duty}"
                )
            if self.ramp_s != 0.0:
                raise ValueError("shape='intermittent' does not use ramp_s")
        return self

    @property
    def is_default(self) -> bool:
        return self == FaultScheduleSpec()


class BcuPumpFaultSpec(StrictModel):
    """Persistent whole-run pump degradation.

    The commanded pump RPM is multiplied by `effectiveness`, before the
    pump transient (PumpDynamics). The drawn severity is constant for
    the run; `schedule` gates when it is felt (default: step at t=0 —
    the original whole-run behavior). 1.0 is a healthy pump. A degraded
    pump *transient* (longer dead time, slower slew) is authored through
    the existing `PlantSpec.pump_response_delay_s` /
    `pump_slew_rpm_per_s` knobs instead.
    """

    effectiveness: float = 1.0
    schedule: FaultScheduleSpec = Field(default_factory=FaultScheduleSpec)

    @model_validator(mode="after")
    def _check_range(self) -> "BcuPumpFaultSpec":
        if not (0.0 < self.effectiveness <= 1.0):
            raise ValueError(
                f"effectiveness must be in (0, 1], got {self.effectiveness}"
            )
        if self.effectiveness == 1.0 and not self.schedule.is_default:
            raise ValueError(
                "a healthy pump (effectiveness == 1.0) cannot carry a schedule"
            )
        return self


class SensorFaultSpec(StrictModel):
    """One persistent measurement-fault archetype on one pressure channel.

    The drawn severity is constant for the run; `schedule` gates when it
    is felt (default: from the first published sample — the original
    whole-run behavior). Kind semantics:

      bias    -> `magnitude` is a signed additive offset [Pa]
      drift   -> `magnitude` is a signed ramp rate [Pa/s] from bridge start
      stuck   -> the channel repeats its first reported sample forever
      dropout -> each publish of this channel is suppressed i.i.d. with
                 probability `drop_prob`

    The bridge applies bias/drift to the true value *before* the
    calibrated noise+quantization chain (a transducer-stage fault; the
    digitizer always runs last); stuck latches the post-chain reading,
    so a frozen channel stays on its quantization comb.
    """

    kind: SensorFaultKind = "none"
    magnitude: float = 0.0
    drop_prob: float = 0.0
    schedule: FaultScheduleSpec = Field(default_factory=FaultScheduleSpec)

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "SensorFaultSpec":
        if self.kind in ("none", "stuck"):
            if self.magnitude != 0.0 or self.drop_prob != 0.0:
                raise ValueError(
                    f"kind={self.kind!r} uses neither magnitude nor drop_prob"
                )
        elif self.kind in ("bias", "drift"):
            if self.magnitude == 0.0:
                raise ValueError(f"kind={self.kind!r} requires a nonzero magnitude")
            if self.drop_prob != 0.0:
                raise ValueError(f"kind={self.kind!r} does not use drop_prob")
        else:  # dropout
            if not (0.0 < self.drop_prob <= 1.0):
                raise ValueError("kind='dropout' requires 0 < drop_prob <= 1")
            if self.magnitude != 0.0:
                raise ValueError("kind='dropout' does not use magnitude")
        if not self.schedule.is_default:
            if self.kind == "none":
                raise ValueError("kind='none' cannot carry a schedule")
            # drift is already a rate-ramp (its schedule only shifts the
            # epoch) and a shaped "stuck" has no unambiguous latch point.
            if self.kind in ("drift", "stuck") and self.schedule.shape != "step":
                raise ValueError(
                    f"kind={self.kind!r} supports only shape='step' schedules"
                )
        return self


class SensorFaultsSpec(StrictModel):
    """Measurement faults, one optional archetype per pressure channel.

    Deliberately pressure-only: the IMU is excluded from sensor-fault
    injection by design (it is still covered by the comms fault).
    """

    external_pressure: SensorFaultSpec = Field(default_factory=SensorFaultSpec)
    tank_pressure: SensorFaultSpec = Field(default_factory=SensorFaultSpec)


class CommsFaultSpec(StrictModel):
    """Uniform bridge-layer link degradation.

    Every bridged sensor/telemetry publish (/imu, /external/pressure,
    /bcu/pressure, /bcu/volume, feedback, flow rate) is dropped i.i.d.
    Bernoulli(drop_prob) — the archetype of frame loss on the shared
    Pi<->STM link. 0.0 = healthy link. The Gazebo-facing plant command
    path and the /anomaly/label ground-truth stream are never gated.
    """

    drop_prob: float = 0.0

    @model_validator(mode="after")
    def _check_range(self) -> "CommsFaultSpec":
        if not (0.0 <= self.drop_prob < 1.0):
            raise ValueError(f"drop_prob must be in [0, 1), got {self.drop_prob}")
        return self


class FaultsSpec(StrictModel):
    """Persistent per-run faults at one constant drawn severity.

    Replaces the old Poisson-MTTF escalation ladder: a run is either
    healthy or carries its fault, whose severity never changes mid-run.
    Pump and sensor faults may carry a `schedule` (onset + step/ramp/
    intermittent envelope) gating WHEN the severity is felt; the default
    schedule is the original active-from-t=0 whole-run behavior. The
    anomaly class of a run is recorded in `Scenario.anomaly` (validated
    for consistency against these blocks).
    """

    bcu_pump: BcuPumpFaultSpec = Field(default_factory=BcuPumpFaultSpec)
    sensors: SensorFaultsSpec = Field(default_factory=SensorFaultsSpec)
    comms: CommsFaultSpec = Field(default_factory=CommsFaultSpec)


class ImuNoiseSpec(StrictModel):
    """Per-axis additive Gaussian sigma for the sim IMU bridge, (x, y, z).

    sigma <= 0 disables that axis (exact passthrough).
    """

    accel_sigma_mps2: tuple[float, float, float] = (0.01215, 0.02372, 0.009544)
    gyro_sigma_rads: tuple[float, float, float] = (0.006541, 0.001033, 0.0008779)


class PressureNoiseSpec(StrictModel):
    """Gaussian sigma + digitizer quantization for one pressure channel.

    Gaussian noise is applied first, then the value is rounded to the
    quantization grid (a real ADC chain). sigma_pa <= 0 disables the
    Gaussian term; quantization_pa <= 0 disables the rounding.
    """

    sigma_pa: float
    quantization_pa: float


class NoiseSpec(StrictModel):
    """Sensor noise injected by the HAL sim bridges (Gaussian + quantization).

    Defaults are the values fitted from the 2026-06-24 lake test —
    provenance: UG-anomaly_detection/lake_test_jun24/investigation/
    noise_characterization.json (dive-4 steady-glide window, rolling-
    median-detrended residuals; Sheppard-corrected sigmas for the
    quantized pressure channels). External pressure is quantization-
    dominated in the real data (sub-LSB Gaussian), so its sigma is 0.0
    and the 100 Pa comb carries the noise.

    `enabled: false` zeroes every channel at compile time, so the
    bridges pass samples through untouched; per-channel sigma/step <= 0
    disables just that term.
    """

    enabled: bool = True
    imu: ImuNoiseSpec = Field(default_factory=ImuNoiseSpec)
    external_pressure: PressureNoiseSpec = Field(
        default_factory=lambda: PressureNoiseSpec(sigma_pa=0.0, quantization_pa=100.0)
    )
    tank_pressure: PressureNoiseSpec = Field(
        default_factory=lambda: PressureNoiseSpec(sigma_pa=353.0, quantization_pa=600.0)
    )


class BcuBridgeSpec(StrictModel):
    """BCU sim-bridge knobs. publish_rate_hz is the rate at which
    pressure/volume telemetry reaches the controllers and so is part
    of the closed-loop data-rate environment a Monte Carlo run sees.
    The same timer steps the simulated pump plant (dead time/slew,
    valve gate, volume integral), so lowering it also coarsens the
    plant integration step.
    """

    publish_rate_hz: int = 10


class ExternalSensorBridgeSpec(StrictModel):
    """External-sensor sim-bridge knob. publish_rate_hz drives the
    pressure stream feeding the depth loop.
    """

    publish_rate_hz: int = 10


class BridgesSpec(StrictModel):
    bcu: BcuBridgeSpec = Field(default_factory=BcuBridgeSpec)
    external_sensor: ExternalSensorBridgeSpec = Field(
        default_factory=ExternalSensorBridgeSpec
    )


class PhysicsKnobs(StrictModel):
    """Sixteen independent physics knobs for domain-randomization sweeps.

    These are the *causes* sampled in log-space; the SDF coefficients
    are computed deterministically from them via the forward map in
    ``compile.py``.  Stored in the emitted YAML for provenance — the
    render path (``render_sdf.py``) ignores this block.

    Defaults are the nominal operating point.  See
    ``doc/hydrodynamic_coefficient_sampling_plan.md`` §1 for derivation.
    """

    # Body geometry (3)
    L: float = 1.50  # hull length [m]
    D: float = 0.15  # hull max diameter [m]
    nabla: float = 0.018  # displaced volume [m³]

    # Horizontal fin pair (3)
    b_f: float = 0.33  # horizontal fin span, root-to-tip [m]
    c_f: float = 0.22  # horizontal fin chord, mean [m]
    x_f: float = 0.70  # horizontal fin lever arm from x_CB [m]

    # Top rudder (3)
    b_r: float = 0.22  # rudder span, root-to-tip [m]
    c_r: float = 0.11  # rudder chord, mean [m]
    x_r: float = 0.939  # rudder lever arm from x_CB [m]

    # Foil profile / fin nonlinear envelope (3 — alpha_stall split)
    t_over_c: float = 0.12  # fin thickness ratio [-]
    alpha_stall_horiz: float = 0.17  # stall angle, horizontal pair [rad]
    alpha_stall_rudder: float = 0.17  # stall angle, top rudder [rad]

    # Empirical / flow-physics (4)
    C_d_c: float = 1.10  # 2D cylinder cross-flow drag coeff [-]
    one_plus_k: float = 1.20  # hull form-factor multiplier [-]
    C_p_base: float = 0.08  # base-pressure drag coefficient [-]
    C_La_mult: float = 1.10  # fin lift-slope correction multiplier [-]


class FinAeroSpec(StrictModel):
    """Lift/drag plugin parameters for one control surface.

    Defaults mirror the three identical `<plugin name="...LiftDrag">`
    blocks in the live `model.sdf`. `area` differs between fins
    (0.0725 m^2) and the top rudder (0.0244 m^2), so it has no class
    default — the parent `HydrodynamicsSpec` supplies it per surface.
    """

    cla: float = 4.13
    cla_stall: float = -1.1
    cda: float = 0.2
    cda_stall: float = 0.03
    alpha_stall: float = 0.17
    a0: float = 0.0
    area: float


class HydrodynamicsSpec(StrictModel):
    """SDF hydrodynamic coefficients, exposed for sampled parameter sweeps.

    Defaults mirror the current `model.sdf` verbatim so that rendering
    the Jinja template with `HydrodynamicsSpec()` reproduces the
    canonical SDF byte-for-byte (locked in by the parity Tier 3 test).

    `RigScenario.hydrodynamics` is `Optional[HydrodynamicsSpec]` —
    scenarios that omit the block (nominal/baseline) skip the Jinja
    render entirely and spawn the canonical SDF unchanged. Scenarios
    that author a block trigger the render path. That's the opt-in
    switch: sampled sweeps add `rig.hydrodynamics:`; everyday runs
    don't. Sampling strategy (LHS, Sobol, Halton, plain MC,
    hand-picked corners) is the orchestrator's call — the spec is the
    same shape regardless.
    """

    # Fluid density seen by the model-side plugins: BuoyancyEngine
    # <fluid_density>, Hydrodynamics <water_density>, and the three
    # LiftDrag <air_density>. Fresh water — the 2026-06-24 lake test is
    # the deployment environment the sim is calibrated against. The
    # world buoyancy plugin's <default_density> (static hull buoyancy)
    # is NOT templated — it is hardcoded in dave_ocean_waves.world and
    # must be changed together with this value when switching between
    # lake and sea water.
    fluid_density: float = 1000.0

    # base_link <fluid_added_mass> (model.sdf lines 39-46)
    added_mass_xx: float = 5.30023995
    added_mass_yy: float = 98.529516
    added_mass_zz: float = 110.543023
    added_mass_pp: float = 1.46943124
    added_mass_qq: float = 20.2676439
    added_mass_rr: float = 16.1139805

    # gz-sim-hydrodynamics-system linear drag diagonals. drag_zW is the
    # lake-fit heave value (quadratic variant of
    # lake_test_jun24/investigation/heave_calibration_fit.json); the
    # other five keep their strip-theory authoring.
    drag_xU: float = -108.0
    drag_yV: float = -8.0
    drag_zW: float = -52.2
    drag_kP: float = -13.0
    drag_mQ: float = -32.0
    drag_nR: float = -20.0

    # gz-sim-hydrodynamics-system quadratic drag diagonals. Deliberately
    # outside the forward-map slot registries (_BODY_SLOTS) — the strip
    # theory map only produces linear coefficients; these are frozen at
    # spec defaults unless a scenario sets them directly. drag_zWabsW is
    # fitted together with drag_zW from the lake dives.
    drag_xUabsU: float = 0.0
    drag_yVabsV: float = 0.0
    drag_zWabsW: float = -436.7
    drag_kPabsP: float = 0.0
    drag_mQabsQ: float = 0.0
    drag_nRabsR: float = 0.0

    # In real life, we also trim the weight slightly for each dive.
    # We have these variables here to do the same thing, and achieve the buoyancy profile
    # we desire.
    trim_mass_bow: float = 3.157
    trim_mass_stern: float = 0.2765
    trim_mass_bladder: float = 0.001

    # BuoyancyEngine <default_volume>: bladder fill at spawn.
    bladder_spawn_volume_m3: float = 0.0022

    # Three gz-sim-lift-drag-system plugins (model.sdf lines 289 / 353 / 462).
    # Fins share 0.0725 m^2; the top rudder is smaller at 0.0244 m^2.
    left_fin: FinAeroSpec = Field(default_factory=lambda: FinAeroSpec(area=0.0725))
    right_fin: FinAeroSpec = Field(default_factory=lambda: FinAeroSpec(area=0.0725))
    top_rudder: FinAeroSpec = Field(default_factory=lambda: FinAeroSpec(area=0.0244))

    # Physics knobs that generated the SDF coefficients above.  Stored
    # for provenance and Sobol analysis; the render path ignores this.
    knobs: Optional[PhysicsKnobs] = None


class RigScenario(StrictModel):
    sim: SimSpec = Field(default_factory=SimSpec)
    plant: PlantSpec = Field(default_factory=PlantSpec)
    faults: FaultsSpec = Field(default_factory=FaultsSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    bridges: BridgesSpec = Field(default_factory=BridgesSpec)
    # Optional by design: when absent, the launch spawns the canonical
    # `model.sdf` unchanged (no Jinja render). When present, the launch
    # renders `model.sdf.jinja` with these values.
    hydrodynamics: Optional[HydrodynamicsSpec] = None
