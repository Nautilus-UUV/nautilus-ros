"""Rig-side scenario schema.

Authors the simulator plant, fault injection, and world identifiers.
Knows nothing about PID gains or mission setpoints — those live in
`control.py`. Defaults at nominal mirror robot_specs and today's
nautilus_params.yaml, so a RigScenario() with no overrides matches
current behaviour. Phase 7 adds a CI parity test that locks the
nominal mirror to robot_specs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

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
    # Default to 10% headroom from the SDF's 2.5 L mechanical max so the
    # control loop can swing through a 2.0 L range without ever pinning the
    # plugin clamp; MC sweeps can widen or shrink either end.
    bladder_min_m3: float = 0.00025
    bladder_max_m3: float = 0.00225
    bladder_nominal_m3: float = BLADDER_VOLUME_M3
    bcu_motor_min_rpm: int = BCU_MOTOR_MIN_RPM
    bcu_motor_max_rpm: int = BCU_MOTOR_MAX_RPM

    # Internal tank pressure sensor model. The tank feeds the
    # bladder, so tank pressure runs INVERSE to bladder fill: oil pushed out
    # into the bladder drains the tank. Dive tests saw a ~0.7-1.5 barg gauge
    # swing, which the BCU bridge maps linearly onto the tank's oil level
    # (full bladder = drained tank = empty endpoint; empty bladder = full tank
    # = full endpoint). Pure fill — depth does not enter. YAML-only knobs with
    # no robot_specs counterpart, same as bladder_min_m3 / bladder_max_m3.
    tank_pressure_empty_pa: float = 70_000.0  # 0.7 barg, tank drained (bladder full)
    tank_pressure_full_pa: float = 150_000.0  # 1.5 barg, tank full of oil (bladder empty)
    
    # Correction for a hull held below atmospheric (partial vacuum). The
    # sensor reads tank-relative-to-hull, so a sub-atmospheric hull inflates
    # the gauge reading by however far it sits below atmospheric. This offset
    # is added to the reading; 0.0 = hull at atmospheric (no correction).
    tank_pressure_vacuum_offset_pa: float = 0.0


class FaultInjectorSpec(StrictModel):
    """Per-injector knobs for the monotonic degradation ladder.

    The actuator walks down `num_levels` equal effectiveness steps
    (100 % -> 0 %) and never recovers; each step is an independent
    Poisson event with mean `mttf_sec`. Defaults are fault-free.
    """

    # Mean time between successive degradation steps (s). <= 0 disables
    # faults entirely (the actuator stays at 100 % forever).
    mttf_sec: float = 0.0
    # Effectiveness ladder resolution: N steps from healthy (level 0,
    # 100 %) to fully broken (level N, 0 %). 5 => 100/80/60/40/20/0.
    num_levels: int = 5


class FaultsSpec(StrictModel):
    # Add more injectors here as they appear (acu_pitch, imu, ...).
    bcu_rpm: FaultInjectorSpec = Field(default_factory=FaultInjectorSpec)


class NoiseSpec(StrictModel):
    """Placeholder for SDF noise-plugin seeds (Phase 8+ work)."""

    pass


class BcuBridgeSpec(StrictModel):
    """BCU sim-bridge knobs. publish_rate_hz is the rate at which
    pressure/volume telemetry reaches the controllers and so is part
    of the closed-loop data-rate environment a Monte Carlo run sees.
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
    L: float = 1.50       # hull length [m]
    D: float = 0.15       # hull max diameter [m]
    nabla: float = 0.018  # displaced volume [m³]

    # Horizontal fin pair (3)
    b_f: float = 0.33     # horizontal fin span, root-to-tip [m]
    c_f: float = 0.22     # horizontal fin chord, mean [m]
    x_f: float = 0.70     # horizontal fin lever arm from x_CB [m]

    # Top rudder (3)
    b_r: float = 0.22     # rudder span, root-to-tip [m]
    c_r: float = 0.11     # rudder chord, mean [m]
    x_r: float = 0.939    # rudder lever arm from x_CB [m]

    # Foil profile / fin nonlinear envelope (3 — alpha_stall split)
    t_over_c: float = 0.12         # fin thickness ratio [-]
    alpha_stall_horiz: float = 0.17  # stall angle, horizontal pair [rad]
    alpha_stall_rudder: float = 0.17  # stall angle, top rudder [rad]

    # Empirical / flow-physics (4)
    C_d_c: float = 1.10        # 2D cylinder cross-flow drag coeff [-]
    one_plus_k: float = 1.20   # hull form-factor multiplier [-]
    C_p_base: float = 0.08     # base-pressure drag coefficient [-]
    C_La_mult: float = 1.10    # fin lift-slope correction multiplier [-]


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

    # base_link <fluid_added_mass> (model.sdf lines 39-46)
    added_mass_xx: float = 5.30023995
    added_mass_yy: float = 98.529516
    added_mass_zz: float = 110.543023
    added_mass_pp: float = 1.46943124
    added_mass_qq: float = 20.2676439
    added_mass_rr: float = 16.1139805

    # gz-sim-hydrodynamics-system linear drag diagonals (model.sdf lines 135-140)
    drag_xU: float = -108.0
    drag_yV: float = -8.0
    drag_zW: float = -162.0
    drag_kP: float = -13.0
    drag_mQ: float = -32.0
    drag_nR: float = -20.0

    # Three gz-sim-lift-drag-system plugins (model.sdf lines 289 / 353 / 462).
    # Fins share 0.0725 m^2; the top rudder is smaller at 0.0244 m^2.
    left_fin: FinAeroSpec = Field(
        default_factory=lambda: FinAeroSpec(area=0.0725)
    )
    right_fin: FinAeroSpec = Field(
        default_factory=lambda: FinAeroSpec(area=0.0725)
    )
    top_rudder: FinAeroSpec = Field(
        default_factory=lambda: FinAeroSpec(area=0.0244)
    )

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
