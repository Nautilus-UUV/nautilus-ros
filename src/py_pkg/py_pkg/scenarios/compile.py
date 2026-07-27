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

from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from rclpy.node import Node
import functools
import logging
import math
from pathlib import Path

import yaml

from .seed import derive_seed
from .spec.control import (
    AcuPitchSpec,
    AcuRollSpec,
    ControlScenario,
    DepthSpec,
    ImuPrefilterSpec,
)
from .spec.rig import (
    FaultScheduleSpec,
    FinAeroSpec,
    HydrodynamicsSpec,
    ImuNoiseSpec,
    NoiseSpec,
    PhysicsKnobs,
    PressureNoiseSpec,
    RigScenario,
    SensorFaultSpec,
)
from .spec.scenario import Scenario

# ---------------------------------------------------------------------------
# Control-side: forward (params_for_*) and inverse (*_spec_from_node)
# ---------------------------------------------------------------------------


def _param(node: Node, name: str, default: Any) -> Any:
    """Declare one ROS parameter and return its override-resolved value."""
    return node.declare_parameter(name, default).value


def params_for_bcu_node(scen: ControlScenario) -> dict[str, Any]:
    d = scen.controllers.depth
    return {
        "frequency_hz": d.frequency_hz,
        "pump_rpm": d.pump_rpm,
        "deadband_pa": d.deadband_pa,
        "tank_stop_band": d.tank_stop_band,
    }


def bcu_spec_from_node(node: Node) -> DepthSpec:
    """Read depth-controller params off `node` into a typed DepthSpec.

    Defaults come from the spec, so a bare `ros2 run bcu_node` with no
    scenario behaves exactly like the installed nominal.yaml.
    """

    default = DepthSpec()
    return DepthSpec(
        frequency_hz=_param(node, "frequency_hz", default.frequency_hz),
        pump_rpm=_param(node, "pump_rpm", default.pump_rpm),
        deadband_pa=_param(node, "deadband_pa", default.deadband_pa),
        tank_stop_band=_param(node, "tank_stop_band", default.tank_stop_band),
    )


def params_for_acu_node(scen: ControlScenario) -> dict[str, Any]:
    # DEPRECATED / NOT IMPLEMENTED IN SIM: the sim ACU actuator was removed
    # (glider_nautilus is now static, symmetric, BCU-only). Retained for the
    # real-hardware ACU path (acu_node -> can_com_node); unused in simulation.
    p = scen.controllers.acu_pitch
    r = scen.controllers.acu_roll
    return {
        "acu_pitch.output_limit_low": p.output_limits[0],
        "acu_pitch.output_limit_high": p.output_limits[1],
        "acu_roll.frequency_hz": r.frequency_hz,
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

    DEPRECATED / NOT IMPLEMENTED IN SIM: the sim ACU actuator was removed;
    retained for the real-hardware ACU path only.

    The pitch axis is bang-bang, so the only knobs are the two
    output_limits (front, back) in metres.
    """

    default = AcuPitchSpec()

    return AcuPitchSpec(
        output_limits=(
            _param(node, "acu_pitch.output_limit_low", default.output_limits[0]),
            _param(node, "acu_pitch.output_limit_high", default.output_limits[1]),
        ),
    )


def acu_roll_spec_from_node(node: Node) -> AcuRollSpec:
    """Read ACU roll-controller params off `node` into a typed AcuRollSpec.

    DEPRECATED / NOT IMPLEMENTED IN SIM: the sim ACU actuator was removed;
    retained for the real-hardware ACU path only.

    Defaults match the literal values that used to live in the
    `init_acu_roll` dict.
    """

    default = AcuRollSpec()

    return AcuRollSpec(
        frequency_hz=_param(node, "acu_roll.frequency_hz", default.frequency_hz),
        kp=_param(node, "acu_roll.kp", default.kp),
        ki=_param(node, "acu_roll.ki", default.ki),
        kd=_param(node, "acu_roll.kd", default.kd),
        command_tolerance=_param(
            node, "acu_roll.command_tolerance", default.command_tolerance
        ),
        integral_limits=(
            _param(node, "acu_roll.integral_limit_low", default.integral_limits[0]),
            _param(node, "acu_roll.integral_limit_high", default.integral_limits[1]),
        ),
        output_limits=(
            _param(node, "acu_roll.output_limit_low", default.output_limits[0]),
            _param(node, "acu_roll.output_limit_high", default.output_limits[1]),
        ),
        derivative_filter=_param(
            node, "acu_roll.derivative_filter", default.derivative_filter
        ),
    )


def params_for_imu_prefilter(scen: ControlScenario) -> dict[str, Any]:
    return {"alpha": scen.estimator.prefilter.alpha}


def imu_prefilter_spec_from_node(node: Node) -> ImuPrefilterSpec:
    """Read prefilter params off `node` into a typed ImuPrefilterSpec."""

    default = ImuPrefilterSpec()

    return ImuPrefilterSpec(alpha=_param(node, "alpha", default.alpha))


# ---------------------------------------------------------------------------
# Rig-side: HAL bridges
# ---------------------------------------------------------------------------


_NOISE_OFF = NoiseSpec(
    enabled=False,
    imu=ImuNoiseSpec(accel_sigma_mps2=(0.0, 0.0, 0.0), gyro_sigma_rads=(0.0, 0.0, 0.0)),
    external_pressure=PressureNoiseSpec(sigma_pa=0.0, quantization_pa=0.0),
    tank_pressure=PressureNoiseSpec(sigma_pa=0.0, quantization_pa=0.0),
)


def _effective_noise(scen: RigScenario) -> NoiseSpec:
    """Resolve the scenario's `enabled` gate to plain per-channel numbers.

    The bridges never see a boolean: a disabled scenario compiles to
    all-zero sigmas/steps, which the noise model treats as exact
    passthrough. Keeps the gate testable at Tier 1.
    """
    return scen.noise if scen.noise.enabled else _NOISE_OFF


def _fault_schedule_params(prefix: str, s: FaultScheduleSpec) -> dict[str, Any]:
    """Wire params for one fault's onset/progression envelope.

    Emitted unconditionally (defaults are the inert step-at-t=0
    schedule) so the wire shape never depends on whether a run is
    faulted.
    """
    return {
        f"{prefix}onset_s": s.onset_s,
        f"{prefix}shape": s.shape,
        f"{prefix}ramp_s": s.ramp_s,
        f"{prefix}period_s": s.period_s,
        f"{prefix}duty": s.duty,
    }


def _sensor_fault_params(
    prefix: str, f: SensorFaultSpec, parent_seed: int, component_id: str
) -> dict[str, Any]:
    """Wire params for one persistent sensor-fault channel.

    The seed is emitted unconditionally (only dropout consumes RNG;
    harmless otherwise) so the wire shape never depends on the kind.
    """
    return {
        f"{prefix}fault_kind": f.kind,
        f"{prefix}fault_magnitude": f.magnitude,
        f"{prefix}fault_drop_prob": f.drop_prob,
        f"{prefix}fault_seed": derive_seed(parent_seed, component_id),
        **_fault_schedule_params(f"{prefix}fault_", f.schedule),
    }


def params_for_bcu_bridge(scen: RigScenario, parent_seed: int = 0) -> dict[str, Any]:
    p = scen.plant
    f = scen.faults
    b = scen.bridges.bcu
    n = _effective_noise(scen)
    return {
        "model_name": scen.sim.model_name,
        "volume_per_rev_m3": p.volume_per_rev_m3,
        "bladder_min_m3": p.bladder_min_m3,
        "bladder_max_m3": p.bladder_max_m3,
        "fault_effectiveness": f.bcu_pump.effectiveness,
        **_fault_schedule_params("fault_", f.bcu_pump.schedule),
        "publish_rate_hz": b.publish_rate_hz,
        "tank_pressure_empty_pa": p.tank_pressure_empty_pa,
        "tank_pressure_full_pa": p.tank_pressure_full_pa,
        "pump_response_delay_s": p.pump_response_delay_s,
        "pump_slew_rpm_per_s": p.pump_slew_rpm_per_s,
        "pump_overshoot_frac": p.pump_overshoot_frac,
        "tank_map_shape": p.tank_map_shape,
        "tank_air_volume_m3": p.tank_air_volume_m3,
        "tank_noise_seed": derive_seed(parent_seed, "tank_pressure_noise"),
        "tank_noise_sigma_pa": n.tank_pressure.sigma_pa,
        "tank_noise_quantization_pa": n.tank_pressure.quantization_pa,
        **_sensor_fault_params(
            "tank_", f.sensors.tank_pressure, parent_seed, "tank_pressure_fault"
        ),
        "comms_drop_prob": f.comms.drop_prob,
        "comms_seed": derive_seed(parent_seed, "bcu_comms_drop"),
    }


def params_for_imu_bridge(scen: RigScenario, parent_seed: int = 0) -> dict[str, Any]:
    n = _effective_noise(scen)
    return {
        "model_name": scen.sim.model_name,
        "noise_seed": derive_seed(parent_seed, "imu_noise"),
        # (x, y, z) as ROS double-array parameters.
        "noise_accel_sigma": list(n.imu.accel_sigma_mps2),
        "noise_gyro_sigma": list(n.imu.gyro_sigma_rads),
        # Comms fault only — the IMU is excluded from sensor-fault
        # injection by design (SensorFaultsSpec).
        "comms_drop_prob": scen.faults.comms.drop_prob,
        "comms_seed": derive_seed(parent_seed, "imu_comms_drop"),
    }


def params_for_external_sensor_bridge(
    scen: RigScenario, parent_seed: int = 0
) -> dict[str, Any]:
    n = _effective_noise(scen)
    return {
        "model_name": scen.sim.model_name,
        "publish_rate_hz": scen.bridges.external_sensor.publish_rate_hz,
        "noise_seed": derive_seed(parent_seed, "external_pressure_noise"),
        "noise_sigma_pa": n.external_pressure.sigma_pa,
        "noise_quantization_pa": n.external_pressure.quantization_pa,
        **_sensor_fault_params(
            "",
            scen.faults.sensors.external_pressure,
            parent_seed,
            "external_pressure_fault",
        ),
        "comms_drop_prob": scen.faults.comms.drop_prob,
        "comms_seed": derive_seed(parent_seed, "external_pressure_comms_drop"),
    }


def params_for_gt_pose_bridge(scen: RigScenario) -> dict[str, Any]:
    return {"model_name": scen.sim.model_name}


def params_for_anomaly_label(scen: Scenario) -> dict[str, Any]:
    """Wire params for the sim-only anomaly_label_bridge.

    The only consumer of `Scenario.anomaly` — the spec's fields ARE the
    wire params (the bridge re-validates them through AnomalyLabelSpec
    and broadcasts them verbatim, plus its per-class `active` gating),
    so a field added to the spec reaches the wire without re-listing.

    The labeled class's fault schedule rides along under a `schedule_`
    prefix so the bridge's `active` flag can honor the onset. Which block
    the label points at is `Scenario.labeled_fault` (the schedule's
    authority is `rig.faults`, and that property sits beside the
    validator enforcing the mapping); nominal/biofouling/comms have no
    such block and carry the inert default.
    """
    fault = scen.labeled_fault
    sched = fault.schedule if fault is not None else FaultScheduleSpec()
    return {
        **scen.anomaly.model_dump(),
        **_fault_schedule_params("schedule_", sched),
    }


def params_for_auto_mission(scen: Scenario) -> dict[str, Any]:
    """Wire params for the sim mission autostart (debug/auto_mission).

    The tank endpoints arm bcu_node's `TankLimitGuard` through the
    same DIVE_INIT path the operator UI uses on hardware. Sampled plant
    truth on purpose: the hardware value is a pre-dive *measurement* of
    the actual tank, so the sim surrogate measures the sampled plant.
    """
    return {
        "dive_init_tank_empty_pa": scen.rig.plant.tank_pressure_empty_pa,
        "dive_init_tank_full_pa": scen.rig.plant.tank_pressure_full_pa,
    }


# ---------------------------------------------------------------------------
# Hydrodynamics: forward map
# ---------------------------------------------------------------------------
#
# Physics knobs -> SDF hydrodynamic surface. Closed-form strip theory
# is computed, then post-multiplied by per-slot calibration scalars
# fit once at the nominal so forward_map(nominal) reproduces the
# canonical SDF exactly. This is pure determinstic.

_LOG = logging.getLogger("py_pkg.scenarios.compile")


class _Constants(NamedTuple):
    """Fluid + structural constants held fixed across every sweep."""

    rho: float
    nu: float
    u_ref: float
    z_r: float  # rudder root z-offset from the roll axis (structural, from SDF)


def _load_nominal_data() -> dict[str, Any]:
    """Read the nominal knob point + fixed constants from nominal_knobs.yaml.

    Tries the source-tree layout first (host-side lhs_sample.py), then the
    ROS 2 install/share layout (inside the container SIF), then falls back to
    the PhysicsKnobs defaults so importing this module never hard-fails.
    """
    library_dir = Path(__file__).resolve().parent / "library"
    nominal_path = library_dir / "nominal_knobs.yaml"

    if not nominal_path.is_file():
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("py_pkg"))
            nominal_path = share_dir / "scenarios" / "library" / "nominal_knobs.yaml"
        except (ImportError, LookupError):
            # ImportError: not in a sourced ROS env. LookupError covers
            # PackageNotFoundError (a KeyError subclass) when py_pkg isn't built.
            pass

    if not nominal_path.is_file():
        _LOG.warning(
            "nominal_knobs.yaml not found at %s; using PhysicsKnobs defaults",
            nominal_path,
        )
        return {
            "physics_knobs": PhysicsKnobs().model_dump(),
            "fluid_constants": {"rho": 1000.0, "nu": 1.05e-6, "u_ref": 0.3},
            "structural_constants": {"z_r": 0.119},
        }

    return yaml.safe_load(nominal_path.read_text())


@functools.lru_cache(maxsize=1)
def _nominal_data() -> dict[str, Any]:
    return _load_nominal_data()


@functools.lru_cache(maxsize=1)
def _nominal_knobs() -> PhysicsKnobs:
    return PhysicsKnobs(**_nominal_data()["physics_knobs"])


@functools.lru_cache(maxsize=1)
def _constants() -> _Constants:
    data = _nominal_data()
    fc = data["fluid_constants"]
    sc = data.get("structural_constants", {})
    return _Constants(
        rho=float(fc["rho"]),
        nu=float(fc["nu"]),
        u_ref=float(fc["u_ref"]),
        z_r=float(sc.get("z_r", 0.119)),
    )


# Slot registry: closed-form keys to the HydrodynamicsSpec surface.
_BODY_SLOTS: tuple[str, ...] = (
    "added_mass_xx",
    "added_mass_yy",
    "added_mass_zz",
    "added_mass_pp",
    "added_mass_qq",
    "added_mass_rr",
    "drag_xU",
    "drag_yV",
    "drag_zW",
    "drag_kP",
    "drag_mQ",
    "drag_nR",
)

_FIN_SLOTS: dict[str, Callable[[HydrodynamicsSpec], float]] = {
    "horiz_cla": lambda s: s.left_fin.cla,
    "horiz_cda": lambda s: s.left_fin.cda,
    "horiz_area": lambda s: s.left_fin.area,
    "vert_cla": lambda s: s.top_rudder.cla,
    "vert_cda": lambda s: s.top_rudder.cda,
    "vert_area": lambda s: s.top_rudder.area,
}

_ALL_SLOTS: frozenset[str] = frozenset(_BODY_SLOTS) | frozenset(_FIN_SLOTS)


def _canonical_values(spec: HydrodynamicsSpec) -> dict[str, float]:
    """Read every registry slot off a HydrodynamicsSpec into a flat dict."""
    can = {k: getattr(spec, k) for k in _BODY_SLOTS}
    can.update({k: reader(spec) for k, reader in _FIN_SLOTS.items()})
    return can


# Closed-form strip theory (§3-§8): one small pure helper per doc section.


def _ittc_cf(Re: float) -> float:
    """ITTC-57 flat-plate friction line, floored for degenerate Reynolds."""
    return 0.075 / (math.log10(Re) - 2) ** 2 if Re > 100 else 0.01


def _lamb_factors(L: float, D: float) -> tuple[float, float, float]:
    """Imlay 1961 added-mass k-factors (axial k1, transverse k2, pitch/yaw k')."""
    if L <= D:
        _LOG.warning(
            "Lamb factors: L (%.3f) <= D (%.3f) is degenerate geometry; "
            "returning zero added-mass factors",
            L,
            D,
        )
        return 0.0, 0.0, 0.0
    e = math.sqrt(1.0 - (D / L) ** 2)
    alpha_0 = (2 * (1 - e**2) / e**3) * (0.5 * math.log((1 + e) / (1 - e)) - e)
    beta_0 = 1 / e**2 - ((1 - e**2) / (2 * e**3)) * math.log((1 + e) / (1 - e))
    k_1 = alpha_0 / (2 - alpha_0)
    k_2 = beta_0 / (2 - beta_0)
    k_prime = (
        e**4
        * (beta_0 - alpha_0)
        / ((2 - e**2) * (2 * e**2 - (2 - e**2) * (beta_0 - alpha_0)))
    )
    return k_1, k_2, k_prime


def _hull_added_mass(g: PhysicsKnobs, c: _Constants) -> dict[str, float]:
    """§3 inviscid slender-body hull added mass (diagonal only)."""
    k_1, k_2, k_prime = _lamb_factors(g.L, g.D)
    I_m_shape = 1.0 / 12.0
    return {
        "X_u_dot": k_1 * c.rho * g.nabla,
        "Y_v_dot": k_2 * c.rho * g.nabla,
        "Z_w_dot": k_2 * c.rho * g.nabla,
        "M_q_dot": k_prime * c.rho * g.nabla * g.L**2 * I_m_shape,
        "N_r_dot": k_prime * c.rho * g.nabla * g.L**2 * I_m_shape,
        "K_p_dot": 0.0,
    }


def _hull_damping(g: PhysicsKnobs, c: _Constants) -> dict[str, float]:
    """§4 cross-flow + skin-friction quadratic damping, linearized at u_ref (§5).

    Hull only — fin damping is generated at runtime by LiftDrag and is never
    summed in here. Returns the linear body-block slots; the quadratic terms
    are intermediate (the SDF body block carries no quadratic slots).
    """
    C_p = (4 * g.nabla) / (math.pi * g.D**2 * g.L)
    sqrt_Cp = math.sqrt(max(0.0, C_p))

    Y_v_v = -0.5 * g.C_d_c * c.rho * g.L * g.D * sqrt_Cp
    M_q_q = -(1.0 / 24.0) * g.C_d_c * c.rho * g.L**3 * g.D * sqrt_Cp

    Re = c.u_ref * g.L / c.nu
    C_F = _ittc_cf(Re)
    C_D = C_F * g.one_plus_k + g.C_p_base
    X_u_u = -(math.pi / 8.0) * c.rho * g.D**2 * C_D

    # Tangent linearization at u_ref: X_u = 2 * X_u|u| * u_ref.
    return {
        "X_u": 2 * X_u_u * c.u_ref,
        "Y_v": 2 * Y_v_v * c.u_ref,
        "Z_w": 2 * Y_v_v * c.u_ref,
        "M_q": 2 * M_q_q * c.u_ref,
        "N_r": 2 * M_q_q * c.u_ref,
        "K_p": 0.0,
    }


def _fin_added_mass(g: PhysicsKnobs, c: _Constants) -> dict[str, float]:
    """§6 thin-plate strip-theory fin added mass, summed into the body block.

    Horizontal pair (lift in z) and the lone top rudder (lift in y) are kept
    separate because of the asymmetric layout; the offset strip integrals use
    the hull-surface root offsets (D/2 for the wings, z_r for the rudder).
    """
    Z_w_dot_horiz = (math.pi / 4.0) * c.rho * g.c_f**2 * g.b_f * 2.0
    M_q_dot_horiz = Z_w_dot_horiz * g.x_f**2
    K_p_dot_horiz = (
        2.0
        * c.rho
        * math.pi
        * (g.c_f / 2.0) ** 2
        * (((g.D / 2.0) + g.b_f) ** 3 - (g.D / 2.0) ** 3)
        / 3.0
    )

    Y_v_dot_rudder = (math.pi / 4.0) * c.rho * g.c_r**2 * g.b_r
    N_r_dot_rudder = Y_v_dot_rudder * g.x_r**2
    K_p_dot_rudder = (
        c.rho * math.pi * (g.c_r / 2.0) ** 2 * ((c.z_r + g.b_r) ** 3 - c.z_r**3) / 3.0
    )

    return {
        "Z_w_dot_horiz": Z_w_dot_horiz,
        "M_q_dot_horiz": M_q_dot_horiz,
        "K_p_dot_horiz": K_p_dot_horiz,
        "Y_v_dot_rudder": Y_v_dot_rudder,
        "N_r_dot_rudder": N_r_dot_rudder,
        "K_p_dot_rudder": K_p_dot_rudder,
    }


def _helmbold(AR: float, mult: float) -> float:
    return mult * (2 * math.pi * AR / (2 + math.sqrt(AR**2 + 4)))


def _fin_lift_slope(g: PhysicsKnobs) -> tuple[float, float]:
    """§7 Helmbold lift slope + interference multiplier (horiz, vert)."""
    AR_f = g.b_f / g.c_f if g.c_f > 0 else 0.0
    AR_r = g.b_r / g.c_r if g.c_r > 0 else 0.0
    return _helmbold(AR_f, g.C_La_mult), _helmbold(AR_r, g.C_La_mult)


def _fin_profile_drag(g: PhysicsKnobs, c: _Constants) -> tuple[float, float]:
    """§8 Hoerner profile drag at fin Reynolds (horiz, vert) — wires in t/c."""
    hoerner = 1 + 2 * g.t_over_c + 60 * g.t_over_c**4
    C_F_f = _ittc_cf(c.u_ref * g.c_f / c.nu)
    C_F_r = _ittc_cf(c.u_ref * g.c_r / c.nu)
    return 2 * C_F_f * hoerner, 2 * C_F_r * hoerner


def _compute_closed_form(g: PhysicsKnobs) -> dict[str, float]:
    """Assemble the closed-form slot dict (§9 composition).

    Body added mass sums hull + fin terms; damping is hull-only; the per-fin
    lift/drag/area land in the fin slots. Keys cover exactly _ALL_SLOTS.
    """
    c = _constants()
    hull_am = _hull_added_mass(g, c)
    damp = _hull_damping(g, c)
    fin_am = _fin_added_mass(g, c)
    cla_h, cla_v = _fin_lift_slope(g)
    cda_h, cda_v = _fin_profile_drag(g, c)

    return {
        "added_mass_xx": hull_am["X_u_dot"],
        "added_mass_yy": hull_am["Y_v_dot"] + fin_am["Y_v_dot_rudder"],
        "added_mass_zz": hull_am["Z_w_dot"] + fin_am["Z_w_dot_horiz"],
        "added_mass_pp": (
            hull_am["K_p_dot"] + fin_am["K_p_dot_horiz"] + fin_am["K_p_dot_rudder"]
        ),
        "added_mass_qq": hull_am["M_q_dot"] + fin_am["M_q_dot_horiz"],
        "added_mass_rr": hull_am["N_r_dot"] + fin_am["N_r_dot_rudder"],
        "drag_xU": damp["X_u"],
        "drag_yV": damp["Y_v"],
        "drag_zW": damp["Z_w"],
        "drag_kP": damp["K_p"],
        "drag_mQ": damp["M_q"],
        "drag_nR": damp["N_r"],
        "horiz_cla": cla_h,
        "horiz_cda": cda_h,
        "horiz_area": g.b_f * g.c_f,
        "vert_cla": cla_v,
        "vert_cda": cda_v,
        "vert_area": g.b_r * g.c_r,
    }


@functools.lru_cache(maxsize=1)
def _calibration() -> tuple[dict[str, float], dict[str, float]]:
    """Build (scalars, canonical_values) once, lazily on first use.

    λ_slot = canonical_SDF / f(g_nominal), so forward_map(g_nominal) lands on
    the canonical SDF exactly. Slots with no closed-form contribution (e.g.
    roll linear damping) get a 0.0 sentinel and fall back to the canonical
    value in forward_map. Logged on first call; |λ|>5 slots warn.
    """
    f_nom = _compute_closed_form(_nominal_knobs())
    canonical = _canonical_values(HydrodynamicsSpec())

    # A slot added to the physics but not the registry (or vice versa) is
    # caught here rather than producing a silently wrong SDF.
    assert (
        set(f_nom) == _ALL_SLOTS
    ), f"closed-form/registry slot mismatch: {set(f_nom) ^ _ALL_SLOTS}"

    scalars: dict[str, float] = {}
    for k, can_val in canonical.items():
        nom_val = f_nom[k]
        if abs(nom_val) < 1e-9:
            scalars[k] = 1.0 if abs(can_val) < 1e-9 else 0.0
        else:
            scalars[k] = can_val / nom_val

    _LOG.info("Hydrodynamic calibration scalars:")
    for k, lam in scalars.items():
        _LOG.info("  %s: %.3f", k, lam)
        if abs(lam) > 5.0:
            _LOG.warning(
                "  %s has a large calibration scalar (|λ|=%.2f > 5); "
                "closed-form physics contributes little to this slot.",
                k,
                abs(lam),
            )
    return scalars, canonical


def calibration_scalars() -> dict[str, float]:
    """Public accessor for the per-slot calibration scalars (G2 gate / tests)."""
    return _calibration()[0]


def _spec_from_slots(out: dict[str, float], knobs: PhysicsKnobs) -> HydrodynamicsSpec:
    """Assemble a HydrodynamicsSpec from the flat slot dict.

    The only place the fin fan-out (horiz -> Left/Right, vert -> TopRudder)
    is written.
    """
    horiz = dict(
        cla=out["horiz_cla"],
        cda=out["horiz_cda"],
        area=out["horiz_area"],
        alpha_stall=knobs.alpha_stall_horiz,
    )
    return HydrodynamicsSpec(
        **{k: out[k] for k in _BODY_SLOTS},
        left_fin=FinAeroSpec(**horiz),
        right_fin=FinAeroSpec(**horiz),
        top_rudder=FinAeroSpec(
            cla=out["vert_cla"],
            cda=out["vert_cda"],
            area=out["vert_area"],
            alpha_stall=knobs.alpha_stall_rudder,
        ),
        knobs=knobs,
    )


def forward_map(knobs: PhysicsKnobs | dict[str, float]) -> HydrodynamicsSpec:
    """Map the 16 physics knobs to an SDF hydrodynamic surface.

    Deterministic: closed-form strip theory (§3-§8) times the per-slot
    calibration scalars (§10). forward_map(nominal) reproduces the canonical
    SDF. Accepts a PhysicsKnobs or a plain dict (coerced + validated at the
    boundary). Sampling jitter (§12.7) is applied separately in the sampler.
    """
    if not isinstance(knobs, PhysicsKnobs):
        knobs = PhysicsKnobs(**knobs)

    scalars, canonical = _calibration()
    f_g = _compute_closed_form(knobs)

    out = {
        # Slots with no closed-form term fall back to the canonical value
        # rather than scaling a zero by the scalar.
        k: canonical[k] if abs(val) < 1e-9 else scalars[k] * val
        for k, val in f_g.items()
    }
    return _spec_from_slots(out, knobs)
