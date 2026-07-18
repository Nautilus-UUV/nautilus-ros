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
    DepthPlantModel,
    DepthSpec,
    PIDPressureSpec,
)
from .spec.rig import (
    FinAeroSpec,
    HydrodynamicsSpec,
    ImuNoiseSpec,
    NoiseSpec,
    PhysicsKnobs,
    PressureNoiseSpec,
    RigScenario,
)

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


def bcu_spec_from_node(node: Node) -> DepthSpec:
    """Read depth-controller params off `node` into a typed DepthSpec.

    Defaults come from the dataclass — absent any scenario override at
    launch time, behaviour matches the literal values that used to live
    in `init_control` / `init_buoyancy_engine` / `init_motor`.
    """

    default = DepthSpec()
    pp = default.pid_pressure
    pm = default.plant_model

    return DepthSpec(
        frequency_hz=_param(node, "frequency_hz", default.frequency_hz),
        pid_pressure=PIDPressureSpec(
            kp=_param(node, "pid_pressure.kp", pp.kp),
            ki=_param(node, "pid_pressure.ki", pp.ki),
            kd=_param(node, "pid_pressure.kd", pp.kd),
            integral_limits=(
                _param(node, "pid_pressure.integral_limit_low", pp.integral_limits[0]),
                _param(node, "pid_pressure.integral_limit_high", pp.integral_limits[1]),
            ),
            output_limits=(
                _param(node, "pid_pressure.output_limit_low", pp.output_limits[0]),
                _param(node, "pid_pressure.output_limit_high", pp.output_limits[1]),
            ),
            derivative_filter=_param(
                node, "pid_pressure.derivative_filter", pp.derivative_filter
            ),
        ),
        plant_model=DepthPlantModel(
            bladder_nominal_m3=_param(
                node, "plant_model.bladder_nominal_m3", pm.bladder_nominal_m3
            ),
            initial_proportion_full=_param(
                node, "plant_model.initial_proportion_full", pm.initial_proportion_full
            ),
            min_rpm=_param(node, "plant_model.min_rpm", pm.min_rpm),
            min_operating_rpm=_param(
                node, "plant_model.min_operating_rpm", pm.min_operating_rpm
            ),
            max_rpm=_param(node, "plant_model.max_rpm", pm.max_rpm),
            pump_efficiency=_param(
                node, "plant_model.pump_efficiency", pm.pump_efficiency
            ),
        ),
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


def params_for_bcu_bridge(scen: RigScenario, parent_seed: int = 0) -> dict[str, Any]:
    p = scen.plant
    f = scen.faults.bcu_rpm
    b = scen.bridges.bcu
    n = _effective_noise(scen)
    return {
        "model_name": scen.sim.model_name,
        "volume_per_rev_m3": p.volume_per_rev_m3,
        "bladder_min_m3": p.bladder_min_m3,
        "bladder_max_m3": p.bladder_max_m3,
        "fault_mttf_sec": f.mttf_sec,
        "fault_num_levels": f.num_levels,
        "rng_seed": derive_seed(parent_seed, "bcu_rpm_fault"),
        "publish_rate_hz": b.publish_rate_hz,
        "tank_pressure_empty_pa": p.tank_pressure_empty_pa,
        "tank_pressure_full_pa": p.tank_pressure_full_pa,
        "pump_response_delay_s": p.pump_response_delay_s,
        "pump_slew_rpm_per_s": p.pump_slew_rpm_per_s,
        "tank_map_shape": p.tank_map_shape,
        "tank_air_volume_m3": p.tank_air_volume_m3,
        "tank_noise_seed": derive_seed(parent_seed, "tank_pressure_noise"),
        "tank_noise_sigma_pa": n.tank_pressure.sigma_pa,
        "tank_noise_quantization_pa": n.tank_pressure.quantization_pa,
    }


def params_for_imu_bridge(scen: RigScenario, parent_seed: int = 0) -> dict[str, Any]:
    n = _effective_noise(scen)
    return {
        "model_name": scen.sim.model_name,
        "noise_seed": derive_seed(parent_seed, "imu_noise"),
        # (x, y, z) as ROS double-array parameters.
        "noise_accel_sigma": list(n.imu.accel_sigma_mps2),
        "noise_gyro_sigma": list(n.imu.gyro_sigma_rads),
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
    }


def params_for_gt_pose_bridge(scen: RigScenario) -> dict[str, Any]:
    return {"model_name": scen.sim.model_name}


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
