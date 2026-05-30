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

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node
import math
import logging
import yaml
from pathlib import Path
import numpy as np

from .seed import derive_seed
from .spec.control import (
    AcuPitchSpec,
    AcuRollSpec,
    ControlScenario,
    DepthPlantModel,
    DepthSpec,
    PIDPressureSpec,
)
from .spec.rig import RigScenario, HydrodynamicsSpec, FinAeroSpec, PhysicsKnobs

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

# ---------------------------------------------------------------------------
# Hydrodynamics: Forward Map
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("py_pkg.scenarios.compile")

def _load_nominal_knobs() -> dict[str, Any]:
    # 1. Try source tree layout (used by host-side lhs_sample.py)
    library_dir = Path(__file__).resolve().parent / "library"
    nominal_path = library_dir / "nominal_knobs.yaml"
    
    if not nominal_path.is_file():
        # 2. Try ROS 2 install layout (used inside the container SIF)
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = Path(get_package_share_directory("py_pkg"))
            nominal_path = share_dir / "scenarios" / "library" / "nominal_knobs.yaml"
        except (ImportError, Exception):
            pass

    if not nominal_path.is_file():
        # Fallback for tests if library isn't packaged properly
        _LOG.warning(f"nominal_knobs.yaml not found at {nominal_path}, using hardcoded defaults")
        return {"physics_knobs": PhysicsKnobs().model_dump(), "fluid_constants": {"rho": 1025.0, "nu": 1.05e-6, "u_ref": 0.3}}
        
    return yaml.safe_load(nominal_path.read_text())

_NOMINAL_DATA = _load_nominal_knobs()
_G_NOMINAL = _NOMINAL_DATA["physics_knobs"]
_FLUID_CONSTANTS = _NOMINAL_DATA["fluid_constants"]
_RHO = float(_FLUID_CONSTANTS["rho"])
_NU = float(_FLUID_CONSTANTS["nu"])
_U_REF = float(_FLUID_CONSTANTS["u_ref"])


def _compute_closed_form(g: dict[str, float]) -> dict[str, float]:
    """Computes hydrodynamic coefficients from physics knobs using strip theory.
    Returns a dict with keys matching HydrodynamicsSpec and FinAeroSpec slots.
    """
    L = g["L"]
    D = g["D"]
    nabla = g["nabla"]
    b_f = g["b_f"]
    c_f = g["c_f"]
    x_f = g["x_f"]
    b_r = g["b_r"]
    c_r = g["c_r"]
    x_r = g["x_r"]
    t_over_c = g["t_over_c"]
    alpha_stall_horiz = g["alpha_stall_horiz"]
    alpha_stall_rudder = g["alpha_stall_rudder"]
    C_d_c = g["C_d_c"]
    one_plus_k = g["one_plus_k"]
    C_p_base = g["C_p_base"]
    C_La_mult = g["C_La_mult"]

    # --- Lamb factors ---
    e = math.sqrt(max(0.0, 1.0 - (D/L)**2)) if L > D else 0.0
    if e > 0:
        alpha_0 = (2 * (1 - e**2) / e**3) * (0.5 * math.log((1 + e)/(1 - e)) - e)
        beta_0 = 1 / e**2 - ((1 - e**2) / (2 * e**3)) * math.log((1 + e)/(1 - e))
        k_1 = alpha_0 / (2 - alpha_0)
        k_2 = beta_0 / (2 - beta_0)
        k_prime = e**4 * (beta_0 - alpha_0) / ((2 - e**2) * (2 * e**2 - (2 - e**2) * (beta_0 - alpha_0)))
    else:
        k_1 = k_2 = k_prime = 0.0

    # --- Hull Added Mass ---
    I_m_shape = 1.0 / 12.0
    X_u_dot_hull = k_1 * _RHO * nabla
    Y_v_dot_hull = Z_w_dot_hull = k_2 * _RHO * nabla
    M_q_dot_hull = N_r_dot_hull = k_prime * _RHO * nabla * L**2 * I_m_shape
    K_p_dot_hull = 0.0

    # --- Hull Quadratic Damping ---
    C_p = (4 * nabla) / (math.pi * D**2 * L)
    Y_v_v_hull = Z_w_w_hull = -0.5 * C_d_c * _RHO * L * D * math.sqrt(max(0.0, C_p))
    M_q_q_hull = N_r_r_hull = -(1.0 / 24.0) * C_d_c * _RHO * L**3 * D * math.sqrt(max(0.0, C_p))

    Re_hull = _U_REF * L / _NU
    C_F_hull = 0.075 / (math.log10(Re_hull) - 2)**2 if Re_hull > 100 else 0.01
    C_D_hull = C_F_hull * one_plus_k + C_p_base
    X_u_u_hull = -(math.pi / 8.0) * _RHO * D**2 * C_D_hull
    K_p_p_hull = 0.0

    # --- Hull Linear Damping (tangent at U_REF) ---
    X_u_hull = 2 * X_u_u_hull * _U_REF
    Y_v_hull = 2 * Y_v_v_hull * _U_REF
    Z_w_hull = 2 * Z_w_w_hull * _U_REF
    M_q_hull = 2 * M_q_q_hull * _U_REF
    N_r_hull = 2 * N_r_r_hull * _U_REF
    K_p_hull = 0.0

    # --- Fin Added Mass ---
    Z_w_dot_horiz = (math.pi / 4.0) * _RHO * c_f**2 * b_f * 2.0
    M_q_dot_horiz = (math.pi / 4.0) * _RHO * c_f**2 * b_f * 2.0 * x_f**2
    Y_v_dot_horiz = N_r_dot_horiz = 0.0
    K_p_dot_horiz = 2.0 * _RHO * math.pi * (c_f / 2.0)**2 * (((D / 2.0) + b_f)**3 - (D / 2.0)**3) / 3.0

    Y_v_dot_rudder = (math.pi / 4.0) * _RHO * c_r**2 * b_r * 1.0
    N_r_dot_rudder = (math.pi / 4.0) * _RHO * c_r**2 * b_r * 1.0 * x_r**2
    Z_w_dot_rudder = M_q_dot_rudder = 0.0
    z_r = 0.119
    K_p_dot_rudder = _RHO * math.pi * (c_r / 2.0)**2 * ((z_r + b_r)**3 - z_r**3) / 3.0

    # --- Fin Lift Slope ---
    AR_f = b_f / c_f if c_f > 0 else 0.0
    AR_r = b_r / c_r if c_r > 0 else 0.0
    cla_horiz = C_La_mult * (2 * math.pi * AR_f / (2 + math.sqrt(AR_f**2 + 4)))
    cla_vert  = C_La_mult * (2 * math.pi * AR_r / (2 + math.sqrt(AR_r**2 + 4)))

    # --- Fin Profile Drag ---
    Re_f = _U_REF * c_f / _NU
    Re_r = _U_REF * c_r / _NU
    C_F_f = 0.075 / (math.log10(Re_f) - 2)**2 if Re_f > 100 else 0.01
    C_F_r = 0.075 / (math.log10(Re_r) - 2)**2 if Re_r > 100 else 0.01
    cda_horiz = 2 * C_F_f * (1 + 2 * t_over_c + 60 * t_over_c**4)
    cda_vert  = 2 * C_F_r * (1 + 2 * t_over_c + 60 * t_over_c**4)

    return {
        "added_mass_xx": X_u_dot_hull,
        "added_mass_yy": Y_v_dot_hull + Y_v_dot_rudder,
        "added_mass_zz": Z_w_dot_hull + Z_w_dot_horiz,
        "added_mass_pp": K_p_dot_hull + K_p_dot_horiz + K_p_dot_rudder,
        "added_mass_qq": M_q_dot_hull + M_q_dot_horiz,
        "added_mass_rr": N_r_dot_hull + N_r_dot_rudder,
        "drag_xU": X_u_hull,
        "drag_yV": Y_v_hull,
        "drag_zW": Z_w_hull,
        "drag_kP": K_p_hull,
        "drag_mQ": M_q_hull,
        "drag_nR": N_r_hull,
        "horiz_cla": cla_horiz,
        "horiz_cda": cda_horiz,
        "horiz_area": b_f * c_f,
        "vert_cla": cla_vert,
        "vert_cda": cda_vert,
        "vert_area": b_r * c_r,
    }


def _compute_calibration_scalars() -> dict[str, float]:
    """Computes calibration scalars: canonical_SDF / f(g_nominal).
    Zeroes out quadratic drag (as canonical SDF has none) implicitly
    because quadratic slots aren't even generated.
    """
    f_nom = _compute_closed_form(_G_NOMINAL)
    sdf_can = HydrodynamicsSpec()
    
    # Map sdf_can fields to the keys generated by _compute_closed_form
    can_dict = {
        "added_mass_xx": sdf_can.added_mass_xx,
        "added_mass_yy": sdf_can.added_mass_yy,
        "added_mass_zz": sdf_can.added_mass_zz,
        "added_mass_pp": sdf_can.added_mass_pp,
        "added_mass_qq": sdf_can.added_mass_qq,
        "added_mass_rr": sdf_can.added_mass_rr,
        "drag_xU": sdf_can.drag_xU,
        "drag_yV": sdf_can.drag_yV,
        "drag_zW": sdf_can.drag_zW,
        "drag_kP": sdf_can.drag_kP,
        "drag_mQ": sdf_can.drag_mQ,
        "drag_nR": sdf_can.drag_nR,
        "horiz_cla": sdf_can.left_fin.cla,
        "horiz_cda": sdf_can.left_fin.cda,
        "horiz_area": sdf_can.left_fin.area,
        "vert_cla": sdf_can.top_rudder.cla,
        "vert_cda": sdf_can.top_rudder.cda,
        "vert_area": sdf_can.top_rudder.area,
    }
    
    scalars = {}
    for k, can_val in can_dict.items():
        nom_val = f_nom[k]
        if abs(nom_val) < 1e-9:
            scalars[k] = 1.0 if abs(can_val) < 1e-9 else 0.0
        else:
            scalars[k] = can_val / nom_val
            
    _LOG.info("Hydrodynamic Calibration Scalars:")
    for k, lam in scalars.items():
        _LOG.info(f"  {k}: {lam:.3f}")
        if abs(lam) > 5.0:
            _LOG.warning(f"  {k} has large calibration scalar (|λ| = {abs(lam):.2f} > 5.0). Closed-form physics dominates less.")
            
    return scalars, can_dict

_CALIBRATION_SCALARS, _CANONICAL_VALUES = _compute_calibration_scalars()


def forward_map(knobs: dict[str, float], jitter_seed: int | None = None, jitter_sigma: float = 0.0) -> HydrodynamicsSpec:
    """Computes the SDF hydrodynamic surface from physics knobs.
    Applies calibration scalars, then applies multiplicative isotropic log-normal jitter if requested.
    """
    f_g = _compute_closed_form(knobs)
    lam = _CALIBRATION_SCALARS
    
    # Apply scalars, fallback to canonical if f_g is 0
    out = {}
    for k in f_g:
        if abs(f_g[k]) < 1e-9:
            out[k] = _CANONICAL_VALUES[k]
        else:
            out[k] = lam[k] * f_g[k]
    
    # Apply jitter AFTER scalars
    if jitter_sigma > 0.0:
        rng = np.random.default_rng(jitter_seed)
        # Jitter applies to the output slots (which includes AM, linear drag, fin cla/cda/area)
        for k in out:
            # Jitter is multiplicative in linear space: val * exp(N(0, sigma^2))
            noise = math.exp(rng.normal(0, jitter_sigma))
            out[k] *= noise
            
        # alpha_stall is also jittered, but it comes straight from knobs, not f_g
        stall_noise_horiz = math.exp(rng.normal(0, jitter_sigma))
        stall_noise_rudder = math.exp(rng.normal(0, jitter_sigma))
        alpha_stall_horiz_final = knobs["alpha_stall_horiz"] * stall_noise_horiz
        alpha_stall_rudder_final = knobs["alpha_stall_rudder"] * stall_noise_rudder
    else:
        alpha_stall_horiz_final = knobs["alpha_stall_horiz"]
        alpha_stall_rudder_final = knobs["alpha_stall_rudder"]
        
    return HydrodynamicsSpec(
        added_mass_xx=out["added_mass_xx"],
        added_mass_yy=out["added_mass_yy"],
        added_mass_zz=out["added_mass_zz"],
        added_mass_pp=out["added_mass_pp"],
        added_mass_qq=out["added_mass_qq"],
        added_mass_rr=out["added_mass_rr"],
        drag_xU=out["drag_xU"],
        drag_yV=out["drag_yV"],
        drag_zW=out["drag_zW"],
        drag_kP=out["drag_kP"],
        drag_mQ=out["drag_mQ"],
        drag_nR=out["drag_nR"],
        left_fin=FinAeroSpec(
            cla=out["horiz_cla"],
            cda=out["horiz_cda"],
            area=out["horiz_area"],
            alpha_stall=alpha_stall_horiz_final,
        ),
        right_fin=FinAeroSpec(
            cla=out["horiz_cla"],
            cda=out["horiz_cda"],
            area=out["horiz_area"],
            alpha_stall=alpha_stall_horiz_final,
        ),
        top_rudder=FinAeroSpec(
            cla=out["vert_cla"],
            cda=out["vert_cda"],
            area=out["vert_area"],
            alpha_stall=alpha_stall_rudder_final,
        ),
        knobs=PhysicsKnobs(**knobs),
    )
