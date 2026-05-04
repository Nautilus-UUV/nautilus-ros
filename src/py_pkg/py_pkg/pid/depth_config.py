# Copied from divetest files: depth_control_node_config.py

from py_pkg.physics import (
    WATER_PRESSURE_GRADIENT_PA_PER_M,
    depth_to_pressure_pa,
    gauge_pressure_pa,
)
from py_pkg.robot_specs import (
    BCU_MOTOR_MAX_RPM,
    BCU_MOTOR_MIN_RPM,
    BLADDER_VOLUME_M3,
)

# Hydrostatic gradient (Pa/m) in salt water. Used to translate the
# legacy depth-domain tuning into pressure-domain gains so the cascade
# stays numerically equivalent: L1/L2 stages have inputs and outputs
# both scaled by RHO_G, L3 maps Pa/s^2 -> q so its gains divide by it.
RHO_G = WATER_PRESSURE_GRADIENT_PA_PER_M

init_control = {}
# Z-positive-down: shallowest gauge pressure (~20 m), deepest (~70 m).
init_control["low_pressure_pa"] = gauge_pressure_pa(depth_to_pressure_pa(20.0))
init_control["high_pressure_pa"] = gauge_pressure_pa(depth_to_pressure_pa(70.0))
init_control["frequency"] = 10
init_control["pid_pressure"] = {
    "kp": 0.1,
    "ki": 0,
    "kd": 0,
    "integral_limits": (-100.0 * RHO_G, 100.0 * RHO_G),
    "output_limits": (-100.0 * RHO_G, 100.0 * RHO_G),
}
init_control["pid_p_dot"] = {
    "kp": 1,
    "ki": 0,
    "kd": 0.1,
    "integral_limits": (-100.0 * RHO_G, 100.0 * RHO_G),
    "output_limits": (-10.0 * RHO_G, 10.0 * RHO_G),
}
init_control["pid_p_ddot"] = {
    "kp": 0.02 / RHO_G,
    "ki": 0.00005 / RHO_G,
    "kd": 0.8 / RHO_G,
    "integral_limits": (-100.0, 100.0),
    "output_limits": (-0.010035, 0.010035),
}
init_pos = {}
init_pos["x"] = 0.0
init_pos["y"] = 0.0
# z carries the controlled variable (gauge Pa); 0 = surface.
init_pos["z"] = 0.0
init_buoyancy_engine = {}
init_buoyancy_engine["tank_volume"] = BLADDER_VOLUME_M3
init_buoyancy_engine["initial_proportion_full"] = 1.0
init_motor = {}
init_motor["min_rpm"] = BCU_MOTOR_MIN_RPM
init_motor["max_rpm"] = BCU_MOTOR_MAX_RPM
