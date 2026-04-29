# Copied from divetest files: depth_control_node_config.py

from py_pkg.robot_specs import (
    BCU_MOTOR_MAX_RPM,
    BCU_MOTOR_MIN_RPM,
    BLADDER_VOLUME_M3,
)

init_control = {}
init_control["high_depth"] = -20.0
init_control["low_depth"] = -70.0
init_control["frequency"] = 10
init_control["pid_depth"] = {
    "kp": 0.1,
    "ki": 0,
    "kd": 0,
    "integral_limits": (-100.0, 100.0),
    "output_limits": (-100.0, 100.0),
}
init_control["pid_v_vel"] = {
    "kp": 1,
    "ki": 0,
    "kd": 0.1,
    "integral_limits": (-100.0, 100.0),
    "output_limits": (-10.0, 10.0),
}
init_control["pid_v_acc"] = {
    "kp": 0.02,
    "ki": 0.00005,
    "kd": 0.8,
    "integral_limits": (-100.0, 100.0),
    "output_limits": (-0.010035, 0.010035),
}
init_pos = {}
init_pos["x"] = 0.0
init_pos["y"] = 0.0
init_pos["z"] = 0.0
init_buoyancy_engine = {}
init_buoyancy_engine["tank_volume"] = BLADDER_VOLUME_M3
init_buoyancy_engine["initial_proportion_full"] = 1.0
init_motor = {}
init_motor["min_rpm"] = BCU_MOTOR_MIN_RPM
init_motor["max_rpm"] = BCU_MOTOR_MAX_RPM
