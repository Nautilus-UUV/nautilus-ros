"""ACU roll controller config.

Output is degrees of ring rotation, clamped to ±ACU_ROLL_MAX_ANGLE_DEG.
ACUControlNode converts deg -> rad at the publish site.
"""

from py_pkg.robot_specs import ACU_ROLL_MAX_ANGLE_DEG

init_acu_roll = {
    "name": "roll",
    "Kp": 0.5,
    "Ki": 0.0,
    "Kd": 0.0,
    "position_tolerance": 1.0,
    "command_tolerance": 0.5,
    "integral_limits": (-30.0, 30.0),
    "output_limits": (-ACU_ROLL_MAX_ANGLE_DEG, ACU_ROLL_MAX_ANGLE_DEG),
    "derivative_filter": 0.0,
}
