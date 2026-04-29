"""Per-axis configuration for the ACU roll controller.

Roll output is bounded by ±ACU_ROLL_MAX_ANGLE_DEG (the physical ring
travel of the roll mechanism).
"""

from py_pkg.robot_specs import (
    ACU_ROLL_MAX_ANGLE_DEG,
    ACU_ROLL_STEPS_PER_DEG,
)

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
    # Conversion from motor-frame degrees to Int32 steps for ACU_ROLL_STEPS.
    "motor_steps_per_unit": ACU_ROLL_STEPS_PER_DEG,
}
