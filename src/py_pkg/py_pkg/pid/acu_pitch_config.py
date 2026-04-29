"""Per-axis configuration for the ACU pitch controller.

Pitch is the glider's primary attitude-for-glide control surface; the
mass-shifter stroke is bounded by ACU_PITCH_OUTPUT_LIMIT_M. Consumed by
MassShifterController, which maps pitch error directly to mass-shifter
stroke — so Kp's units are m/deg. Ki/Kd are 0 today (P-only behaviour)
but the keys are present so they can be tuned without code changes —
see the design note about long 35° holds needing a small Ki with
conservative integral_limits.
"""

from py_pkg.robot_specs import (
    ACU_PITCH_OUTPUT_LIMIT_M,
    ACU_PITCH_STEPS_PER_M,
)

init_acu_pitch = {
    "name": "pitch",
    "Kp": 0.5,
    "Ki": 0.0,
    "Kd": 0.0,
    "position_tolerance": 1.0,
    "command_tolerance": 0.5,
    "integral_limits": (-10.0, 10.0),
    "output_limits": (-ACU_PITCH_OUTPUT_LIMIT_M, ACU_PITCH_OUTPUT_LIMIT_M),
    "derivative_filter": 0.0,
    # Conversion from motor-frame metres to Int32 steps for ACU_PITCH_STEPS.
    "motor_steps_per_unit": ACU_PITCH_STEPS_PER_M,
}
