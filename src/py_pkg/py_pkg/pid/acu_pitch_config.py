"""ACU pitch controller config.

MassShifterController maps pitch error (deg) directly to stroke (m), so
Kp is m/deg. P-only today; Ki/Kd kept so a small Ki can be added for
sustained 35° holds without code changes. ACUControlNode converts m -> mm
at the publish site.
"""

from py_pkg.robot_specs import ACU_PITCH_OUTPUT_LIMIT_M

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
}
