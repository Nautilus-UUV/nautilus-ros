"""ACU pitch controller config.

MassShifterController maps pitch error (deg) directly to stroke (m), so
Kp is m/deg, Ki is m/(deg*s), Kd is m*s/deg. ACUControlNode converts
m -> mm at the publish site.

Full PID. Ki cancels the steady-state offset that a P-only controller
leaves at the SAWTOOTH glide angles (the body's natural drag/buoyancy
moment is non-zero at +/-35 deg, so without I the shifter sits
saturated and the body trims a few deg past the setpoint). Kd damps
the hard leg-flip transition where the setpoint jumps from -35 to +35
in a single tick. Integral_limits are kept tight against the actuator
clamp so anti-windup stays effective when the shifter is railed.
"""

from py_pkg.robot_specs import ACU_PITCH_OUTPUT_LIMIT_M

init_acu_pitch = {
    "name": "pitch",
    "Kp": 0.1,
    "Ki": 0.0,
    "Kd": 0.5,
    "position_tolerance": 1.0,
    "command_tolerance": 0.5,
    "integral_limits": (-10.0, 10.0),
    "output_limits": (-ACU_PITCH_OUTPUT_LIMIT_M, ACU_PITCH_OUTPUT_LIMIT_M),
    "derivative_filter": 0.2,
}
