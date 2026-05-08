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

`command_tolerance` is the sole publish deadband — if the clamped
target hasn't moved more than this since the last publish, we don't
publish again. 5 mm matches the wire-format quantization step (the mm
Int16) and sits comfortably above EKF pitch noise mapped through Kp,
so noise alone never republishes.
"""

from py_pkg.robot_specs import ACU_PITCH_OUTPUT_LIMITS_M

init_acu_pitch = {
    "name": "pitch",
    "Kp": 0.1,
    "Ki": 0.0,
    "Kd": 0.5,
    "command_tolerance": 0.005,
    "integral_limits": (-10.0, 10.0),
    "output_limits": ACU_PITCH_OUTPUT_LIMITS_M,
    "derivative_filter": 0.2,
}
