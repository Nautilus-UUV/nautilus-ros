"""ACU roll controller config.

Output is degrees of ring rotation, clamped to ±ACU_ROLL_MAX_ANGLE_DEG.
ACUControlNode converts deg -> centidegrees at the publish site.

Full PID. I cancels the steady-state offset under any constant rolling
moment (asymmetric drag during a glide, residual rolling momentum from
a leg flip), D damps fast transitions. Kept conservative so the roll
axis doesn't induce coupling into the pitch dynamics.

`command_tolerance` is the sole publish deadband. 0.5° sits above the
EKF's typical roll-orientation noise (~0.5°) so steady-state sensor
noise alone doesn't republish, and well below any operationally
meaningful roll setpoint change. Both consequences keep the EPOS bus
quiet at trim.
"""

from py_pkg.robot_specs import ACU_ROLL_MAX_ANGLE_DEG

init_acu_roll = {
    "name": "roll",
    "Kp": 0.5,
    "Ki": 0.005,
    "Kd": 0.05,
    "command_tolerance": 0.5,
    "integral_limits": (-30.0, 30.0),
    "output_limits": (-ACU_ROLL_MAX_ANGLE_DEG, ACU_ROLL_MAX_ANGLE_DEG),
    "derivative_filter": 0.2,
}