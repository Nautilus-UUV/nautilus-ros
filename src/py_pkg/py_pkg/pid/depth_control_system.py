"""Outer-loop depth controller for the glider.

This is the math that turns "we want to be at this depth" into "we
want to inflate or deflate the bladder this fast." A single PID
compares the target pressure against the measured pressure (both in
gauge Pa) and outputs a bladder flow ratio q in 1/s — how fast to
move oil between the internal tank and the external bladder. Sign
convention: positive q means descend (sink), negative q means ascend
(rise). The mechanism is displacement, not ballast — to sink we
deflate the bladder (oil back into the tank) so it displaces less
water; to rise we inflate it.

The conversion from this flow command to the actual pump RPM happens
in ``depth_node``, not here. Note the sign flip there: on the BCU bus
a *positive* RPM inflates the bladder (rise), so the controller's
"positive q = descend" is delivered as a *negative* RPM command. This
file stays at the level of "what should the bladder be doing" and
leaves the wire-level details to the node that publishes.
"""

from py_pkg.scenarios.spec.control import DepthSpec
from py_pkg.utils_controls import PIDController


class DepthControlSystem:
    """Single-PID pressure tracker that drives the BCU.

    Everything in and around this controller is in the same unit —
    gauge pascals. The target depth, the measured pressure, and the
    PID's internal error are all gauge Pa, so there's no unit
    conversion happening inside. The output is a bladder flow ratio q
    in 1/s: positive q means "deflate the bladder, sink", negative q
    means "inflate the bladder, rise."

    A single PID is enough here because the physics from the flow
    command all the way down to actual depth behaves like a
    damped double integrator (flow changes bladder volume, which
    changes buoyancy, which changes depth, with drag in the mix).
    That's a shape a well-tuned PID handles cleanly.

    Note: this class doesn't run on its own clock. The caller decides
    when to advance it — each call to ``calc_acc`` is one control
    step. ``depth_node`` ticks it at 10 Hz.
    """

    def __init__(self, config: DepthSpec) -> None:
        # Single PID: gauge Pa -> q (1/s). Derivative-on-measurement
        # (Pa/s rate) and a derivative filter handle the 10 Hz +
        # quantized-pressure noise floor.
        pp = config.pid_pressure
        self.pid_pressure = PIDController(
            kp=pp.kp,
            ki=pp.ki,
            kd=pp.kd,
            integral_limits=pp.integral_limits,
            output_limits=pp.output_limits,
            derivative_filter=pp.derivative_filter,
        )

        # depth_node's control_loop gates calc_acc on its own
        # target_pressure_pa-is-None check, so this is only ever read
        # after target_pose_callback has overwritten it. Surface (gauge 0)
        # is the safe default in case that invariant is ever violated.
        self.target_pressure_pa: float = 0.0

    def calc_acc(self, pressure_pa: float, time: float) -> float:
        """Run one control step and return the bladder flow command q (1/s)."""
        return self.pid_pressure.update(self.target_pressure_pa, pressure_pa, time)

    def reset(self) -> None:
        """Wipe controller memory back to construction state.

        Clears the PID's integrator, derivative filter and timing history,
        and re-arms the surface-safe default target. Used when the depth
        loop is told to go fresh (Do-Nothing mission) so no windup or stale
        setpoint carries over from a previous mission.
        """
        self.pid_pressure.reset()
        self.target_pressure_pa = 0.0
