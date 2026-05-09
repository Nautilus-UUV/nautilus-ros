"""Outer-loop depth controller for the glider.

This is the math that turns "we want to be at this depth" into "we
want to fill or empty the bladder this fast." A single PID compares
the target pressure against the measured pressure (both in gauge Pa)
and outputs a bladder flow ratio q in 1/s — the fraction of the
bladder to fill or empty per second. Positive q means we are filling
the bladder, which makes the glider denser and sinks it; negative q
empties it.

The conversion from this flow command to the actual pump RPM
(including the sign flip — the pump is wired so that *filling* the
bladder requires a *negative* RPM command) happens in ``depth_node``,
not here. This file stays at the level of "what should the bladder be
doing" and leaves the wire-level details to the node that publishes.
"""

from py_pkg.utils_controls import PIDController


class DepthControlSystem:
    """Single-PID pressure tracker that drives the BCU.

    Everything in and around this controller is in the same unit —
    gauge pascals. The target depth, the measured pressure, and the
    PID's internal error are all gauge Pa, so there's no unit
    conversion happening inside. The output is a bladder flow ratio q
    in 1/s: positive q means "fill the bladder, sink", negative q
    means "empty the bladder, rise."

    A single PID is enough here because the physics from the flow
    command all the way down to actual depth behaves like a
    damped double integrator (flow changes bladder volume, which
    changes buoyancy, which changes depth, with drag in the mix).
    That's a shape a well-tuned PID handles cleanly.

    Note: this class doesn't run on its own clock. The caller decides
    when to advance it — each call to ``calc_acc`` is one control
    step. ``depth_node`` ticks it at 10 Hz.
    """

    def __init__(self, config: dict) -> None:
        # Single PID: gauge Pa -> q (1/s). Derivative-on-measurement
        # (Pa/s rate) and a derivative filter handle the 10 Hz +
        # quantized-pressure noise floor.
        self.pid_pressure = PIDController(**config["pid_pressure"])

        # Glide path setpoints, gauge Pa.
        self.low_pressure_pa: float = config["low_pressure_pa"]
        self.high_pressure_pa: float = config["high_pressure_pa"]
        self.target_pressure_pa: float = self.high_pressure_pa

    def calc_acc(self, pressure_pa: float, time: float) -> float:
        """Run one control step and return the bladder flow command q (1/s)."""
        return self.pid_pressure.update(self.target_pressure_pa, pressure_pa, time)
