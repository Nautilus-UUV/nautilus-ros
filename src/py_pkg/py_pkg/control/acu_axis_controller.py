"""ACU axis controllers (PID + mass-shifter helpers).

DEPRECATED / NOT IMPLEMENTED IN SIM: these back the ACU node, whose simulated
actuator was removed (glider_nautilus is now static, symmetric, BCU-only).
Retained for the real-hardware ACU path only.
"""

from py_pkg.math_utils import clamp
from py_pkg.utils_controls import PIDController


class AxisController:
    """Controls a single axis (roll or pitch) of the vehicle.

    A PID inside, an output clamp at the motor frame, and a
    redundant-publish guard at the back. The guard is what keeps the
    EPOS bus quiet when the axis is sitting saturated against its clamp:
    if the motor-frame target hasn't moved by more than `command_tolerance`
    since we last published, we just don't republish.

    Both the vehicle's angle and the motor's position are measured in
    degrees, so the controller calculates an *incremental* adjustment:
    it adds the required correction to the motor's current position.

    Time is passed in externally in seconds. This ensures the control
    math (ki/kd values) remains consistent regardless of how fast the
    main program loops.
    """

    def __init__(
        self,
        kp,
        command_tolerance,
        ki=0.0,
        kd=0.0,
        integral_limits=(-1000.0, 1000.0),
        output_limits=(-1000.0, 1000.0),
        derivative_filter=0.0,
    ):
        self.command_tolerance = command_tolerance
        self.output_limits = output_limits

        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            integral_limits=integral_limits,
            output_limits=(-float("inf"), float("inf")),
            derivative_filter=derivative_filter,
        )

        self.current_pos = 0.0
        self.target_pos = 0.0
        # `None` = "we have never published anything yet", which forces
        # the first post-prime call to emit. Without this, an axis whose
        # prime-call target happens to equal its first-real-call target
        # (asymmetric clamps where clamp(0) ≠ 0, saturated targets, etc.)
        # would never publish.
        self.last_commanded_pos: float | None = None
        self._primed = False

    def reset(self):
        """Wipe controller memory back to construction state.

        Clears the PID's integrator/filter/timing and the redundant-publish
        guard so the next `update` re-primes exactly as it did at boot. Used
        when the ACU is told to go fresh (mission stop).
        """
        self.pid.reset()
        self.current_pos = 0.0
        self.target_pos = 0.0
        self.last_commanded_pos = None
        self._primed = False

    def update_sensor(self, measured_pos):
        self.current_pos = measured_pos

    def update(self, desired_value, time):
        """Returns a motor-frame command, or None if the redundant-publish
        guard determines the bus has nothing new to hear."""
        # Motor and target angle share a unit (degrees): apply the PID
        # correction relative to where the motor currently sits.
        correction = self.pid.update(desired_value, self.current_pos, time)
        self.target_pos = clamp(self.current_pos + correction, *self.output_limits)

        # Prime: PID's first call returns 0 to seed prev_time, which would
        # show up as a phantom "go to zero" command. Swallow it.
        if not self._primed:
            self._primed = True
            return None

        if (
            self.last_commanded_pos is None
            or abs(self.target_pos - self.last_commanded_pos) > self.command_tolerance
        ):
            self.last_commanded_pos = self.target_pos
            return self.target_pos

        return None
