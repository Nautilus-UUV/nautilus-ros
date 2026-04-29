"""Per-axis PID controller + state machine + deadband for the ACU.

Generates motor-frame target positions for a single axis (roll or pitch).
The PID acts on UUV-frame angle (degrees) and produces a positional
correction; the new target is `current + correction`, clamped to the
physical motor-frame stroke/angle, and gated by a state machine + command
deadband to avoid unnecessary motor traffic.
"""

from py_pkg.utils_controls import PIDController


class AxisController:
    class State:
        STEADY = 0
        SHIFTING = 1

    def __init__(
        self,
        name,
        Kp,
        position_tolerance,
        command_tolerance,
        Ki=0.0,
        Kd=0.0,
        integral_limits=(-1000.0, 1000.0),
        output_limits=(-1000.0, 1000.0),
        derivative_filter=0.0,
    ):
        self.name = name
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.position_tolerance = position_tolerance
        self.command_tolerance = command_tolerance
        self.output_limits = output_limits

        self.pid = PIDController(
            kp=Kp,
            ki=Ki,
            kd=Kd,
            integral_limits=integral_limits,
            output_limits=(-float("inf"), float("inf")),
            derivative_filter=derivative_filter,
        )
        # Seed the PID so the first user-facing update() produces a
        # non-zero correction (prev_time gets initialised here).
        self.pid.update(0.0, 0.0, 0.0)
        self._tick = 1.0

        self.state = AxisController.State.STEADY
        self.current_pos = 0.0
        self.target_pos = 0.0
        self.last_commanded_pos = 0.0

    def update_sensor(self, measured_pos):
        self.current_pos = measured_pos

    def _pid_correction(self, desired_value):
        correction = self.pid.update(desired_value, self.current_pos, self._tick)
        self._tick += 1.0
        return correction

    def _clamp_motor(self, val):
        lo, hi = self.output_limits
        if val > hi:
            return hi
        if val < lo:
            return lo
        return val

    def _compute_new_target(self, desired_value):
        # Roll-style: slew current_pos toward desired_value in the same frame.
        # MassShifterController overrides this for pitch (frame change).
        return self.current_pos + self._pid_correction(desired_value)

    def compute_control(self, desired_value):
        """new_target_pre_clamp from the configured controller law."""
        return self._compute_new_target(desired_value)

    def update(self, desired_value):
        """Run the state machine and return a motor-frame command, or None."""
        new_target = self._compute_new_target(desired_value)
        self.target_pos = self._clamp_motor(new_target)
        error = abs(desired_value - self.current_pos)

        if self.state == AxisController.State.STEADY:
            if error > self.position_tolerance:
                self.state = AxisController.State.SHIFTING
                self.last_commanded_pos = self.current_pos
                return self.target_pos

        elif self.state == AxisController.State.SHIFTING:
            if error <= self.position_tolerance:
                self.state = AxisController.State.STEADY

        if abs(self.target_pos - self.last_commanded_pos) > self.command_tolerance:
            self.last_commanded_pos = self.target_pos
            return self.target_pos

        return None


class MassShifterController(AxisController):
    """Pitch-axis variant: PID error -> absolute mass-shifter stroke (m).

    Roll's actuator is itself an angular position with feedback, so the
    base controller slews `current_pos` toward `desired_value` in the same
    frame. Pitch's actuator is a mass-shifter whose stroke is set directly
    from pitch error. Kp's units are m/deg.
    """

    def _compute_new_target(self, desired_value):
        return self._pid_correction(desired_value)
