from py_pkg.utils_controls import PIDController


class AxisController:
    """Controls a single axis (roll or pitch) of the vehicle.

    This controller calculates how much the motor needs to move to reach a desired
    angle. To prevent overworking the motor with tiny, continuous adjustments,
    it uses a state machine and tolerance checks. It stops sending commands if the
    vehicle is already close enough to the target angle (`position_tolerance`) or
    if the required motor movement is very small (`command_tolerance`).

    By default (used for Roll), both the vehicle's angle and the motor's position
    are measured in degrees. Therefore, the controller calculates an *incremental*
    adjustment: it adds the required correction to the motor's current position.

    Note: Time is passed in externally in seconds. This ensures the control math
    (Ki/Kd values) remains consistent regardless of how fast the main program loops.
    """

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

        self.state = AxisController.State.STEADY
        self.current_pos = 0.0
        self.target_pos = 0.0
        self.last_commanded_pos = 0.0
        self._primed = False

    def update_sensor(self, measured_pos):
        self.current_pos = measured_pos

    def _pid_correction(self, desired_value, time):
        return self.pid.update(desired_value, self.current_pos, time)

    def _clamp_motor(self, val):
        lo, hi = self.output_limits
        if val > hi:
            return hi
        if val < lo:
            return lo
        return val

    def _compute_new_target(self, desired_value, time):
        # By default, the motor and the target angle use the same unit (degrees).
        # We calculate the relative correction and add it to our current position.
        return self.current_pos + self._pid_correction(desired_value, time)

    def compute_control(self, desired_value, time):
        """Calculates the raw target position before safety limits are applied."""
        return self._compute_new_target(desired_value, time)

    def update(self, desired_value, time):
        """Evaluates whether to move the motor based on current tolerances, returning a command or None."""
        new_target = self._compute_new_target(desired_value, time)
        self.target_pos = self._clamp_motor(new_target)
        if not self._primed:
            self._primed = True
            return None
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
    """Controls the pitch axis by linearly moving a weight (a mass-shifter).

    Unlike the roll axis, this motor moves in meters (linear stroke) to change
    an angle measured in degrees. The tuning parameter (Kp) acts as the conversion
    factor between meters and degrees.

    Because of this physical difference, the mathematical output is the *exact
    absolute position* (in meters) the weight needs to move to. We do not add
    this to the current position, as that would apply the correction twice.
    """

    def _compute_new_target(self, desired_value, time):
        return self._pid_correction(desired_value, time)
