from py_pkg.utils_controls import PIDController


class AxisController:
    """Controls a single axis (roll or pitch) of the vehicle.

    A PID inside, an output clamp at the motor frame, and a
    redundant-publish guard at the back. The guard is what keeps the
    EPOS bus quiet when the axis is sitting saturated against its clamp:
    if the motor-frame target hasn't moved by more than `command_tolerance`
    since we last published, we just don't republish.

    By default (used for Roll), both the vehicle's angle and the motor's
    position are measured in degrees. Therefore, the controller
    calculates an *incremental* adjustment: it adds the required
    correction to the motor's current position. `MassShifterController`
    overrides this for the pitch axis, where the motor frame (m) is
    different from the sensor frame (deg).

    Time is passed in externally in seconds. This ensures the control
    math (ki/kd values) remains consistent regardless of how fast the
    main program loops.
    """

    def __init__(
        self,
        name,
        kp,
        command_tolerance,
        ki=0.0,
        kd=0.0,
        integral_limits=(-1000.0, 1000.0),
        output_limits=(-1000.0, 1000.0),
        derivative_filter=0.0,
    ):
        self.name = name
        self.kp = kp
        self.ki = ki
        self.kd = kd
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
        """Returns a motor-frame command, or None if the redundant-publish
        guard determines the bus has nothing new to hear."""
        self.target_pos = self._clamp_motor(
            self._compute_new_target(desired_value, time)
        )

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


class MassShifterController(AxisController):
    """Controls the pitch axis by linearly moving a weight (a mass-shifter).

    Unlike the roll axis, this motor moves in meters (linear stroke) to change
    an angle measured in degrees. The tuning parameter (kp) acts as the conversion
    factor between meters and degrees.

    Because of this physical difference, the mathematical output is the *exact
    absolute position* (in meters) the weight needs to move to. We do not add
    this to the current position, as that would apply the correction twice.
    """

    def _compute_new_target(self, desired_value, time):
        return self._pid_correction(desired_value, time)
