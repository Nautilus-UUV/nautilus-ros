class PIDController:
    """PID controller with derivative-on-measurement and integral windup protection."""

    def __init__(
        self,
        kp: float = 1,
        ki: float = 0,
        kd: float = 0,
        integral_limits: tuple[float, float] = (-1000, 1000),
        output_limits: tuple[float, float] = (-1000, 1000),
        derivative_filter: float = 0.0,
    ) -> None:
        """
        Initialize PID controller.

        Parameters
        ----------
        kp : float, optional
            Proportional gain.
        ki : float, optional
            Integral gain.
        kd : float, optional
            Derivative gain.
        integral_limits : tuple[float, float], optional
            (min, max) integral term limits.
        output_limits : tuple[float, float], optional
            (min, max) output limits.
        derivative_filter : float, optional
            Low-pass filter coefficient (0=no filter, 0.1=light, 0.5=heavy).

        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_min, self.integral_max = integral_limits
        self.output_min, self.output_max = output_limits
        self.derivative_filter_coeff = derivative_filter

        self.prev_error = 0.0
        self.prev_input = 0.0
        self.prev_time = None
        self.integral = 0.0
        self.filtered_derivative = 0.0

    def update(self, target: float, input: float, time: float) -> float:
        """
        Update PID controller.

        Parameters
        ----------
        target : float
            Desired setpoint.
        input : float
            Current measured value.
        time : float
            Current timestamp (seconds).

        Returns
        -------
        float
            Control output.

        """
        error = target - input

        # First call initialization
        if self.prev_time is None:
            self.prev_error = error
            self.prev_input = input
            self.prev_time = time
            return 0.0

        time_delta = time - self.prev_time

        # Prevent division by zero
        if time_delta <= 0:
            return self.kp * error + self.ki * self.integral

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * time_delta
        self.integral = max(self.integral_min, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral

        # Derivative term (on measurement to prevent kickback)
        raw_derivative = -(input - self.prev_input) / time_delta

        if self.derivative_filter_coeff > 0:
            # Apply exponential moving average filter
            self.filtered_derivative = (
                self.derivative_filter_coeff * raw_derivative
                + (1 - self.derivative_filter_coeff) * self.filtered_derivative
            )
            derivative = self.filtered_derivative
        else:
            derivative = raw_derivative

        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        output = max(self.output_min, min(self.output_max, output))

        # Store previous values
        self.prev_error = error
        self.prev_input = input
        self.prev_time = time

        return output

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.filtered_derivative = 0.0
        self.prev_error = 0.0
        self.prev_input = 0.0
        self.prev_time = None
