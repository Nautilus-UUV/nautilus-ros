"""Tier 1 unit tests for the PIDController in py_pkg.utils_controls.

Pure math, no ROS, no hardware. Verifies first-call initialization, the
P/I/D terms in isolation, the integral/output clamps, derivative-on-
measurement (no kick on a setpoint step), the dt<=0 guard, and reset().
"""

import pytest
from py_pkg.utils_controls import PIDController


class TestFirstCall:
    def test_first_call_returns_zero(self):
        pid = PIDController(kp=2.0)
        assert pid.update(target=10.0, input=0.0, time=0.0) == 0.0

    def test_first_call_initializes_state(self):
        pid = PIDController(kp=2.0)
        pid.update(target=10.0, input=0.0, time=0.0)
        assert pid.prev_time == 0.0
        assert pid.prev_error == 10.0
        assert pid.prev_input == 0.0


class TestProportionalOnly:
    def test_pure_p_returns_kp_times_error(self):
        pid = PIDController(kp=2.0, ki=0, kd=0)
        pid.update(target=10.0, input=0.0, time=0.0)  # init
        out = pid.update(target=10.0, input=0.0, time=0.1)
        assert out == pytest.approx(20.0)

    def test_zero_error_gives_zero_output(self):
        pid = PIDController(kp=2.0, ki=0, kd=0)
        pid.update(target=5.0, input=5.0, time=0.0)
        assert pid.update(target=5.0, input=5.0, time=0.1) == 0.0

    def test_negative_error_gives_negative_output(self):
        pid = PIDController(kp=2.0, ki=0, kd=0)
        pid.update(target=0.0, input=10.0, time=0.0)
        out = pid.update(target=0.0, input=10.0, time=0.1)
        assert out == pytest.approx(-20.0)


class TestOutputClamping:
    def test_output_clamped_to_positive_limit(self):
        pid = PIDController(kp=100.0, output_limits=(-10.0, 10.0))
        pid.update(target=10.0, input=0.0, time=0.0)
        # Unclamped would be 1000; clamped to +10
        assert pid.update(target=10.0, input=0.0, time=0.1) == 10.0

    def test_output_clamped_to_negative_limit(self):
        pid = PIDController(kp=100.0, output_limits=(-10.0, 10.0))
        pid.update(target=0.0, input=10.0, time=0.0)
        assert pid.update(target=0.0, input=10.0, time=0.1) == -10.0

    def test_asymmetric_output_limits(self):
        # Asymmetric limits — possible with the new tuple API
        pid = PIDController(kp=100.0, output_limits=(-3.0, 7.0))
        pid.update(target=10.0, input=0.0, time=0.0)
        assert pid.update(target=10.0, input=0.0, time=0.1) == 7.0
        pid2 = PIDController(kp=100.0, output_limits=(-3.0, 7.0))
        pid2.update(target=0.0, input=10.0, time=0.0)
        assert pid2.update(target=0.0, input=10.0, time=0.1) == -3.0


class TestIntegral:
    def test_integral_accumulates(self):
        pid = PIDController(kp=0, ki=1.0, kd=0)
        pid.update(target=10.0, input=0.0, time=0.0)  # init, error=10
        # step 1: integral += 10 * 0.1 = 1.0, output = 1.0
        out1 = pid.update(target=10.0, input=0.0, time=0.1)
        # step 2: integral += 10 * 0.1 = 2.0, output = 2.0
        out2 = pid.update(target=10.0, input=0.0, time=0.2)
        assert out1 == pytest.approx(1.0)
        assert out2 == pytest.approx(2.0)

    def test_integral_clamped_to_limit(self):
        pid = PIDController(
            kp=0,
            ki=1.0,
            kd=0,
            integral_limits=(-5.0, 5.0),
            output_limits=(-1000, 1000),
        )
        pid.update(target=100.0, input=0.0, time=0.0)
        # error*dt = 100 * 0.1 = 10, clamped to 5 → output = 5
        out = pid.update(target=100.0, input=0.0, time=0.1)
        assert out == pytest.approx(5.0)


class TestDerivativeOnMeasurement:
    """The new PID uses derivative-on-measurement (-d(input)/dt), not
    derivative-on-error. A pure setpoint step must NOT cause a derivative
    spike — that was the latent bug in the old implementation.
    """

    def test_no_kick_on_setpoint_step(self):
        pid = PIDController(kp=0, ki=0, kd=1.0)
        pid.update(target=0.0, input=0.0, time=0.0)  # init
        # Setpoint jumps; measurement is unchanged → no derivative kick
        out = pid.update(target=100.0, input=0.0, time=0.1)
        assert out == pytest.approx(0.0)

    def test_responds_to_measurement_change(self):
        pid = PIDController(kp=0, ki=0, kd=1.0)
        pid.update(target=0.0, input=0.0, time=0.0)  # init
        # Measurement rises by 1 over dt=0.1 → derivative = -10
        # (negative because the derivative is on -input)
        out = pid.update(target=0.0, input=1.0, time=0.1)
        assert out == pytest.approx(-10.0)

    def test_derivative_zero_when_measurement_constant(self):
        pid = PIDController(kp=0, ki=0, kd=1.0)
        pid.update(target=10.0, input=5.0, time=0.0)  # init
        # input unchanged → derivative term is zero
        assert pid.update(target=10.0, input=5.0, time=0.1) == 0.0


class TestTimeDeltaGuard:
    """When dt is non-positive (clock skew, duplicate stamps), update()
    must not divide by zero or flip the derivative's sign.
    """

    def test_zero_dt_does_not_raise(self):
        pid = PIDController(kp=2.0, ki=1.0, kd=1.0)
        pid.update(target=10.0, input=0.0, time=1.0)  # init
        # Same timestamp — dt = 0
        out = pid.update(target=10.0, input=0.0, time=1.0)
        # Returns kp*error + ki*integral; integral wasn't updated this call
        assert out == pytest.approx(2.0 * 10.0)

    def test_negative_dt_does_not_raise(self):
        pid = PIDController(kp=2.0, ki=0, kd=1.0)
        pid.update(target=10.0, input=0.0, time=2.0)  # init
        out = pid.update(target=10.0, input=0.0, time=1.0)
        # No exception, no NaN, no negative-dt sign flip
        assert out == pytest.approx(20.0)


class TestReset:
    """reset() should clear integral, prev_*, and the filtered derivative."""

    def test_reset_clears_integral_and_prev_state(self):
        pid = PIDController(kp=0, ki=1.0, kd=0)
        pid.update(target=10.0, input=0.0, time=0.0)
        pid.update(target=10.0, input=0.0, time=0.1)  # integral builds
        assert pid.integral != 0.0

        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_time is None
        assert pid.prev_error == 0.0
        assert pid.prev_input == 0.0

    def test_first_call_after_reset_returns_zero(self):
        pid = PIDController(kp=2.0)
        pid.update(target=10.0, input=0.0, time=0.0)
        pid.update(target=10.0, input=0.0, time=0.1)
        pid.reset()
        # Behaves like a fresh controller
        assert pid.update(target=10.0, input=0.0, time=0.5) == 0.0
