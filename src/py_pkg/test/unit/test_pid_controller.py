"""Tier 1 unit tests for the PIDController used inside DepthControlSystem.

Pure math, no ROS, no hardware. Verifies first-call initialization, the
P/I/D terms in isolation, and the integral/output clamping.

Note: this is the older PID implementation in pid.depth_control_system,
not the cleaner one in utils_controls.py (which is currently unused).
"""

import pytest
from py_pkg.pid.depth_control_system import PIDController


class TestFirstCall:
    def test_first_call_returns_zero(self):
        pid = PIDController(kp=2.0)
        assert pid.update(target=10.0, input=0.0, time=0.0) == 0

    def test_first_call_initializes_state(self):
        pid = PIDController(kp=2.0)
        pid.update(target=10.0, input=0.0, time=0.0)
        assert pid.prev_time == 0.0
        assert pid.prev_error == 10.0


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
        pid = PIDController(kp=100.0, output_limit=10.0)
        pid.update(target=10.0, input=0.0, time=0.0)
        # Unclamped would be 1000; clamped to +10
        assert pid.update(target=10.0, input=0.0, time=0.1) == 10.0

    def test_output_clamped_to_negative_limit(self):
        pid = PIDController(kp=100.0, output_limit=10.0)
        pid.update(target=0.0, input=10.0, time=0.0)
        assert pid.update(target=0.0, input=10.0, time=0.1) == -10.0


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
        pid = PIDController(kp=0, ki=1.0, kd=0, integral_limit=5.0, output_limit=1000)
        pid.update(target=100.0, input=0.0, time=0.0)
        # error*dt = 100 * 0.1 = 10, clamped to 5 → output = 5
        out = pid.update(target=100.0, input=0.0, time=0.1)
        assert out == pytest.approx(5.0)


class TestDerivative:
    def test_derivative_responds_to_error_change(self):
        pid = PIDController(kp=0, ki=0, kd=1.0)
        pid.update(target=0.0, input=0.0, time=0.0)  # init, prev_error=0
        # error = 1 - 0 = 1, dt = 0.1, derivative = (1 - 0) / 0.1 = 10
        out = pid.update(target=1.0, input=0.0, time=0.1)
        assert out == pytest.approx(10.0)

    def test_derivative_zero_when_error_constant(self):
        pid = PIDController(kp=0, ki=0, kd=1.0)
        pid.update(target=10.0, input=0.0, time=0.0)  # init, prev_error=10
        # error = 10, prev_error = 10, derivative = 0
        assert pid.update(target=10.0, input=0.0, time=0.1) == 0.0
