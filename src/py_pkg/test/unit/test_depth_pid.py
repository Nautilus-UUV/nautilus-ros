"""Tier 1 unit tests for the depth loop's PID as bcu_node configures it.

The depth loop is a single PID on gauge pressure (Pa) -> bladder flow
ratio q (1/s), built straight from DepthSpec.pid_pressure (the same
construction bcu_node performs). Pure logic; no ROS context, no
hardware. Uses a default DepthSpec so the tests track the same gains
and limits as the real node would see absent a scenario override.

Z-positive-down throughout: the target is gauge Pa at the desired hold
depth.
"""

import pytest

from py_pkg.physics import depth_to_pressure_pa, gauge_pressure_pa
from py_pkg.scenarios.spec.control import DepthSpec
from py_pkg.utils_controls import PIDController


def _gauge_pa_for_depth(depth_m: float) -> float:
    return gauge_pressure_pa(depth_to_pressure_pa(depth_m))


def _make_pid() -> PIDController:
    pp = DepthSpec().pid_pressure
    return PIDController(
        kp=pp.kp,
        ki=pp.ki,
        kd=pp.kd,
        integral_limits=pp.integral_limits,
        output_limits=pp.output_limits,
        derivative_filter=pp.derivative_filter,
    )


class TestCadence:
    """Every call advances the PID — the caller owns cadence.

    The depth loop used to gate updates internally on a 1/frequency
    period; that gate hid stalled callbacks and coupled gain units to
    the loop rate. Now each call is one PID step at the supplied time.
    """

    def test_first_call_returns_zero(self):
        # PIDController returns 0 on the first call (initialises prev_time).
        pid = _make_pid()
        target = _gauge_pa_for_depth(70.0)
        assert pid.update(target, 0.0, time=0.0) == pytest.approx(0.0)

    def test_subsequent_call_produces_nonzero_command_under_error(self):
        pid = _make_pid()
        target = _gauge_pa_for_depth(70.0)
        pid.update(target, 0.0, time=0.0)
        # Same wallclock-style cadence the node would supply.
        assert pid.update(target, 0.0, time=0.1) != 0.0

    def test_repeated_calls_within_same_period_still_step(self):
        # No internal throttle — three calls inside a single 0.1 s window
        # all advance the PID. Sign stays positive (target deeper).
        pid = _make_pid()
        target = _gauge_pa_for_depth(70.0)
        pid.update(target, 0.0, time=0.0)
        a = pid.update(target, 0.0, time=0.01)
        b = pid.update(target, 0.0, time=0.02)
        assert a > 0 and b > 0


class TestSign:
    """The PID drives bladder flow toward closing the pressure error.

    Z-positive-down throughout, so the natural sign already matches q's
    semantic — no negation between the PID and the flow demand:
      target deeper (higher Pa)  -> positive q (fill bladder, sink)
      target shallower (lower Pa) -> negative q (drain bladder, rise)
    """

    def test_zero_error_zero_command(self):
        pid = _make_pid()
        for t in (0.0, 0.1, 0.2, 0.3):
            result = pid.update(0.0, 0.0, time=t)
        assert result == pytest.approx(0.0)

    def test_command_positive_when_target_deeper(self):
        pid = _make_pid()
        target = _gauge_pa_for_depth(70.0)
        for t in (0.0, 0.1, 0.2):
            result = pid.update(target, 0.0, time=t)
        assert result > 0

    def test_command_negative_when_target_shallower(self):
        pid = _make_pid()
        for t in (0.0, 0.1, 0.2):
            result = pid.update(0.0, _gauge_pa_for_depth(50.0), time=t)
        assert result < 0


class TestReset:
    """`reset()` returns the PID to construction state so no windup
    carries over (mission stop; the node drops its own target to None)."""

    def test_reset_clears_pid_state(self):
        pid = _make_pid()
        target = _gauge_pa_for_depth(70.0)
        # Wind the integrator up with a sustained error.
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            pid.update(target, 0.0, time=t)
        assert pid.integral != 0.0

        pid.reset()
        assert pid.integral == pytest.approx(0.0)
        assert pid.prev_time is None
        # Post-reset, the first call re-seeds prev_time and returns 0 again,
        # exactly like a freshly constructed controller.
        assert pid.update(target, 0.0, time=10.0) == pytest.approx(0.0)
