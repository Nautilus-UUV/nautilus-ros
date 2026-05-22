"""Tier 1 unit tests for DepthControlSystem.

The controller is a thin shell around a single PID on gauge pressure
(Pa) -> bladder flow ratio q (1/s). Pure logic; no ROS context, no
hardware. Uses a default DepthSpec so the tests track the same gains
and limits as the real node would see absent a scenario override.

Z-positive-down throughout: target_pressure_pa is gauge Pa at the
desired hold depth.
"""

import pytest

from py_pkg.physics import depth_to_pressure_pa, gauge_pressure_pa
from py_pkg.pid.depth_control_system import DepthControlSystem
from py_pkg.scenarios.spec.control import DepthSpec


def _gauge_pa_for_depth(depth_m: float) -> float:
    return gauge_pressure_pa(depth_to_pressure_pa(depth_m))


def _make_system(target_pressure_pa: float = _gauge_pa_for_depth(70.0)):
    cs = DepthControlSystem(DepthSpec())
    cs.target_pressure_pa = target_pressure_pa
    return cs


class TestCalcAccCadence:
    """Every call advances the PID — the caller owns cadence.

    The controller used to gate updates internally on a 1/frequency
    period; that gate hid stalled callbacks and coupled gain units to
    the loop rate. Now each call is one PID step at the supplied time.
    """

    def test_first_call_returns_zero(self):
        # PIDController returns 0 on the first call (initialises prev_time).
        cs = _make_system()
        assert cs.calc_acc(pressure_pa=0.0, time=0.0) == pytest.approx(0.0)

    def test_subsequent_call_produces_nonzero_command_under_error(self):
        cs = _make_system(target_pressure_pa=_gauge_pa_for_depth(70.0))
        cs.calc_acc(pressure_pa=0.0, time=0.0)
        # Same wallclock-style cadence the node would supply.
        cmd = cs.calc_acc(pressure_pa=0.0, time=0.1)
        assert cmd != 0.0

    def test_repeated_calls_within_same_period_still_step(self):
        # No internal throttle — three calls inside a single 0.1 s window
        # all advance the PID. Sign stays positive (target deeper).
        cs = _make_system(target_pressure_pa=_gauge_pa_for_depth(70.0))
        cs.calc_acc(pressure_pa=0.0, time=0.0)
        a = cs.calc_acc(pressure_pa=0.0, time=0.01)
        b = cs.calc_acc(pressure_pa=0.0, time=0.02)
        assert a > 0 and b > 0


class TestCalcAccSign:
    """PID drives bladder flow toward closing the pressure error.

    Z-positive-down throughout, so the natural sign already matches q's
    semantic — no negation in calc_acc:
      target deeper (higher Pa)  -> positive q (fill bladder, sink)
      target shallower (lower Pa) -> negative q (drain bladder, rise)
    """

    def test_zero_error_zero_command(self):
        cs = _make_system(target_pressure_pa=0.0)
        for t in (0.0, 0.1, 0.2, 0.3):
            result = cs.calc_acc(pressure_pa=0.0, time=t)
        assert result == pytest.approx(0.0)

    def test_command_positive_when_target_deeper(self):
        cs = _make_system(target_pressure_pa=_gauge_pa_for_depth(70.0))
        for t in (0.0, 0.1, 0.2):
            result = cs.calc_acc(pressure_pa=0.0, time=t)
        assert result > 0

    def test_command_negative_when_target_shallower(self):
        cs = _make_system(target_pressure_pa=0.0)
        for t in (0.0, 0.1, 0.2):
            result = cs.calc_acc(pressure_pa=_gauge_pa_for_depth(50.0), time=t)
        assert result < 0


class TestReset:
    """`reset()` returns the controller to construction state so no PID
    windup or stale setpoint carries over (Do-Nothing mission)."""

    def test_reset_clears_target_to_surface(self):
        cs = _make_system(target_pressure_pa=_gauge_pa_for_depth(70.0))
        cs.reset()
        assert cs.target_pressure_pa == pytest.approx(0.0)

    def test_reset_clears_pid_state(self):
        cs = _make_system(target_pressure_pa=_gauge_pa_for_depth(70.0))
        # Wind the integrator up with a sustained error.
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            cs.calc_acc(pressure_pa=0.0, time=t)
        assert cs.pid_pressure.integral != 0.0

        cs.reset()
        assert cs.pid_pressure.integral == pytest.approx(0.0)
        assert cs.pid_pressure.prev_time is None
        # Post-reset, the first call re-seeds prev_time and returns 0 again,
        # exactly like a freshly constructed controller.
        assert cs.calc_acc(pressure_pa=0.0, time=10.0) == pytest.approx(0.0)
