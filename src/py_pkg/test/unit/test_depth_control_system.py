"""Tier 1 unit tests for DepthControlSystem.

Covers the finite-difference estimators (estimate_velocity,
estimate_acceleration) and the cascaded calc_acc loop. Pure logic; no
ROS context, no hardware. Uses init_control from depth_config so the
tests track the same gains and limits as the real node.

The cascade is pressure-native: position.z() carries gauge Pa, the
target is gauge Pa, and the rate / acceleration estimates are Pa/s
and Pa/s^2.
"""

import pytest

from py_pkg.math_utils import Vector
from py_pkg.physics import depth_to_pressure_pa, gauge_pressure_pa
from py_pkg.pid.depth_config import init_control
from py_pkg.pid.depth_control_system import DepthControlSystem


def _gauge_pa_for_depth(depth_m: float) -> float:
    return gauge_pressure_pa(depth_to_pressure_pa(depth_m))


def _make_system(target_pressure_pa: float = _gauge_pa_for_depth(70.0)):
    # Z-positive-down: target_pressure_pa = gauge pressure at the depth
    # we want to hold; default tracks the deepest config extreme (~70 m).
    cs = DepthControlSystem(init_control)
    cs.target_pressure_pa = target_pressure_pa
    return cs


class TestEstimateVelocity:
    """Forward-difference on the last two stored positions."""

    def test_returns_none_with_zero_points(self):
        cs = _make_system()
        assert cs.estimate_velocity([], []) is None

    def test_returns_none_with_one_point(self):
        cs = _make_system()
        assert cs.estimate_velocity([Vector(0, 0, 0)], [0.0]) is None

    def test_two_points_forward_difference(self):
        cs = _make_system()
        positions = [Vector(0, 0, 0), Vector(0, 0, -2)]
        times = [0.0, 0.1]
        # (-2 - 0) / (0.1 - 0) = -20  (Pa/s; the math is unit-agnostic)
        assert cs.estimate_velocity(positions, times) == pytest.approx(-20.0)

    def test_uses_only_last_two_points(self):
        cs = _make_system()
        positions = [Vector(0, 0, 0), Vector(0, 0, 5), Vector(0, 0, 7)]
        times = [0.0, 0.5, 1.0]
        # (7 - 5) / (1.0 - 0.5) = 4.0 — earlier points ignored
        assert cs.estimate_velocity(positions, times) == pytest.approx(4.0)


class TestEstimateAcceleration:
    """Central second-difference on the last three stored positions."""

    def test_returns_none_with_fewer_than_three_points(self):
        cs = _make_system()
        assert cs.estimate_acceleration([], []) is None
        assert cs.estimate_acceleration([Vector(0, 0, 0)], [0.0]) is None
        assert cs.estimate_acceleration(
            [Vector(0, 0, 0), Vector(0, 0, 1)], [0.0, 0.1]
        ) is None

    def test_constant_velocity_returns_zero_acceleration(self):
        cs = _make_system()
        # Linear z(t) → second derivative is 0
        positions = [Vector(0, 0, 0), Vector(0, 0, 1), Vector(0, 0, 2)]
        times = [0.0, 0.1, 0.2]
        assert cs.estimate_acceleration(positions, times) == pytest.approx(0.0)

    def test_constant_acceleration_recovered(self):
        cs = _make_system()
        # z(t) = 0.5 * a * t^2 with a = 4 (Pa/s^2 in this regime; uniform spacing)
        a = 4.0
        positions = [Vector(0, 0, 0.5 * a * t**2) for t in (0.0, 0.1, 0.2)]
        times = [0.0, 0.1, 0.2]
        assert cs.estimate_acceleration(positions, times) == pytest.approx(a)


class TestCalcAccThrottling:
    """calc_acc returns prev_command if called more often than 1/frequency."""

    def test_first_call_at_t0_throttles_returns_zero(self):
        cs = _make_system()
        result = cs.calc_acc(
            position=Vector(0, 0, 0),
            tank=0.0,
            time=0.0,
        )
        # init: prev_command=0.0, throttle hits at t=0 since prev_update_time=0
        assert result == 0.0

    def test_throttles_within_period(self):
        cs = _make_system()
        zero = Vector(0, 0, 0)
        cs.calc_acc(zero, 0.0, 0.0)
        # Period is 1/frequency = 0.1; t=0.05 still in cooldown
        result = cs.calc_acc(zero, 0.0, 0.05)
        assert result == 0.0


class TestCalcAccSign:
    """Cascaded PID drives bladder flow toward closing the pressure error.

    Z-positive-down throughout (target_pressure_pa, position.z(), rate,
    accel), so the cascade's natural sign already matches q's semantic —
    no negation in calc_acc:
      target deeper (higher Pa) than current  → positive q (fill bladder, glider sinks)
      target shallower (lower Pa) than current → negative q (drain bladder, glider rises)
    """

    def test_zero_error_zero_command(self):
        cs = _make_system(target_pressure_pa=0.0)
        zero = Vector(0, 0, 0)
        for t in (0.0, 0.1, 0.2, 0.3):
            result = cs.calc_acc(zero, 0.0, t)
        # At target = current, every cascade stage sees zero error → output 0
        assert result == pytest.approx(0.0)

    def test_command_positive_when_target_deeper(self):
        cs = _make_system(target_pressure_pa=_gauge_pa_for_depth(70.0))
        zero = Vector(0, 0, 0)
        for t in (0.0, 0.1, 0.2):
            result = cs.calc_acc(zero, 0.0, t)
        # Need to sink → fill bladder → q > 0
        assert result > 0

    def test_command_negative_when_target_shallower(self):
        # Already deep, asked to come up to the surface.
        cs = _make_system(target_pressure_pa=0.0)
        deep = Vector(0, 0, _gauge_pa_for_depth(50.0))
        for t in (0.0, 0.1, 0.2):
            result = cs.calc_acc(deep, 0.0, t)
        # Need to rise → drain bladder → q < 0
        assert result < 0
