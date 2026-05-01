"""Tier 1 unit tests for DepthControlSystem.

Covers the finite-difference estimators (estimate_velocity,
estimate_acceleration) and the cascaded calc_acc loop. Pure logic; no
ROS context, no hardware. Uses init_control from depth_config so the
tests track the same gains and limits as the real node.
"""

import pytest

from py_pkg.math_utils import Vector
from py_pkg.pid.depth_config import init_control
from py_pkg.pid.depth_control_system import DepthControlSystem


def _make_system(target_depth=70.0):
    # Z-positive-down: 70.0 = "70 m below the surface" (deepest extreme).
    cs = DepthControlSystem(init_control)
    cs.target_depth = target_depth
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
        # (-2 - 0) / (0.1 - 0) = -20
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
        # z(t) = 0.5 * a * t^2 with a = 4 m/s^2 (uniform spacing)
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
    """Cascaded PID drives bladder flow toward closing the depth error.

    Z-positive-down throughout (target_depth, position.z(), velocity, accel),
    so the cascade's natural sign already matches q's semantic — no
    negation in calc_acc:
      target deeper than current  → positive q (fill bladder, glider sinks)
      target shallower than current → negative q (drain bladder, glider rises)
    """

    def test_zero_error_zero_command(self):
        cs = _make_system(target_depth=0.0)
        zero = Vector(0, 0, 0)
        for t in (0.0, 0.1, 0.2, 0.3):
            result = cs.calc_acc(zero, 0.0, t)
        # At target = current, every cascade stage sees zero error → output 0
        assert result == pytest.approx(0.0)

    def test_command_positive_when_target_deeper(self):
        cs = _make_system(target_depth=70.0)
        zero = Vector(0, 0, 0)
        for t in (0.0, 0.1, 0.2):
            result = cs.calc_acc(zero, 0.0, t)
        # Need to sink → fill bladder → q > 0
        assert result > 0

    def test_command_negative_when_target_shallower(self):
        # Already deep, asked to come up to the surface.
        cs = _make_system(target_depth=0.0)
        deep = Vector(0, 0, 50)
        for t in (0.0, 0.1, 0.2):
            result = cs.calc_acc(deep, 0.0, t)
        # Need to rise → drain bladder → q < 0
        assert result < 0
