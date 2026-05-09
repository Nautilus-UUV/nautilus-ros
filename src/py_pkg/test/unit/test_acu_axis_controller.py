"""Tier 1 unit tests for AxisController.

The pitch axis is now bang-bang inside ``pid/acu_node.py`` (no separate
class), so this file exercises the surface that remains: AxisController
as the roll-only PID + motor-frame clamp + redundant-publish guard.

Time is supplied explicitly to every step. Tests prime each axis at
t=0.0 (PIDController returns 0 on its first call to seed prev_time) and
assert behaviour at t>=0.1, the same cadence the 10 Hz node uses.
"""

import pytest

from py_pkg.pid.acu_axis_controller import AxisController


PRIME_TIME = 0.0
DT = 0.1


def _t(step: int) -> float:
    """10 Hz monotonic clock; t=0 is the prime call."""
    return PRIME_TIME + DT * step


def _prime(axis: AxisController, current_pos: float = 0.0) -> None:
    """Consume the seed-call. PIDController returns 0 on first update; the
    axis swallows it via _primed so this leaves last_commanded_pos at None."""
    axis.update_sensor(current_pos)
    assert axis.update(desired_value=current_pos, time=PRIME_TIME) is None


def make_axis(Kp=0.5, command_tolerance=0.0, output_limits=(-1000.0, 1000.0)):
    return AxisController(
        name="test_axis",
        Kp=Kp,
        command_tolerance=command_tolerance,
        output_limits=output_limits,
    )


class TestPrimeCall:
    """The seed call is swallowed: PIDController's first update returns 0
    (just to latch prev_time) and that 0 must not appear on the wire."""

    def test_first_update_returns_none(self):
        axis = make_axis()
        axis.update_sensor(0.0)
        assert axis.update(desired_value=10.0, time=PRIME_TIME) is None

    def test_first_real_call_emits_even_when_target_unchanged(self):
        # last_commanded_pos starts at None, so the first post-prime tick
        # always emits regardless of how close target_pos is to 0.
        axis = make_axis(Kp=0.5, command_tolerance=10.0)
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=0.0, time=_t(1))
        assert cmd == pytest.approx(0.0)


class TestProportionalIncrement:
    """new_target = current_pos + Kp * (desired - current_pos).

    Roll's actuator and sensor share units (degrees), so the controller
    treats Kp's output as an *increment* on the current position rather
    than an absolute target.
    """

    def test_proportional_step_toward_target(self):
        axis = make_axis(Kp=0.5)
        _prime(axis, current_pos=0.0)
        # error = 10, correction = 5, target = 0 + 5 = 5.
        assert axis.update(desired_value=10.0, time=_t(1)) == pytest.approx(5.0)

    def test_zero_error_holds_position(self):
        axis = make_axis(Kp=0.5)
        _prime(axis, current_pos=5.0)
        assert axis.update(desired_value=5.0, time=_t(1)) == pytest.approx(5.0)

    def test_negative_error_steps_backward(self):
        axis = make_axis(Kp=0.5)
        _prime(axis, current_pos=10.0)
        # error = -10, correction = -5, target = 10 - 5 = 5.
        assert axis.update(desired_value=0.0, time=_t(1)) == pytest.approx(5.0)

    def test_kp_one_jumps_to_target(self):
        axis = make_axis(Kp=1.0)
        _prime(axis, current_pos=0.0)
        assert axis.update(desired_value=10.0, time=_t(1)) == pytest.approx(10.0)


class TestOutputClamp:
    """``output_limits`` clamps the motor-frame target. Roll's shipped
    config uses ±ACU_ROLL_MAX_ANGLE_DEG; here we just stress the
    saturation behaviour with a tight range."""

    def _make(self):
        return AxisController(
            name="roll",
            Kp=0.5,
            command_tolerance=0.0,
            output_limits=(-25.0, 25.0),
        )

    def test_large_positive_error_clamped_to_positive_limit(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        # Unclamped target = 0 + 0.5*100 = 50 → clamped to +25.
        assert axis.update(desired_value=100.0, time=_t(1)) == pytest.approx(25.0)

    def test_large_negative_error_clamped_to_negative_limit(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        assert axis.update(desired_value=-100.0, time=_t(1)) == pytest.approx(-25.0)

    def test_small_error_below_clamp_passes_through(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        # target = 0 + 0.5*4 = 2, well within ±25.
        assert axis.update(desired_value=4.0, time=_t(1)) == pytest.approx(2.0)


class TestRedundantPublishGuard:
    """``command_tolerance`` keeps the EPOS bus quiet when the motor-frame
    target hasn't moved by more than the threshold since the last publish.
    """

    def test_first_post_prime_emits_unconditionally(self):
        # last_commanded_pos starts at None, so the deadband is bypassed
        # for the first real call. Without this, an axis whose first
        # target happens to equal 0 (or a clamped value) would never
        # publish.
        axis = make_axis(Kp=0.5, command_tolerance=100.0)
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=10.0, time=_t(1))
        assert cmd == pytest.approx(5.0)

    def test_target_change_within_tolerance_suppressed(self):
        axis = make_axis(Kp=0.5, command_tolerance=1.0)
        _prime(axis, current_pos=0.0)

        # Tick 1 emits; latches last_commanded_pos = 5.0.
        cmd1 = axis.update(desired_value=10.0, time=_t(1))
        # Tick 2: target unchanged, |5 - 5| = 0 ≤ 1.0 → suppressed.
        cmd2 = axis.update(desired_value=10.0, time=_t(2))

        assert cmd1 == pytest.approx(5.0)
        assert cmd2 is None

    def test_target_change_above_tolerance_emits(self):
        axis = make_axis(Kp=0.5, command_tolerance=0.1)
        _prime(axis, current_pos=0.0)

        cmd1 = axis.update(desired_value=10.0, time=_t(1))
        # current_pos still 0, desired now 20 → target = 0 + 0.5*20 = 10.
        # |10 - 5| = 5 > 0.1 → emits.
        cmd2 = axis.update(desired_value=20.0, time=_t(2))

        assert cmd1 == pytest.approx(5.0)
        assert cmd2 == pytest.approx(10.0)

    def test_quiescent_at_clamp(self):
        # Saturated against the output clamp: subsequent ticks keep
        # producing the same clamped target, so the guard suppresses
        # everything after the first emission.
        axis = AxisController(
            name="roll",
            Kp=0.5,
            command_tolerance=0.0,
            output_limits=(-25.0, 25.0),
        )
        _prime(axis, current_pos=0.0)
        cmd1 = axis.update(desired_value=100.0, time=_t(1))
        cmd2 = axis.update(desired_value=100.0, time=_t(2))
        cmd3 = axis.update(desired_value=100.0, time=_t(3))

        assert cmd1 == pytest.approx(25.0)
        assert cmd2 is None
        assert cmd3 is None


class TestPIDPathWired:
    """Smoke that the underlying PIDController is actually feeding the
    axis. Not re-testing PIDController itself — test_pid_controller.py
    covers that. With Kp=0 we isolate the integral contribution and watch
    the output grow under persistent error.
    """

    def test_integral_accumulates_under_persistent_error(self):
        axis = AxisController(
            name="roll",
            Kp=0.0,
            Ki=0.5,
            Kd=0.0,
            command_tolerance=0.0,
            integral_limits=(-1000.0, 1000.0),
            output_limits=(-1000.0, 1000.0),
        )
        _prime(axis, current_pos=0.0)

        cmd1 = axis.update(desired_value=10.0, time=_t(1))
        cmd2 = axis.update(desired_value=10.0, time=_t(2))
        cmd3 = axis.update(desired_value=10.0, time=_t(3))

        assert cmd1 is not None and cmd2 is not None and cmd3 is not None
        assert cmd2 > cmd1
        assert cmd3 > cmd2
