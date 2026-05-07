"""Tier 1 unit tests for the ACU per-axis controllers.

Covers AxisController (roll: degrees-in / degrees-out, slewed) and
MassShifterController (pitch: degrees-in / metres-out, error-direct).
Both: PIDController-backed correction + state-machine + deadband +
motor-frame output clamp.

Time is supplied explicitly to every step. Tests prime each axis at
t=0.0 (PIDController returns 0 on its first call to initialise prev_time)
and assert behaviour at t>=0.1, the same cadence the 10 Hz node uses.
"""

import pytest
from py_pkg.pid.acu_axis_controller import AxisController, MassShifterController


# Prime the PID at t=0 then step at fixed dt; the node sees the same
# pattern (first 10 Hz tick after startup returns 0, subsequent ticks
# carry real PID output).
PRIME_TIME = 0.0
DT = 0.1


def _t(step: int) -> float:
    """Test clock: 10 Hz monotonic, t=0 reserved for the prime call."""
    return PRIME_TIME + DT * step


def _prime(axis: AxisController, current_pos: float = 0.0) -> None:
    """First update() returns 0 from PIDController; consume it here so
    follow-up assertions exercise the steady-state PID path."""
    axis.update_sensor(current_pos)
    axis.update(desired_value=current_pos, time=PRIME_TIME)


def make_axis(Kp=0.5, position_tolerance=1.0, command_tolerance=0.5):
    return AxisController(
        name="test_axis",
        Kp=Kp,
        position_tolerance=position_tolerance,
        command_tolerance=command_tolerance,
    )


class TestComputeControl:
    """new_target = current + Kp * (desired - current)."""

    def test_proportional_step_toward_target(self):
        axis = make_axis(Kp=0.5)
        _prime(axis, current_pos=0.0)
        # error = 10, correction = 5, new_target = 5
        assert axis.compute_control(10.0, _t(1)) == pytest.approx(5.0)

    def test_zero_error_no_movement(self):
        axis = make_axis(Kp=0.5)
        _prime(axis, current_pos=5.0)
        assert axis.compute_control(5.0, _t(1)) == pytest.approx(5.0)

    def test_negative_error_steps_backward(self):
        axis = make_axis(Kp=0.5)
        _prime(axis, current_pos=10.0)
        # error = -10, correction = -5, new_target = 5
        assert axis.compute_control(0.0, _t(1)) == pytest.approx(5.0)

    def test_kp_one_jumps_to_target(self):
        axis = make_axis(Kp=1.0)
        _prime(axis, current_pos=0.0)
        assert axis.compute_control(10.0, _t(1)) == pytest.approx(10.0)


class TestStateMachine:
    """STEADY → SHIFTING when |error| > position_tolerance, back when within."""

    def test_starts_steady(self):
        axis = make_axis()
        assert axis.state == AxisController.State.STEADY

    def test_steady_stays_steady_when_error_small(self):
        axis = make_axis(position_tolerance=1.0)
        _prime(axis, current_pos=0.0)
        axis.update(desired_value=0.5, time=_t(1))  # |error| = 0.5 < 1.0
        assert axis.state == AxisController.State.STEADY

    def test_steady_transitions_to_shifting_on_large_error(self):
        axis = make_axis(position_tolerance=1.0)
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=10.0, time=_t(1))  # |error| = 10 > 1
        assert axis.state == AxisController.State.SHIFTING
        assert cmd is not None

    def test_shifting_returns_to_steady_when_error_drops(self):
        axis = make_axis(position_tolerance=1.0)
        _prime(axis, current_pos=0.0)
        axis.update(desired_value=10.0, time=_t(1))  # → SHIFTING
        assert axis.state == AxisController.State.SHIFTING

        axis.update_sensor(9.5)  # |error| = 0.5, within tolerance
        axis.update(desired_value=10.0, time=_t(2))
        assert axis.state == AxisController.State.STEADY


class TestCommandDeadband:
    """A new motor command is only issued when the target moves more than command_tolerance.

    Important quirk: the STEADY→SHIFTING transition sets last_commanded_pos to
    current_pos (not target_pos), so the deadband only engages once the
    bottom-path executes and synchronises last_commanded_pos to target_pos.
    """

    def test_first_large_error_emits_command(self):
        axis = make_axis(position_tolerance=1.0, command_tolerance=0.5)
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=10.0, time=_t(1))
        assert cmd is not None

    def test_shifting_emits_until_target_stops_moving(self):
        axis = make_axis(Kp=0.5, position_tolerance=1.0, command_tolerance=0.5)
        _prime(axis, current_pos=0.0)

        # Call 1: STEADY→SHIFTING, returns target_pos=5, last_commanded=0
        cmd1 = axis.update(desired_value=10.0, time=_t(1))
        # Call 2: SHIFTING, target still 5 (current=0, desired=10 unchanged),
        #         bottom path: |5-0|=5 > 0.5, emits and sets last_commanded=5
        cmd2 = axis.update(desired_value=10.0, time=_t(2))
        # Call 3: SHIFTING, target still 5, |5-5|=0 ≤ 0.5, returns None
        cmd3 = axis.update(desired_value=10.0, time=_t(3))

        assert cmd1 == pytest.approx(5.0)
        assert cmd2 == pytest.approx(5.0)
        assert cmd3 is None

    def test_steady_small_target_change_below_deadband_suppressed(self):
        # Stay in STEADY (error within position_tolerance), so we exercise the
        # bottom path directly without going through the SHIFTING quirk.
        axis = make_axis(Kp=0.5, position_tolerance=2.0, command_tolerance=1.0)
        _prime(axis, current_pos=0.0)
        # error = 1 < position_tolerance, target = 0 + 0.5*1 = 0.5,
        # |0.5 - 0| = 0.5 < command_tolerance=1.0 → None
        cmd = axis.update(desired_value=1.0, time=_t(1))
        assert cmd is None
        assert axis.state == AxisController.State.STEADY

    def test_steady_target_change_above_deadband_emits(self):
        axis = make_axis(Kp=0.5, position_tolerance=2.0, command_tolerance=0.1)
        _prime(axis, current_pos=0.0)
        # error = 1, target = 0.5, |0.5-0| = 0.5 > 0.1 → emits 0.5
        cmd = axis.update(desired_value=1.0, time=_t(1))
        assert cmd == pytest.approx(0.5)


class TestPIDPathWired:
    """Smoke test that the PIDController is actually feeding the axis.

    Not re-testing PIDController itself (test_pid_controller.py covers
    that); just confirming Ki>0 produces growing output under persistent
    error. With Kp=0 we isolate the integral contribution.
    """

    def test_integral_accumulates_under_persistent_error(self):
        axis = AxisController(
            name="pitch",
            Kp=0.0,
            Ki=0.5,
            Kd=0.0,
            position_tolerance=0.01,
            command_tolerance=0.0,
            integral_limits=(-1000.0, 1000.0),
            output_limits=(-1000.0, 1000.0),
        )
        _prime(axis, current_pos=0.0)

        # Each successive tick the integral grows -> the correction grows
        # -> target_pos grows.
        cmd1 = axis.update(desired_value=10.0, time=_t(1))
        cmd2 = axis.update(desired_value=10.0, time=_t(2))
        cmd3 = axis.update(desired_value=10.0, time=_t(3))

        assert cmd1 is not None and cmd2 is not None and cmd3 is not None
        assert cmd2 > cmd1
        assert cmd3 > cmd2


class TestMotorFrameOutputClamp:
    """Roll-style clamped-P at the motor-frame output."""

    def _make(self, position_tolerance=1.0):
        return AxisController(
            name="roll",
            Kp=0.5,
            position_tolerance=position_tolerance,
            command_tolerance=0.0,
            output_limits=(-25.0, 25.0),
        )

    def test_large_positive_error_clamped_to_positive_limit(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        # Error = 100°. Unclamped new_target = 0 + 0.5*100 = 50 — clamped to +25.
        cmd = axis.update(desired_value=100.0, time=_t(1))
        assert cmd == pytest.approx(25.0)

    def test_large_negative_error_clamped_to_negative_limit(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=-100.0, time=_t(1))
        assert cmd == pytest.approx(-25.0)

    def test_small_error_below_clamp_passes_through(self):
        axis = self._make(position_tolerance=0.01)
        _prime(axis, current_pos=0.0)
        # Error = 4°, new_target = 0 + 0.5*4 = 2 — well within ±25.
        cmd = axis.update(desired_value=4.0, time=_t(1))
        assert cmd == pytest.approx(2.0)


class TestMassShifterController:
    """Pitch variant: cmd = clamp(Kp_m_per_deg * pitch_error_deg).

    Kp's units are m/deg. Output is the PID correction directly, not
    `current + correction` — the actuator (mass-shifter stroke) is in a
    different frame from the sensor (pitch angle).
    """

    def _make(self, Kp=0.5, output_limits=(-0.07, 0.07), position_tolerance=1.0):
        return MassShifterController(
            name="pitch",
            Kp=Kp,
            position_tolerance=position_tolerance,
            command_tolerance=0.0,
            output_limits=output_limits,
        )

    def test_zero_error_returns_zero_correction(self):
        # Distinguishes from base AxisController, which would return current_pos.
        axis = self._make()
        _prime(axis, current_pos=5.0)
        assert axis.compute_control(5.0, _t(1)) == pytest.approx(0.0)

    def test_positive_error_returns_positive_correction(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        # error = 10°, Kp = 0.5 m/deg → 5 m
        assert axis.compute_control(10.0, _t(1)) == pytest.approx(5.0)

    def test_negative_error_returns_negative_correction(self):
        axis = self._make()
        _prime(axis, current_pos=10.0)
        # error = -10° → -5 m
        assert axis.compute_control(0.0, _t(1)) == pytest.approx(-5.0)

    def test_clamped_at_positive_limit(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        # 0.5 * 35 = 17.5 m; clamp to +0.07.
        cmd = axis.update(desired_value=35.0, time=_t(1))
        assert cmd == pytest.approx(0.07)

    def test_clamped_at_negative_limit(self):
        axis = self._make()
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=-35.0, time=_t(1))
        assert cmd == pytest.approx(-0.07)

    def test_proportional_region_below_clamp(self):
        # Now observable: |err| < 0.14° produces a sub-saturation response.
        # Pre-fix, the bogus 'current_deg + correction_deg' branch saturated
        # the clamp at any |err| > 0.14° regardless of the proportional region.
        axis = self._make(position_tolerance=0.01)
        _prime(axis, current_pos=0.0)
        cmd = axis.update(desired_value=0.1, time=_t(1))
        # 0.5 * 0.1 = 0.05 m, well within ±0.07.
        assert cmd == pytest.approx(0.05)
