"""Tier 1 unit tests for the ACU per-axis controller.

Covers AxisController only (the P-controller + state-machine + deadband).
ACUController.update is not tested here because it routes through
ACUController.uuv_to_motor, which calls SimMath.clamp_mag with 3 args
(it accepts 2) — a separate bug to fix before integration tests.
"""

import pytest
from py_pkg.pid.acu_control_system import AxisController


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
        axis.update_sensor(0.0)
        # error = 10, correction = 5, new_target = 5
        assert axis.compute_control(10.0) == pytest.approx(5.0)

    def test_zero_error_no_movement(self):
        axis = make_axis(Kp=0.5)
        axis.update_sensor(5.0)
        assert axis.compute_control(5.0) == pytest.approx(5.0)

    def test_negative_error_steps_backward(self):
        axis = make_axis(Kp=0.5)
        axis.update_sensor(10.0)
        # error = -10, correction = -5, new_target = 5
        assert axis.compute_control(0.0) == pytest.approx(5.0)

    def test_kp_one_jumps_to_target(self):
        axis = make_axis(Kp=1.0)
        axis.update_sensor(0.0)
        assert axis.compute_control(10.0) == pytest.approx(10.0)


class TestStateMachine:
    """STEADY → SHIFTING when |error| > position_tolerance, back when within."""

    def test_starts_steady(self):
        axis = make_axis()
        assert axis.state == AxisController.State.STEADY

    def test_steady_stays_steady_when_error_small(self):
        axis = make_axis(position_tolerance=1.0)
        axis.update_sensor(0.0)
        axis.update(desired_value=0.5)  # |error| = 0.5 < 1.0
        assert axis.state == AxisController.State.STEADY

    def test_steady_transitions_to_shifting_on_large_error(self):
        axis = make_axis(position_tolerance=1.0)
        axis.update_sensor(0.0)
        cmd = axis.update(desired_value=10.0)  # |error| = 10 > 1
        assert axis.state == AxisController.State.SHIFTING
        assert cmd is not None

    def test_shifting_returns_to_steady_when_error_drops(self):
        axis = make_axis(position_tolerance=1.0)
        axis.update_sensor(0.0)
        axis.update(desired_value=10.0)  # → SHIFTING
        assert axis.state == AxisController.State.SHIFTING

        axis.update_sensor(9.5)  # |error| = 0.5, within tolerance
        axis.update(desired_value=10.0)
        assert axis.state == AxisController.State.STEADY


class TestCommandDeadband:
    """A new motor command is only issued when the target moves more than command_tolerance.

    Important quirk: the STEADY→SHIFTING transition sets last_commanded_pos to
    current_pos (not target_pos), so the deadband only engages once the
    bottom-path executes and synchronises last_commanded_pos to target_pos.
    """

    def test_first_large_error_emits_command(self):
        axis = make_axis(position_tolerance=1.0, command_tolerance=0.5)
        axis.update_sensor(0.0)
        cmd = axis.update(desired_value=10.0)
        assert cmd is not None

    def test_shifting_emits_until_target_stops_moving(self):
        axis = make_axis(Kp=0.5, position_tolerance=1.0, command_tolerance=0.5)
        axis.update_sensor(0.0)

        # Call 1: STEADY→SHIFTING, returns target_pos=5, last_commanded=0
        cmd1 = axis.update(desired_value=10.0)
        # Call 2: SHIFTING, target still 5 (current=0, desired=10 unchanged),
        #         bottom path: |5-0|=5 > 0.5, emits and sets last_commanded=5
        cmd2 = axis.update(desired_value=10.0)
        # Call 3: SHIFTING, target still 5, |5-5|=0 ≤ 0.5, returns None
        cmd3 = axis.update(desired_value=10.0)

        assert cmd1 == pytest.approx(5.0)
        assert cmd2 == pytest.approx(5.0)
        assert cmd3 is None

    def test_steady_small_target_change_below_deadband_suppressed(self):
        # Stay in STEADY (error within position_tolerance), so we exercise the
        # bottom path directly without going through the SHIFTING quirk.
        axis = make_axis(Kp=0.5, position_tolerance=2.0, command_tolerance=1.0)
        axis.update_sensor(0.0)
        # error = 1 < position_tolerance, target = 0 + 0.5*1 = 0.5,
        # |0.5 - 0| = 0.5 < command_tolerance=1.0 → None
        cmd = axis.update(desired_value=1.0)
        assert cmd is None
        assert axis.state == AxisController.State.STEADY

    def test_steady_target_change_above_deadband_emits(self):
        axis = make_axis(Kp=0.5, position_tolerance=2.0, command_tolerance=0.1)
        axis.update_sensor(0.0)
        # error = 1, target = 0.5, |0.5-0| = 0.5 > 0.1 → emits 0.5
        cmd = axis.update(desired_value=1.0)
        assert cmd == pytest.approx(0.5)
