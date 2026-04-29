"""Tier 2 in-process rclpy tests for ACUControlNode.

Black-box: drive the node via published POSITION_TARGET / POSITION_ESTIMATION
Pose messages and assert on what it publishes on ACU_PITCH_STEPS /
ACU_ROLL_STEPS.

The node has a 10 Hz control timer. Unlike DepthControlNode, the ACU only
emits when its per-axis controller's state machine + command-deadband return
non-None — pure quiescence (target == current) produces no traffic.
"""

import pytest

from py_pkg.robot_specs import (
    ACU_PITCH_OUTPUT_LIMIT_M,
    ACU_PITCH_STEPS_PER_M,
    ACU_ROLL_MAX_ANGLE_DEG,
    ACU_ROLL_STEPS_PER_DEG,
)


PITCH_MAX_STEPS = int(round(ACU_PITCH_OUTPUT_LIMIT_M * ACU_PITCH_STEPS_PER_M))
ROLL_MAX_STEPS = int(round(ACU_ROLL_MAX_ANGLE_DEG * ACU_ROLL_STEPS_PER_DEG))


class TestWiringSmoke:
    """Construction + topic graph wiring."""

    def test_node_constructs(self, acu_node_harness):
        assert acu_node_harness.node is not None

    def test_position_target_subscription_present(self, acu_node_harness):
        names = [sub.topic_name for sub in acu_node_harness.node.subscriptions]
        assert "/position/target" in names

    def test_position_estimation_subscription_present(self, acu_node_harness):
        names = [sub.topic_name for sub in acu_node_harness.node.subscriptions]
        assert "/position/estimation" in names

    def test_pitch_steps_publisher_present(self, acu_node_harness):
        names = [pub.topic_name for pub in acu_node_harness.node.publishers]
        assert "/acu/pitch/steps" in names

    def test_roll_steps_publisher_present(self, acu_node_harness):
        names = [pub.topic_name for pub in acu_node_harness.node.publishers]
        assert "/acu/roll/steps" in names


class TestTargetIngress:
    """POSITION_TARGET messages flow into target_roll_deg / target_pitch_deg."""

    def test_target_pose_updates_roll_state(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=15.0, pitch_deg=0.0)
        h.spin_until(
            lambda: h.node.target_roll_deg == pytest.approx(15.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.target_roll_deg == pytest.approx(15.0, abs=1e-4)
        assert h.node.target_pitch_deg == pytest.approx(0.0, abs=1e-4)

    def test_target_pose_updates_pitch_state(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=12.0)
        h.spin_until(
            lambda: h.node.target_pitch_deg == pytest.approx(12.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.target_pitch_deg == pytest.approx(12.0, abs=1e-4)
        assert h.node.target_roll_deg == pytest.approx(0.0, abs=1e-4)

    def test_target_pose_updates_both_axes(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=-7.5, pitch_deg=20.0)
        h.spin_until(
            lambda: (
                h.node.target_roll_deg == pytest.approx(-7.5, abs=1e-4)
                and h.node.target_pitch_deg == pytest.approx(20.0, abs=1e-4)
            ),
            timeout=1.0,
        )
        assert h.node.target_roll_deg == pytest.approx(-7.5, abs=1e-4)
        assert h.node.target_pitch_deg == pytest.approx(20.0, abs=1e-4)


class TestEstimationIngress:
    """POSITION_ESTIMATION messages flow into current_roll_deg / current_pitch_deg."""

    def test_estimation_pose_updates_current_roll(self, acu_node_harness):
        h = acu_node_harness
        h.publish_current_attitude(roll_deg=8.0, pitch_deg=0.0)
        h.spin_until(
            lambda: h.node.current_roll_deg == pytest.approx(8.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.current_roll_deg == pytest.approx(8.0, abs=1e-4)

    def test_estimation_pose_updates_current_pitch(self, acu_node_harness):
        h = acu_node_harness
        h.publish_current_attitude(roll_deg=0.0, pitch_deg=-3.5)
        h.spin_until(
            lambda: h.node.current_pitch_deg == pytest.approx(-3.5, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.current_pitch_deg == pytest.approx(-3.5, abs=1e-4)


class TestTimerEmits:
    """Node publishes step commands once a non-trivial error is set up."""

    def test_emits_roll_steps_within_one_second(self, acu_node_harness):
        h = acu_node_harness
        # current pose stays at 0; target roll 20° creates error > position_tolerance
        # so the roll axis goes SHIFTING and publishes.
        h.publish_target_attitude(roll_deg=20.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_steps) >= 1, timeout=1.5)
        assert len(h.received_roll_steps) >= 1

    def test_emits_pitch_steps_within_one_second(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=20.0)
        h.spin_until(lambda: len(h.received_pitch_steps) >= 1, timeout=1.5)
        assert len(h.received_pitch_steps) >= 1

    def test_no_emission_when_target_matches_current(self, acu_node_harness):
        # Both target and current at 0; |error| stays inside position_tolerance.
        # Axes never leave STEADY, so neither publisher should emit.
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=0.0)
        h.publish_current_attitude(roll_deg=0.0, pitch_deg=0.0)
        h.spin_for(0.6)
        assert h.received_roll_steps == []
        assert h.received_pitch_steps == []


class TestRollSignConvention:
    """Positive desired roll (relative to current=0) → positive ACU_ROLL_STEPS.

    Roll axis: target_pos = current + Kp*(desired - current); steps =
    target_pos * ACU_ROLL_STEPS_PER_DEG. No sign inversion in the publish path.
    """

    def test_positive_target_publishes_positive_steps(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=20.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_steps) >= 1, timeout=1.5)
        first = h.received_roll_steps[0]
        assert first > 0, f"expected positive roll steps, got {h.received_roll_steps}"
        assert abs(first) <= ROLL_MAX_STEPS

    def test_negative_target_publishes_negative_steps(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=-20.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_steps) >= 1, timeout=1.5)
        first = h.received_roll_steps[0]
        assert first < 0, f"expected negative roll steps, got {h.received_roll_steps}"
        assert abs(first) <= ROLL_MAX_STEPS


class TestPitchSignConvention:
    """Positive desired pitch → positive ACU_PITCH_STEPS; negative → negative."""

    def test_positive_target_publishes_positive_steps(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=20.0)
        h.spin_until(lambda: len(h.received_pitch_steps) >= 1, timeout=1.5)
        first = h.received_pitch_steps[0]
        assert first > 0, f"expected positive pitch steps, got {h.received_pitch_steps}"
        assert abs(first) <= PITCH_MAX_STEPS

    def test_negative_target_publishes_negative_steps(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=-20.0)
        h.spin_until(lambda: len(h.received_pitch_steps) >= 1, timeout=1.5)
        first = h.received_pitch_steps[0]
        assert first < 0, f"expected negative pitch steps, got {h.received_pitch_steps}"
        assert abs(first) <= PITCH_MAX_STEPS


class TestSaturation:
    """Large errors must clamp to ±output_limit * steps_per_unit."""

    def test_roll_saturates_at_max_steps(self, acu_node_harness):
        # Desired roll 100° far exceeds ACU_ROLL_MAX_ANGLE_DEG (25°) but stays
        # within (-180°, 180°) so the quaternion -> roll/pitch round-trip is
        # unambiguous (atan2 wraps inputs outside that range). Every emission
        # must be |steps| <= ROLL_MAX_STEPS.
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=100.0, pitch_deg=0.0)
        h.spin_for(0.6)
        assert len(h.received_roll_steps) >= 1
        for s in h.received_roll_steps:
            assert abs(s) <= ROLL_MAX_STEPS, (
                f"published roll steps {s} exceeds ROLL_MAX_STEPS={ROLL_MAX_STEPS}"
            )
        # Steady-state command should hit the positive bound.
        assert h.received_roll_steps[-1] == ROLL_MAX_STEPS

    def test_roll_saturates_at_min_steps_for_negative_target(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=-100.0, pitch_deg=0.0)
        h.spin_for(0.6)
        assert len(h.received_roll_steps) >= 1
        for s in h.received_roll_steps:
            assert abs(s) <= ROLL_MAX_STEPS
        assert h.received_roll_steps[-1] == -ROLL_MAX_STEPS

    def test_pitch_saturates_at_max_steps(self, acu_node_harness):
        # Desired pitch 60° far exceeds the mass-shifter output limit (0.07);
        # 60° stays well below the asin gimbal-lock pole at ±90°.
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=60.0)
        h.spin_for(0.6)
        assert len(h.received_pitch_steps) >= 1
        for s in h.received_pitch_steps:
            assert abs(s) <= PITCH_MAX_STEPS, (
                f"published pitch steps {s} exceeds PITCH_MAX_STEPS={PITCH_MAX_STEPS}"
            )
        assert h.received_pitch_steps[-1] == PITCH_MAX_STEPS


class TestPositionToleranceDeadband:
    """Errors within ±position_tolerance keep the axis STEADY → no emission.

    Roll/pitch position_tolerance is 1.0° in the shipped configs. With a
    target offset of 0.5° from current, the axis must stay STEADY and the
    publisher must remain silent.
    """

    def test_roll_within_tolerance_silent(self, acu_node_harness):
        h = acu_node_harness
        # current=0, target=0.5° → |error|=0.5 <= 1.0 → STEADY, no emit.
        h.publish_target_attitude(roll_deg=0.5, pitch_deg=0.0)
        h.spin_for(0.6)
        assert h.received_roll_steps == []

    def test_pitch_within_tolerance_silent(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=0.5)
        h.spin_for(0.6)
        assert h.received_pitch_steps == []


class TestQuiescenceAfterShift:
    """Once the axis reaches its clamped target and the command stops moving,
    publication ceases (command_tolerance gates further emissions).

    This pins the eventual-quiescence behaviour: a saturating error produces
    a few step commands that converge to the clamp, then the publisher goes
    quiet — it does not flood the bus at 10 Hz forever.
    """

    def test_roll_saturated_settles_to_no_new_emissions(self, acu_node_harness):
        h = acu_node_harness
        # 100° stays within the unambiguous quaternion->Euler range; the
        # roll axis still saturates at ACU_ROLL_MAX_ANGLE_DEG (25°).
        h.publish_target_attitude(roll_deg=100.0, pitch_deg=0.0)
        # Burst phase: collect a few emissions while target is shifting.
        h.spin_until(lambda: len(h.received_roll_steps) >= 2, timeout=1.5)
        early_count = len(h.received_roll_steps)
        # Quiescence phase: spin further; with current still 0 and target
        # already at clamp, |target_pos - last_commanded| stays 0 → no new
        # emissions. Allow at most 1 trailing message for timing slack.
        h.spin_for(0.6)
        late_count = len(h.received_roll_steps)
        assert late_count - early_count <= 1, (
            f"expected quiescence after saturation, got {h.received_roll_steps}"
        )


