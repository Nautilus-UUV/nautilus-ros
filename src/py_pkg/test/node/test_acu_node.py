"""Tier 2 tests for ACUControlNode.

Drive POSITION_TARGET / POSITION_ESTIMATION; assert on ACU_PITCH (Int16
mm) and ACU_ROLL (Int16 centidegrees, scale = ACU_ROLL_CDEG_PER_DEG).
The ACU only emits while its state machine is shifting, so target ==
current produces no traffic.
"""

import pytest
from py_pkg.robot_specs import (
    ACU_PITCH_OUTPUT_LIMIT_M,
    ACU_ROLL_CDEG_PER_DEG,
    ACU_ROLL_MAX_ANGLE_DEG,
)

# Saturation bounds in publish units. Mirror ACUControlNode's m -> mm and
# deg -> centidegrees conversions; update if wire units change.
PITCH_MAX_MM = int(round(ACU_PITCH_OUTPUT_LIMIT_M * 1000.0))
ROLL_MAX_CDEG = int(round(ACU_ROLL_MAX_ANGLE_DEG * ACU_ROLL_CDEG_PER_DEG))


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

    def test_pitch_publisher_present(self, acu_node_harness):
        names = [pub.topic_name for pub in acu_node_harness.node.publishers]
        assert "/acu/pitch" in names

    def test_roll_publisher_present(self, acu_node_harness):
        names = [pub.topic_name for pub in acu_node_harness.node.publishers]
        assert "/acu/roll" in names


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
    """Node publishes physical-position commands once a non-trivial error is set up."""

    def test_emits_roll_within_one_second(self, acu_node_harness):
        h = acu_node_harness
        # current pose stays at 0; target roll 20° creates error > position_tolerance
        # so the roll axis goes SHIFTING and publishes.
        h.publish_target_attitude(roll_deg=20.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        assert len(h.received_roll_cdeg) >= 1

    def test_emits_pitch_within_one_second(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=20.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert len(h.received_pitch_mm) >= 1

    def test_no_emission_when_target_matches_current(self, acu_node_harness):
        # Both target and current at 0; |error| stays inside position_tolerance.
        # Axes never leave STEADY, so neither publisher should emit.
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=0.0)
        h.publish_current_attitude(roll_deg=0.0, pitch_deg=0.0)
        h.spin_for(0.6)
        assert h.received_roll_cdeg == []
        assert h.received_pitch_mm == []


class TestRollSignConvention:
    """Positive desired roll (relative to current=0) -> positive ACU_ROLL (cdeg).

    Roll axis: target_pos_deg = current + Kp*(desired - current); published
    value is int(round(target_pos_deg * ACU_ROLL_CDEG_PER_DEG)). No sign
    inversion in the publish path.
    """

    def test_positive_target_publishes_positive_value(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=20.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        first = h.received_roll_cdeg[0]
        assert first > 0, f"expected positive roll cdeg, got {h.received_roll_cdeg}"
        assert abs(first) <= ROLL_MAX_CDEG

    def test_negative_target_publishes_negative_value(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=-20.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        first = h.received_roll_cdeg[0]
        assert first < 0, f"expected negative roll cdeg, got {h.received_roll_cdeg}"
        assert abs(first) <= ROLL_MAX_CDEG


class TestPitchSignConvention:
    """Positive desired pitch -> positive ACU_PITCH (mm); negative -> negative."""

    def test_positive_target_publishes_positive_value(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=20.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        first = h.received_pitch_mm[0]
        assert first > 0, f"expected positive pitch mm, got {h.received_pitch_mm}"
        assert abs(first) <= PITCH_MAX_MM

    def test_negative_target_publishes_negative_value(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=-20.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        first = h.received_pitch_mm[0]
        assert first < 0, f"expected negative pitch mm, got {h.received_pitch_mm}"
        assert abs(first) <= PITCH_MAX_MM


class TestSaturation:
    """Large errors must clamp to the configured output limit (in publish units)."""

    def test_roll_saturates_at_max(self, acu_node_harness):
        # 100° far exceeds the 30° clamp but stays inside (-180°, 180°)
        # so the quaternion->Euler round-trip is unambiguous.
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=100.0, pitch_deg=0.0)
        h.spin_for(0.6)
        assert len(h.received_roll_cdeg) >= 1
        for v in h.received_roll_cdeg:
            assert abs(v) <= ROLL_MAX_CDEG, (
                f"published roll {v} cdeg exceeds ROLL_MAX_CDEG={ROLL_MAX_CDEG}"
            )
        assert h.received_roll_cdeg[-1] == ROLL_MAX_CDEG

    def test_roll_saturates_at_min_for_negative_target(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=-100.0, pitch_deg=0.0)
        h.spin_for(0.6)
        assert len(h.received_roll_cdeg) >= 1
        for v in h.received_roll_cdeg:
            assert abs(v) <= ROLL_MAX_CDEG
        assert h.received_roll_cdeg[-1] == -ROLL_MAX_CDEG

    def test_pitch_saturates_at_max(self, acu_node_harness):
        # 60° far exceeds the 0.07 m mass-shifter limit, well below the
        # asin gimbal-lock pole at ±90°.
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=60.0)
        h.spin_for(0.6)
        assert len(h.received_pitch_mm) >= 1
        for v in h.received_pitch_mm:
            assert abs(v) <= PITCH_MAX_MM, (
                f"published pitch {v} mm exceeds PITCH_MAX_MM={PITCH_MAX_MM}"
            )
        assert h.received_pitch_mm[-1] == PITCH_MAX_MM


class TestPositionToleranceDeadband:
    """Errors within ±position_tolerance (1° in the shipped configs) keep
    the axis STEADY and the publisher silent."""

    def test_roll_within_tolerance_silent(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.5, pitch_deg=0.0)
        h.spin_for(0.6)
        assert h.received_roll_cdeg == []

    def test_pitch_within_tolerance_silent(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=0.0, pitch_deg=0.5)
        h.spin_for(0.6)
        assert h.received_pitch_mm == []


class TestQuiescenceAfterShift:
    """Once a saturating axis settles to its clamp, command_tolerance
    silences further emissions — the publisher does not flood at 10 Hz."""

    def test_roll_saturated_settles_to_no_new_emissions(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target_attitude(roll_deg=100.0, pitch_deg=0.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 2, timeout=1.5)
        early_count = len(h.received_roll_cdeg)
        # After settling at the clamp, no new emissions; allow 1 for slack.
        h.spin_for(0.6)
        late_count = len(h.received_roll_cdeg)
        assert late_count - early_count <= 1, (
            f"expected quiescence after saturation, got {h.received_roll_cdeg}"
        )
