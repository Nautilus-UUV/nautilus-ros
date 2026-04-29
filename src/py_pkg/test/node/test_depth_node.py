"""Tier 2 in-process rclpy tests for DepthControlNode.

Black-box behavioral tests: drive the node via published TARGET_DEPTH /
EXTERNAL_PRESSURE messages and assert on what it publishes on BCU_RPM.

The node has a 10 Hz control timer, so most tests need ~0.3-0.5s of spin
time to see one or more emissions.
"""

import pytest

from py_pkg.physics import pressure_to_depth
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM, BCU_MOTOR_MIN_RPM


# A pressure that yields current_depth = 0 (atmospheric).
PRESSURE_AT_SURFACE_PA = 101_325

# Pressure giving current_depth ~= -50m (sign matches depth_node usage:
# pressure_to_depth returns (pa - atm)/(rho*g), positive when pa > atm).
# To get a strongly-negative current_depth we feed sub-atmospheric pa.
PRESSURE_FOR_DEEP_NEGATIVE_PA = -401_213  # ≈ pressure_to_depth(...) ≈ -50.0


class TestWiringSmoke:
    """Construction + topic graph wiring."""

    def test_node_constructs(self, depth_node_harness):
        assert depth_node_harness.node is not None

    def test_target_depth_subscription_present(self, depth_node_harness):
        names = [
            sub.topic_name
            for sub in depth_node_harness.node.subscriptions
        ]
        assert "/target_depth" in names

    def test_external_pressure_subscription_present(self, depth_node_harness):
        names = [
            sub.topic_name
            for sub in depth_node_harness.node.subscriptions
        ]
        assert "/external/pressure" in names

    def test_bcu_rpm_publisher_present(self, depth_node_harness):
        names = [
            pub.topic_name
            for pub in depth_node_harness.node.publishers
        ]
        assert "/bcu/rpm" in names


class TestTargetDepthIngress:
    """TARGET_DEPTH messages flow into node.target_depth and the inner control system."""

    def test_target_depth_updates_node_state(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(-42.0)
        h.spin_until(lambda: h.node.target_depth == -42.0, timeout=1.0)
        assert h.node.target_depth == pytest.approx(-42.0)

    def test_target_depth_propagates_to_control_system(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(-15.5)
        h.spin_until(
            lambda: h.node.control_system.target_depth == -15.5, timeout=1.0
        )
        assert h.node.control_system.target_depth == pytest.approx(-15.5)


class TestPressureIngress:
    """EXTERNAL_PRESSURE messages flow through pressure_to_depth into current_depth."""

    def test_atmospheric_pressure_yields_zero_depth(self, depth_node_harness):
        h = depth_node_harness
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: h.node.current_depth != 0.0 or _seen_any(h), timeout=1.0)
        # atmospheric → depth 0 to within float tolerance
        expected = pressure_to_depth(PRESSURE_AT_SURFACE_PA)
        assert h.node.current_depth == pytest.approx(expected, abs=1e-6)

    def test_overpressure_matches_pressure_to_depth(self, depth_node_harness):
        h = depth_node_harness
        pa = PRESSURE_AT_SURFACE_PA + 50_000
        h.publish_external_pressure(pa)
        expected = pressure_to_depth(pa)
        h.spin_until(
            lambda: h.node.current_depth == pytest.approx(expected, abs=1e-6),
            timeout=1.0,
        )
        assert h.node.current_depth == pytest.approx(expected, abs=1e-6)


class TestTimerEmits:
    """Node publishes on BCU_RPM at 10Hz once it has inputs."""

    def test_emits_within_one_second(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(-30.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 1, timeout=1.5)
        assert len(h.received_rpm) >= 1

    def test_emits_multiple_at_10hz(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(-30.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        # 0.6s @ 10 Hz should give ~6 emissions; assert at least 3 to leave margin
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3


class TestSignConvention:
    """Pump wiring inverts q→rpm: positive q is published as a negative Int32.

    Walkthrough (target deeper than current):
      target=-70, current=0  → calc_acc returns q > 0 (fill bladder, sink)
      → q_to_rpm preserves sign → motor_rpm > 0
      → msg.data = int(-1 * motor_rpm) → published RPM is NEGATIVE.

    Inverted case (target shallower than current):
      target=0, current=-50  → q < 0 → motor_rpm < 0
      → published RPM is POSITIVE.

    Note: the cascaded PID's first tick can emit 0 before its derivative /
    integral state has settled, so we assert on the last emission after the
    cascade has had several ticks to reach steady state.
    """

    def test_target_deeper_publishes_negative_rpm(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(-70.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)  # current_depth ≈ 0
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        # Once the cascade has settled, the steady command must be negative.
        last = h.received_rpm[-1]
        assert last < 0, f"expected negative steady-state rpm, got {h.received_rpm}"
        assert abs(last) <= BCU_MOTOR_MAX_RPM

    def test_target_shallower_publishes_positive_rpm(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(0.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_NEGATIVE_PA)  # current ≈ -50
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        last = h.received_rpm[-1]
        assert last > 0, f"expected positive steady-state rpm, got {h.received_rpm}"
        assert abs(last) <= BCU_MOTOR_MAX_RPM


class TestClamping:
    """|published_rpm| must not exceed BCU_MOTOR_MAX_RPM regardless of input."""

    def test_large_error_clamped_to_max(self, depth_node_harness):
        h = depth_node_harness
        # Aggressive setpoint: target very deep, currently at surface — drives
        # the cascade into saturation.
        h.publish_target_depth(-1000.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert abs(r) <= BCU_MOTOR_MAX_RPM, f"published {r} exceeds max"

    def test_published_rpm_respects_min_deadband(self, depth_node_harness):
        # depth_node zeros out commands whose magnitude is below min_rpm (the
        # pump can't run reliably below that), so every emission must be
        # either 0 or have |rpm| >= min_rpm — no values inside (0, min_rpm).
        h = depth_node_harness
        h.publish_target_depth(0.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert r == 0 or abs(r) >= BCU_MOTOR_MIN_RPM, (
                f"published {r} falls inside the (0, {BCU_MOTOR_MIN_RPM}) deadband"
            )


def _seen_any(harness) -> bool:
    """Helper for spin_until: stop spinning once the timer has emitted at least once."""
    return len(harness.received_rpm) >= 1
