"""Tier 2 in-process rclpy tests for DepthControlNode.

Black-box behavioral tests: drive the node via published POSITION_TARGET
(Pose, depth in `position.z`) / EXTERNAL_PRESSURE messages and assert on
what it publishes on BCU_RPM.

The node has a 10 Hz control timer, so most tests need ~0.3-0.5s of spin
time to see one or more emissions.
"""

import pytest

from py_pkg.physics import pressure_to_depth
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM, BCU_MOTOR_MIN_RPM


# A pressure that yields current_depth = 0 (atmospheric).
PRESSURE_AT_SURFACE_PA = 101_325

# Pressure giving current_depth ~= +50 m (Z-positive-down). pressure_to_depth
# returns (pa - atm) / (rho * g), so 50 m * 1025 kg/m^3 * 9.806 m/s^2 ≈
# 502_538 Pa above atmospheric.
PRESSURE_FOR_DEEP_PA = 101_325 + 502_538


class TestWiringSmoke:
    """Construction + topic graph wiring."""

    def test_node_constructs(self, depth_node_harness):
        assert depth_node_harness.node is not None

    def test_target_pose_subscription_present(self, depth_node_harness):
        names = [
            sub.topic_name
            for sub in depth_node_harness.node.subscriptions
        ]
        assert "/position/target" in names

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

    def test_bcu_valves_publisher_present(self, depth_node_harness):
        names = [
            pub.topic_name
            for pub in depth_node_harness.node.publishers
        ]
        assert "/bcu/valves" in names


class TestTargetDepthIngress:
    """POSITION_TARGET.position.z flows into node.target_depth and the inner control system."""

    def test_target_depth_updates_node_state(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(42.0)
        h.spin_until(lambda: h.node.target_depth == 42.0, timeout=1.0)
        assert h.node.target_depth == pytest.approx(42.0)

    def test_target_depth_propagates_to_control_system(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(15.5)
        h.spin_until(
            lambda: h.node.control_system.target_depth == 15.5, timeout=1.0
        )
        assert h.node.control_system.target_depth == pytest.approx(15.5)


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
        h.publish_target_depth(30.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 1, timeout=1.5)
        assert len(h.received_rpm) >= 1

    def test_emits_multiple_at_10hz(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(30.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        # 0.6s @ 10 Hz should give ~6 emissions; assert at least 3 to leave margin
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3


class TestSignConvention:
    """Pump wiring inverts q→rpm: positive q is published as a negative Int32.

    Z-positive-down throughout. Walkthrough (target deeper than current):
      target=+70, current=0  → calc_acc returns q > 0 (fill bladder, sink)
      → q_to_rpm preserves sign → motor_rpm > 0
      → msg.data = int(-1 * motor_rpm) → published RPM is NEGATIVE.

    Inverted case (target shallower than current):
      target=0, current=+50  → q < 0 → motor_rpm < 0
      → published RPM is POSITIVE.

    Note: the cascaded PID's first tick can emit 0 before its derivative /
    integral state has settled, so we assert on the last emission after the
    cascade has had several ticks to reach steady state.
    """

    def test_target_deeper_publishes_negative_rpm(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(70.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)  # current_depth ≈ 0
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        # Once the cascade has settled, the steady command must be negative.
        last = h.received_rpm[-1]
        assert last < 0, f"expected negative steady-state rpm, got {h.received_rpm}"
        assert abs(last) <= BCU_MOTOR_MAX_RPM

    def test_target_shallower_publishes_positive_rpm(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(0.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)  # current ≈ +50
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
        h.publish_target_depth(1000.0)
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


class TestValveEmission:
    """Node publishes BCU_VALVES alongside BCU_RPM at 10 Hz."""

    def test_emits_within_one_second(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_depth(30.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_valves) >= 1, timeout=1.5)
        assert len(h.received_valves) >= 1

    def test_valves_track_rpm_emissions(self, depth_node_harness):
        # Per-callback the node publishes RPM then valves; counts should
        # stay in lockstep within a sample of the timer.
        h = depth_node_harness
        h.publish_target_depth(30.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        # Allow at most one sample of skew (RPM may have been delivered
        # without the valves message yet, but not the other way around).
        assert abs(len(h.received_rpm) - len(h.received_valves)) <= 1


class TestValveSelection:
    """End-to-end: depth + descent intent shape the BCU_VALVES bitmask.

    Bitmask layout: bit0 = valve 1 (pump path), bit1 = valve 2 (passive
    vent). The cascaded PID needs several ticks to settle, so assertions
    use the last emission after spin.
    """

    def test_shallow_descend_uses_valve1(self, depth_node_harness):
        # At surface with target deep → pump active driving descent
        # → valve1=1, valve2=0 → bitmask = 0b01.
        h = depth_node_harness
        h.publish_target_depth(70.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert h.received_valves[-1] == 0b01, (
            f"expected pump-via-valve1 (0b01), got history {h.received_valves}"
        )

    def test_deep_descend_passively_vents(self, depth_node_harness):
        # Below threshold with descent intent → pump forced off and
        # valve 2 vents → bitmask = 0b10, RPM = 0.
        h = depth_node_harness
        h.publish_target_depth(100.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)  # current ≈ +50
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert h.received_valves[-1] == 0b10, (
            f"expected passive-vent (0b10), got history {h.received_valves}"
        )
        assert h.received_rpm[-1] == 0, (
            f"deep-descend must zero the pump, got rpm history {h.received_rpm}"
        )

    def test_deep_ascend_uses_valve1(self, depth_node_harness):
        # Below threshold but ascending → pump active, valve 1 carries
        # flow, valve 2 closed → bitmask = 0b01.
        h = depth_node_harness
        h.publish_target_depth(0.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)  # current ≈ +50
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert h.received_valves[-1] == 0b01, (
            f"expected pump-via-valve1 (0b01), got history {h.received_valves}"
        )

    def test_quiescent_closes_both_valves(self, depth_node_harness):
        # Target == current at the surface → q ≈ 0, pump idle → both
        # valves closed → bitmask = 0b00.
        h = depth_node_harness
        h.publish_target_depth(0.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_valves) >= 3
        # All emissions must be 0; a stray valve open here would mean the
        # node opened a passive vent without a descent intent.
        assert all(v == 0 for v in h.received_valves), (
            f"expected all-closed history, got {h.received_valves}"
        )

    def test_valve2_implies_zero_rpm(self, depth_node_harness):
        # Cross-check the invariant from select_pump_and_valves: any
        # sample where valve 2 is open must have a zero pump command.
        h = depth_node_harness
        h.publish_target_depth(100.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)
        h.spin_for(0.6)
        # Pair-wise alignment: zip stops at the shorter list, which
        # absorbs at-most-one-sample skew between the two topics.
        for rpm, valves in zip(h.received_rpm, h.received_valves):
            if valves & 0b10:
                assert rpm == 0, (
                    f"valve2 open with non-zero rpm={rpm} "
                    f"(rpm history {h.received_rpm}, "
                    f"valves history {h.received_valves})"
                )


def _seen_any(harness) -> bool:
    """Helper for spin_until: stop spinning once the timer has emitted at least once."""
    return len(harness.received_rpm) >= 1
