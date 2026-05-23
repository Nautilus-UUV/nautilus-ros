"""Tier 2 in-process rclpy tests for DepthControlNode.

Black-box behavioral tests: drive the node via published POSITION_TARGET
(Pose, gauge pressure in `position.z`, Pa) / EXTERNAL_PRESSURE messages
and assert on what it publishes on BCU_RPM.

The node has a 10 Hz control timer, so most tests need ~0.3-0.5s of spin
time to see one or more emissions.
"""

import pytest

from py_pkg.physics import (
    ATMOSPHERIC_PRESSURE_PA,
    depth_to_pressure_pa,
    gauge_pressure_pa,
)
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM


# Absolute Pa at the surface (atmospheric); yields current_pressure_pa = 0.
PRESSURE_AT_SURFACE_PA = int(ATMOSPHERIC_PRESSURE_PA)

# Absolute Pa for current depth ~+50 m (Z-positive-down).
PRESSURE_FOR_DEEP_PA = int(depth_to_pressure_pa(50.0))

# Gauge-Pa setpoints used by the tests, expressed via depth equivalents
# so the intent ("70 m below the surface", "30 m") stays readable.
TARGET_PA_70M = gauge_pressure_pa(depth_to_pressure_pa(70.0))
TARGET_PA_30M = gauge_pressure_pa(depth_to_pressure_pa(30.0))
TARGET_PA_100M = gauge_pressure_pa(depth_to_pressure_pa(100.0))
TARGET_PA_DEEP_HUGE = gauge_pressure_pa(depth_to_pressure_pa(1000.0))


class TestManualOverride:
    """Engaging CONTROL_MANUAL_OVERRIDE hands the BCU off at a safe stop --
    one 0-RPM + valves-closed command -- then depth_node goes silent so the
    manual driver (bcu_debug) owns /bcu/rpm + /bcu/valves without the loop
    racing it."""

    def test_engage_commands_safe_stop_then_pauses(self, depth_node_harness):
        h = depth_node_harness
        h.publish_manual_override(True)
        h.spin_until(lambda: h.node._manual_override is True, timeout=1.0)
        h.spin_for(0.2)  # let the one-shot safe-stop emission land

        assert h.received_rpm and h.received_rpm[-1] == 0
        assert h.received_valves and h.received_valves[-1] == 0

        # After the safe stop the loop stays silent -- no periodic zero-hold.
        h.received_rpm.clear()
        h.received_valves.clear()
        h.spin_for(0.5)  # 5+ control ticks at 10 Hz
        assert h.received_rpm == [], (
            f"depth_node must stay silent after the safe stop, got {h.received_rpm}"
        )
        assert h.received_valves == []

    def test_engage_mid_pump_commands_safe_stop(self, depth_node_harness):
        # The point of the safe stop: a mission mid-pump must not leave its
        # last RPM running when manual mode takes over.
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        assert h.received_rpm[-1] != 0, "precondition: pump actively commanded"

        h.publish_manual_override(True)
        h.spin_until(lambda: h.node._manual_override is True, timeout=1.0)
        h.spin_for(0.2)
        assert h.received_rpm[-1] == 0, (
            f"engaging manual mid-pump must command 0 RPM, got {h.received_rpm[-5:]}"
        )
        assert h.received_valves[-1] == 0

    def test_publish_resumes_after_override_released(self, depth_node_harness):
        h = depth_node_harness
        h.publish_manual_override(True)
        h.spin_until(lambda: h.node._manual_override is True, timeout=1.0)
        h.spin_for(0.3)

        h.publish_manual_override(False)
        h.spin_until(lambda: h.node._manual_override is False, timeout=1.0)

        baseline = len(h.received_rpm)
        h.spin_for(0.4)
        assert len(h.received_rpm) > baseline, (
            "depth_node must resume the zero-hold publish once override is released"
        )


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


class TestTargetPressureIngress:
    """POSITION_TARGET.position.z (gauge Pa) flows into node.target_pressure_pa
    and the inner control system."""

    def test_target_pressure_updates_node_state(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(42.0)
        h.spin_until(lambda: h.node.target_pressure_pa == 42.0, timeout=1.0)
        assert h.node.target_pressure_pa == pytest.approx(42.0)

    def test_target_pressure_propagates_to_control_system(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(15.5)
        h.spin_until(
            lambda: h.node.control_system.target_pressure_pa == 15.5, timeout=1.0
        )
        assert h.node.control_system.target_pressure_pa == pytest.approx(15.5)


class TestPressureIngress:
    """EXTERNAL_PRESSURE messages flow through gauge_pressure_pa
    into current_pressure_pa."""

    def test_atmospheric_pressure_yields_zero_gauge(self, depth_node_harness):
        h = depth_node_harness
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(
            lambda: h.node.current_pressure_pa != 0.0 or _seen_any(h),
            timeout=1.0,
        )
        # atmospheric → gauge 0 to within float tolerance
        expected = gauge_pressure_pa(PRESSURE_AT_SURFACE_PA)
        assert h.node.current_pressure_pa == pytest.approx(expected, abs=1e-6)

    def test_overpressure_matches_gauge_pressure_pa(self, depth_node_harness):
        h = depth_node_harness
        pa = PRESSURE_AT_SURFACE_PA + 50_000
        h.publish_external_pressure(pa)
        expected = gauge_pressure_pa(pa)
        h.spin_until(
            lambda: h.node.current_pressure_pa == pytest.approx(expected, abs=1e-6),
            timeout=1.0,
        )
        assert h.node.current_pressure_pa == pytest.approx(expected, abs=1e-6)


class TestTimerEmits:
    """Node publishes on BCU_RPM at 10Hz once it has inputs."""

    def test_emits_within_one_second(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 1, timeout=1.5)
        assert len(h.received_rpm) >= 1

    def test_emits_multiple_at_10hz(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        # 0.6s @ 10 Hz should give ~6 emissions; assert at least 3 to leave margin
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3


class TestSignConvention:
    """Pump wiring inverts q→rpm: positive q is published as a negative Int32.

    Z-positive-down throughout. Walkthrough (target deeper than current):
      target=70m gauge Pa, current=0  → calc_acc returns q > 0 (fill bladder, sink)
      → q_to_rpm preserves sign → motor_rpm > 0
      → msg.data = int(-1 * motor_rpm) → published RPM is NEGATIVE.

    Inverted case (target shallower than current):
      target=0, current=+50m gauge Pa  → q < 0 → motor_rpm < 0
      → published RPM is POSITIVE.

    Note: the cascaded PID's first tick can emit 0 before its derivative /
    integral state has settled, so we assert on the last emission after the
    cascade has had several ticks to reach steady state.
    """

    def test_target_deeper_publishes_negative_rpm(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)  # current_pressure_pa ≈ 0
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        # Once the cascade has settled, the steady command must be negative.
        last = h.received_rpm[-1]
        assert last < 0, f"expected negative steady-state rpm, got {h.received_rpm}"
        assert abs(last) <= BCU_MOTOR_MAX_RPM

    def test_target_shallower_publishes_positive_rpm(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(0.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)  # current ≈ +50 m gauge Pa
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
        h.publish_target_pressure(TARGET_PA_DEEP_HUGE)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert abs(r) <= BCU_MOTOR_MAX_RPM, f"published {r} exceeds max"

    def test_published_rpm_respects_min_deadband(self, depth_node_harness):
        # The pump deadband is on by default (min_rpm=500,
        # min_operating_rpm=1000), so deadband_snap either suppresses a
        # command to 0 or snaps it up to +/-min_operating_rpm — no emission
        # may land inside (0, min_operating_rpm). The exact three-region
        # mapping is proved in test_math_utils.TestDeadbandSnap.
        h = depth_node_harness
        edge = h.node._min_operating_rpm
        h.publish_target_pressure(0.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert r == 0 or abs(r) >= edge, (
                f"published {r} falls inside the (0, {edge}) deadband"
            )


class TestValveEmission:
    """Node publishes BCU_VALVES alongside BCU_RPM at 10 Hz."""

    def test_emits_within_one_second(self, depth_node_harness):
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_valves) >= 1, timeout=1.5)
        assert len(h.received_valves) >= 1

    def test_valves_track_rpm_emissions(self, depth_node_harness):
        # Per-callback the node publishes RPM then valves; counts should
        # stay in lockstep within a sample of the timer.
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        # Allow at most one sample of skew (RPM may have been delivered
        # without the valves message yet, but not the other way around).
        assert abs(len(h.received_rpm) - len(h.received_valves)) <= 1


class TestValveSelection:
    """End-to-end: pressure + descent intent shape the BCU_VALVES bitmask.

    Bitmask layout: bit0 = valve 1 (pump path), bit1 = valve 2 (passive
    vent). The cascaded PID needs several ticks to settle, so assertions
    use the last emission after spin.
    """

    def test_shallow_descend_uses_valve1(self, depth_node_harness):
        # At surface with target deep → pump active driving descent
        # → valve1=1, valve2=0 → bitmask = 0b01.
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert h.received_valves[-1] == 0b01, (
            f"expected pump-via-valve1 (0b01), got history {h.received_valves}"
        )

    def test_deep_descend_passively_vents(self, depth_node_harness):
        # Below threshold with descent intent → pump forced off and
        # valve 2 vents → bitmask = 0b10, RPM = 0.
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_100M)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)  # current ≈ +50 m gauge Pa
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
        h.publish_target_pressure(0.0)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)  # current ≈ +50 m gauge Pa
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert h.received_valves[-1] == 0b01, (
            f"expected pump-via-valve1 (0b01), got history {h.received_valves}"
        )

    def test_quiescent_closes_both_valves(self, depth_node_harness):
        # Target == current at the surface → q ≈ 0, pump idle → both
        # valves closed → bitmask = 0b00.
        h = depth_node_harness
        h.publish_target_pressure(0.0)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_valves) >= 3
        # All emissions must be 0; a stray valve open here would mean the
        # node opened a passive vent without a descent intent.
        assert all(v == 0 for v in h.received_valves), (
            f"expected all-closed history, got {h.received_valves}"
        )

    def test_deep_quiescent_closes_both_valves(self, depth_node_harness):
        # Boundary on the strict `q > 0` in select_pump_and_valves. Deep +
        # target == current settles to q ≈ 0; strict `>` keeps the vent
        # closed, but a `>=` slip — or a cascade sign flip producing a
        # tiny positive q at zero error — would open valve 2 here. The
        # shallow-quiescent test above can't catch this because deep=False
        # short-circuits the q-sign branch entirely.
        h = depth_node_harness
        target_gauge = gauge_pressure_pa(PRESSURE_FOR_DEEP_PA)
        h.publish_target_pressure(target_gauge)
        h.publish_external_pressure(PRESSURE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_valves) >= 6, timeout=1.5)
        # Tail-of-history: ignore transients while target/pressure subs
        # land out of order. After settling, every sample must be both
        # valves closed with the pump idle.
        tail_valves = h.received_valves[-3:]
        tail_rpm = h.received_rpm[-3:]
        assert all(v == 0b00 for v in tail_valves), (
            f"expected steady all-closed valves at deep quiescent, got tail "
            f"{tail_valves} (full history {h.received_valves})"
        )
        assert all(r == 0 for r in tail_rpm), (
            f"expected zero pump rpm at deep quiescent, got tail {tail_rpm} "
            f"(full history {h.received_rpm})"
        )

    def test_valve2_implies_zero_rpm(self, depth_node_harness):
        # Cross-check the invariant from select_pump_and_valves: any
        # sample where valve 2 is open must have a zero pump command.
        h = depth_node_harness
        h.publish_target_pressure(TARGET_PA_100M)
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


class TestControlReset:
    """CONTROL_RESET drops the held target and wipes controller state, so
    depth_node falls back to its no-target zero-RPM / valves-closed hold --
    exactly as it sat at boot before any mission (Do-Nothing mission)."""

    def test_reset_returns_to_zero_hold(self, depth_node_harness):
        h = depth_node_harness
        # Drive a real descent command first.
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_external_pressure(PRESSURE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        assert h.received_rpm[-1] != 0, "precondition: pump actively commanded"

        # Reset -> target back to None, controller state wiped.
        h.publish_reset()
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        assert h.node.target_pressure_pa is None

        # Every subsequent emission is the zero-RPM / valves-closed hold.
        rpm_before = len(h.received_rpm)
        valves_before = len(h.received_valves)
        h.spin_for(0.4)
        post_rpm = h.received_rpm[rpm_before:]
        post_valves = h.received_valves[valves_before:]
        assert len(post_rpm) >= 2, "node must keep emitting the zero-hold"
        assert all(r == 0 for r in post_rpm), f"expected zero-hold, got {post_rpm}"
        assert all(v == 0 for v in post_valves), (
            f"expected valves closed, got {post_valves}"
        )


def _seen_any(harness) -> bool:
    """Helper for spin_until: stop spinning once the timer has emitted at least once."""
    return len(harness.received_rpm) >= 1
