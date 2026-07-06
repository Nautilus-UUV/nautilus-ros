"""Tier 2 in-process rclpy tests for BCUNode.

Black-box behavioral tests: drive the node via published POSITION_TARGET
(Pose, gauge pressure in `position.z`, Pa) and POSITION_ESTIMATION (Pose,
gauge depth in `position.z`, Pa) messages and assert on what it publishes
on BCU_RPM.

The depth measurement arrives already gauged on POSITION_ESTIMATION.position.z
-- attitude_node owns the absolute->gauge conversion now -- so the tests feed
gauge Pa straight in via ``publish_depth_gauge`` (no SurfaceReference on the
node anymore). DIVE_INIT still carries the tank endpoints for the output clamp.

The node has a 10 Hz control timer, so most tests need ~0.3-0.5s of spin
time to see one or more emissions.
"""

import pytest

from py_pkg.physics import (
    depth_to_pressure_pa,
    gauge_pressure_pa,
)
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM
from py_pkg.scenarios.spec.rig import PlantSpec


# Gauge Pa at the surface: 0 (Z-positive-down). The depth measurement is
# already gauged, so "at the surface" is simply 0.
GAUGE_AT_SURFACE_PA = 0.0

# Tank endpoints for the output-clamp tests, taken from the sim plant
# (rig.py): empty = drained tank (bladder full), full = tank full of oil.
_PLANT = PlantSpec()
TANK_EMPTY_PA = int(_PLANT.tank_pressure_empty_pa)
TANK_FULL_PA = int(_PLANT.tank_pressure_full_pa)
TANK_MID_PA = (TANK_EMPTY_PA + TANK_FULL_PA) // 2

# Gauge Pa for current depth ~+50 m (Z-positive-down).
GAUGE_FOR_DEEP_PA = gauge_pressure_pa(depth_to_pressure_pa(50.0))

# Gauge-Pa setpoints used by the tests, expressed via depth equivalents
# so the intent ("70 m below the surface", "30 m") stays readable.
TARGET_PA_70M = gauge_pressure_pa(depth_to_pressure_pa(70.0))
TARGET_PA_30M = gauge_pressure_pa(depth_to_pressure_pa(30.0))
TARGET_PA_100M = gauge_pressure_pa(depth_to_pressure_pa(100.0))
TARGET_PA_DEEP_HUGE = gauge_pressure_pa(depth_to_pressure_pa(1000.0))


class TestWiringSmoke:
    """Construction + topic graph wiring."""

    def test_node_constructs(self, bcu_node_harness):
        assert bcu_node_harness.node is not None

    def test_target_pose_subscription_present(self, bcu_node_harness):
        names = [sub.topic_name for sub in bcu_node_harness.node.subscriptions]
        assert "/position/target" in names

    def test_position_estimation_subscription_present(self, bcu_node_harness):
        names = [sub.topic_name for sub in bcu_node_harness.node.subscriptions]
        assert "/position/estimation" in names

    def test_bcu_rpm_publisher_present(self, bcu_node_harness):
        names = [pub.topic_name for pub in bcu_node_harness.node.publishers]
        assert "/bcu/rpm" in names

    def test_bcu_valves_publisher_present(self, bcu_node_harness):
        names = [pub.topic_name for pub in bcu_node_harness.node.publishers]
        assert "/bcu/valves" in names


class TestTargetPressureIngress:
    """POSITION_TARGET.position.z (gauge Pa) flows into node.target_pressure_pa
    and the inner control system."""

    def test_target_pressure_updates_node_state(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(42.0)
        h.spin_until(lambda: h.node.target_pressure_pa == 42.0, timeout=1.0)
        assert h.node.target_pressure_pa == pytest.approx(42.0)

    def test_target_pressure_propagates_to_control_system(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(15.5)
        h.spin_until(
            lambda: h.node.control_system.target_pressure_pa == 15.5, timeout=1.0
        )
        assert h.node.control_system.target_pressure_pa == pytest.approx(15.5)


class TestPressureIngress:
    """POSITION_ESTIMATION.position.z (gauge Pa) flows straight into
    current_pressure_pa -- no conversion on the node now that attitude_node
    owns the absolute->gauge step."""

    def test_surface_gauge_lands_at_zero(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        # The node is silent without a target, and gauge ~0 is
        # indistinguishable from the init value, so just spin to let the
        # estimation callback run, then assert the value landed.
        h.spin_for(0.3)
        assert h.node.current_pressure_pa == pytest.approx(
            GAUGE_AT_SURFACE_PA, abs=1e-6
        )

    def test_gauge_depth_stored_verbatim(self, bcu_node_harness):
        h = bcu_node_harness
        gauge = gauge_pressure_pa(depth_to_pressure_pa(50.0))
        h.publish_depth_gauge(gauge)
        h.spin_until(
            lambda: h.node.current_pressure_pa == pytest.approx(gauge, abs=1e-6),
            timeout=1.0,
        )
        assert h.node.current_pressure_pa == pytest.approx(gauge, abs=1e-6)


class TestTimerEmits:
    """Node publishes on BCU_RPM at 10Hz once it has inputs."""

    def test_emits_within_one_second(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 1, timeout=1.5)
        assert len(h.received_rpm) >= 1

    def test_emits_multiple_at_10hz(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
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

    def test_target_deeper_publishes_negative_rpm(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)  # current_pressure_pa ≈ 0
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        # Once the cascade has settled, the steady command must be negative.
        last = h.received_rpm[-1]
        assert last < 0, f"expected negative steady-state rpm, got {h.received_rpm}"
        assert abs(last) <= BCU_MOTOR_MAX_RPM

    def test_target_shallower_publishes_positive_rpm(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)  # current ≈ +50 m gauge Pa
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        last = h.received_rpm[-1]
        assert last > 0, f"expected positive steady-state rpm, got {h.received_rpm}"
        assert abs(last) <= BCU_MOTOR_MAX_RPM


class TestClamping:
    """|published_rpm| must not exceed BCU_MOTOR_MAX_RPM regardless of input."""

    def test_large_error_clamped_to_max(self, bcu_node_harness):
        h = bcu_node_harness
        # Aggressive setpoint: target very deep, currently at surface — drives
        # the cascade into saturation.
        h.publish_target_pressure(TARGET_PA_DEEP_HUGE)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert abs(r) <= BCU_MOTOR_MAX_RPM, f"published {r} exceeds max"

    def test_published_rpm_respects_min_deadband(self, bcu_node_harness):
        # The pump deadband is on by default (min_rpm=500,
        # min_operating_rpm=1000), so deadband_snap either suppresses a
        # command to 0 or snaps it up to +/-min_operating_rpm — no emission
        # may land inside (0, min_operating_rpm). The exact three-region
        # mapping is proved in test_math_utils.TestDeadbandSnap.
        h = bcu_node_harness
        edge = h.node._min_operating_rpm
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert (
                r == 0 or abs(r) >= edge
            ), f"published {r} falls inside the (0, {edge}) deadband"


class TestValveEmission:
    """Node publishes BCU_VALVES alongside BCU_RPM at 10 Hz."""

    def test_emits_within_one_second(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_valves) >= 1, timeout=1.5)
        assert len(h.received_valves) >= 1

    def test_valves_track_rpm_emissions(self, bcu_node_harness):
        # Per-callback the node publishes RPM then valves; counts should
        # stay in lockstep within a sample of the timer.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        # Allow at most one sample of skew (RPM may have been delivered
        # without the valves message yet, but not the other way around).
        assert abs(len(h.received_rpm) - len(h.received_valves)) <= 1


class TestValveSelection:
    """End-to-end: pressure + descent intent shape the BCU_VALVES bitmask.

    Bitmask layout: bit0 = motor way (operator "valve 2", pump path),
    bit1 = free/bypass way (operator "valve 1", passive vent). The cascaded
    PID needs several ticks to settle, so assertions use the last emission
    after spin.
    """

    def test_shallow_descend_uses_motor_valve(self, bcu_node_harness):
        # At surface with target deep → pump active driving descent
        # → motor=1, free=0 → bitmask = 0b01.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert (
            h.received_valves[-1] == 0b01
        ), f"expected pump-via-motor-valve (0b01), got history {h.received_valves}"

    def test_deep_descend_passively_vents(self, bcu_node_harness):
        # Below threshold with descent intent → pump forced off and
        # the free/bypass way vents → bitmask = 0b10, RPM = 0.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_100M)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)  # current ≈ +50 m gauge Pa
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert (
            h.received_valves[-1] == 0b10
        ), f"expected passive-vent (0b10), got history {h.received_valves}"
        assert (
            h.received_rpm[-1] == 0
        ), f"deep-descend must zero the pump, got rpm history {h.received_rpm}"

    def test_deep_ascend_uses_motor_valve(self, bcu_node_harness):
        # Below threshold but ascending → pump active, the motor way carries
        # flow, the free/bypass way closed → bitmask = 0b01.
        h = bcu_node_harness
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)  # current ≈ +50 m gauge Pa
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert (
            h.received_valves[-1] == 0b01
        ), f"expected pump-via-motor-valve (0b01), got history {h.received_valves}"

    def test_quiescent_closes_both_valves(self, bcu_node_harness):
        # Target == current at the surface → q ≈ 0, pump idle → both
        # valves closed → bitmask = 0b00.
        h = bcu_node_harness
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_valves) >= 3
        # All emissions must be 0; a stray valve open here would mean the
        # node opened a passive vent without a descent intent.
        assert all(
            v == 0 for v in h.received_valves
        ), f"expected all-closed history, got {h.received_valves}"

    def test_deep_quiescent_closes_both_valves(self, bcu_node_harness):
        # Boundary on the strict `q > 0` in select_pump_and_valves. Deep +
        # target == current settles to q ≈ 0; strict `>` keeps the vent
        # closed, but a `>=` slip — or a cascade sign flip producing a
        # tiny positive q at zero error — would open the free vent here. The
        # shallow-quiescent test above can't catch this because deep=False
        # short-circuits the q-sign branch entirely.
        h = bcu_node_harness
        h.publish_target_pressure(GAUGE_FOR_DEEP_PA)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
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

    def test_passive_vent_implies_zero_rpm(self, bcu_node_harness):
        # Cross-check the invariant from select_pump_and_valves: any sample
        # where the free/bypass vent (bit1) is open must have a zero pump
        # command.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_100M)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_for(0.6)
        # Pair-wise alignment: zip stops at the shorter list, which
        # absorbs at-most-one-sample skew between the two topics.
        for rpm, valves in zip(h.received_rpm, h.received_valves):
            if valves & 0b10:
                assert rpm == 0, (
                    f"passive vent open with non-zero rpm={rpm} "
                    f"(rpm history {h.received_rpm}, "
                    f"valves history {h.received_valves})"
                )


class TestTankLimitClamp:
    """The output clamp zeroes RPM and closes the valves when the
    commanded oil flow is headed at a registered tank endpoint that's
    been reached -- and stays inert without a registration."""

    def _register(self, h) -> None:
        # bcu_node's _on_dive_init now consumes only the tank endpoints; the
        # surface_pressure_pa field is ignored here (it gauges nothing on this
        # node anymore), but a valid value keeps the payload well-formed.
        h.publish_dive_init(
            surface_pressure_pa=101_325.0,
            tank_empty_pa=TANK_EMPTY_PA,
            tank_full_pa=TANK_FULL_PA,
        )
        h.spin_until(lambda: h.node._tank_full_pa is not None, timeout=1.0)

    def test_empty_limit_clamps_ascend(self, bcu_node_harness):
        # Ascend stimulus (target shallower than current) drives a positive
        # bus RPM (TestSignConvention) -- that inflates the bladder and
        # drains the tank, so a tank already at empty must clamp it.
        h = bcu_node_harness
        self._register(h)
        h.publish_tank_pressure(TANK_EMPTY_PA)
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 6, timeout=1.5)
        tail_rpm = h.received_rpm[-3:]
        tail_valves = h.received_valves[-3:]
        assert all(
            r == 0 for r in tail_rpm
        ), f"ascend at the empty-tank limit must clamp to 0, got {h.received_rpm}"
        assert all(
            v == 0 for v in tail_valves
        ), f"clamp must close the valves, got {h.received_valves}"

    def test_full_limit_gates_passive_vent(self, bcu_node_harness):
        # The deep-descend passive vent (0b10) lets ambient push oil INTO
        # the tank; at the full endpoint the clamp must shut it.
        h = bcu_node_harness
        self._register(h)
        h.publish_tank_pressure(TANK_FULL_PA)
        h.publish_target_pressure(TARGET_PA_100M)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_valves) >= 6, timeout=1.5)
        tail_rpm = h.received_rpm[-3:]
        tail_valves = h.received_valves[-3:]
        assert all(v == 0 for v in tail_valves), (
            f"passive vent at the full-tank limit must close, "
            f"got {h.received_valves}"
        )
        assert all(r == 0 for r in tail_rpm)

    def test_unregistered_tank_pressure_changes_nothing(self, bcu_node_harness):
        # Tank pressure flowing in WITHOUT a registration must leave the
        # existing behavior untouched: deep descend still passively vents.
        h = bcu_node_harness
        h.publish_tank_pressure(TANK_FULL_PA)
        h.publish_target_pressure(TARGET_PA_100M)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_valves) >= 4, timeout=1.5)
        assert h.received_valves[-1] == 0b10, (
            f"without a registration the vent must stay open, "
            f"got {h.received_valves}"
        )

    def test_clamp_releases_when_tank_recovers(self, bcu_node_harness):
        # Per-tick clamp, not a latch: tank back inside the range ->
        # commands resume on the next tick.
        h = bcu_node_harness
        self._register(h)
        h.publish_tank_pressure(TANK_EMPTY_PA)
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 6, timeout=1.5)
        assert h.received_rpm[-1] == 0, "precondition: clamp active"

        h.received_rpm.clear()
        h.publish_tank_pressure(TANK_MID_PA)
        h.spin_until(
            lambda: any(r > 0 for r in h.received_rpm),
            timeout=1.5,
        )
        assert any(r > 0 for r in h.received_rpm), (
            f"ascend command must resume once the tank recovers, "
            f"got {h.received_rpm}"
        )


class TestStopResetsAndSilences:
    """/command=false drops the held target, wipes controller state, and
    re-asserts the safe-stop (0 RPM + valves closed) for a bounded burst
    (~STOP_REASSERT_S) so the STM latches the zero even if a single message
    is dropped -- then goes silent so a debug node can own the wire."""

    def test_stop_reasserts_zero_burst_then_silent(self, bcu_node_harness):
        h = bcu_node_harness
        # Drive a real descent command first.
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        assert h.received_rpm[-1] != 0, "precondition: pump actively commanded"

        # Stop -> target cleared, controller state wiped.
        h.publish_command(False)
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        assert h.node.target_pressure_pa is None

        # The safe-stop is re-asserted as a burst: several 0-RPM / closed-valve
        # emissions land, and every one is zero (no stray command after stop).
        h.received_rpm.clear()
        h.received_valves.clear()
        h.spin_for(0.5)  # inside the ~1 s burst window
        assert (
            len(h.received_rpm) >= 2
        ), f"stop must re-assert the safe-stop for a burst, got {h.received_rpm}"
        assert all(r == 0 for r in h.received_rpm), h.received_rpm
        assert all(v == 0 for v in h.received_valves), h.received_valves

        # Once the burst is spent the loop goes silent -- no perpetual zero-hold.
        h.spin_for(1.2)  # let the rest of the burst drain
        h.received_rpm.clear()
        h.received_valves.clear()
        h.spin_for(0.5)
        assert (
            h.received_rpm == []
        ), f"bcu_node must go silent after the burst, got {h.received_rpm}"
        assert h.received_valves == []

    def test_repeated_stop_does_not_rearm_burst(self, bcu_node_harness):
        # Edge-trigger: the UI sends /command=false before every manual command,
        # so a stop while already stopped must be a no-op -- otherwise each one
        # re-arms the burst and chatters the wire against the manual driver.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)

        # First stop fires the burst; let it drain to silence.
        h.publish_command(False)
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        h.spin_for(1.3)
        h.received_rpm.clear()
        h.received_valves.clear()

        # A second stop while already stopped must emit nothing.
        h.publish_command(False)
        h.spin_for(0.5)
        assert (
            h.received_rpm == []
        ), f"repeated stop must not re-arm the burst, got {h.received_rpm}"
        assert h.received_valves == []

    def test_manual_command_during_stop_yields_instead_of_bursting(
        self, bcu_node_harness
    ):
        # A manual command landing with the stop makes bcu_node yield the wire to
        # bcu_debug: at most the single immediate safe-stop sample, never a burst.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        h.received_rpm.clear()
        h.received_valves.clear()

        # The engageManual order: stop, then the manual command, back to back.
        h.publish_command(False)
        h.publish_debug_valves(0b10)  # free/vent valve open
        h.spin_for(0.5)  # a full burst would land ~5 zeros here

        assert len(h.received_rpm) <= 2, (
            f"manual command must cancel the burst (<=1 safe-stop sample), "
            f"got {h.received_rpm}"
        )
        assert all(r == 0 for r in h.received_rpm), h.received_rpm


class TestCommandGate:
    """The near-setpoint command gate is wired in: configured from the spec,
    and its deadband holds the pump idle even for a command that the rpm
    deadband alone would let through."""

    def test_gate_configured_from_spec_defaults(self, bcu_node_harness):
        from py_pkg.scenarios.spec.control import DepthSpec

        d = DepthSpec()
        g = bcu_node_harness.node._gate
        # Proves the inverse param mapping (bcu_spec_from_node) carried the
        # three new gate fields into the node.
        assert g.error_arm_pa == pytest.approx(d.error_arm_pa)
        assert g.error_disarm_pa == pytest.approx(d.error_disarm_pa)
        assert g.min_valve_dwell_s == pytest.approx(d.min_valve_dwell_s)

    def test_disarm_band_holds_pump_idle(self, bcu_node_harness):
        from py_pkg.pid.bcu_command_gate import BcuCommandGate

        h = bcu_node_harness
        # Widen the deadband so a normally-active descent sits inside it: a
        # 30 m error would drive a big rpm, but the gate must suppress it.
        h.node._gate = BcuCommandGate(
            error_arm_pa=400_000.0, error_disarm_pa=400_000.0, min_valve_dwell_s=0.0
        )
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        assert all(
            r == 0 for r in h.received_rpm
        ), f"gate disarm band must hold the pump idle, got {h.received_rpm}"
        assert all(v == 0 for v in h.received_valves), h.received_valves
