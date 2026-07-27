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

from py_pkg.math_utils import span_band_guards
from py_pkg.physics import (
    depth_to_pressure_pa,
    gauge_pressure_pa,
)
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM
from py_pkg.scenarios.spec.control import DepthSpec
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

# The cutoff insets each endpoint by `tank_stop_band` of the span: it fires
# once the tank is within the guard, before the raw endpoint. Read from
# DepthSpec -- the same value the node under test builds its guard from via
# bcu_spec_from_node -- so retuning the band can't silently decouple these
# constants from the node. At the 5% default the guards land at ~102410
# (empty side) / ~185390 (full side).
TANK_LOW_GUARD_PA, TANK_HIGH_GUARD_PA = span_band_guards(
    float(TANK_EMPTY_PA), float(TANK_FULL_PA), DepthSpec().tank_stop_band
)
# Inside the empty-side band: ABOVE the empty endpoint but at/below the low
# guard -> must still clamp (proves the inset, not merely the endpoint).
TANK_IN_LOW_BAND_PA = int((TANK_EMPTY_PA + TANK_LOW_GUARD_PA) / 2)  # ~102410
# Just clear of the low guard -> must NOT clamp (pins the guard boundary).
TANK_ABOVE_LOW_GUARD_PA = int(TANK_LOW_GUARD_PA) + 3_000  # ~110020
# Inside the full-side band: BELOW the full endpoint but at/above the high
# guard -> must still clamp.
TANK_IN_HIGH_BAND_PA = int((TANK_HIGH_GUARD_PA + TANK_FULL_PA) / 2)  # ~185390

# Gauge Pa for current depth ~+50 m (Z-positive-down).
GAUGE_FOR_DEEP_PA = gauge_pressure_pa(depth_to_pressure_pa(50.0))

# Gauge-Pa setpoints used by the tests, expressed via depth equivalents
# so the intent ("70 m below the surface", "30 m") stays readable.
TARGET_PA_70M = gauge_pressure_pa(depth_to_pressure_pa(70.0))
TARGET_PA_30M = gauge_pressure_pa(depth_to_pressure_pa(30.0))
TARGET_PA_10M = gauge_pressure_pa(depth_to_pressure_pa(10.0))
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
    """POSITION_TARGET.position.z (gauge Pa) flows into node.target_pressure_pa,
    which the control loop feeds straight to the depth PID."""

    def test_target_pressure_updates_node_state(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(42.0)
        h.spin_until(lambda: h.node.target_pressure_pa == 42.0, timeout=1.0)
        assert h.node.target_pressure_pa == pytest.approx(42.0)


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
    """Bus-RPM sign convention, load-bearing all the way down to the STM.

    Z-positive-down throughout, and `solve_bcu_command` is bang-bang on the
    sign of `target - current`:
      target=70m gauge Pa, current=0  → error > 0 (target is deeper)
      → deflate the bladder to sink → published RPM is NEGATIVE.

    Inverted case (target shallower than current):
      target=0, current=+50m gauge Pa → error < 0
      → inflate the bladder to rise  → published RPM is POSITIVE.

    Note: the node arms a safe-stop burst at boot, so the first emissions on
    the wire are zeros that predate the target. We spin past them and assert on
    the last emission, which under bang-bang is the steady leg command.
    """

    def test_target_deeper_publishes_negative_rpm(self, bcu_node_harness):
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)  # current_pressure_pa ≈ 0
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        # Past the boot burst, the steady leg command must be negative.
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
        # Aggressive setpoint: target very deep, currently at surface — the
        # largest error the solver can be handed.
        h.publish_target_pressure(TARGET_PA_DEEP_HUGE)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        for r in h.received_rpm:
            assert abs(r) <= BCU_MOTOR_MAX_RPM, f"published {r} exceeds max"

    def test_published_rpm_is_only_ever_one_of_three_values(self, bcu_node_harness):
        # The defining bang-bang property, asserted on the wire rather than
        # on the pure law: nothing between 0 and +/-pump_rpm may ever be
        # published, because nothing in the node modulates. An intermediate
        # value here means a proportional stage crept back in.
        h = bcu_node_harness
        rpm = DepthSpec().pump_rpm
        for target, depth in (
            (TARGET_PA_30M, GAUGE_AT_SURFACE_PA),  # descend
            (0.0, GAUGE_FOR_DEEP_PA),  # ascend
            (0.0, GAUGE_AT_SURFACE_PA),  # nothing to do
        ):
            h.received_rpm.clear()
            h.publish_target_pressure(target)
            h.publish_depth_gauge(depth)
            h.spin_for(0.6)
            assert len(h.received_rpm) >= 3
            assert set(h.received_rpm) <= {-rpm, 0, rpm}, h.received_rpm

    def test_idle_inside_the_deadband(self, bcu_node_harness):
        # Target == current depth: the pump must be silent and both valves
        # shut. This is the surface case SURFACE sits in for its whole
        # completion dwell.
        h = bcu_node_harness
        h.publish_target_pressure(GAUGE_AT_SURFACE_PA)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        assert all(r == 0 for r in h.received_rpm), h.received_rpm
        assert all(v == 0 for v in h.received_valves), h.received_valves


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
    bit1 = free/bypass way (operator "valve 1", passive vent). The node's boot
    safe-stop burst puts zeros on the wire before the first target lands, so
    assertions use the last emission after spin.
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
        # Boundary on the strict `error > deadband_pa` in solve_bcu_command.
        # Deep + target == current gives zero error; the strict `>` keeps the
        # vent closed, but a `>=` slip — or a sign flip producing a tiny
        # positive error at zero — would open the free vent here. The
        # shallow-quiescent test above can't catch this because the deep
        # branch is what selects the passive vent at all.
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

    def test_latch_releases_on_leg_reversal(self, bcu_node_harness):
        # The latch is held until the command reverses -- which is exactly
        # what a mission leg change does. Rail the tank on an ascend, then
        # command a descent: the pump must run again on the next tick.
        h = bcu_node_harness
        self._register(h)
        h.publish_tank_pressure(TANK_EMPTY_PA)
        h.publish_target_pressure(0.0)  # ascend: drains the tank toward empty
        # Shallow of BCU_DEEP_THRESHOLD_PA on purpose: past it a descend
        # command passively vents at 0 rpm, which would prove nothing about
        # the pump resuming.
        h.publish_depth_gauge(TARGET_PA_10M)
        h.spin_until(lambda: len(h.received_rpm) >= 6, timeout=1.5)
        assert h.received_rpm[-1] == 0, "precondition: latched at the rail"

        h.received_rpm.clear()
        h.publish_target_pressure(TARGET_PA_30M)  # reverse: descend
        h.spin_until(lambda: any(r < 0 for r in h.received_rpm), timeout=1.5)
        assert any(r < 0 for r in h.received_rpm), (
            f"the reversed command must flow straight through, "
            f"got {h.received_rpm}"
        )

    def test_latch_holds_through_a_tank_reading_that_retreats(self, bcu_node_harness):
        # Deliberate asymmetry vs. the old release band: while latched the
        # pump is stopped and both valves shut, so the tank CANNOT move on
        # its own. A reading that says it did is noise, and must not restart
        # the pump into the rail.
        h = bcu_node_harness
        self._register(h)
        h.publish_tank_pressure(TANK_EMPTY_PA)
        h.publish_target_pressure(0.0)
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 6, timeout=1.5)
        assert h.received_rpm[-1] == 0, "precondition: latched at the rail"

        h.received_rpm.clear()
        h.publish_tank_pressure(TANK_MID_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        assert all(r == 0 for r in h.received_rpm), h.received_rpm

    def test_low_guard_band_clamps_before_empty_endpoint(self, bcu_node_harness):
        # A tank reading inside the empty-side band (above the 97800 endpoint
        # but at/below the ~107020 guard) must clamp the drain-the-tank ascend
        # command -- proving the clamp fires on the 10% inset, not just at the
        # raw endpoint.
        h = bcu_node_harness
        self._register(h)
        assert TANK_EMPTY_PA < TANK_IN_LOW_BAND_PA <= TANK_LOW_GUARD_PA
        h.publish_tank_pressure(TANK_IN_LOW_BAND_PA)
        h.publish_target_pressure(0.0)  # ascend: positive bus rpm, drains tank
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 6, timeout=1.5)
        tail_rpm = h.received_rpm[-3:]
        tail_valves = h.received_valves[-3:]
        assert all(r == 0 for r in tail_rpm), (
            f"ascend inside the empty-side guard band must clamp to 0, "
            f"got {h.received_rpm}"
        )
        assert all(v == 0 for v in tail_valves), (
            f"clamp must close the valves, got {h.received_valves}"
        )

    def test_just_above_low_guard_does_not_clamp(self, bcu_node_harness):
        # A hair above the low guard (~110020 > ~107020) is outside the band:
        # the ascend command must flow. Pins the boundary at the guard, not
        # somewhere between the guard and the endpoint.
        h = bcu_node_harness
        self._register(h)
        assert TANK_ABOVE_LOW_GUARD_PA > TANK_LOW_GUARD_PA
        h.publish_tank_pressure(TANK_ABOVE_LOW_GUARD_PA)
        h.publish_target_pressure(0.0)  # ascend stimulus
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 6, timeout=1.5)
        assert any(r > 0 for r in h.received_rpm[-3:]), (
            f"just above the guard the ascend must not clamp, got {h.received_rpm}"
        )

    def test_high_guard_band_gates_passive_vent_before_full_endpoint(
        self, bcu_node_harness
    ):
        # Symmetric high-side case: a tank reading inside the full-side band
        # (below the 190000 endpoint but at/above the ~180780 guard) must shut
        # the deep-descend passive vent that would push more oil into the tank.
        h = bcu_node_harness
        self._register(h)
        assert TANK_HIGH_GUARD_PA <= TANK_IN_HIGH_BAND_PA < TANK_FULL_PA
        h.publish_tank_pressure(TANK_IN_HIGH_BAND_PA)
        h.publish_target_pressure(TARGET_PA_100M)  # deep descend -> passive vent
        h.publish_depth_gauge(GAUGE_FOR_DEEP_PA)
        h.spin_until(lambda: len(h.received_valves) >= 6, timeout=1.5)
        tail_rpm = h.received_rpm[-3:]
        tail_valves = h.received_valves[-3:]
        assert all(v == 0 for v in tail_valves), (
            f"passive vent inside the full-side guard band must close, "
            f"got {h.received_valves}"
        )
        assert all(r == 0 for r in tail_rpm), (
            f"clamp must keep the pump idle, got {h.received_rpm}"
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


class TestMissionCompleteStops:
    """A finished mission stops the BCU through MISSION_COMPLETE alone.

    Completion and an operator abort are different events on different topics,
    and the controller's response to each is the same safe-stop. This pins the
    completion half: with NO /command traffic at all, the latched Bool(true) on
    /mission/complete must drop the target and park the wire. Before the
    controllers subscribed here, pathfinding had to forge a /command=false to
    get this behavior -- which made the two events indistinguishable downstream.
    """

    def test_completion_drops_target_and_parks_the_wire(self, bcu_node_harness):
        h = bcu_node_harness
        # Drive a real descent first -- the pump must actually be running, or
        # the stop would prove nothing.
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)
        assert h.received_rpm[-1] != 0, "precondition: pump actively commanded"

        # Completion only. No /command is published anywhere in this test.
        h.publish_mission_complete()
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        assert h.node.target_pressure_pa is None

        # Same safe-stop burst the operator stop produces: zeros, valves closed.
        h.received_rpm.clear()
        h.received_valves.clear()
        h.spin_for(0.5)  # inside the ~1 s burst window
        assert len(h.received_rpm) >= 2, (
            f"completion must re-assert the safe-stop for a burst, "
            f"got {h.received_rpm}"
        )
        assert all(r == 0 for r in h.received_rpm), h.received_rpm
        assert all(v == 0 for v in h.received_valves), h.received_valves

        # And then silence -- the finished mission's target is not re-chased
        # even though depth samples keep arriving.
        h.spin_for(1.2)  # let the rest of the burst drain
        h.received_rpm.clear()
        h.received_valves.clear()
        for _ in range(8):
            h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
            h.spin_for(0.05)
        assert h.received_rpm == [], (
            f"bcu_node must stay silent after completion, got {h.received_rpm}"
        )
        assert h.received_valves == []

    def test_completion_after_an_operator_stop_is_a_no_op(self, bcu_node_harness):
        # Ordering is not guaranteed: an operator can stop a run in the same
        # instant it finishes, and MISSION_COMPLETE is TRANSIENT_LOCAL, so a
        # latched replay can land on an already-stopped controller. The
        # edge-trigger in _stop must absorb it rather than re-arm the burst and
        # chatter against whatever owns the wire by then.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)

        h.publish_command(False)
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        h.spin_for(1.3)  # drain the burst to silence
        h.received_rpm.clear()
        h.received_valves.clear()

        h.publish_mission_complete()
        h.spin_for(0.5)
        assert h.received_rpm == [], (
            f"completion on an already-stopped BCU must emit nothing, "
            f"got {h.received_rpm}"
        )
        assert h.received_valves == []

    def test_completion_false_is_ignored(self, bcu_node_harness):
        # Nothing publishes Bool(false) on MISSION_COMPLETE today. If something
        # ever does, it is not a completion and must not stop a running mission.
        h = bcu_node_harness
        h.publish_target_pressure(TARGET_PA_70M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_until(lambda: len(h.received_rpm) >= 4, timeout=1.5)

        h.publish_mission_complete(False)
        h.spin_for(0.4)
        assert h.node.target_pressure_pa == pytest.approx(TARGET_PA_70M), (
            "Bool(false) on MISSION_COMPLETE must not clear the target"
        )
        assert h.received_rpm[-1] != 0, "the pump must still be commanded"


class TestSpecWiring:
    """The inverse param mapping (bcu_spec_from_node) reaches the node."""

    def test_bang_bang_knobs_carried_from_the_spec(self, bcu_node_harness):
        d = DepthSpec()
        node = bcu_node_harness.node
        assert node._pump_rpm == d.pump_rpm
        assert node._deadband_pa == pytest.approx(d.deadband_pa)
        assert node._tank_guard.stop_band == pytest.approx(d.tank_stop_band)

    def test_widening_the_deadband_silences_a_real_descent(self, bcu_node_harness):
        # A 30 m error normally commands full-speed descent; widen the
        # deadband past it and the node must go fully idle. Proves the
        # deadband is genuinely the thing gating the pump, not a constant.
        h = bcu_node_harness
        h.node._deadband_pa = 400_000.0
        h.publish_target_pressure(TARGET_PA_30M)
        h.publish_depth_gauge(GAUGE_AT_SURFACE_PA)
        h.spin_for(0.6)
        assert len(h.received_rpm) >= 3
        assert all(r == 0 for r in h.received_rpm), h.received_rpm
        assert all(v == 0 for v in h.received_valves), h.received_valves
