"""Tier 2: BcuDebugNode end-to-end via in-process rclpy harness.

There's no override gate now: the node drives /bcu/rpm + /bcu/valves whenever
it holds a command. Contention with depth_node is avoided upstream (the UI
stops the mission first, so depth_node goes silent). When idle the node
publishes nothing; a short trailing-zero flush carries the terminal 0 out after
a command ends. Emergency surface is the safety path and always acts.

Asserts the behaviours the bench operator depends on:
* a pump command reaches /bcu/rpm immediately (no gate),
* a 0 follows after the requested duration (motor stops),
* a fresh command supersedes an in-flight stop without a 0 in between,
* when idle the node is silent (depth_node owns the BCU),
* a session end flushes trailing zeros so the reset-to-0 reaches the UI stream,
* /debug/reset all-stops the node (zero, then silent),
* pump-until-pressure runs without a feasibility pre-check and stops when the
  tank reading crosses the target,
* valve commands latch and are re-asserted by the heartbeat,
* a pump-only command never touches the valves,
* emergency surface blows ballast until surfaced.
"""

from py_pkg.physics import ATMOSPHERIC_PRESSURE_PA
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM, BCU_MOTOR_VALVE_MASK


def test_command_drives_wire_immediately(bcu_debug_node_harness):
    # No override gate: a pump command reaches /bcu/rpm right away.
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=0.5)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)
    assert 500 in h.received_rpm


def test_pump_publishes_rpm_then_zero(bcu_debug_node_harness):
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=0.2)

    # First, we must see the requested rpm on /bcu/rpm.
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)

    # Then, after the duration elapses, we must see a 0 stop.
    h.spin_until(
        lambda: h.received_rpm and h.received_rpm[-1] == 0,
        timeout=1.5,
    )

    # The node heartbeats the commanded value while the session is live and
    # flushes trailing 0s when it ends, so the only *nonzero* command must be
    # the requested 500, and the stream must end at 0.
    nonzero = [r for r in h.received_rpm if r != 0]
    assert nonzero and all(r == 500 for r in nonzero), (
        f"only 500 should have been commanded: {h.received_rpm}"
    )
    assert h.received_rpm[-1] == 0


def test_negative_rpm_passes_through(bcu_debug_node_harness):
    # "Pump out" sends a negative rpm; the debug node must not flip the
    # sign or otherwise interpret it -- the wire-level direction flip lives in
    # stm_com, not here.
    h = bcu_debug_node_harness
    h.publish_pump(rpm=-300, duration_s=0.15)

    h.spin_until(lambda: -300 in h.received_rpm, timeout=1.0)
    h.spin_until(
        lambda: h.received_rpm and h.received_rpm[-1] == 0,
        timeout=1.5,
    )

    nonzero = [r for r in h.received_rpm if r != 0]
    assert nonzero and all(r == -300 for r in nonzero), (
        f"debug node must pass the negative rpm through unchanged: {h.received_rpm}"
    )
    assert h.received_rpm[-1] == 0


def test_second_command_cancels_pending_stop(bcu_debug_node_harness):
    # Issue a long pump, then a short one before the first stop fires.
    # The first stop timer must be cancelled, so the sequence reaching
    # /bcu/rpm is [400..., 800..., 0] -- no stray 0 between the two rpms.
    h = bcu_debug_node_harness

    h.publish_pump(rpm=400, duration_s=2.0)
    h.spin_until(lambda: 400 in h.received_rpm, timeout=1.0)

    h.spin_for(0.1)
    h.publish_pump(rpm=800, duration_s=0.15)
    h.spin_until(lambda: 800 in h.received_rpm, timeout=1.0)

    # Wait past the second stop window with margin.
    h.spin_until(
        lambda: h.received_rpm and h.received_rpm[-1] == 0,
        timeout=1.5,
    )

    rpms = h.received_rpm
    idx_400 = rpms.index(400)
    idx_800 = rpms.index(800)
    assert idx_400 < idx_800
    # No stop-0 between the two commands: while the first pump is active the
    # heartbeat re-asserts 400, and the new command supersedes it via
    # _clear_pump_state() without arming the flush -- so it never drops to 0
    # until the (cancelled) timer would have fired. (No leading idle 0s now --
    # the node is silent until the first command.)
    assert all(v != 0 for v in rpms[idx_400:idx_800]), (
        f"unexpected stop fired before the second command was honoured: {rpms}"
    )
    assert rpms[-1] == 0


def test_reset_zeros_motor_and_silences(bcu_debug_node_harness):
    # The red Reset all-stops the debug node: zero the motor (with a short
    # trailing-zero flush), then go silent so depth_node can reclaim the wire.
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=5.0)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)

    h.publish_reset()
    # Immediate 0 + the trailing-zero flush, then silence.
    h.spin_for(0.8)
    assert h.received_rpm and h.received_rpm[-1] == 0
    assert all(r == 0 for r in h.received_rpm[-5:]), (
        f"reset must emit only 0s, got {h.received_rpm}"
    )

    # Now past the flush: the node is fully silent.
    h.received.clear()
    h.spin_for(0.5)
    assert h.received_rpm == [], (
        f"debug node kept driving the wire after the reset flush: {h.received_rpm}"
    )


def test_valve_command_latches_and_heartbeats(bcu_debug_node_harness):
    # An open valve must reach /bcu/valves and keep being re-asserted by the
    # heartbeat so it survives the bridge egress throttle.
    h = bcu_debug_node_harness
    h.publish_valves(BCU_MOTOR_VALVE_MASK)

    h.spin_until(lambda: BCU_MOTOR_VALVE_MASK in h.received_valves, timeout=1.0)

    h.received_valves.clear()
    h.spin_for(0.3)
    assert h.received_valves, "valve mask should be re-asserted by the heartbeat"
    assert all(v == BCU_MOTOR_VALVE_MASK for v in h.received_valves)


def test_pump_only_does_not_clobber_valves(bcu_debug_node_harness):
    # An RPM-only pump must not touch /bcu/valves -- otherwise it would slam
    # the valves shut underneath the operator (and against the pump flow). The
    # trailing-zero flush is RPM-only for exactly this reason.
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=0.2)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)
    h.spin_for(0.6)  # past the duration and the flush window
    assert h.received_valves == []


def test_emergency_surface_blows_ballast(bcu_debug_node_harness):
    # Emergency is the safety path: it acts immediately, no command staging.
    h = bcu_debug_node_harness
    # Start deep: gauge pressure well above the surface threshold.
    h.publish_external_pressure(int(ATMOSPHERIC_PRESSURE_PA) + 50_000)
    h.spin_for(0.1)
    h.publish_emergency(True)

    # Blow ballast: full positive RPM with valve 2 (the motor way) open.
    h.spin_until(lambda: BCU_MOTOR_MAX_RPM in h.received_rpm, timeout=1.0)
    h.spin_until(lambda: BCU_MOTOR_VALVE_MASK in h.received_valves, timeout=1.0)

    # Report we've reached the surface -> the pump must stop.
    h.publish_external_pressure(int(ATMOSPHERIC_PRESSURE_PA))
    h.spin_until(lambda: h.received_rpm and h.received_rpm[-1] == 0, timeout=2.0)
    assert h.received_rpm[-1] == 0


def test_idle_is_silent(bcu_debug_node_harness):
    # With no active command the node publishes nothing, leaving /bcu/rpm to
    # depth_node. (The trailing-zero flush only runs right after a command ends.)
    h = bcu_debug_node_harness
    h.spin_for(0.5)
    assert h.received_rpm == [], (
        f"idle debug node must stay silent, got {h.received_rpm}"
    )


def test_session_end_flushes_trailing_zeros(bcu_debug_node_harness):
    # When a pump session ends the node carries the terminal 0 for a few ticks
    # so the throttled UI egress reliably lands it, then goes silent. Without
    # the flush a lone terminal 0 could lose the race and freeze the strip
    # chart at the last nonzero rpm.
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=0.15)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)
    h.spin_for(0.6)  # past the stop and the flush

    last_500 = max(i for i, r in enumerate(h.received_rpm) if r == 500)
    trailing = h.received_rpm[last_500 + 1:]
    assert trailing and all(r == 0 for r in trailing), (
        f"expected a trailing-zero flush after the session, got {h.received_rpm}"
    )
    assert len(trailing) >= 2, f"flush too short: {h.received_rpm}"

    # Then silent.
    h.received.clear()
    h.spin_for(0.5)
    assert h.received_rpm == []


def test_pump_until_pressure_runs_without_preflight_refusal(bcu_debug_node_harness):
    # Issue 4: the bench bug. With the tank reading near 0 and a target like
    # 110000 Pa, "pump in" (positive rpm) is "already past" (tank <= target),
    # which the old code refused outright. It must now run the command (emit the
    # requested rpm) rather than instantly zeroing.
    h = bcu_debug_node_harness
    h.publish_tank_pressure(0)  # hardware tank sits near zero gauge
    h.spin_until(lambda: h.node._tank_pa is not None, timeout=1.0)

    h.publish_pump_until_pressure(rpm=500, target_pressure_pa=110000)
    # The command must reach the wire -- proof we did not pre-refuse.
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)
    assert 500 in h.received_rpm


def test_pump_until_pressure_stops_when_target_crossed(bcu_debug_node_harness):
    # Closed-loop stop: inflate (positive rpm) until the tank reading drops to
    # the target. Starts above target (keeps pumping), then a fresh sample
    # below the target ends it.
    h = bcu_debug_node_harness
    h.publish_tank_pressure(8000)
    h.spin_until(lambda: h.node._tank_pa is not None, timeout=1.0)

    h.publish_pump_until_pressure(rpm=500, target_pressure_pa=5000)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)

    # Tank crosses below the target -> the pump must stop.
    h.publish_tank_pressure(4000)
    h.spin_until(lambda: h.received_rpm and h.received_rpm[-1] == 0, timeout=1.5)
    nonzero = [r for r in h.received_rpm if r != 0]
    assert nonzero and all(r == 500 for r in nonzero), (
        f"only the commanded 500 should appear: {h.received_rpm}"
    )
    assert h.received_rpm[-1] == 0
