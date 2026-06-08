"""Tier 2: BcuDebugNode end-to-end via in-process rclpy harness.

The override is operator-owned now (the slider raises CONTROL_MANUAL_OVERRIDE),
so the node is a gated relay: it drives /bcu/rpm + /bcu/valves only while the
flag is up, and hands the wire back to depth_node when it drops. Emergency
surface is the one exception -- it acts regardless of the flag.

Asserts the behaviours the bench operator depends on:
* commands are dropped while the override is off (depth_node owns the BCU),
* once in manual mode the requested rpm reaches /bcu/rpm immediately,
* a 0 follows after the requested duration (motor stops),
* a fresh command supersedes an in-flight stop without a 0 in between,
* while in manual the node heartbeats the current command (0 when idle) so the
  reset-to-0 always reaches the throttled UI stream,
* dropping the override zeros the motor and goes silent (depth_node takes over),
* pump-until-pressure runs without a feasibility pre-check and stops when the
  tank reading crosses the target,
* valve commands latch and are re-asserted by the heartbeat,
* emergency surface blows ballast until surfaced, regardless of the flag.
"""

from py_pkg.physics import ATMOSPHERIC_PRESSURE_PA
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM, BCU_MOTOR_VALVE_MASK


def _enter_manual(h) -> None:
    """Raise the operator override and wait for the node to see it."""
    h.publish_manual_override(True)
    h.spin_until(lambda: h.node._manual_override is True, timeout=1.0)


def test_command_ignored_without_override(bcu_debug_node_harness):
    # With the override off, depth_node owns /bcu/rpm -- the debug node must
    # stay silent rather than racing it (the UI also disables the buttons).
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=0.5)
    h.spin_for(0.3)
    assert h.received_rpm == [], (
        f"debug node drove the wire while override was off: {h.received_rpm}"
    )


def test_pump_publishes_rpm_then_zero(bcu_debug_node_harness):
    h = bcu_debug_node_harness
    _enter_manual(h)
    h.publish_pump(rpm=500, duration_s=0.2)

    # First, we must see the requested rpm on /bcu/rpm.
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)

    # Then, after the duration elapses, we must see a 0 stop.
    h.spin_until(
        lambda: h.received_rpm and h.received_rpm[-1] == 0,
        timeout=1.5,
    )

    # In manual mode the node heartbeats the commanded value (0 once idle), so
    # the stream is padded with leading/trailing 0s -- the only *nonzero*
    # command must be the requested 500, and it must end at 0.
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
    _enter_manual(h)
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
    # /bcu/rpm is [400, 800, 0] -- no stray 0 between the two rpms.
    h = bcu_debug_node_harness
    _enter_manual(h)

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
    # Crucially no stop-0 between the two commands: while the first pump is
    # active the heartbeat re-asserts 400, it never drops to 0 until the
    # (cancelled) timer would have fired. Leading 0s from the idle heartbeat
    # before the first command are fine.
    assert all(v != 0 for v in rpms[idx_400:idx_800]), (
        f"unexpected stop fired before the second command was honoured: {rpms}"
    )
    assert rpms[-1] == 0


def test_dropping_override_zeros_motor_and_silences(bcu_debug_node_harness):
    # Turning the slider off hands the BCU back to depth_node: the node must
    # zero the motor once and then go quiet (no heartbeat in autonomous mode).
    h = bcu_debug_node_harness
    _enter_manual(h)
    h.publish_pump(rpm=500, duration_s=5.0)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)

    h.publish_manual_override(False)
    h.spin_until(lambda: h.node._manual_override is False, timeout=1.0)
    h.spin_until(lambda: h.received_rpm and h.received_rpm[-1] == 0, timeout=1.0)

    h.received.clear()
    h.spin_for(0.3)
    assert h.received_rpm == [], (
        f"debug node kept driving the wire after override dropped: {h.received_rpm}"
    )


def test_valve_command_latches_and_heartbeats(bcu_debug_node_harness):
    # In manual mode an open valve must reach /bcu/valves and keep being
    # re-asserted by the heartbeat so it survives the bridge egress throttle.
    h = bcu_debug_node_harness
    _enter_manual(h)
    h.publish_valves(BCU_MOTOR_VALVE_MASK)

    h.spin_until(lambda: BCU_MOTOR_VALVE_MASK in h.received_valves, timeout=1.0)

    h.received_valves.clear()
    h.spin_for(0.3)
    assert h.received_valves, "valve mask should be re-asserted by the heartbeat"
    assert all(v == BCU_MOTOR_VALVE_MASK for v in h.received_valves)


def test_pump_only_does_not_clobber_valves(bcu_debug_node_harness):
    # An RPM-only pump must not touch /bcu/valves -- otherwise it would slam
    # the valves shut underneath the operator (and against the pump flow).
    h = bcu_debug_node_harness
    _enter_manual(h)
    h.publish_pump(rpm=500, duration_s=0.2)
    h.spin_until(lambda: 500 in h.received_rpm, timeout=1.0)
    h.spin_for(0.2)
    assert h.received_valves == []


def test_emergency_surface_blows_ballast_regardless_of_override(bcu_debug_node_harness):
    # Emergency is the safety path: it must act even with the override off
    # (the UI raises the override first, but the node doesn't depend on it).
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


def test_idle_manual_heartbeats_zero(bcu_debug_node_harness):
    # Issue 5: while in manual with no active pump, the node heartbeats 0 every
    # tick. Without this the lone terminal 0 after a pump can lose the race
    # against the UI egress throttle and the strip chart freezes at the last
    # nonzero rpm.
    h = bcu_debug_node_harness
    _enter_manual(h)
    h.spin_for(0.4)  # several tick periods
    assert h.received_rpm, "expected a 0 heartbeat while idle in manual mode"
    assert all(r == 0 for r in h.received_rpm), (
        f"idle manual mode should heartbeat only 0: {h.received_rpm}"
    )


def test_pump_until_pressure_runs_without_preflight_refusal(bcu_debug_node_harness):
    # Issue 4: the bench bug. With the tank reading near 0 and a target like
    # 110000 Pa, "pump in" (positive rpm) is "already past" (tank <= target),
    # which the old code refused outright. It must now run the command (emit the
    # requested rpm) rather than instantly zeroing.
    h = bcu_debug_node_harness
    _enter_manual(h)
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
    _enter_manual(h)
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
