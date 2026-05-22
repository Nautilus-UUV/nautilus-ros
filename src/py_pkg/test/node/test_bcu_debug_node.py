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
* dropping the override zeros the motor and silences the heartbeat,
* valve commands latch and are re-asserted by the heartbeat,
* emergency surface blows ballast until surfaced, regardless of the flag.
"""

from py_pkg.physics import ATMOSPHERIC_PRESSURE_PA
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM

VALVE1_OPEN_MASK = 0b01


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

    assert h.received_rpm[0] == 500
    assert h.received_rpm[-1] == 0


def test_negative_rpm_passes_through(bcu_debug_node_harness):
    # "Pump out" sends a negative rpm; the debug node must not flip the
    # sign or otherwise interpret it -- the motor driver downstream does
    # the only direction-aware thing in the stack.
    h = bcu_debug_node_harness
    _enter_manual(h)
    h.publish_pump(rpm=-300, duration_s=0.15)

    h.spin_until(lambda: -300 in h.received_rpm, timeout=1.0)
    h.spin_until(
        lambda: h.received_rpm and h.received_rpm[-1] == 0,
        timeout=1.5,
    )

    assert h.received_rpm[0] == -300
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
    assert rpms[0] == 400
    idx_800 = rpms.index(800)
    # Nothing -- and crucially no stop-0 -- between the two commands.
    assert all(v != 0 for v in rpms[:idx_800]), (
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
    h.publish_valves(VALVE1_OPEN_MASK)

    h.spin_until(lambda: VALVE1_OPEN_MASK in h.received_valves, timeout=1.0)

    h.received_valves.clear()
    h.spin_for(0.3)
    assert h.received_valves, "valve mask should be re-asserted by the heartbeat"
    assert all(v == VALVE1_OPEN_MASK for v in h.received_valves)


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

    # Blow ballast: full positive RPM with valve 1 open.
    h.spin_until(lambda: BCU_MOTOR_MAX_RPM in h.received_rpm, timeout=1.0)
    h.spin_until(lambda: VALVE1_OPEN_MASK in h.received_valves, timeout=1.0)

    # Report we've reached the surface -> the pump must stop.
    h.publish_external_pressure(int(ATMOSPHERIC_PRESSURE_PA))
    h.spin_until(lambda: h.received_rpm and h.received_rpm[-1] == 0, timeout=2.0)
    assert h.received_rpm[-1] == 0
