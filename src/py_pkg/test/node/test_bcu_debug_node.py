"""Tier 2: BcuDebugNode end-to-end via in-process rclpy harness.

Asserts the three behaviours the bench operator depends on:
* the requested rpm reaches /bcu/rpm immediately,
* a 0 follows after the requested duration (motor stops),
* a fresh command supersedes an in-flight stop without a 0 in between.
"""


def test_override_engages_around_pump_window(bcu_debug_node_harness):
    # The whole point of the override is to let depth_node know "step
    # aside" -- it must go True before the rpm goes out and False after
    # the stop, so depth_node never publishes against us mid-pump.
    h = bcu_debug_node_harness
    h.publish_pump(rpm=500, duration_s=0.15)

    h.spin_until(
        lambda: True in h.override_states and False in h.override_states,
        timeout=2.0,
    )

    assert h.override_states[0] is True, (
        f"override must engage before/with the first rpm publish, got {h.override_states}"
    )
    assert h.override_states[-1] is False, (
        f"override must release after the stop, got {h.override_states}"
    )


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

    assert h.received_rpm[0] == 500
    assert h.received_rpm[-1] == 0


def test_negative_rpm_passes_through(bcu_debug_node_harness):
    # "Pump out" sends a negative rpm; the debug node must not flip the
    # sign or otherwise interpret it -- the motor driver downstream does
    # the only direction-aware thing in the stack.
    h = bcu_debug_node_harness
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
