"""Tier 2: AcuDebugNode end-to-end via in-process rclpy harness.

The override is operator-owned now (the slider raises CONTROL_ACU_OVERRIDE),
so the ACU manual driver is a gated relay: it holds an operator-set pitch/roll
position on the wire (heartbeated) only while the flag is up, and drops the
held setpoints when it goes down -- turning the slider off is the release.
"""


def _enter_manual(h) -> None:
    """Raise the operator ACU override and wait for the node to see it."""
    h.publish_acu_override(True)
    h.spin_until(lambda: h.node._manual_override is True, timeout=1.0)


def test_command_ignored_without_override(acu_debug_node_harness):
    # With the override off, acu_node owns /acu/pitch -- the debug node must
    # stay silent (the UI also disables the buttons).
    h = acu_debug_node_harness
    h.publish_pitch(-60)
    h.spin_for(0.3)
    assert h.received_pitch_mm == [], (
        f"debug node drove the ACU while override was off: {h.received_pitch_mm}"
    )


def test_pitch_command_holds_while_manual(acu_debug_node_harness):
    h = acu_debug_node_harness
    _enter_manual(h)
    h.publish_pitch(-60)

    h.spin_until(lambda: -60 in h.received_pitch_mm, timeout=1.0)

    # The held setpoint is re-asserted by the heartbeat -- that's what keeps
    # the position parked while acu_node is silenced.
    h.received_pitch_mm.clear()
    h.spin_for(0.3)
    assert h.received_pitch_mm, "pitch setpoint should be re-asserted by the heartbeat"
    assert all(v == -60 for v in h.received_pitch_mm)


def test_roll_command_holds_wire_value(acu_debug_node_harness):
    # The wire value is centidegrees; the node passes it straight through.
    h = acu_debug_node_harness
    _enter_manual(h)
    h.publish_roll(1500)

    h.spin_until(lambda: 1500 in h.received_roll_cdeg, timeout=1.0)
    assert h.received_roll_cdeg[-1] == 1500


def test_dropping_override_hands_control_back(acu_debug_node_harness):
    h = acu_debug_node_harness
    _enter_manual(h)
    h.publish_pitch(-40)
    h.spin_until(lambda: -40 in h.received_pitch_mm, timeout=1.0)

    # Slider off -> the node drops the held setpoint and stops heartbeating,
    # so acu_node owns the axes again.
    h.publish_acu_override(False)
    h.spin_until(lambda: h.node._manual_override is False, timeout=1.0)

    h.received_pitch_mm.clear()
    h.spin_for(0.3)
    assert h.received_pitch_mm == []
