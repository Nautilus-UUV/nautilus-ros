"""Tier 2: AcuDebugNode end-to-end via in-process rclpy harness.

There's no override gate now: the node holds whatever pitch/roll position it's
commanded and heartbeats it on the wire. Contention with acu_node is avoided
upstream (the UI stops the mission first, so acu_node goes silent). When no axis
is commanded the node is silent. /debug/reset releases both axes to neutral.
"""


def test_command_drives_wire_immediately(acu_debug_node_harness):
    # No override gate: a pitch command reaches /acu/pitch right away.
    h = acu_debug_node_harness
    h.publish_pitch(-60)
    h.spin_until(lambda: -60 in h.received_pitch_mm, timeout=1.0)
    assert -60 in h.received_pitch_mm


def test_pitch_command_holds(acu_debug_node_harness):
    h = acu_debug_node_harness
    h.publish_pitch(-60)

    h.spin_until(lambda: -60 in h.received_pitch_mm, timeout=1.0)

    # The held setpoint is re-asserted by the heartbeat -- that's what keeps
    # the position parked on the wire.
    h.received_pitch_mm.clear()
    h.spin_for(0.3)
    assert h.received_pitch_mm, "pitch setpoint should be re-asserted by the heartbeat"
    assert all(v == -60 for v in h.received_pitch_mm)


def test_roll_command_holds_wire_value(acu_debug_node_harness):
    # The wire value is centidegrees; the node passes it straight through.
    h = acu_debug_node_harness
    h.publish_roll(1500)

    h.spin_until(lambda: 1500 in h.received_roll_cdeg, timeout=1.0)
    assert h.received_roll_cdeg[-1] == 1500


def test_idle_is_silent(acu_debug_node_harness):
    # No command -> the node never touches /acu/pitch or /acu/roll.
    h = acu_debug_node_harness
    h.spin_for(0.4)
    assert h.received_pitch_mm == []
    assert h.received_roll_cdeg == []


def test_reset_neutralizes_and_silences(acu_debug_node_harness):
    # /debug/reset releases both axes: command neutral (0/0) once, carry it out
    # with a short flush, then go silent so acu_node owns the axes again.
    h = acu_debug_node_harness
    h.publish_pitch(-40)
    h.spin_until(lambda: -40 in h.received_pitch_mm, timeout=1.0)

    h.publish_reset()
    h.spin_for(0.8)  # neutral + flush, then silence
    assert h.received_pitch_mm and h.received_pitch_mm[-1] == 0
    assert all(v == 0 for v in h.received_pitch_mm[-3:]), (
        f"reset must emit only neutral pitch, got {h.received_pitch_mm}"
    )

    # Past the flush: fully silent on both axes.
    h.received_pitch_mm.clear()
    h.received_roll_cdeg.clear()
    h.spin_for(0.5)
    assert h.received_pitch_mm == []
    assert h.received_roll_cdeg == []
