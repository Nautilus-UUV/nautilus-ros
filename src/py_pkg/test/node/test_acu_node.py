"""Tier 2 tests for ACUControlNode (bang-bang pitch + PID roll).

Pitch ignores attitude entirely now — it's a pressure-error bang-bang
publishing one of two ``Int16`` mm extremes pulled from
``AcuPitchSpec.output_limits``. Roll keeps the same PID/AxisController
contract on ``ACU_ROLL`` (Int16 centidegrees).

POSITION_ESTIMATION is now the single vehicle-state input: its orientation
drives current_roll_deg (roll PID) and its position.z carries the current
gauge depth (Pa) the bang-bang pitch loop compares against. attitude_node
owns the absolute->gauge conversion, so the node has no EXTERNAL_PRESSURE /
DIVE_INIT / SurfaceReference path anymore — the harness feeds gauge depth
directly via ``publish_current_attitude(gauge_pa=...)``.

The harness publishes:
* ``POSITION_TARGET``  — orientation = roll target, position.z = target
  gauge pressure (Pa, pathfinding's TRIM convention).
* ``POSITION_ESTIMATION`` — orientation = current roll, position.z =
  current gauge depth (Pa). Pitch dimension is deliberately ignored.
"""

import pytest

from py_pkg.robot_specs import (
    ACU_PITCH_MM_PER_M,
    ACU_ROLL_CDEG_PER_DEG,
    ACU_ROLL_MAX_ANGLE_DEG,
)
from py_pkg.scenarios.spec.control import AcuPitchSpec

# Bang-bang pitch wire values, mirrored from acu_node.py. Index 0 is the
# "back" extreme (selected when shallower than setpoint), index 1 is the
# "front" extreme (selected when deeper than or equal to setpoint).
_PITCH_OUTPUT_LIMITS_M = AcuPitchSpec().output_limits
PITCH_BACK_MM = int(round(_PITCH_OUTPUT_LIMITS_M[0] * ACU_PITCH_MM_PER_M))
PITCH_FRONT_MM = int(round(_PITCH_OUTPUT_LIMITS_M[1] * ACU_PITCH_MM_PER_M))

ROLL_MAX_CDEG = int(round(ACU_ROLL_MAX_ANGLE_DEG * ACU_ROLL_CDEG_PER_DEG))


class TestWiringSmoke:
    """Construction + topic graph wiring."""

    def test_node_constructs(self, acu_node_harness):
        assert acu_node_harness.node is not None

    def test_position_target_subscription_present(self, acu_node_harness):
        names = [sub.topic_name for sub in acu_node_harness.node.subscriptions]
        assert "/position/target" in names

    def test_position_estimation_subscription_present(self, acu_node_harness):
        names = [sub.topic_name for sub in acu_node_harness.node.subscriptions]
        assert "/position/estimation" in names

    def test_pitch_publisher_present(self, acu_node_harness):
        names = [pub.topic_name for pub in acu_node_harness.node.publishers]
        assert "/acu/pitch" in names

    def test_roll_publisher_present(self, acu_node_harness):
        names = [pub.topic_name for pub in acu_node_harness.node.publishers]
        assert "/acu/roll" in names


class TestTargetIngress:
    """``POSITION_TARGET`` carries roll-target (orientation) and pressure-
    target (position.z, gauge Pa)."""

    def test_target_pose_updates_roll_state(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(roll_deg=15.0)
        h.spin_until(
            lambda: h.node.target_roll_deg == pytest.approx(15.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.target_roll_deg == pytest.approx(15.0, abs=1e-4)

    def test_target_pose_updates_target_pressure(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=65332.0)
        h.spin_until(
            lambda: h.node.target_pressure_pa == pytest.approx(65332.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.target_pressure_pa == pytest.approx(65332.0, abs=1e-4)

    def test_target_pose_updates_both_simultaneously(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(roll_deg=-7.5, target_pressure_pa=50000.0)
        h.spin_until(
            lambda: (
                h.node.target_roll_deg == pytest.approx(-7.5, abs=1e-4)
                and h.node.target_pressure_pa == pytest.approx(50000.0, abs=1e-4)
            ),
            timeout=1.0,
        )
        assert h.node.target_roll_deg == pytest.approx(-7.5, abs=1e-4)
        assert h.node.target_pressure_pa == pytest.approx(50000.0, abs=1e-4)


class TestEstimationIngress:
    """``POSITION_ESTIMATION`` is the single vehicle-state input: orientation
    drives current_roll_deg (roll PID) and position.z drives
    current_pressure_pa (gauge Pa) for the bang-bang pitch leg select. Pitch
    off the pose is deliberately ignored."""

    def test_estimation_pose_updates_current_roll(self, acu_node_harness):
        h = acu_node_harness
        h.publish_current_attitude(roll_deg=8.0)
        h.spin_until(
            lambda: h.node.current_roll_deg == pytest.approx(8.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.current_roll_deg == pytest.approx(8.0, abs=1e-4)

    def test_estimation_pose_updates_current_pressure(self, acu_node_harness):
        # position.z is gauge Pa, stored verbatim -- attitude_node already
        # gauged it, so the node does no conversion.
        h = acu_node_harness
        gauge = 50000.0
        h.publish_current_attitude(gauge_pa=gauge)
        h.spin_until(
            lambda: h.node.current_pressure_pa is not None,
            timeout=1.0,
        )
        assert h.node.current_pressure_pa == pytest.approx(gauge, abs=1e-6)


class TestTimerEmits:
    """Once both inputs (pressure + target) have been seen, the 10 Hz
    loop publishes pitch and a non-trivial roll error drives roll too."""

    def test_emits_pitch_after_pressure_and_target(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=50000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert len(h.received_pitch_mm) >= 1

    def test_emits_roll_within_one_second(self, acu_node_harness):
        h = acu_node_harness
        # current pose stays at 0; target roll 20° creates an error large
        # enough that the roll PID + redundant-publish guard emits.
        h.publish_target(roll_deg=20.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        assert len(h.received_roll_cdeg) >= 1


class TestPitchGatedOnInputs:
    """Bang-bang refuses to pick a side from uninitialised zeros: it
    waits for both a gauge depth (POSITION_ESTIMATION.position.z) and a
    setpoint (POSITION_TARGET)."""

    def test_no_pitch_until_pressure_seen(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=50000.0)
        # Drive only the target — no estimation pose, so no gauge depth.
        h.spin_for(0.6)
        assert h.received_pitch_mm == []

    def test_no_pitch_until_target_seen(self, acu_node_harness):
        h = acu_node_harness
        h.publish_current_attitude(gauge_pa=20000.0)
        # Drive only the estimation pose — target never arrives.
        h.spin_for(0.6)
        assert h.received_pitch_mm == []


class TestPitchBangBang:
    """Output is one of two extremes from ``AcuPitchSpec.output_limits``;
    it never lands anywhere in between regardless of error magnitude."""

    def test_shallower_than_target_emits_back(self, acu_node_harness):
        # current gauge < target gauge → diving leg → BACK extreme.
        h = acu_node_harness
        h.publish_target(target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert h.received_pitch_mm[-1] == PITCH_BACK_MM

    def test_deeper_than_target_emits_front(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=20000.0)
        h.publish_current_attitude(gauge_pa=80000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert h.received_pitch_mm[-1] == PITCH_FRONT_MM

    def test_pitch_only_takes_two_values(self, acu_node_harness):
        # Even a small error keeps pitch railed — that's the whole point
        # of bang-bang. No proportional region, no clamp story.
        h = acu_node_harness
        h.publish_target(target_pressure_pa=50000.0)
        h.publish_current_attitude(gauge_pa=50000.0 - 100.0)
        h.spin_for(0.6)
        assert len(h.received_pitch_mm) >= 1
        for v in h.received_pitch_mm:
            assert v in (PITCH_BACK_MM, PITCH_FRONT_MM), (
                f"unexpected pitch wire value {v}; expected one of "
                f"{(PITCH_BACK_MM, PITCH_FRONT_MM)}"
            )

    def test_pitch_flips_when_setpoint_crosses(self, acu_node_harness):
        # Same physical glider, two different setpoints either side of
        # the same pressure reading. The two ticks must produce
        # different outputs.
        h = acu_node_harness
        h.publish_current_attitude(gauge_pa=50000.0)

        h.publish_target(target_pressure_pa=80000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        first = h.received_pitch_mm[-1]

        h.publish_target(target_pressure_pa=20000.0)
        h.spin_until(
            lambda: len(h.received_pitch_mm) >= 1 and h.received_pitch_mm[-1] != first,
            timeout=1.5,
        )
        assert h.received_pitch_mm[-1] != first
        assert {first, h.received_pitch_mm[-1]} == {PITCH_BACK_MM, PITCH_FRONT_MM}


class TestRollSignConvention:
    """Positive desired roll (relative to current=0) → positive ACU_ROLL
    (cdeg). Roll axis target_pos_deg = current + kp*(desired - current);
    published value is int(round(target_pos_deg * ACU_ROLL_CDEG_PER_DEG))
    with no sign inversion in the publish path."""

    def test_positive_target_publishes_positive_value(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(roll_deg=20.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        first = h.received_roll_cdeg[0]
        assert first > 0, f"expected positive roll cdeg, got {h.received_roll_cdeg}"
        assert abs(first) <= ROLL_MAX_CDEG

    def test_negative_target_publishes_negative_value(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(roll_deg=-20.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        first = h.received_roll_cdeg[0]
        assert first < 0, f"expected negative roll cdeg, got {h.received_roll_cdeg}"
        assert abs(first) <= ROLL_MAX_CDEG


class TestRollSaturation:
    """Large roll errors clamp to ±ACU_ROLL_MAX_ANGLE_DEG (in cdeg on
    the wire)."""

    def test_roll_saturates_at_max(self, acu_node_harness):
        # 100° far exceeds the clamp but stays inside (-180°, 180°) so
        # the quaternion → Euler round-trip is unambiguous.
        h = acu_node_harness
        h.publish_target(roll_deg=100.0)
        h.spin_for(0.6)
        assert len(h.received_roll_cdeg) >= 1
        for v in h.received_roll_cdeg:
            assert (
                abs(v) <= ROLL_MAX_CDEG
            ), f"published roll {v} cdeg exceeds ROLL_MAX_CDEG={ROLL_MAX_CDEG}"
        assert h.received_roll_cdeg[-1] == ROLL_MAX_CDEG

    def test_roll_saturates_at_min_for_negative_target(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(roll_deg=-100.0)
        h.spin_for(0.6)
        assert len(h.received_roll_cdeg) >= 1
        for v in h.received_roll_cdeg:
            assert abs(v) <= ROLL_MAX_CDEG
        assert h.received_roll_cdeg[-1] == -ROLL_MAX_CDEG


class TestRollQuiescence:
    """Saturated against the clamp, the redundant-publish guard
    silences further emissions — no flooding at 10 Hz."""

    def test_roll_saturated_settles_to_no_new_emissions(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(roll_deg=100.0)
        h.spin_until(lambda: len(h.received_roll_cdeg) >= 1, timeout=1.5)
        early_count = len(h.received_roll_cdeg)
        # After settling at the clamp, no new emissions; allow 1 for slack.
        h.spin_for(0.6)
        late_count = len(h.received_roll_cdeg)
        assert (
            late_count - early_count <= 1
        ), f"expected quiescence after saturation, got {h.received_roll_cdeg}"


class TestStopResetsAndSilences:
    """/command=false drops the held target, re-primes the roll axis, emits ONE
    neutral (0 pitch + 0 roll), then gates the whole loop off -- exactly as at
    boot before any mission. Silence (pitch AND roll) frees the ACU wire for
    acu_debug with no contention."""

    def test_stop_emits_neutral_then_silent(self, acu_node_harness):
        h = acu_node_harness
        # Drive a pitch + roll command first (shallower than target -> diving).
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 2, timeout=1.5)

        # Stop -> target cleared, one neutral (0/0) emitted, loop gated off.
        h.publish_command(False)
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        assert h.node.target_pressure_pa is None
        h.spin_for(0.2)  # let the one-shot neutral land
        assert h.received_pitch_mm and h.received_pitch_mm[-1] == 0
        assert h.received_roll_cdeg and h.received_roll_cdeg[-1] == 0

        # The loop is fully silent now (pitch AND roll), even with pressure
        # still flowing -- no new target has arrived to re-arm it.
        h.received_pitch_mm.clear()
        h.received_roll_cdeg.clear()
        for _ in range(8):
            h.publish_current_attitude(gauge_pa=20000.0)
            h.spin_for(0.05)
        assert (
            h.received_pitch_mm == []
        ), f"pitch must stay silent after stop, got {h.received_pitch_mm}"
        assert (
            h.received_roll_cdeg == []
        ), f"roll must stay silent after stop, got {h.received_roll_cdeg}"


class TestMissionCompleteStops:
    """A finished mission neutralizes the ACU through MISSION_COMPLETE alone.

    Same contract as the operator stop above, reached by the other event. It
    matters most here because the STM re-ships the last value forever: without
    this subscription, a completed mission would leave its final mass-shifter
    position latched on the wire with nothing left to clear it."""

    def test_completion_emits_neutral_then_silent(self, acu_node_harness):
        h = acu_node_harness
        # Drive a pitch + roll command first (shallower than target -> diving).
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 2, timeout=1.5)
        assert h.received_pitch_mm[-1] != 0, "precondition: pitch actively driven"

        # Completion only -- no /command anywhere in this test.
        h.publish_mission_complete()
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        assert h.node.target_pressure_pa is None
        h.spin_for(0.2)  # let the one-shot neutral land
        assert h.received_pitch_mm and h.received_pitch_mm[-1] == 0
        assert h.received_roll_cdeg and h.received_roll_cdeg[-1] == 0

        # Loop gated off: attitude keeps flowing, nothing new goes out.
        h.received_pitch_mm.clear()
        h.received_roll_cdeg.clear()
        for _ in range(8):
            h.publish_current_attitude(gauge_pa=20000.0)
            h.spin_for(0.05)
        assert (
            h.received_pitch_mm == []
        ), f"pitch must stay silent after completion, got {h.received_pitch_mm}"
        assert (
            h.received_roll_cdeg == []
        ), f"roll must stay silent after completion, got {h.received_roll_cdeg}"

    def test_completion_false_is_ignored(self, acu_node_harness):
        # Bool(false) on MISSION_COMPLETE is not a completion; a running
        # mission must survive it.
        h = acu_node_harness
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 2, timeout=1.5)

        h.publish_mission_complete(False)
        h.spin_for(0.3)
        assert h.node.target_pressure_pa == pytest.approx(80000.0)
        assert h.received_pitch_mm[-1] != 0, "pitch must still be driven"


class TestStopIsEdgeTriggered:
    """A stop while already stopped must put NOTHING on the ACU wire.

    There is no arbiter on /acu/pitch + /acu/roll -- last writer wins at the
    STM -- and the UI sends /command=false before every manual command. Without
    the edge trigger each of those stops drops a neutral on top of the
    operator's pitch/roll and silently undoes the command they just sent. Note
    this is a stronger requirement than the BCU's version of the same guard:
    there the duplicate stop re-arms a safe-stop burst, here a single stray
    sample is already enough to move the mass shifter."""

    def _drive_then_stop(self, h):
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 2, timeout=1.5)
        h.publish_command(False)
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        h.spin_for(0.2)  # let the one legitimate neutral land

    def test_repeated_stop_emits_nothing(self, acu_node_harness):
        h = acu_node_harness
        self._drive_then_stop(h)
        h.received_pitch_mm.clear()
        h.received_roll_cdeg.clear()

        h.publish_command(False)
        h.spin_for(0.4)
        assert (
            h.received_pitch_mm == []
        ), f"a repeated stop must not re-emit pitch, got {h.received_pitch_mm}"
        assert (
            h.received_roll_cdeg == []
        ), f"a repeated stop must not re-emit roll, got {h.received_roll_cdeg}"

    def test_stop_does_not_clobber_a_manual_command(self, acu_node_harness):
        # The engageManual order the UI actually sends: stop, then the manual
        # setpoint. A second stop arrives before the NEXT manual command -- and
        # must not land a neutral between them. Modelled here by checking that
        # acu_node contributes nothing at all once stopped, so whatever
        # acu_debug put on the wire is still the last word.
        h = acu_node_harness
        self._drive_then_stop(h)
        h.received_pitch_mm.clear()
        h.received_roll_cdeg.clear()

        for _ in range(4):
            h.publish_command(False)
            h.publish_current_attitude(roll_deg=15.0, gauge_pa=20000.0)
            h.spin_for(0.1)
        assert h.received_pitch_mm == [] and h.received_roll_cdeg == [], (
            "acu_node must stay off the wire while stopped, leaving the manual "
            f"driver's value latched; got pitch={h.received_pitch_mm} "
            f"roll={h.received_roll_cdeg}"
        )

    def test_a_fresh_target_re_arms_the_stop(self, acu_node_harness):
        # The guard must not wedge the node: a new mission target makes the
        # next stop emit its neutral again.
        h = acu_node_harness
        self._drive_then_stop(h)
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: h.node.target_pressure_pa is not None, timeout=1.0)
        h.received_pitch_mm.clear()
        h.received_roll_cdeg.clear()

        h.publish_command(False)
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        h.spin_for(0.2)
        assert h.received_pitch_mm and h.received_pitch_mm[-1] == 0
        assert h.received_roll_cdeg and h.received_roll_cdeg[-1] == 0


class TestTeardownParksTheWire:
    """destroy_node emits a neutral, mirroring bcu_node's teardown safe-stop.

    The STM latches the last value it was sent and has no staleness watchdog,
    so a node exiting mid-mission (Ctrl-C, launch shutdown, crash-restart)
    would otherwise leave the mass shifter parked at the last commanded
    pitch/roll with nothing left running to move it."""

    def test_destroy_node_emits_neutral(self, acu_node_harness):
        h = acu_node_harness
        # A live mission with a non-neutral command on the wire.
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_current_attitude(gauge_pa=20000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 2, timeout=1.5)
        assert h.received_pitch_mm[-1] != 0, "precondition: pitch off neutral"
        h.received_pitch_mm.clear()
        h.received_roll_cdeg.clear()

        # Exit without any stop at all. The tester keeps spinning, so what the
        # node emitted on its way out is observable.
        h.destroy_node_under_test()
        h.spin_for(0.5)

        assert h.received_pitch_mm[-1:] == [
            0
        ], f"teardown must park pitch at neutral, got {h.received_pitch_mm}"
        assert h.received_roll_cdeg[-1:] == [
            0
        ], f"teardown must park roll at neutral, got {h.received_roll_cdeg}"
