"""Tier 2 tests for ACUControlNode (bang-bang pitch + PID roll).

Pitch ignores attitude entirely now — it's a pressure-error bang-bang
publishing one of two ``Int16`` mm extremes pulled from
``AcuPitchSpec.output_limits``. Roll keeps the same PID/AxisController
contract on ``ACU_ROLL`` (Int16 centidegrees).

The harness publishes:
* ``POSITION_TARGET``  — orientation = roll target, position.z = target
  gauge pressure (Pa, pathfinding's TRIM convention).
* ``POSITION_ESTIMATION`` — orientation = current roll. Pitch dimension
  is deliberately ignored by the node.
* ``EXTERNAL_PRESSURE`` — absolute pressure sensor reading (Pa, Int32).
"""

import pytest

from py_pkg.physics import ATMOSPHERIC_PRESSURE_PA
from py_pkg.robot_specs import ACU_ROLL_CDEG_PER_DEG, ACU_ROLL_MAX_ANGLE_DEG
from py_pkg.scenarios.spec.control import AcuPitchSpec

# Bang-bang pitch wire values, mirrored from acu_node.py. Index 0 is the
# "back" extreme (selected when shallower than setpoint), index 1 is the
# "front" extreme (selected when deeper than or equal to setpoint).
_PITCH_OUTPUT_LIMITS_M = AcuPitchSpec().output_limits
PITCH_BACK_MM = int(round(_PITCH_OUTPUT_LIMITS_M[0] * 1000.0))
PITCH_FRONT_MM = int(round(_PITCH_OUTPUT_LIMITS_M[1] * 1000.0))

ROLL_MAX_CDEG = int(round(ACU_ROLL_MAX_ANGLE_DEG * ACU_ROLL_CDEG_PER_DEG))


def _abs_pa_for_gauge(gauge_pa: float) -> int:
    """Build an EXTERNAL_PRESSURE (absolute Pa) reading that, after the
    node's `gauge_pressure_pa` ingress conversion, lands at ``gauge_pa``.
    """
    return int(round(gauge_pa + ATMOSPHERIC_PRESSURE_PA))


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

    def test_external_pressure_subscription_present(self, acu_node_harness):
        names = [sub.topic_name for sub in acu_node_harness.node.subscriptions]
        assert "/external/pressure" in names

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
    """``POSITION_ESTIMATION`` orientation drives current_roll_deg only —
    pitch deliberately ignores the pose estimate (the bang-bang loop
    reads pressure directly)."""

    def test_estimation_pose_updates_current_roll(self, acu_node_harness):
        h = acu_node_harness
        h.publish_current_attitude(roll_deg=8.0)
        h.spin_until(
            lambda: h.node.current_roll_deg == pytest.approx(8.0, abs=1e-4),
            timeout=1.0,
        )
        assert h.node.current_roll_deg == pytest.approx(8.0, abs=1e-4)


class TestPressureIngress:
    """``EXTERNAL_PRESSURE`` is absolute Pa on the wire; the node converts
    to gauge before storing. Without that conversion the bang-bang would
    never flip legs against gauge-frame setpoints."""

    def test_pressure_message_stored_as_gauge(self, acu_node_harness):
        h = acu_node_harness
        gauge_target = 50000.0
        h.publish_external_pressure(_abs_pa_for_gauge(gauge_target))
        h.spin_until(
            lambda: h.node.current_pressure_pa is not None,
            timeout=1.0,
        )
        assert h.node.current_pressure_pa == pytest.approx(gauge_target, abs=1e-3)


class TestTimerEmits:
    """Once both inputs (pressure + target) have been seen, the 10 Hz
    loop publishes pitch and a non-trivial roll error drives roll too."""

    def test_emits_pitch_after_pressure_and_target(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=50000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(20000.0))
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
    waits for both EXTERNAL_PRESSURE and POSITION_TARGET."""

    def test_no_pitch_until_pressure_seen(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=50000.0)
        # Drive only the target — pressure never arrives.
        h.spin_for(0.6)
        assert h.received_pitch_mm == []

    def test_no_pitch_until_target_seen(self, acu_node_harness):
        h = acu_node_harness
        h.publish_external_pressure(_abs_pa_for_gauge(20000.0))
        # Drive only pressure — target never arrives.
        h.spin_for(0.6)
        assert h.received_pitch_mm == []


class TestPitchBangBang:
    """Output is one of two extremes from ``AcuPitchSpec.output_limits``;
    it never lands anywhere in between regardless of error magnitude."""

    def test_shallower_than_target_emits_back(self, acu_node_harness):
        # current gauge < target gauge → diving leg → BACK extreme.
        h = acu_node_harness
        h.publish_target(target_pressure_pa=80000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(20000.0))
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert h.received_pitch_mm[-1] == PITCH_BACK_MM

    def test_deeper_than_target_emits_front(self, acu_node_harness):
        h = acu_node_harness
        h.publish_target(target_pressure_pa=20000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(80000.0))
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert h.received_pitch_mm[-1] == PITCH_FRONT_MM

    def test_pitch_only_takes_two_values(self, acu_node_harness):
        # Even a small error keeps pitch railed — that's the whole point
        # of bang-bang. No proportional region, no clamp story.
        h = acu_node_harness
        h.publish_target(target_pressure_pa=50000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(50000.0 - 100.0))
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
        h.publish_external_pressure(_abs_pa_for_gauge(50000.0))

        h.publish_target(target_pressure_pa=80000.0)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        first = h.received_pitch_mm[-1]

        h.publish_target(target_pressure_pa=20000.0)
        h.spin_until(
            lambda: len(h.received_pitch_mm) >= 1
            and h.received_pitch_mm[-1] != first,
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
            assert abs(v) <= ROLL_MAX_CDEG, (
                f"published roll {v} cdeg exceeds ROLL_MAX_CDEG={ROLL_MAX_CDEG}"
            )
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
        assert late_count - early_count <= 1, (
            f"expected quiescence after saturation, got {h.received_roll_cdeg}"
        )


class TestManualOverride:
    """CONTROL_ACU_OVERRIDE silences the controller so acu_debug can own
    /acu/pitch and /acu/roll without the loop racing it on the wire."""

    def test_override_suppresses_pitch_and_roll(self, acu_node_harness):
        h = acu_node_harness
        h.publish_acu_override(True)
        # Let the flag land before the inputs that would otherwise drive a tick.
        h.spin_for(0.1)
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(20000.0))
        h.spin_for(0.6)
        assert h.received_pitch_mm == []
        assert h.received_roll_cdeg == []

    def test_release_resumes_publishing(self, acu_node_harness):
        h = acu_node_harness
        h.publish_acu_override(True)
        h.spin_for(0.1)
        h.publish_target(roll_deg=20.0, target_pressure_pa=80000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(20000.0))
        h.spin_for(0.4)
        assert h.received_pitch_mm == []

        h.publish_acu_override(False)
        h.spin_until(lambda: len(h.received_pitch_mm) >= 1, timeout=1.5)
        assert len(h.received_pitch_mm) >= 1


class TestControlReset:
    """CONTROL_RESET drops the held target and re-primes the roll axis, so
    the bang-bang pitch loop re-gates on a fresh target -- exactly as at
    boot before any mission (Do-Nothing mission)."""

    def test_reset_clears_target_and_gates_pitch_off(self, acu_node_harness):
        h = acu_node_harness
        # Drive a pitch command first (current shallower than target -> diving).
        h.publish_target(target_pressure_pa=80000.0)
        h.publish_external_pressure(_abs_pa_for_gauge(20000.0))
        h.spin_until(lambda: len(h.received_pitch_mm) >= 2, timeout=1.5)

        h.publish_reset()
        h.spin_until(lambda: h.node.target_pressure_pa is None, timeout=1.0)
        assert h.node.target_pressure_pa is None

        # With the target cleared, the bang-bang pitch loop is gated off: no
        # new pitch emissions even though pressure is still flowing.
        pitch_before = len(h.received_pitch_mm)
        h.spin_for(0.4)
        assert len(h.received_pitch_mm) == pitch_before, (
            f"pitch must stay silent after reset, got "
            f"{h.received_pitch_mm[pitch_before:]}"
        )
