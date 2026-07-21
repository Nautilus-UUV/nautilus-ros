"""Tier 1 unit tests for SawtoothMission (py_pkg.path.missions.sawtooth).

Pins the hysteresis state machine, leg-boundary setpoint contract, resurface-count termination,
and `start()` re-arm semantics. The pathfinding executor calls
`update -> reference -> is_done` once per tick; tests mirror that order.
"""

import math

import pytest
from geometry_msgs.msg import Pose

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.path.missions.profile import MissionState
from py_pkg.path.missions.sawtooth import (
    DESCEND_TOLERANCE_PA,
    SURFACE_THRESHOLD_PA,
    SawtoothMission,
)


def _make_state(target_pa=200_000.0, angle_rad=0.4, n_resurfaces=2):
    return MissionState(
        target_pressure_pa=target_pa,
        angle_rad=angle_rad,
        n_resurfaces=n_resurfaces,
    )


def _pitch_of(pose: Pose) -> float:
    _, pitch = quaternion_to_roll_pitch(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return pitch


class TestStartInitialState:
    """`start()` must arm the descending leg with a zeroed resurface count,
    even if a previous mission left the state machine mid-cycle.
    """

    def test_starts_descending(self):
        m = SawtoothMission()
        m.start(_make_state())
        # First tick before any update: should be descending.
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(200_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.4)

    def test_resurface_count_starts_at_zero(self):
        m = SawtoothMission()
        m.start(_make_state(n_resurfaces=1))
        assert m.is_done(0.0) is False

    def test_restart_rearms_descend_after_ascend_state(self):
        # Drive to an ascending leg, then `start()` again — the new mission
        # must start descending (regression: stale `_descending=False`).
        m = SawtoothMission()
        m.start(_make_state())
        m.update(200_000.0)  # at depth -> flip to ascending
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)
        m.start(_make_state(target_pa=50_000.0, angle_rad=0.2, n_resurfaces=3))
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(50_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.2)


class TestDescendLegSetpoint:
    """While descending the executor must publish `z = target_pa` and a
    negative pitch of magnitude `angle_rad`."""

    def test_descend_setpoint_is_target_depth(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=150_000.0, angle_rad=0.3))
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(150_000.0)

    def test_descend_pitch_is_negative_angle(self):
        m = SawtoothMission()
        m.start(_make_state(angle_rad=0.5))
        assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.5)

    def test_zero_roll_and_yaw_on_descend(self):
        m = SawtoothMission()
        m.start(_make_state())
        pose = m.reference(0.0)
        roll, _ = quaternion_to_roll_pitch(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        assert roll == pytest.approx(0.0)


class TestAscendLegSetpoint:
    def test_ascend_setpoint_is_surface(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        m.update(200_000.0)  # trigger flip
        assert m.reference(0.0).position.z == pytest.approx(0.0)

    def test_ascend_pitch_is_positive_angle(self):
        m = SawtoothMission()
        m.start(_make_state(angle_rad=0.5))
        m.update(200_000.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.5)


class TestDescendToAscendTransition:
    """Hysteresis on the deep extremum: leg flips when current pressure
    enters the descend tolerance band, not before."""

    def test_no_flip_before_tolerance(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        # Just outside the band — must remain descending.
        m.update(200_000.0 - DESCEND_TOLERANCE_PA - 1.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)

    def test_flip_at_tolerance_boundary(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        # Exactly on the band edge — predicate uses `>=`, so this flips.
        m.update(200_000.0 - DESCEND_TOLERANCE_PA)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)

    def test_flip_past_target(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        # Overshoot — still flips, even past the target.
        m.update(220_000.0)
        assert m.reference(0.0).position.z == pytest.approx(0.0)


class TestAscendToDescendTransition:
    def test_no_flip_above_surface_threshold(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        m.update(200_000.0)  # now ascending
        m.update(SURFACE_THRESHOLD_PA + 1.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)

    def test_flip_at_surface_boundary(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        m.update(200_000.0)
        # `<=` predicate — the boundary itself flips back to descending.
        m.update(SURFACE_THRESHOLD_PA)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)

    def test_resurface_count_increments_on_resurface_only(self):
        # The count increments when ascending -> descending, NOT when
        # descending -> ascending. A full down-leg alone must not count.
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0, n_resurfaces=5))
        m.update(200_000.0)  # at depth -> ascending; count unchanged
        assert m._resurface_count == 0
        m.update(0.0)        # back at surface -> count++
        assert m._resurface_count == 1


class TestHysteresisInsideBands:
    """Inside either tolerance band the leg must hold — no oscillation
    on noisy pressure samples that straddle the threshold."""

    def test_descending_holds_through_shallow_noise(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        # Many shallow samples — none cross into the descend tolerance band.
        for p in (0.0, 5_000.0, 50_000.0, 100_000.0, 150_000.0):
            m.update(p)
            assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)

    def test_ascending_holds_through_deep_noise(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0))
        m.update(200_000.0)  # ascending
        for p in (199_000.0, 100_000.0, 50_000.0, 10_000.0):
            m.update(p)
            assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)


class TestTermination:
    """`is_done` is `count >= n_resurfaces`; the executor checks it AFTER
    `reference` so a single full cycle ends a 1-resurface mission."""

    def test_n_zero_is_immediately_done(self):
        m = SawtoothMission()
        m.start(_make_state(n_resurfaces=0))
        assert m.is_done(0.0) is True

    def test_n_one_done_after_single_cycle(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0, n_resurfaces=1))
        assert not m.is_done(0.0)
        m.update(200_000.0)        # depth reached, ascending
        assert not m.is_done(0.0)
        m.update(0.0)              # resurfaced, count=1
        assert m.is_done(0.0) is True

    def test_n_three_requires_three_full_cycles(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0, n_resurfaces=3))
        for cycle in range(3):
            m.update(200_000.0)
            m.update(0.0)
            expected_done = cycle == 2
            assert m.is_done(0.0) is expected_done


class TestReferenceIgnoresMissionT:
    """`mission_t` is unused — sawtooth is event-driven, not time-driven.
    Pinning this prevents a regression where someone wires a time-based
    schedule and breaks the threshold-crossing contract."""

    def test_reference_independent_of_t(self):
        m = SawtoothMission()
        m.start(_make_state())
        p0 = m.reference(0.0)
        p1 = m.reference(1e6)
        assert p0.position.z == p1.position.z
        assert _pitch_of(p0) == pytest.approx(_pitch_of(p1))


class TestQuaternionUnitNorm:
    """Every published quaternion must be unit-norm (downstream ACU PID
    decodes orientation; non-unit quats produce silent angle errors)."""

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, 1.0, math.pi / 2])
    def test_descend_quat_is_unit(self, angle):
        m = SawtoothMission()
        m.start(_make_state(angle_rad=angle))
        q = m.reference(0.0).orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        assert norm == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, 1.0, math.pi / 2])
    def test_ascend_quat_is_unit(self, angle):
        m = SawtoothMission()
        m.start(_make_state(angle_rad=angle))
        m.update(1e9)  # force ascend
        q = m.reference(0.0).orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        assert norm == pytest.approx(1.0, abs=1e-9)


def _make_dwell_state(dwell_s, target_pa=200_000.0, angle_rad=0.4, n_resurfaces=1):
    return MissionState(
        target_pressure_pa=target_pa,
        angle_rad=angle_rad,
        n_resurfaces=n_resurfaces,
        dwell_s=dwell_s,
    )


class TestDwellAtDepth:
    """`dwell_s > 0` station-keeps level at the deep extremum before the
    ascend leg. The timer runs from FIRST band entry with no restart on a
    bob out of the band (unlike SurfaceMission's surface dwell), so every
    hold is bounded at `dwell_s`."""

    def test_dwell_holds_target_depth_level(self):
        m = SawtoothMission()
        m.start(_make_dwell_state(dwell_s=30.0))
        m.update(200_000.0)  # entered the band -> dwelling
        pose = m.reference(100.0)
        assert pose.position.z == pytest.approx(200_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)

    def test_flip_to_ascend_exactly_at_dwell_elapse(self):
        m = SawtoothMission()
        m.start(_make_dwell_state(dwell_s=30.0))
        m.update(200_000.0)
        m.reference(100.0)  # first reference stamps the timer
        # Just short of the hold — still station-keeping.
        pose = m.reference(129.9)
        assert pose.position.z == pytest.approx(200_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)
        # `>=` predicate — the boundary itself flips to the ascend leg.
        pose = m.reference(130.0)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)

    def test_bob_during_dwell_does_not_restart_timer(self):
        m = SawtoothMission()
        m.start(_make_dwell_state(dwell_s=30.0))
        m.update(200_000.0)
        m.reference(100.0)  # timer stamped at t=100
        # Bob shallow of the band mid-hold — the timer must keep running
        # from first entry, so the flip still lands at t=130.
        m.update(200_000.0 - DESCEND_TOLERANCE_PA - 5_000.0)
        assert _pitch_of(m.reference(115.0)) == pytest.approx(0.0)
        assert m.reference(130.0).position.z == pytest.approx(0.0)

    def test_bob_during_dwell_does_not_trigger_resurface(self):
        # Even a full surface reading mid-hold must not count as a
        # resurface or terminate the mission — the state machine is frozen.
        m = SawtoothMission()
        m.start(_make_dwell_state(dwell_s=30.0, n_resurfaces=1))
        m.update(200_000.0)
        m.reference(100.0)
        m.update(0.0)
        assert m._resurface_count == 0
        assert m.is_done(115.0) is False
        pose = m.reference(115.0)
        assert pose.position.z == pytest.approx(200_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)

    def test_full_cycle_with_dwell_terminates(self):
        # depth -> hold -> ascend -> surface completes a 1-resurface mission.
        m = SawtoothMission()
        m.start(_make_dwell_state(dwell_s=30.0, n_resurfaces=1))
        m.update(200_000.0)
        m.reference(100.0)
        m.reference(130.0)  # dwell elapsed -> ascending
        m.update(0.0)  # resurfaced, count=1
        assert m._resurface_count == 1
        assert m.is_done(131.0) is True

    def test_dwell_quat_is_unit(self):
        m = SawtoothMission()
        m.start(_make_dwell_state(dwell_s=30.0))
        m.update(200_000.0)
        q = m.reference(0.0).orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_dwell_zero_flips_immediately(self):
        # `dwell_s=0` (the `MissionState` default) preserves the historical
        # contract: band entry flips straight to the ascend leg with no
        # station-keep tick in between.
        m = SawtoothMission()
        m.start(_make_state())  # default dwell_s=0.0
        m.update(200_000.0)
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)

    def test_dwell_zero_full_cycle_unchanged(self):
        m = SawtoothMission()
        m.start(_make_state(target_pa=200_000.0, n_resurfaces=1))
        assert not m.is_done(0.0)
        m.update(200_000.0)  # depth reached, ascending
        assert not m.is_done(0.0)
        m.update(0.0)  # resurfaced, count=1
        assert m.is_done(0.0) is True
