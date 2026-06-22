"""Tier 1 unit tests for SawtoothMission (py_pkg.path.missions.sawtooth).

Pins the two-pressure hysteresis state machine, the leg-boundary setpoint
contract (deep / shallow / final-surface legs), dive-count termination via a
final ascent to the surface, the shallow=0 backward-compat profile, and
`start()` re-arm semantics. The pathfinding executor calls
`update -> reference -> is_done` once per tick; tests mirror that order.
"""

import math

import pytest
from geometry_msgs.msg import Pose

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.path.missions.profile import MissionState
from py_pkg.path.missions.sawtooth import (
    DESCEND_TOLERANCE_PA,
    SHALLOW_TOLERANCE_PA,
    SURFACE_THRESHOLD_PA,
    SawtoothMission,
)

DEEP_PA = 200_000.0
SHALLOW_PA = 50_000.0


def _make_state(
    target_pa=DEEP_PA, shallow_pa=SHALLOW_PA, angle_rad=0.4, n_oscillations=2
):
    return MissionState(
        target_pressure_pa=target_pa,
        shallow_pressure_pa=shallow_pa,
        angle_rad=angle_rad,
        n_oscillations=n_oscillations,
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
    """`start()` must arm the descending leg with a zeroed dive count,
    even if a previous mission left the state machine mid-cycle.
    """

    def test_starts_descending(self):
        m = SawtoothMission()
        m.start(_make_state())
        # First tick before any update: should be descending to the deep extremum.
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(DEEP_PA)
        assert _pitch_of(pose) == pytest.approx(-0.4)

    def test_not_done_at_start(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=1))
        assert m.is_done(0.0) is False

    def test_restart_rearms_descend_after_ascend_state(self):
        # Drive to an ascending leg, then `start()` again — the new mission
        # must start descending (regression: stale `_descending=False`).
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=3))
        m.update(DEEP_PA)  # at depth -> flip to ascending
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)
        m.start(_make_state(target_pa=80_000.0, shallow_pa=20_000.0, angle_rad=0.2))
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(80_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.2)


class TestDescendLegSetpoint:
    """While descending the executor must publish `z = deep_pa` and a
    negative pitch of magnitude `angle_rad`."""

    def test_descend_setpoint_is_deep_extremum(self):
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
    """A non-final ascend climbs to the shallow extremum (not the surface)."""

    def test_ascend_setpoint_is_shallow_extremum(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=2))
        m.update(DEEP_PA)  # first dive done -> ascending toward shallow
        assert m.reference(0.0).position.z == pytest.approx(SHALLOW_PA)

    def test_ascend_pitch_is_positive_angle(self):
        m = SawtoothMission()
        m.start(_make_state(angle_rad=0.5, n_oscillations=2))
        m.update(DEEP_PA)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.5)


class TestDescendToAscendTransition:
    """Hysteresis on the deep extremum: leg flips when current pressure
    enters the descend tolerance band, not before."""

    def test_no_flip_before_tolerance(self):
        m = SawtoothMission()
        m.start(_make_state())
        # Just outside the band — must remain descending.
        m.update(DEEP_PA - DESCEND_TOLERANCE_PA - 1.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)

    def test_flip_at_tolerance_boundary(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=2))
        # Exactly on the band edge — predicate uses `>=`, so this flips.
        m.update(DEEP_PA - DESCEND_TOLERANCE_PA)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)
        assert m.reference(0.0).position.z == pytest.approx(SHALLOW_PA)

    def test_flip_past_deep_extremum(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=2))
        # Overshoot — still flips, even past the deep extremum.
        m.update(DEEP_PA + 20_000.0)
        assert m.reference(0.0).position.z == pytest.approx(SHALLOW_PA)


class TestAscendToDescendTransition:
    def test_no_flip_above_shallow_band(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=3))
        m.update(DEEP_PA)  # now ascending toward shallow
        m.update(SHALLOW_PA + SHALLOW_TOLERANCE_PA + 1.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)

    def test_flip_at_shallow_boundary(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=3))
        m.update(DEEP_PA)
        # `<=` predicate — the boundary itself flips back to descending.
        m.update(SHALLOW_PA + SHALLOW_TOLERANCE_PA)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)
        assert m.reference(0.0).position.z == pytest.approx(DEEP_PA)

    def test_dive_count_increments_on_reaching_deep_only(self):
        # The count increments when descending -> ascending (a dive lands),
        # NOT when ascending -> descending. A shallow turn alone must not count.
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=5))
        m.update(DEEP_PA)  # dive 1 lands
        assert m._dive_count == 1
        m.update(SHALLOW_PA)  # shallow turn -> descending; no increment
        assert m._dive_count == 1
        m.update(DEEP_PA)  # dive 2 lands
        assert m._dive_count == 2


class TestHysteresisInsideBands:
    """Inside either tolerance band the leg must hold — no oscillation
    on noisy pressure samples that straddle the threshold."""

    def test_descending_holds_through_shallow_noise(self):
        m = SawtoothMission()
        m.start(_make_state())
        # Many shallow samples — none cross into the descend tolerance band.
        for p in (0.0, 5_000.0, 60_000.0, 120_000.0, 180_000.0):
            m.update(p)
            assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)

    def test_ascending_holds_through_deep_noise(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=3))
        m.update(DEEP_PA)  # ascending toward shallow
        # All above the shallow band edge (SHALLOW_PA + tolerance = 55 000).
        for p in (199_000.0, 120_000.0, 60_000.0):
            m.update(p)
            assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)
            assert m.reference(0.0).position.z == pytest.approx(SHALLOW_PA)


class TestFinalSurfacing:
    """After the N-th dive the final ascend targets the surface (z = 0) and
    the mission completes only once the vehicle is back at the surface."""

    def test_final_leg_targets_surface_not_shallow(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=1))
        m.update(DEEP_PA)  # 1st (and last) dive -> surfacing leg
        assert m.reference(0.0).position.z == pytest.approx(0.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(+0.4)

    def test_not_done_until_surfaced(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=1))
        m.update(DEEP_PA)  # surfacing leg begins
        assert m.is_done(0.0) is False
        # Still descending through the surfacing ascent — not yet surfaced.
        m.update(SHALLOW_PA)
        assert m.is_done(0.0) is False
        m.update(SURFACE_THRESHOLD_PA)  # crossed the surface gate
        assert m.is_done(0.0) is True

    def test_two_oscillations_then_surface(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=2))
        m.update(DEEP_PA)  # dive 1 -> ascend to shallow
        assert m.reference(0.0).position.z == pytest.approx(SHALLOW_PA)
        m.update(SHALLOW_PA)  # shallow turn -> descend
        assert m.reference(0.0).position.z == pytest.approx(DEEP_PA)
        m.update(DEEP_PA)  # dive 2 (== N) -> surfacing leg
        assert m.reference(0.0).position.z == pytest.approx(0.0)
        assert not m.is_done(0.0)
        m.update(0.0)  # surfaced
        assert m.is_done(0.0) is True


class TestTermination:
    """`is_done` becomes true only after the final surfacing ascent reaches
    the surface gate, having completed `n_oscillations` dives first."""

    def test_n_zero_surfaces_immediately(self):
        # Zero dives requested -> go straight to the surfacing leg; done once
        # the vehicle is at the surface.
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=0))
        assert m.reference(0.0).position.z == pytest.approx(0.0)
        assert not m.is_done(0.0)
        m.update(0.0)
        assert m.is_done(0.0) is True

    def test_n_one_done_after_single_dive_and_surface(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=1))
        assert not m.is_done(0.0)
        m.update(DEEP_PA)  # dive lands -> surfacing
        assert not m.is_done(0.0)
        m.update(0.0)  # surfaced -> done
        assert m.is_done(0.0) is True

    def test_n_three_requires_three_dives_then_surface(self):
        m = SawtoothMission()
        m.start(_make_state(n_oscillations=3))
        # Dives 1 and 2: deep -> shallow turn, no completion.
        for _ in range(2):
            m.update(DEEP_PA)
            assert not m.is_done(0.0)
            m.update(SHALLOW_PA)
            assert not m.is_done(0.0)
        # Dive 3 reaches N -> surfacing leg; complete only once surfaced.
        m.update(DEEP_PA)
        assert not m.is_done(0.0)
        m.update(0.0)
        assert m.is_done(0.0) is True


class TestShallowZeroBackwardCompat:
    """`shallow_pressure_pa = 0` reproduces the legacy dive-to-surface
    profile: every climb goes to the surface, where it both turns and
    (on the final dive) completes."""

    def test_ascend_targets_surface_when_shallow_zero(self):
        m = SawtoothMission()
        m.start(_make_state(shallow_pa=0.0, n_oscillations=2))
        m.update(DEEP_PA)  # first dive -> non-final ascend
        assert m.reference(0.0).position.z == pytest.approx(0.0)

    def test_inverted_shallow_falls_back_to_surface(self):
        # shallow >= deep is degenerate -> clamp to the legacy 0 (surface).
        m = SawtoothMission()
        m.start(_make_state(target_pa=DEEP_PA, shallow_pa=DEEP_PA, n_oscillations=2))
        m.update(DEEP_PA)
        assert m.reference(0.0).position.z == pytest.approx(0.0)

    def test_negative_shallow_falls_back_to_surface(self):
        m = SawtoothMission()
        m.start(_make_state(shallow_pa=-10_000.0, n_oscillations=2))
        m.update(DEEP_PA)
        assert m.reference(0.0).position.z == pytest.approx(0.0)


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
        m.start(_make_state(angle_rad=angle, n_oscillations=3))
        m.update(DEEP_PA)  # force ascend
        q = m.reference(0.0).orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        assert norm == pytest.approx(1.0, abs=1e-9)
