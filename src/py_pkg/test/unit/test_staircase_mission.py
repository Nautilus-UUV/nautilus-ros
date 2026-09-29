"""Tier 1 unit tests for StaircaseMission (py_pkg.path.missions.staircase).

Pins the ladder construction (`target_pa * i / n` steps + surface leg),
per-step arrival tolerance, surface termination, and `start()` re-arm
semantics. The pathfinding executor calls `update -> reference -> is_done`
once per tick; tests mirror that order.

Every step advance is a turn for the bang-bang BCU -- the new target sits
below the vehicle, so the depth error keeps one sign until the next
arrival. Arrival advances the leg on the same tick; there is no hold.
"""

import math

import pytest
from geometry_msgs.msg import Pose

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.path.missions.profile import (
    DESCEND_TOLERANCE_PA,
    SURFACE_THRESHOLD_PA,
    MissionState,
)
from py_pkg.path.missions.staircase import StaircaseMission


def _make_state(target_pa=200_000.0, angle_rad=0.4, n_steps=4):
    return MissionState(
        target_pressure_pa=target_pa,
        angle_rad=angle_rad,
        n_steps=n_steps,
    )


def _pitch_of(pose: Pose) -> float:
    _, pitch = quaternion_to_roll_pitch(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return pitch


def _run_to_surface_leg(m: StaircaseMission, steps_pa):
    """Arrive at every descent step, leaving the mission on the surface leg."""
    for step_pa in steps_pa:
        m.update(step_pa)  # inside the arrival band -> advance


class TestLadderTargets:
    """`start()` must build evenly spaced steps `target_pa * i / n` and a
    trailing surface leg."""

    def test_step_targets_equal_spacing(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        assert m._targets_pa == pytest.approx(
            [50_000.0, 100_000.0, 150_000.0, 200_000.0, 0.0]
        )

    def test_first_leg_setpoint_is_first_step(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, angle_rad=0.3, n_steps=4))
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(50_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.3)

    def test_deepest_step_is_target(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=180_000.0, n_steps=3))
        assert m._targets_pa[-2] == pytest.approx(180_000.0)


class TestPerStepArrival:
    """A down-leg arrives when the observed pressure enters the shared
    descend tolerance band, not before -- and advances on that same tick."""

    def test_no_arrival_before_tolerance(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        m.update(50_000.0 - DESCEND_TOLERANCE_PA - 1.0)
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(50_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.4)

    def test_arrival_at_tolerance_boundary_advances(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        # Exactly on the band edge — predicate uses `>=`, so this arrives,
        # and the setpoint is already the NEXT step.
        m.update(50_000.0 - DESCEND_TOLERANCE_PA)
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(100_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.4)

    def test_arrival_past_step_advances(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        # Overshoot — still counts as arriving at the *current* step.
        m.update(60_000.0)
        assert m.reference(0.0).position.z == pytest.approx(100_000.0)

    def test_steps_advance_one_per_arrival(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        for arrived, expected_next in (
            (50_000.0, 100_000.0),
            (100_000.0, 150_000.0),
            (150_000.0, 200_000.0),
            (200_000.0, 0.0),  # -> surface leg
        ):
            m.update(arrived)
            assert m.reference(0.0).position.z == pytest.approx(expected_next)

    def test_reference_is_time_invariant(self):
        # No timer left in the state machine: the setpoint depends only on
        # the leg, so replaying `reference` at any t gives the same pose.
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        m.update(50_000.0)
        poses = [m.reference(t) for t in (0.0, 1.0, 60.0, 600.0, 1e6)]
        assert {p.position.z for p in poses} == {100_000.0}


class TestSurfaceLegTermination:
    """The final leg ascends to the surface; `is_done` flips once the
    observed pressure reads at the surface."""

    def test_surface_leg_setpoint_and_pitch(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, angle_rad=0.4, n_steps=4))
        _run_to_surface_leg(m, [50_000.0, 100_000.0, 150_000.0, 200_000.0])
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)

    def test_done_at_surface_boundary(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        _run_to_surface_leg(m, [50_000.0, 100_000.0, 150_000.0, 200_000.0])
        assert m.is_done(0.0) is False
        m.update(SURFACE_THRESHOLD_PA + 1.0)  # not yet surfaced
        assert m.is_done(0.0) is False
        m.update(SURFACE_THRESHOLD_PA)  # `<=` predicate — boundary is done
        assert m.is_done(0.0) is True


class TestSingleStep:
    """`n_steps=1`: one dive to the operator target, then surface."""

    def test_dive_then_surface(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=150_000.0, n_steps=1))
        # Dive straight to the operator target.
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(150_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.4)
        # Arriving turns straight around.
        m.update(150_000.0)
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)
        m.update(0.0)
        assert m.is_done(0.0) is True

    def test_n_steps_zero_treated_as_one(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=150_000.0, n_steps=0))
        assert m._targets_pa == pytest.approx([150_000.0, 0.0])
        assert m.reference(0.0).position.z == pytest.approx(150_000.0)


class TestQuaternionUnitNorm:
    """Every published quaternion must be unit-norm (downstream ACU PID
    decodes orientation; non-unit quats produce silent angle errors)."""

    @pytest.mark.parametrize("angle", [0.0, 0.1, 0.5, 1.0, math.pi / 2])
    def test_all_phases_unit_quat(self, angle):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, angle_rad=angle, n_steps=1))
        descending = m.reference(0.0)
        m.update(200_000.0)
        ascending = m.reference(0.0)
        for pose in (descending, ascending):
            q = pose.orientation
            norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
            assert norm == pytest.approx(1.0, abs=1e-9)


class TestRestartRearms:
    """`start()` must fully re-arm the ladder, even after a completed run
    (regression guard: stale `_done` / `_leg` state)."""

    def test_start_after_completed_run(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=2))
        _run_to_surface_leg(m, [100_000.0, 200_000.0])
        m.update(0.0)
        assert m.is_done(0.0) is True
        m.start(_make_state(target_pa=90_000.0, angle_rad=0.2, n_steps=3))
        assert m.is_done(0.0) is False
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(30_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.2)
