"""Tier 1 unit tests for StaircaseMission (py_pkg.path.missions.staircase).

Pins the ladder construction (`target_pa * i / n` steps + surface leg),
per-step arrival tolerance, per-step dwell timing, surface termination,
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
)
from py_pkg.path.missions.staircase import StaircaseMission


def _make_state(target_pa=200_000.0, angle_rad=0.4, dwell_s=30.0, n_steps=4):
    return MissionState(
        target_pressure_pa=target_pa,
        angle_rad=angle_rad,
        dwell_s=dwell_s,
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


def _run_to_surface_leg(m: StaircaseMission, steps_pa, dwell_s=30.0, t0=0.0):
    """Arrive + dwell out every descent step; returns mission time on the
    final surface leg."""
    t = t0
    for step_pa in steps_pa:
        m.update(step_pa)  # inside the arrival band
        m.reference(t)  # stamps the dwell timer
        t += dwell_s
        m.reference(t)  # dwell elapsed -> next leg
    return t


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
    descend tolerance band, not before."""

    def test_no_arrival_before_tolerance(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        m.update(50_000.0 - DESCEND_TOLERANCE_PA - 1.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(-0.4)

    def test_arrival_at_tolerance_boundary(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        # Exactly on the band edge — predicate uses `>=`, so this arrives.
        m.update(50_000.0 - DESCEND_TOLERANCE_PA)
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(50_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)

    def test_arrival_past_step(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        # Overshoot — still arrives at the *current* step.
        m.update(60_000.0)
        assert _pitch_of(m.reference(0.0)) == pytest.approx(0.0)


class TestPerStepDwell:
    """Arrived steps station-keep level for `dwell_s`, timed from the
    first reference after band entry, then advance to the next leg."""

    def test_hold_until_dwell_elapses(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, dwell_s=30.0, n_steps=4))
        m.update(50_000.0)
        m.reference(100.0)  # stamps the timer
        pose = m.reference(129.9)
        assert pose.position.z == pytest.approx(50_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)

    def test_advance_exactly_at_dwell_elapse(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, angle_rad=0.4, n_steps=4))
        m.update(50_000.0)
        m.reference(100.0)
        # `>=` predicate — the boundary itself advances to step 2.
        pose = m.reference(130.0)
        assert pose.position.z == pytest.approx(100_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.4)

    def test_bob_during_hold_does_not_restart_timer(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        m.update(50_000.0)
        m.reference(100.0)  # timer stamped at t=100
        # Bob shallow of the band mid-hold — advance still lands at t=130.
        m.update(50_000.0 - DESCEND_TOLERANCE_PA - 5_000.0)
        assert _pitch_of(m.reference(115.0)) == pytest.approx(0.0)
        assert m.reference(130.0).position.z == pytest.approx(100_000.0)

    def test_each_step_dwells_independently(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, dwell_s=30.0, n_steps=2))
        m.update(100_000.0)
        m.reference(0.0)
        m.reference(30.0)  # -> step 2 (200 kPa)
        m.update(200_000.0)
        m.reference(50.0)  # second dwell stamps fresh at t=50
        pose = m.reference(79.9)
        assert pose.position.z == pytest.approx(200_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)
        # -> surface leg.
        pose = m.reference(80.0)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)


class TestSurfaceLegTermination:
    """The final leg ascends to the surface; `is_done` flips once the
    observed pressure reads at the surface."""

    def test_surface_leg_setpoint_and_pitch(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, angle_rad=0.4, n_steps=4))
        t = _run_to_surface_leg(m, [50_000.0, 100_000.0, 150_000.0, 200_000.0])
        pose = m.reference(t)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)

    def test_done_at_surface_boundary(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=4))
        t = _run_to_surface_leg(m, [50_000.0, 100_000.0, 150_000.0, 200_000.0])
        assert m.is_done(t) is False
        m.update(SURFACE_THRESHOLD_PA + 1.0)  # not yet surfaced
        assert m.is_done(t) is False
        m.update(SURFACE_THRESHOLD_PA)  # `<=` predicate — boundary is done
        assert m.is_done(t) is True


class TestSingleStep:
    """`n_steps=1` IS the station-keep profile: dive -> hold -> surface."""

    def test_dive_hold_surface(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=150_000.0, dwell_s=60.0, n_steps=1))
        # Dive straight to the operator target.
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(150_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.4)
        # Hold level for dwell_s.
        m.update(150_000.0)
        m.reference(10.0)
        pose = m.reference(69.9)
        assert pose.position.z == pytest.approx(150_000.0)
        assert _pitch_of(pose) == pytest.approx(0.0)
        # Surface and terminate.
        pose = m.reference(70.0)
        assert pose.position.z == pytest.approx(0.0)
        assert _pitch_of(pose) == pytest.approx(+0.4)
        m.update(0.0)
        assert m.is_done(80.0) is True

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
        for pose in (
            m.reference(0.0),  # descending
            self._arrive(m),  # holding
            m.reference(100.0),  # ascending (dwell elapsed)
        ):
            q = pose.orientation
            norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
            assert norm == pytest.approx(1.0, abs=1e-9)

    @staticmethod
    def _arrive(m: StaircaseMission) -> Pose:
        m.update(200_000.0)
        return m.reference(0.0)


class TestRestartRearms:
    """`start()` must fully re-arm the ladder, even after a completed run
    (regression guard: stale `_done` / `_leg` / dwell state)."""

    def test_start_after_completed_run(self):
        m = StaircaseMission()
        m.start(_make_state(target_pa=200_000.0, n_steps=2))
        t = _run_to_surface_leg(m, [100_000.0, 200_000.0])
        m.update(0.0)
        assert m.is_done(t) is True
        m.start(_make_state(target_pa=90_000.0, angle_rad=0.2, n_steps=3))
        assert m.is_done(0.0) is False
        pose = m.reference(0.0)
        assert pose.position.z == pytest.approx(30_000.0)
        assert _pitch_of(pose) == pytest.approx(-0.2)
