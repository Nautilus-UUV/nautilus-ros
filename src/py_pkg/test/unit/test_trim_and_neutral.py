"""Tier 1 unit tests for TrimAndNeutralBuoyancyMission.

Pure logic — no rclpy, no executor, no sim. Pins the open-loop hold
contract: x/y from the start-time pose, z = `target_pressure_pa`,
identity orientation, never-done. The cascaded depth controller and the
ACU drive the actual hold; the mission only emits the constant setpoint.
"""

import math

import pytest
from geometry_msgs.msg import Pose

from py_pkg.path.missions.profile import MissionState
from py_pkg.path.missions.trim_and_neutral import TrimAndNeutralBuoyancyMission


def _pose(x=0.0, y=0.0, z=0.0):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.position.z = z
    p.orientation.w = 1.0
    return p


class TestStartCapturesTargetDepth:
    def test_target_pressure_used_as_z_setpoint(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=123_456.0))
        # `position.z` is gauge Pa per the MissionProfile contract; the
        # cascaded depth controller converts to meters downstream.
        assert m.reference(0.0).position.z == pytest.approx(123_456.0)

    def test_zero_target_is_surface_hold(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=0.0))
        assert m.reference(0.0).position.z == pytest.approx(0.0)


class TestStartCapturesHorizontalPosition:
    """`x`/`y` come from the start-time estimator pose so the ACU holds station
    instead of drifting (relevant once horizontal control lands)."""

    def test_pose_xy_captured(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(pose=_pose(x=10.0, y=-5.0, z=99.0), target_pressure_pa=80_000.0))
        ref = m.reference(0.0)
        assert ref.position.x == pytest.approx(10.0)
        assert ref.position.y == pytest.approx(-5.0)

    def test_pose_z_is_ignored(self):
        # The pose's z is meters; only `target_pressure_pa` (Pa) feeds z.
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(pose=_pose(z=42.0), target_pressure_pa=10_000.0))
        assert m.reference(0.0).position.z == pytest.approx(10_000.0)

    def test_pose_none_defaults_xy_to_zero(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(pose=None, target_pressure_pa=80_000.0))
        ref = m.reference(0.0)
        assert ref.position.x == pytest.approx(0.0)
        assert ref.position.y == pytest.approx(0.0)


class TestOrientationIsIdentity:
    """Identity quaternion -> roll = pitch = yaw = 0. The ACU drives mass
    to flat trim; any drift here would silently command an attitude."""

    def test_orientation_is_unit_w(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=80_000.0))
        q = m.reference(0.0).orientation
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)
        assert q.w == pytest.approx(1.0)

    def test_orientation_unit_norm(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=80_000.0))
        q = m.reference(0.0).orientation
        norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        assert norm == pytest.approx(1.0)


class TestUpdateIsNoop:
    """Open-loop hold: `update` never mutates state, so `reference` must
    return the same Pose regardless of pressure samples in between."""

    def test_pressure_samples_do_not_change_setpoint(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=80_000.0))
        before = m.reference(0.0)
        for p in (0.0, 50_000.0, 80_000.0, 200_000.0, -1.0, float("nan")):
            m.update(p)
        after = m.reference(0.0)
        assert after.position.z == before.position.z
        assert after.orientation.w == before.orientation.w


class TestReferenceIsTimeInvariant:
    """No internal clock — `mission_t` is irrelevant to the setpoint."""

    def test_reference_constant_in_t(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(pose=_pose(x=3.0, y=4.0), target_pressure_pa=80_000.0))
        for t in (0.0, 1.0, 60.0, 3600.0, 1e9):
            ref = m.reference(t)
            assert ref.position.x == pytest.approx(3.0)
            assert ref.position.y == pytest.approx(4.0)
            assert ref.position.z == pytest.approx(80_000.0)
            assert ref.orientation.w == pytest.approx(1.0)


class TestNeverDone:
    """The mission ends only on operator stop/abort — `is_done` is
    permanently False so the executor never auto-terminates the hold."""

    def test_is_done_false_at_t_zero(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=80_000.0))
        assert m.is_done(0.0) is False

    @pytest.mark.parametrize("t", [0.0, 60.0, 3600.0, 86_400.0, 1e9])
    def test_is_done_false_for_all_t(self, t):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=80_000.0))
        assert m.is_done(t) is False


class TestRestartUpdatesSetpoint:
    """Calling `start()` again must adopt the new operator command — a
    second mission with a deeper target overrides the first."""

    def test_second_start_replaces_target(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(target_pressure_pa=50_000.0))
        m.start(MissionState(target_pressure_pa=200_000.0))
        assert m.reference(0.0).position.z == pytest.approx(200_000.0)

    def test_second_start_replaces_xy(self):
        m = TrimAndNeutralBuoyancyMission()
        m.start(MissionState(pose=_pose(x=1.0, y=2.0), target_pressure_pa=10_000.0))
        m.start(MissionState(pose=_pose(x=-7.0, y=11.0), target_pressure_pa=10_000.0))
        ref = m.reference(0.0)
        assert ref.position.x == pytest.approx(-7.0)
        assert ref.position.y == pytest.approx(11.0)
