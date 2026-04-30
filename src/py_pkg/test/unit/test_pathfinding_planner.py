"""Tier 1 unit tests for the pathfinding planner.

Pure geometry, no ROS spinning. Verifies plan_straight_segment:

* yaw is never commanded (roll=0, yaw=0 round-trip; quaternion has qx=qz=0),
* pitch is constant across each segment and equals atan2(dz, horiz),
* the final waypoint is exactly the goal,
* sampling step controls waypoint count,
* edge cases: degenerate (start≈goal), purely vertical (horiz≈0), invalid step.
"""

import math

import pytest

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.path.pathfinding import plan_straight_segment


def _rp(pose):
    """Extract (roll_rad, pitch_rad) from a Pose's quaternion."""
    q = pose.orientation
    return quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)


class TestStraightLineGeometry:
    def test_last_waypoint_is_goal(self):
        poses = plan_straight_segment((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), step=2.0)
        last = poses[-1]
        assert last.position.x == pytest.approx(10.0)
        assert last.position.y == pytest.approx(0.0)
        assert last.position.z == pytest.approx(0.0)

    def test_first_waypoint_is_one_step_from_start(self):
        # 10m segment with step=2 → first waypoint at (2, 0, 0)
        poses = plan_straight_segment((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), step=2.0)
        first = poses[0]
        assert first.position.x == pytest.approx(2.0)
        assert first.position.y == pytest.approx(0.0)
        assert first.position.z == pytest.approx(0.0)

    def test_waypoints_are_collinear(self):
        poses = plan_straight_segment((0.0, 0.0, 0.0), (10.0, 5.0, -3.0), step=1.0)
        # Each waypoint should lie on the straight line from start to goal.
        for p in poses:
            t = p.position.x / 10.0
            assert p.position.y == pytest.approx(5.0 * t, abs=1e-9)
            assert p.position.z == pytest.approx(-3.0 * t, abs=1e-9)

    def test_waypoint_count_scales_with_step(self):
        # 10m segment, step=2 → ceil(10/2) = 5 waypoints
        assert len(plan_straight_segment((0, 0, 0), (10, 0, 0), step=2.0)) == 5
        # step=4 → ceil(10/4) = 3 waypoints
        assert len(plan_straight_segment((0, 0, 0), (10, 0, 0), step=4.0)) == 3
        # step=10 → 1 waypoint at the goal
        assert len(plan_straight_segment((0, 0, 0), (10, 0, 0), step=10.0)) == 1

    def test_step_larger_than_segment_returns_single_waypoint_at_goal(self):
        poses = plan_straight_segment((0, 0, 0), (1, 0, 0), step=5.0)
        assert len(poses) == 1
        assert poses[0].position.x == pytest.approx(1.0)


class TestPitchFromGeometry:
    def test_horizontal_segment_has_zero_pitch(self):
        poses = plan_straight_segment((0, 0, 0), (10, 0, 0), step=2.0)
        for p in poses:
            _, pitch = _rp(p)
            assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_dive_at_35_degrees(self):
        # User's target scenario: pitch = -35° on the way down.
        # tan(35°) = dz/horiz; pick horiz=20, dz=-20*tan(35°).
        horiz = 20.0
        dz = -horiz * math.tan(math.radians(35.0))
        poses = plan_straight_segment((0, 0, 0), (horiz, 0, dz), step=2.0)
        for p in poses:
            _, pitch = _rp(p)
            assert math.degrees(pitch) == pytest.approx(-35.0, abs=1e-6)

    def test_climb_at_35_degrees(self):
        horiz = 20.0
        dz = horiz * math.tan(math.radians(35.0))
        poses = plan_straight_segment((0, 0, 0), (horiz, 0, dz), step=2.0)
        for p in poses:
            _, pitch = _rp(p)
            assert math.degrees(pitch) == pytest.approx(35.0, abs=1e-6)

    def test_pitch_constant_across_segment(self):
        poses = plan_straight_segment((0, 0, 0), (15.0, 7.0, -4.0), step=1.0)
        pitches = [_rp(p)[1] for p in poses]
        # All pitches in a single segment must agree to floating-point precision.
        for pitch in pitches[1:]:
            assert pitch == pytest.approx(pitches[0], abs=1e-12)

    def test_pitch_matches_atan2_dz_horiz(self):
        # Mixed XY direction must not change the pitch — it's atan2(dz, horiz)
        # over horiz = sqrt(dx^2 + dy^2), independent of the XY heading.
        dx, dy, dz = 6.0, 8.0, -5.0  # horiz = 10
        expected = math.atan2(dz, math.hypot(dx, dy))
        poses = plan_straight_segment((0, 0, 0), (dx, dy, dz), step=1.0)
        _, pitch = _rp(poses[-1])
        assert pitch == pytest.approx(expected, abs=1e-9)


class TestNoYawNoRoll:
    """Yaw is never commanded; roll is always 0."""

    @pytest.mark.parametrize(
        "goal",
        [
            (10.0, 0.0, -3.0),  # +X dive
            (-10.0, 0.0, -3.0),  # -X dive
            (0.0, 10.0, 2.0),  # +Y climb
            (5.0, -5.0, -2.0),  # diagonal dive
        ],
    )
    def test_quaternion_has_only_pitch_component(self, goal):
        # _pitch_to_quaternion produces qx=qz=0 (roll=yaw=0). The XY heading
        # of the segment is encoded in position progression, not orientation.
        poses = plan_straight_segment((0, 0, 0), goal, step=2.0)
        for p in poses:
            assert p.orientation.x == pytest.approx(0.0, abs=1e-12)
            assert p.orientation.z == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "goal",
        [
            (10.0, 0.0, -3.0),
            (-10.0, 0.0, -3.0),
            (0.0, 10.0, 2.0),
            (5.0, -5.0, -2.0),
        ],
    )
    def test_roll_round_trips_to_zero(self, goal):
        poses = plan_straight_segment((0, 0, 0), goal, step=2.0)
        for p in poses:
            roll, _ = _rp(p)
            assert roll == pytest.approx(0.0, abs=1e-9)


class TestEdgeCases:
    def test_start_equals_goal_returns_single_waypoint(self):
        poses = plan_straight_segment((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), step=2.0)
        assert len(poses) == 1
        assert poses[0].position.x == pytest.approx(1.0)
        assert poses[0].position.y == pytest.approx(2.0)
        assert poses[0].position.z == pytest.approx(3.0)
        _, pitch = _rp(poses[0])
        assert pitch == pytest.approx(0.0, abs=1e-12)

    def test_pure_vertical_segment_uses_zero_pitch(self):
        # horiz ≈ 0, dz ≠ 0: planner has no body-pitch authority, so pitch=0
        # and the BCU drives the descent.
        poses = plan_straight_segment((0, 0, 0), (0, 0, -10.0), step=2.0)
        for p in poses:
            _, pitch = _rp(p)
            assert pitch == pytest.approx(0.0, abs=1e-12)

    def test_pure_vertical_segment_interpolates_z(self):
        poses = plan_straight_segment((0, 0, 0), (0, 0, -10.0), step=2.0)
        zs = [p.position.z for p in poses]
        # Monotonically decreasing, last is -10.
        assert zs == sorted(zs, reverse=True)
        assert zs[-1] == pytest.approx(-10.0)

    def test_negative_step_rejected(self):
        with pytest.raises(ValueError):
            plan_straight_segment((0, 0, 0), (1, 0, 0), step=-1.0)

    def test_zero_step_rejected(self):
        with pytest.raises(ValueError):
            plan_straight_segment((0, 0, 0), (1, 0, 0), step=0.0)


class TestYoyoScenario:
    """A sinusoidal yo-yo encoded as a chain of straight segments.

    Mean depth 25 m, peak amplitude ±5 m, pitch = ±35°. Each leg is a single
    call to plan_straight_segment; chained together they form the yo-yo
    pattern the planner is meant to express.
    """

    def _leg(self, start, goal):
        return plan_straight_segment(start, goal, step=2.0)

    def test_yoyo_alternates_pitch_sign(self):
        amplitude = 5.0
        pitch_deg = 35.0
        # Each leg traverses 2*amplitude in z (peak↔trough), so the
        # corresponding horizontal distance for a 35° pitch is 2A/tan(35°).
        horiz = 2.0 * amplitude / math.tan(math.radians(pitch_deg))

        # Down — Up — Down — Up
        a = (0.0, 0.0, -25.0 + amplitude)
        b = (horiz, 0.0, -25.0 - amplitude)
        c = (2 * horiz, 0.0, -25.0 + amplitude)
        d = (3 * horiz, 0.0, -25.0 - amplitude)
        e = (4 * horiz, 0.0, -25.0 + amplitude)

        legs = [self._leg(a, b), self._leg(b, c), self._leg(c, d), self._leg(d, e)]
        pitches_deg = [math.degrees(_rp(leg[-1])[1]) for leg in legs]

        assert pitches_deg[0] == pytest.approx(-pitch_deg, abs=1e-6)
        assert pitches_deg[1] == pytest.approx(+pitch_deg, abs=1e-6)
        assert pitches_deg[2] == pytest.approx(-pitch_deg, abs=1e-6)
        assert pitches_deg[3] == pytest.approx(+pitch_deg, abs=1e-6)

    def test_yoyo_z_envelope_bounded_by_amplitude(self):
        amplitude = 5.0
        pitch_deg = 35.0
        # Each leg traverses 2*amplitude in z (peak↔trough), so the
        # corresponding horizontal distance for a 35° pitch is 2A/tan(35°).
        horiz = 2.0 * amplitude / math.tan(math.radians(pitch_deg))

        a = (0.0, 0.0, -25.0 + amplitude)
        b = (horiz, 0.0, -25.0 - amplitude)
        c = (2 * horiz, 0.0, -25.0 + amplitude)

        for leg in [self._leg(a, b), self._leg(b, c)]:
            for pose in leg:
                assert pose.position.z >= -25.0 - amplitude - 1e-9
                assert pose.position.z <= -25.0 + amplitude + 1e-9
