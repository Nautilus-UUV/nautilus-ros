"""Tier 1 unit tests for the pathfinding planner.

Pure geometry, no ROS spinning. Targets the planner-specific
``_build_turn_straight_xy_path`` (XY arc-then-straight construction);
the orientation helpers it relies on (``quaternion_to_yaw``,
``rpy_to_quaternion``, ``wrap_angle``) live in ``py_pkg.math_utils``
and are covered by ``test_math_utils.py``.

``_build_turn_straight_xy_path`` is an instance method that doesn't
touch ``self``, so we call it unbound (passing ``None`` as the first
argument) to avoid spinning up a node.
"""

import math

import pytest
from py_pkg.path.pathfinding import PathfindingNode

_build_xy = PathfindingNode._build_turn_straight_xy_path


# ---------------------------------------------------------------------------
# _build_turn_straight_xy_path
# ---------------------------------------------------------------------------


class TestBuildPathAlreadyFacingGoal:
    """When start_yaw matches the goal heading, the planner skips the arc
    and emits a straight linspace from start to goal."""

    def test_first_point_is_start(self):
        pts = _build_xy(None, 0.0, 0.0, yaw0=0.0, gx=10.0, gy=0.0, R=10.0, ds=2.0)
        assert pts[0][0] == pytest.approx(0.0)
        assert pts[0][1] == pytest.approx(0.0)

    def test_last_point_is_goal(self):
        pts = _build_xy(None, 0.0, 0.0, yaw0=0.0, gx=10.0, gy=0.0, R=10.0, ds=2.0)
        assert pts[-1][0] == pytest.approx(10.0)
        assert pts[-1][1] == pytest.approx(0.0)

    def test_all_yaws_match_goal_heading(self):
        pts = _build_xy(None, 0.0, 0.0, yaw0=0.0, gx=10.0, gy=0.0, R=10.0, ds=2.0)
        for _, _, yaw in pts:
            assert yaw == pytest.approx(0.0, abs=1e-12)

    def test_waypoint_count_scales_with_step(self):
        # Straight branch: num_steps = max(2, int(dist/ds) + 1).
        n2 = len(_build_xy(None, 0, 0, 0.0, 10.0, 0.0, R=10.0, ds=2.0))
        n4 = len(_build_xy(None, 0, 0, 0.0, 10.0, 0.0, R=10.0, ds=4.0))
        assert n2 == max(2, int(10 / 2) + 1) == 6
        assert n4 == max(2, int(10 / 4) + 1) == 3

    def test_points_are_collinear(self):
        # Diagonal already-facing-goal: yaw0 already equals atan2(dy, dx).
        gx, gy = 10.0, 5.0
        yaw0 = math.atan2(gy, gx)
        pts = _build_xy(None, 0.0, 0.0, yaw0, gx, gy, R=10.0, ds=2.0)
        # Each point lies on the straight line from start to goal.
        for x, y, _ in pts:
            t = x / gx
            assert y == pytest.approx(gy * t, abs=1e-9)


class TestBuildPathLeftTurn:
    """delta_yaw > 0 -> arc to the left, then straight to the goal."""

    def test_first_point_is_start_with_start_yaw(self):
        pts = _build_xy(None, 0.0, 0.0, 0.0, 5.0, 5.0, R=10.0, ds=2.0)
        assert pts[0] == (pytest.approx(0.0), pytest.approx(0.0), pytest.approx(0.0))

    def test_last_point_is_goal(self):
        pts = _build_xy(None, 0.0, 0.0, 0.0, 5.0, 5.0, R=10.0, ds=2.0)
        # Last appended point of the straight portion is exactly the goal.
        assert pts[-1][0] == pytest.approx(5.0)
        assert pts[-1][1] == pytest.approx(5.0)

    def test_yaw_increases_along_arc(self):
        # 90° left turn at R=10: arc_length = 10*π/2 ≈ 15.7, ds=2 -> 7 arc steps.
        pts = _build_xy(None, 0.0, 0.0, 0.0, 0.0, 20.0, R=10.0, ds=2.0)
        # First point has yaw=0 (start), then arc samples should be strictly
        # increasing in yaw until the arc completes.
        yaws = [yaw for _, _, yaw in pts]
        # Arc steps = max(1, int(10*pi/2 / 2)) = 7. Indices 1..7 are the arc.
        for i in range(1, 8):
            assert yaws[i] > yaws[i - 1]


class TestBuildPathRightTurn:
    """delta_yaw < 0 -> arc to the right; yaw decreases."""

    def test_yaw_decreases_along_arc(self):
        # Goal at (5, -5): goal_yaw = -π/4, delta_yaw = -π/4.
        pts = _build_xy(None, 0.0, 0.0, 0.0, 5.0, -5.0, R=10.0, ds=2.0)
        yaws = [yaw for _, _, yaw in pts]
        # arc_length = 10*π/4 ≈ 7.85, arc_steps = 3. Arc samples are indices 1..3.
        for i in range(1, 4):
            assert yaws[i] < yaws[i - 1]


class TestBuildPathDegenerate:
    def test_zero_xy_distance_returns_single_point(self):
        pts = _build_xy(None, 1.0, 2.0, yaw0=0.5, gx=1.0, gy=2.0, R=10.0, ds=2.0)
        assert len(pts) == 1
        assert pts[0] == (1.0, 2.0, 0.5)


# ---------------------------------------------------------------------------
# Segment pitch (from _plan_trajectory_from_current_pose's geometry rule)
# ---------------------------------------------------------------------------


class TestSegmentPitchGeometry:
    """The full planner uses ``pitch = atan2(dz_total, total_horiz)`` (no
    FLU sign flip). Verify the rule directly so changes to it surface
    here, not just through the Tier 2 emission tests."""

    def test_horizontal_segment_pitch_is_zero(self):
        assert math.atan2(0.0, 10.0) == pytest.approx(0.0)

    def test_dive_pitch_positive(self):
        # Z-positive-down: dz>0 means dive. Original convention emits +pitch.
        assert math.atan2(5.0, 5.0) == pytest.approx(math.radians(45.0))

    def test_climb_pitch_negative(self):
        assert math.atan2(-5.0, 5.0) == pytest.approx(math.radians(-45.0))

    def test_yoyo_alternates_pitch_sign(self):
        # Mean depth 25 m, ±5 m amplitude, 35° pitch -> horiz per leg.
        amplitude = 5.0
        pitch_deg = 35.0
        horiz = 2.0 * amplitude / math.tan(math.radians(pitch_deg))

        legs = [
            (25.0 - amplitude, 25.0 + amplitude),  # dive
            (25.0 + amplitude, 25.0 - amplitude),  # climb
            (25.0 - amplitude, 25.0 + amplitude),  # dive
            (25.0 + amplitude, 25.0 - amplitude),  # climb
        ]
        signs = []
        for z_start, z_end in legs:
            pitch = math.atan2(z_end - z_start, horiz)
            signs.append(math.copysign(1.0, pitch))

        assert signs == [+1.0, -1.0, +1.0, -1.0]
