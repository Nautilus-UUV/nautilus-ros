"""Tier 2 in-process rclpy tests for PathfindingNode.

Black-box: drive the node via published POSITION_ESTIMATION / COMMAND /
PATH messages and assert on what it publishes on POSITION_TARGET.

The planner emits the first POSITION_TARGET synchronously inside the
``start`` command callback (no timer wait), so most happy-path assertions
resolve in well under a second. Timer-driven advance tests still need
``timer_dt = 0.5s``-class spin time to see the next emission.

Targets the turn-then-straight (Dubins-style) planner: each segment
starts with the EKF pose itself (so the first emission is at the start
position), pitch is ``atan2(dz, horiz)`` (no FLU sign flip), and yaw is
commanded along the arc + straight portion.
"""

import math

import pytest

from py_pkg.math_utils import quaternion_to_roll_pitch


def _rp(pose):
    q = pose.orientation
    return quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)


def _yaw(pose):
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TestWiringSmoke:
    def test_node_constructs(self, pathfinding_node_harness):
        assert pathfinding_node_harness.node is not None

    def test_position_estimation_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/position/estimation" in names

    def test_command_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/command" in names

    def test_path_subscription_present(self, pathfinding_node_harness):
        names = [s.topic_name for s in pathfinding_node_harness.node.subscriptions]
        assert "/path" in names

    def test_position_target_publisher_present(self, pathfinding_node_harness):
        names = [p.topic_name for p in pathfinding_node_harness.node.publishers]
        assert "/position/target" in names

    def test_initial_mode_is_idle(self, pathfinding_node_harness):
        assert pathfinding_node_harness.node.mode == "IDLE"


class TestPathIngress:
    def test_path_message_populates_keypoints(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_path([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        h.spin_until(lambda: len(h.node.keypoints) == 2, timeout=1.0)
        assert h.node.keypoints == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        assert h.node.current_keypoint_idx == 0

    def test_path_with_bad_length_is_rejected(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # First load a valid path.
        h.publish_path([(1.0, 2.0, 3.0)])
        h.spin_until(lambda: len(h.node.keypoints) == 1, timeout=1.0)
        # Then send a malformed one (length not a multiple of 3) directly.
        from std_msgs.msg import Float32MultiArray

        bad = Float32MultiArray()
        bad.data = [1.0, 2.0]  # length 2 — invalid
        h.tester.path_pub.publish(bad)
        h.spin_for(0.3)
        # Keypoints unchanged (still the prior valid path).
        assert h.node.keypoints == [(1.0, 2.0, 3.0)]


class TestStartCommand:
    def test_start_without_pose_emits_nothing(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_path([(10.0, 0.0, 0.0)])
        h.spin_until(lambda: len(h.node.keypoints) == 1, timeout=1.0)
        h.publish_command("start")
        h.spin_for(0.5)
        assert h.received_targets == []
        assert h.node.mode == "IDLE"

    def test_start_without_path_emits_nothing(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.spin_until(lambda: h.node.current_pose is not None, timeout=1.0)
        h.publish_command("start")
        h.spin_for(0.5)
        assert h.received_targets == []
        assert h.node.mode == "IDLE"

    def test_start_happy_path_emits_first_target(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(10.0, 0.0, 0.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        first = h.received_targets[0]
        # Dubins-style planner emits the start position itself as the first
        # waypoint (start_yaw=0 already faces the goal -> straight-line case,
        # whose linspace includes the start point).
        assert first.position.x == pytest.approx(0.0, abs=1e-6)
        assert first.position.y == pytest.approx(0.0, abs=1e-6)
        assert first.position.z == pytest.approx(0.0, abs=1e-6)
        assert h.node.mode == "RUNNING"


class TestPlannedPitchInEmissions:
    """The pitch baked into the published Pose matches segment geometry."""

    def test_dive_at_35_degrees_round_trips_through_emission(
        self, pathfinding_node_harness
    ):
        h = pathfinding_node_harness
        horiz = 20.0
        # World frame is Z-positive-down (dz>0 means dive). The Dubins
        # planner uses pitch = atan2(dz_total, total_horiz) with no sign
        # flip, so a 35° dive emits +35° pitch.
        dz = horiz * math.tan(math.radians(35.0))

        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(horiz, 0.0, dz)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        # Goal is straight ahead in XY (y=0), so yaw=0 throughout and
        # quaternion_to_roll_pitch reads pitch cleanly.
        roll, pitch = _rp(h.received_targets[0])
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert math.degrees(pitch) == pytest.approx(35.0, abs=1e-6)


class TestYawCommandedOnDiagonal:
    """Diagonal segments require a heading change; the planner commands yaw
    along the arc-then-straight construction."""

    def test_diagonal_segment_emits_nonzero_yaw(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # XY-diagonal segment with non-zero dz: start_yaw=0, goal_yaw=π/4,
        # so the planner runs the arc branch and emits a yaw-bearing
        # quaternion on every waypoint past the start.
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(5.0, 5.0, 3.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        # First emission is the start point with yaw=0; the arc kicks in
        # from the second waypoint onward.
        h.spin_until(lambda: len(h.received_targets) >= 2, timeout=2.5)

        yaw_second = _yaw(h.received_targets[1])
        assert abs(yaw_second) > 1e-3, (
            f"Expected non-zero yaw after the start point, got {yaw_second}"
        )


class TestEKFAdvance:
    """When the EKF reports we've reached the current waypoint, the timer advances."""

    def test_reaching_waypoint_emits_next_target(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(10.0, 0.0, 0.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        # First emission is the start point (0,0,0); EKF still claims
        # (0,0,0), so the next timer tick sees us at the waypoint and
        # advances to the next sample (dubins_step=2.0 along x).
        h.spin_until(lambda: len(h.received_targets) >= 2, timeout=2.5)

        second = h.received_targets[1]
        assert second.position.x == pytest.approx(2.0, abs=1e-6)


class TestStopCommand:
    def test_stop_freezes_mode_no_advance(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(10.0, 0.0, 0.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        # Issue start + stop back-to-back so no timer tick can sneak an
        # advance in between (first waypoint is the start point itself,
        # which the timer would otherwise immediately advance past).
        h.publish_command("start")
        h.publish_command("stop")
        h.spin_until(lambda: h.node.mode == "STOPPED", timeout=1.0)
        # Drain any in-flight POSITION_TARGET emissions sitting in the
        # tester's subscription queue before snapshotting; once mode is
        # STOPPED the timer callback no-ops, so further emissions would
        # be a real advance.
        h.spin_for(0.6)
        emissions_after_stop = len(h.received_targets)

        h.spin_for(1.5)
        assert len(h.received_targets) == emissions_after_stop


class TestAbortCommand:
    def test_abort_clears_keypoints_and_trajectory(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 2,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        h.publish_command("abort")
        h.spin_until(lambda: h.node.mode == "ABORTED", timeout=1.0)

        assert h.node.keypoints == []
        assert h.node.trajectory_poses == []
        assert h.node.current_keypoint_idx is None
