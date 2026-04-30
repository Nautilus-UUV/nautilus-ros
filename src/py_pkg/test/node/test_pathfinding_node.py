"""Tier 2 in-process rclpy tests for PathfindingNode.

Black-box: drive the node via published POSITION_ESTIMATION / COMMAND /
PATH messages and assert on what it publishes on POSITION_TARGET.

The planner emits the first POSITION_TARGET synchronously inside the
``start`` command callback (no timer wait), so most happy-path assertions
resolve in well under a second. Timer-driven advance tests still need
``timer_dt = 0.5s``-class spin time to see the next emission.
"""

import math

import pytest

from py_pkg.math_utils import quaternion_to_roll_pitch


def _rp(pose):
    q = pose.orientation
    return quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)


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
        # path_step=2.0 over a 10m segment: first emission is one step in.
        assert first.position.x == pytest.approx(2.0, abs=1e-6)
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
        dz = -horiz * math.tan(math.radians(35.0))

        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(horiz, 0.0, dz)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        roll, pitch = _rp(h.received_targets[0])
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert math.degrees(pitch) == pytest.approx(-35.0, abs=1e-6)


class TestNoYawCommanded:
    """Pose orientation never carries a yaw component."""

    def test_quaternion_qx_qz_are_zero(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        # XY-diagonal segment with a Z component would produce non-zero yaw
        # in the OLD planner; the new planner must not.
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(5.0, 5.0, -3.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)

        for pose in h.received_targets:
            assert pose.orientation.x == pytest.approx(0.0, abs=1e-12)
            assert pose.orientation.z == pytest.approx(0.0, abs=1e-12)


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
        # First target is (2, 0, 0). Now claim we're there.
        h.publish_pose_estimation(2.0, 0.0, 0.0)
        # timer_dt = 0.5s — give it a couple of cycles to fire.
        h.spin_until(lambda: len(h.received_targets) >= 2, timeout=2.5)

        second = h.received_targets[1]
        assert second.position.x == pytest.approx(4.0, abs=1e-6)


class TestStopCommand:
    def test_stop_freezes_mode_no_advance(self, pathfinding_node_harness):
        h = pathfinding_node_harness
        h.publish_pose_estimation(0.0, 0.0, 0.0)
        h.publish_path([(10.0, 0.0, 0.0)])
        h.spin_until(
            lambda: h.node.current_pose is not None and len(h.node.keypoints) == 1,
            timeout=1.0,
        )
        h.publish_command("start")
        h.spin_until(lambda: len(h.received_targets) >= 1, timeout=1.0)
        emissions_after_start = len(h.received_targets)

        h.publish_command("stop")
        h.spin_until(lambda: h.node.mode == "STOPPED", timeout=1.0)
        # Even if EKF says we reached the first waypoint, no advance fires.
        h.publish_pose_estimation(2.0, 0.0, 0.0)
        h.spin_for(1.5)
        assert len(h.received_targets) == emissions_after_start


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
