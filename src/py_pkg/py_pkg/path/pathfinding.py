#!/usr/bin/env python3
"""Waypoint follower.

The glider has no yaw authority, so each segment is a straight 3D line
with roll=0, yaw=0, and pitch = atan2(dz, horiz). Downstream: depth_node
reads position.z, acu_node reads roll/pitch from the quaternion.
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


def _pitch_to_quaternion(pitch: float) -> tuple:
    # Quaternion for (roll=0, pitch, yaw=0), ZYX Tait-Bryan.
    half = 0.5 * pitch
    return 0.0, math.sin(half), 0.0, math.cos(half)


def _make_pose(x: float, y: float, z: float, pitch: float) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    qx, qy, qz, qw = _pitch_to_quaternion(pitch)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


def plan_straight_segment(
    start: tuple,
    goal: tuple,
    step: float,
) -> list:
    """Sample a straight line from ``start`` to ``goal`` every ``step`` m.

    World frame is Z-positive-down (deeper = larger z). Body-frame pitch
    follows REP-103 FLU (positive pitch = nose up), so the segment pitch
    is ``atan2(-dz, horiz)`` — diving (dz>0) maps to negative pitch
    (nose-down), climbing maps to positive pitch.

    First waypoint is one step in, last is exactly ``goal``. Pure-vertical
    or degenerate segments use pitch=0 (BCU drives those directly).
    """
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")

    sx, sy, sz = start
    gx, gy, gz = goal
    dx, dy, dz = gx - sx, gy - sy, gz - sz
    horiz = math.hypot(dx, dy)
    total = math.hypot(horiz, dz)

    if total < 1e-6:
        return [_make_pose(gx, gy, gz, pitch=0.0)]

    pitch = 0.0 if horiz < 1e-6 else math.atan2(-dz, horiz)
    num_steps = max(1, int(math.ceil(total / step)))

    poses = []
    for i in range(1, num_steps + 1):
        t = i / num_steps
        poses.append(_make_pose(sx + t * dx, sy + t * dy, sz + t * dz, pitch))
    return poses


class PathfindingNode(Node):
    """Mission-level waypoint follower.

    SUBSCRIBES:
      * POSITION_ESTIMATION (Pose) — current EKF pose.
      * COMMAND (String) — "start" / "stop" / "abort".
      * PATH (Float32MultiArray) — flattened keypoints
        ``[x1, y1, z1, x2, y2, z2, ...]``.

    PUBLISHES:
      * POSITION_TARGET (Pose) — current waypoint, sent once per
        advance event (waypoint reached or replan).
    """

    def __init__(self):
        super().__init__("pathfinding_node")

        self.speed = 0.4  # [m/s] assumed glide speed (timeout heuristic only)
        self.position_tolerance = 1.0  # [m] EKF acceptance radius
        self.segment_timeout_factor = 3.0
        self.timer_dt = 0.5  # [s] EKF/timeout check rate
        self.path_step = 2.0  # [m] sampling step along the segment

        self.state_sub = create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self._state_callback
        )
        self.command_sub = create_subscription_for_topic(
            self, UUVTopics.COMMAND, self._command_callback
        )
        self.path_sub = create_subscription_for_topic(
            self, UUVTopics.PATH, self._path_callback
        )
        self.position_target_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )

        self.keypoints: list = []
        self.current_keypoint_idx = None
        self.current_pose: Pose = None

        self.trajectory_poses: list = []
        self.current_index = 0

        self.last_waypoint_time = None
        self.last_segment_length = None

        self.mode = "IDLE"

        self.timer = self.create_timer(self.timer_dt, self._timer_callback)

        self.get_logger().info("PathfindingNode started")

    # ── subscribers ──────────────────────────────────────────────────────

    def _state_callback(self, msg: Pose):
        self.current_pose = msg

    def _path_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) % 3 != 0:
            self.get_logger().error(
                f"Received /path with length {len(data)}, not a multiple of 3."
            )
            return

        self.keypoints = [
            (float(data[i]), float(data[i + 1]), float(data[i + 2]))
            for i in range(0, len(data), 3)
        ]
        self.current_keypoint_idx = 0 if self.keypoints else None

        self.trajectory_poses = []
        self.current_index = 0
        self.last_waypoint_time = None
        self.last_segment_length = None

        self.get_logger().info(
            f"Received new path with {len(self.keypoints)} keypoints."
        )

    def _command_callback(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"Received /command: '{command}'")
        if command == "start":
            self._handle_start()
        elif command == "stop":
            self._handle_stop()
        elif command == "abort":
            self._handle_abort()
        else:
            self.get_logger().warn(
                f"Unknown command '{command}'. Expected 'start', 'stop', or 'abort'."
            )

    # ── command handlers ─────────────────────────────────────────────────

    def _handle_start(self):
        if self.current_pose is None:
            self.get_logger().warn("Cannot start: no current pose from EKF yet.")
            return
        if not self.keypoints:
            self.get_logger().warn("Cannot start: no keypoints received on /path.")
            return
        if self.current_keypoint_idx is None:
            self.current_keypoint_idx = 0

        self.mode = "RUNNING"
        self.get_logger().info("Mode set to RUNNING.")

        if not self.trajectory_poses:
            self._plan_trajectory_to_current_keypoint()

    def _handle_stop(self):
        self.mode = "STOPPED"
        self.get_logger().info("Mode set to STOPPED. Holding current target pose.")

    def _handle_abort(self):
        self.mode = "ABORTED"
        self.trajectory_poses = []
        self.current_index = 0
        self.last_waypoint_time = None
        self.last_segment_length = None
        self.keypoints = []
        self.current_keypoint_idx = None
        self.get_logger().info("Mode set to ABORTED. Trajectory cleared.")

    # ── planning ─────────────────────────────────────────────────────────

    def _plan_trajectory_to_current_keypoint(self):
        if self.current_pose is None:
            self.get_logger().warn("Cannot plan: no current pose.")
            return
        if (
            self.current_keypoint_idx is None
            or self.current_keypoint_idx >= len(self.keypoints)
        ):
            self.get_logger().warn("Cannot plan: invalid current_keypoint_idx.")
            return

        start = (
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z,
        )
        goal = self.keypoints[self.current_keypoint_idx]
        self.trajectory_poses = plan_straight_segment(start, goal, self.path_step)
        self.current_index = 0

        self.get_logger().info(
            f"Planned segment to keypoint {self.current_keypoint_idx} with "
            f"{len(self.trajectory_poses)} waypoints."
        )
        self._send_current_waypoint()

    def _send_current_waypoint(self):
        if not self.trajectory_poses:
            self.get_logger().warn("No trajectory poses to send.")
            return

        pose = self.trajectory_poses[self.current_index]
        self.position_target_pub.publish(pose)
        self.last_waypoint_time = self.get_clock().now()

        if self.current_index == 0:
            if self.current_pose is not None:
                prev = self.current_pose.position
                self.last_segment_length = math.dist(
                    (prev.x, prev.y, prev.z),
                    (pose.position.x, pose.position.y, pose.position.z),
                )
            else:
                self.last_segment_length = None
        else:
            prev_pose = self.trajectory_poses[self.current_index - 1]
            self.last_segment_length = math.dist(
                (prev_pose.position.x, prev_pose.position.y, prev_pose.position.z),
                (pose.position.x, pose.position.y, pose.position.z),
            )

        self.get_logger().info(
            f"Sent waypoint {self.current_index + 1}/{len(self.trajectory_poses)} "
            f"for keypoint {self.current_keypoint_idx}: "
            f"({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        )

    def _advance_to_next_waypoint(self):
        if not self.trajectory_poses:
            return

        if self.current_index < len(self.trajectory_poses) - 1:
            self.current_index += 1
            self._send_current_waypoint()
            return

        self.get_logger().info("Local trajectory to current keypoint completed.")

        if (
            self.keypoints
            and self.current_keypoint_idx is not None
            and self.current_keypoint_idx < len(self.keypoints) - 1
            and self.mode == "RUNNING"
        ):
            self.current_keypoint_idx += 1
            self.trajectory_poses = []
            self.current_index = 0
            self._plan_trajectory_to_current_keypoint()
        else:
            self.get_logger().info(
                "All keypoints completed or not running. Mission complete."
            )
            self.trajectory_poses = []
            self.current_index = 0
            self.last_waypoint_time = None
            self.last_segment_length = None

    # ── timer ────────────────────────────────────────────────────────────

    def _timer_callback(self):
        if self.mode != "RUNNING":
            return
        if self.current_pose is None:
            return
        if not self.trajectory_poses:
            return

        pose = self.trajectory_poses[self.current_index]
        cur = self.current_pose.position
        dist = math.dist(
            (cur.x, cur.y, cur.z),
            (pose.position.x, pose.position.y, pose.position.z),
        )

        if dist <= self.position_tolerance:
            self.get_logger().info(
                f"Waypoint reached via EKF (dist={dist:.2f} m <= "
                f"{self.position_tolerance:.2f} m)."
            )
            self._advance_to_next_waypoint()
            return

        if self.last_waypoint_time is not None:
            now = self.get_clock().now()
            elapsed = (now.nanoseconds - self.last_waypoint_time.nanoseconds) * 1e-9

            if self.last_segment_length is not None and self.last_segment_length > 1e-3:
                expected_time = self.last_segment_length / self.speed
            else:
                expected_time = self.timer_dt

            timeout = self.segment_timeout_factor * expected_time

            if elapsed > timeout:
                self.get_logger().warn(
                    f"Timeout on waypoint (elapsed={elapsed:.1f}s > {timeout:.1f}s). "
                    f"Replanning from current pose to current keypoint."
                )
                self._plan_trajectory_to_current_keypoint()


def main(args=None):
    rclpy.init(args=args)
    node = PathfindingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
