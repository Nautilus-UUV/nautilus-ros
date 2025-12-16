#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, String
from .uuv_ros_core import UUVTopics, UUVQoS, TOPIC_MESSAGE_MAP


class PathfindingNode(Node):
    """
    Pathfinding / trajectory following node for Nautilus.

    SUBSCRIBES:
      - /position/estimation   (UUVTopics.POSITION_ESTIMATION, Pose)
          Current estimated pose from EKF.
      - /command               (UUVTopics.COMMAND, String)
          High-level commands: "start", "stop", "abort".
      - /path                  (UUVTopics.PATH, Float32MultiArray)
          Flattened list of keypoints: [x1, y1, z1, x2, y2, z2, ...]

    PUBLISHES:
      - /position/target       (UUVTopics.POSITION_TARGET, Pose)
          Current target pose for the controller.
    """

    def __init__(self):
        super().__init__("pathfinding_node")
        self.speed = 0.4  # [m/s] assumed forward speed
        self.position_tolerance = 1.0  # [m] distance to accept waypoint as reached
        self.segment_timeout_factor = 3.0  # multiplier on expected time per segment
        self.timer_dt = 0.5  # [s] how often we check EKF vs. current waypoint
        self.turn_radius = 10.0  # [m] turning radius for the arc
        self.dubins_step = 2.0  # [m] step size along the path (arc + straight)

        # Current estimated pose (from EKF)
        self.state_sub = self.create_subscription(
            TOPIC_MESSAGE_MAP[UUVTopics.POSITION_ESTIMATION],  # Pose
            UUVTopics.POSITION_ESTIMATION,
            self._state_callback,
            UUVQoS.SENSOR_STREAM,
        )

        # High-level commands: "start", "stop", "abort"
        self.command_sub = self.create_subscription(
            TOPIC_MESSAGE_MAP[UUVTopics.COMMAND],  # String
            UUVTopics.COMMAND,
            self._command_callback,
            UUVQoS.COMMAND,
        )

        # Full path (flattened list of Poses)
        self.path_sub = self.create_subscription(
            TOPIC_MESSAGE_MAP[UUVTopics.PATH],  # Float32MultiArray
            UUVTopics.PATH,
            self._path_callback,
            UUVQoS.CONTROL,
        )

        # Next waypoint Pose (what control should aim at)
        self.position_target_pub = self.create_publisher(
            TOPIC_MESSAGE_MAP[UUVTopics.POSITION_TARGET],  # Pose
            UUVTopics.POSITION_TARGET,
            UUVQoS.CONTROL,
        )

        self.keypoints: list[tuple[float, float, float]] = []
        self.current_keypoint_idx: int | None = None

        self.current_pose: Pose = None

        # List of Pose waypoints along our trajectory
        self.trajectory_poses: list[Pose] = []
        self.current_index = 0

        # Timing / monitoring for each segment
        self.last_waypoint_time = None
        self.last_segment_length = None

        self.mode = "IDLE"

        # Timer to periodically check progress towards current waypoint
        self.timer = self.create_timer(self.timer_dt, self._timer_callback)

        self.get_logger().info("PathfindingNode started")

    def _state_callback(self, msg: Pose):
        """Receive the latest EKF pose estimate."""
        self.current_pose = msg

    def _path_callback(self, msg: Float32MultiArray):
        """
        Receive the path of keypoints from /path.

        msg.data = [x1, y1, z1, x2, y2, z2, ...]
        """
        data = list(msg.data)
        if len(data) % 3 != 0:
            self.get_logger().error(
                f"Received /path with length {len(data)}, not a multiple of 3."
            )
            return

        keypoints = []
        for i in range(0, len(data), 3):
            x = float(data[i])
            y = float(data[i + 1])
            z = float(data[i + 2])
            keypoints.append((x, y, z))

        self.keypoints = keypoints
        self.current_keypoint_idx = 0 if keypoints else None

        # When a new path arrives, clear existing local trajectory.
        self.trajectory_poses = []
        self.current_index = 0
        self.last_waypoint_time = None
        self.last_segment_length = None

        self.get_logger().info(
            f"Received new path with {len(self.keypoints)} keypoints from /path."
        )

    def _command_callback(self, msg: String):
        """Handle high-level commands: "start", "stop", "abort"."""
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

    def _handle_start(self):
        """Start or resume following the keypoints."""
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

        # If we don't have a local trajectory, plan one to the current keypoint.
        if not self.trajectory_poses:
            self._plan_trajectory_to_current_keypoint()

    def _handle_stop(self):
        """Stop advancing to new waypoints, but keep the current target pose."""
        self.mode = "STOPPED"
        self.get_logger().info("Mode set to STOPPED. Holding current target pose.")

    def _handle_abort(self):
        """Abort the current mission: clear trajectory and keypoints."""
        self.mode = "ABORTED"
        self.trajectory_poses = []
        self.current_index = 0
        self.last_waypoint_time = None
        self.last_segment_length = None
        # clear keypoints as well:
        self.keypoints = []
        self.current_keypoint_idx = None
        self.get_logger().info("Mode set to ABORTED. Trajectory cleared.")

    def _plan_trajectory_to_current_keypoint(self):
        """Plan a local trajectory from current_pose to the current keypoint in self.keypoints."""
        if self.current_pose is None:
            self.get_logger().warn("Cannot plan: no current pose.")
            return
        if self.current_keypoint_idx is None or self.current_keypoint_idx >= len(
            self.keypoints
        ):
            self.get_logger().warn("Cannot plan: invalid current_keypoint_idx.")
            return

        gx, gy, gz = self.keypoints[self.current_keypoint_idx]
        self._plan_trajectory_from_current_pose((gx, gy, gz))

    def _build_turn_straight_xy_path(self, x0, y0, yaw0, gx, gy, R, ds):
        """
        Build a turn straight path to the destination.

        Returns: list of (x, y, yaw) along the path.
        """
        points = []

        # Vector to goal in XY
        dx = gx - x0
        dy = gy - y0
        dist_xy = math.hypot(dx, dy)
        if dist_xy < 1e-6:
            # Already at goal horizontally
            return [(x0, y0, yaw0)]

        # the heading we need to turn to in order to go straight to the goal
        goal_yaw = math.atan2(dy, dx)

        # Normalize angle difference goal_yaw - yaw0 to [-pi, pi]
        def normalize_angle(angle):
            while angle > math.pi:
                angle -= 2.0 * math.pi
            while angle < -math.pi:
                angle += 2.0 * math.pi
            return angle

        # angle difference of where we are facing and where we should face
        delta_yaw = normalize_angle(goal_yaw - yaw0)

        # If already roughly facing the goal, skip the turn
        if abs(delta_yaw) < 1e-3:
            # Just do a straight line
            num_steps = max(2, int(dist_xy / ds) + 1)
            xs = np.linspace(x0, gx, num_steps)
            ys = np.linspace(y0, gy, num_steps)
            return [(float(x), float(y), goal_yaw) for x, y in zip(xs, ys)]

        # Otherwise, we do an arc first
        # if > 0 we need to turn left / if < 0 we need to turn right
        turn_dir = 1.0 if delta_yaw > 0.0 else -1.0
        # magnitude of the angle we need to rotate
        delta_yaw_abs = abs(delta_yaw)

        # Arc length and steps
        arc_length = R * delta_yaw_abs
        arc_steps = max(1, int(arc_length / ds))

        # start of the arc
        yaw = yaw0
        x = x0
        y = y0

        points.append((x, y, yaw))

        # Kinematic integration along the arc
        for _ in range(arc_steps):
            # arc step size along the path
            step = arc_length / arc_steps
            # heading change for this step
            dyaw = turn_dir * (step / R)

            yaw += dyaw
            x += step * math.cos(yaw)
            y += step * math.sin(yaw)

            points.append((x, y, yaw))

        # After the arc, yaw is approximately goal_yaw
        # Now do straight to the goal
        dx2 = gx - x
        dy2 = gy - y
        dist2 = math.hypot(dx2, dy2)
        if dist2 < 1e-6:
            return points

        straight_steps = max(1, int(dist2 / ds))
        straight_yaw = math.atan2(dy2, dx2)

        for i in range(straight_steps):
            step = dist2 / straight_steps
            x += step * math.cos(straight_yaw)
            y += step * math.sin(straight_yaw)
            points.append((x, y, straight_yaw))

        # Ensure final point is exactly the goal
        points.append((gx, gy, straight_yaw))

        return points

    def _quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """Extract yaw (rotation around Z) from a quaternion assuming ZYX convention."""
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _plan_trajectory_from_current_pose(self, goal_position):
        """
        Compute a trajectory from current_pose to goal_position.

        Uses a turn–then–straight path in the XY plane and constant pitch in Z.
        """
        if self.current_pose is None or goal_position is None:
            self.get_logger().warn("Missing current pose or goal; cannot plan.")
            return

        # 1) Start pose
        start = self.current_pose.position
        start_x, start_y, start_z = start.x, start.y, start.z

        # Extract yaw from quaternion (we ignore roll & pitch for planning here)
        q = self.current_pose.orientation
        start_yaw = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # 2) Goal position
        gx, gy, gz = goal_position

        # If start and goal are basically identical
        dx0 = gx - start_x
        dy0 = gy - start_y
        dz0 = gz - start_z
        horiz_dist0 = math.hypot(dx0, dy0)
        if horiz_dist0 < 1e-6 and abs(dz0) < 1e-6:
            self.get_logger().warn(
                "Start and goal are essentially identical. Single-point trajectory."
            )
            pose = Pose()
            pose.position.x = start_x
            pose.position.y = start_y
            pose.position.z = start_z
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0

            self.trajectory_poses = [pose]
            self.current_index = 0
            self._send_current_waypoint()
            return

        # 3) Build XY path with turn–then–straight
        xy_yaw_points = self._build_turn_straight_xy_path(
            start_x, start_y, start_yaw, gx, gy, self.turn_radius, self.dubins_step
        )

        if len(xy_yaw_points) < 2:
            self.get_logger().warn(
                "Failed to generate XY path; falling back to straight line."
            )
            # Fallback: simple straight line
            xs = np.linspace(start_x, gx, 10)
            ys = np.linspace(start_y, gy, 10)
            yaw_straight = math.atan2(gy - start_y, gx - start_x)
            xy_yaw_points = [(float(x), float(y), yaw_straight) for x, y in zip(xs, ys)]

        # 4) Compute horizontal distance and z interpolation
        horiz_dists = [0.0]
        total_horiz = 0.0
        for i in range(1, len(xy_yaw_points)):
            x_prev, y_prev, _ = xy_yaw_points[i - 1]
            x_curr, y_curr, _ = xy_yaw_points[i]
            ds = math.hypot(x_curr - x_prev, y_curr - y_prev)
            total_horiz += ds
            horiz_dists.append(total_horiz)

        if total_horiz < 1e-6:
            total_horiz = horiz_dist0  # fallback

        # Vertical change
        dz_total = gz - start_z

        # 5) Constant pitch angle
        pitch = math.atan2(dz_total, total_horiz)

        # 6) Build Pose waypoints
        poses: list[Pose] = []
        for i, (x, y, yaw) in enumerate(xy_yaw_points):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)

            # Linear interpolation of z along the horizontal distance
            if total_horiz > 1e-6:
                t = horiz_dists[i] / total_horiz
            else:
                t = 0.0
            pose.position.z = start_z + t * dz_total

            # Orientation from roll=0, constant pitch, yaw_i
            qx, qy, qz, qw = self._rpy_to_quaternion(0.0, pitch, yaw)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw

            poses.append(pose)

        self.trajectory_poses = poses
        self.current_index = 0

        self.get_logger().info(
            f"Planned turn–straight trajectory with {len(self.trajectory_poses)} waypoints, "
            f"horizontal distance ~{total_horiz:.2f} m"
        )

        # 7) Publish full path once & send first waypoint
        self._send_current_waypoint()

    def _send_current_waypoint(self):
        """Publish the current waypoint Pose on /position/target."""
        if not self.trajectory_poses:
            self.get_logger().warn("No trajectory poses to send.")
            return

        pose = self.trajectory_poses[self.current_index]
        self.position_target_pub.publish(pose)

        # Update timing and segment length tracking
        self.last_waypoint_time = self.get_clock().now()
        if self.current_index == 0:
            # Approximate length from current EKF pose to first waypoint
            if self.current_pose is not None:
                prev = self.current_pose.position
                dx = pose.position.x - prev.x
                dy = pose.position.y - prev.y
                dz = pose.position.z - prev.z
                self.last_segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)
            else:
                self.last_segment_length = None
        else:
            prev_pose = self.trajectory_poses[self.current_index - 1]
            dx = pose.position.x - prev_pose.position.x
            dy = pose.position.y - prev_pose.position.y
            dz = pose.position.z - prev_pose.position.z
            self.last_segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)

        self.get_logger().info(
            f"Sent waypoint {self.current_index + 1}/{len(self.trajectory_poses)} "
            f"for keypoint {self.current_keypoint_idx}: "
            f"({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        )

    def _advance_to_next_waypoint(self):
        """
        Move to next waypoint and send it.

        Or, if at the end of local trajectory, go to next keypoint (if any).
        """
        if not self.trajectory_poses:
            return

        # Still have more fine-grained waypoints for this keypoint
        if self.current_index < len(self.trajectory_poses) - 1:
            self.current_index += 1
            self._send_current_waypoint()
            return

        # Finished local trajectory to this keypoint
        self.get_logger().info("Local trajectory to current keypoint completed.")

        # Move to next keypoint, if available and in RUNNING mode
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

    def _timer_callback(self):
        """
        Every timer_dt seconds: If mode != RUNNING: do nothing.

        Check distance from EKF pose to current waypoint.
        If within tolerance -> send next waypoint.
        If too long without progress -> replan local trajectory.
        """
        if self.mode != "RUNNING":
            return

        if self.current_pose is None:
            return
        if not self.trajectory_poses:
            return

        pose = self.trajectory_poses[self.current_index]
        wp_x, wp_y, wp_z = pose.position.x, pose.position.y, pose.position.z

        cur = self.current_pose.position
        dx = wp_x - cur.x
        dy = wp_y - cur.y
        dz = wp_z - cur.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Reached via EKF check
        if dist <= self.position_tolerance:
            self.get_logger().info(
                f"Waypoint reached via EKF (dist={dist:.2f} m <= {self.position_tolerance:.2f} m)."
            )
            self._advance_to_next_waypoint()
            return

        # Timeout check
        if self.last_waypoint_time is not None:
            now = self.get_clock().now()
            elapsed = (
                now.nanoseconds - self.last_waypoint_time.nanoseconds
            ) * 1e-9  # sec

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

    def _rpy_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """Convert roll, pitch, yaw (in radians) to a quaternion (x, y, z, w)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = PathfindingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
