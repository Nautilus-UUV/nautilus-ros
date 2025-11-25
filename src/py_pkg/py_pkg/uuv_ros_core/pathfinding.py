#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, String
from uuv_ros_core import UUVTopics, UUVQoS, TOPIC_MESSAGE_MAP
from uuv_ros_core import create_publisher_for_topic


class PathfindingNode(Node):
    """
    Pathfinding / trajectory planning node for Nautilus.

    SUBSCRIBES:
      - /position/estimation   (UUVTopics.POSITION_ESTIMATION, Pose)
      - /command               (UUVTopics.COMMAND, String)     # final desired position

    PUBLISHES:
      - /path                  (UUVTopics.PATH, Float32MultiArray)  # full trajectory (flattened Poses)
      - /position/target       (UUVTopics.POSITION_TARGET, Pose)    # next waypoint Pose only
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

        # Final desired position (mission command)
        # For now we assume message.data = "x y z" (three floats as text)
        self.command_sub = self.create_subscription(
            TOPIC_MESSAGE_MAP[UUVTopics.COMMAND],  # String
            UUVTopics.COMMAND,
            self._command_callback,
            UUVQoS.COMMAND,
        )

        # Full path (flattened list of Poses)
        self.path_pub = create_publisher_for_topic(
            self, UUVTopics.PATH
        )  # Float32MultiArray

        # Next waypoint Pose (what control should aim at)
        self.position_target_pub = self.create_publisher(
            TOPIC_MESSAGE_MAP[UUVTopics.POSITION_TARGET],  # Pose
            UUVTopics.POSITION_TARGET,
            UUVQoS.CONTROL,
        )

        self.current_pose: Pose = None

        # Goal position (x, y, z) from /command
        self.goal_position = None  # (x, y, z) tuple

        # List of Pose waypoints along our trajectory
        self.trajectory_poses: list[Pose] = []
        self.current_index = 0

        # Timing / monitoring for each segment
        self.last_waypoint_time = None
        self.last_segment_length = None

        # Timer to periodically check progress towards current waypoint
        self.timer = self.create_timer(self.timer_dt, self._timer_callback)

        self.get_logger().info("PathfindingNode started")

    def _state_callback(self, msg: Pose):
        """Receive the latest EKF pose estimate."""
        self.current_pose = msg

    def _command_callback(self, msg: String):
        """
        Handle new mission command from /command.

        For now we assume the format is:
            "x y z"
        Example: "100.0 50.0 -20.0"
        """
        try:
            parts = msg.data.strip().split()
            if len(parts) != 3:
                raise ValueError("Expected three numbers: x y z")
            x, y, z = map(float, parts)
        except Exception as e:
            self.get_logger().error(f"Failed to parse /command: '{msg.data}' ({e})")
            return

        self.goal_position = (x, y, z)
        self.get_logger().info(f"New goal from /command: ({x:.2f}, {y:.2f}, {z:.2f})")

        if self.current_pose is None:
            self.get_logger().warn("Cannot plan trajectory yet: no current pose.")
            return

        self._plan_trajectory_from_current_pose()

    def _build_turn_straight_xy_path(self, x0, y0, yaw0, gx, gy, R, ds):
        """
        Build a simple XY path:

        1) Constant-radius turn (left or right) from yaw0 until
            the heading points directly at the goal.
        2) Straight line from end of arc to goal.

        Arguments:
        x0, y0:   start position
        yaw0:     start heading (rad)
        gx, gy:   goal position
        R:        turn radius
        ds:       step length along the path (m)

        Returns:
        list of (x, y, yaw) along the path
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
        """
        Extract yaw (rotation around Z) from a quaternion assuming ZYX convention.
        """
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _plan_trajectory_from_current_pose(self):
        """
        Compute a trajectory from current_pose to goal_position using
        a turn–then–straight path in the XY plane and constant pitch in Z.

        Steps:
        1) Get start (x,y,z) and start yaw from current_pose.
        2) Get goal (gx, gy, gz) from self.goal_position.
        3) Build an XY polyline with heading using a constant-radius turn
            until facing the goal, then straight line.
        4) Interpolate z along the path from start.z to gz.
        5) Use a constant pitch = atan2(Δz, horizontal_distance).
        6) For each path sample, build a Pose with (x,y,z) and orientation
            from roll=0, constant pitch, yaw_i.
        7) Store as self.trajectory_poses, publish full path, send first waypoint.
        """
        if self.current_pose is None or self.goal_position is None:
            self.get_logger().warn("Missing current pose or goal; cannot plan.")
            return

        # 1) Start pose
        start = self.current_pose.position
        start_x, start_y, start_z = start.x, start.y, start.z

        # Extract yaw from quaternion (we ignore roll & pitch for planning here)
        q = self.current_pose.orientation
        start_yaw = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # 2) Goal position
        gx, gy, gz = self.goal_position

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
            self._publish_full_path()
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
        self._publish_full_path()
        self._send_current_waypoint()

    def _publish_full_path(self):
        """
        Publish the full trajectory on /path as a Float32MultiArray.

        Format per Pose:
          [x, y, z, qx, qy, qz, qw]
        So the full data is:
          [x0,y0,z0,qx0,qy0,qz0,qw0, x1,y1,z1,qx1, ...]
        """
        if not self.trajectory_poses:
            return

        arr = []
        for pose in self.trajectory_poses:
            arr.append(float(pose.position.x))
            arr.append(float(pose.position.y))
            arr.append(float(pose.position.z))
            arr.append(float(pose.orientation.x))
            arr.append(float(pose.orientation.y))
            arr.append(float(pose.orientation.z))
            arr.append(float(pose.orientation.w))

        msg = Float32MultiArray()
        msg.data = arr
        self.path_pub.publish(msg)

    def _send_current_waypoint(self):
        """
        Publish the current waypoint Pose on /position/target.
        """
        if not self.trajectory_poses:
            self.get_logger().warn("No trajectory poses to send.")
            return

        pose = self.trajectory_poses[self.current_index]
        self.position_target_pub.publish(pose)

        # Update timing and segment length tracking
        self.last_waypoint_time = self.get_clock().now()
        if self.current_index == 0:
            self.last_segment_length = None
        else:
            prev_pose = self.trajectory_poses[self.current_index - 1]
            dx = pose.position.x - prev_pose.position.x
            dy = pose.position.y - prev_pose.position.y
            dz = pose.position.z - prev_pose.position.z
            self.last_segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)

        self.get_logger().info(
            f"Sent waypoint {self.current_index + 1}/{len(self.trajectory_poses)}: "
            f"({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        )

    def _advance_to_next_waypoint(self):
        """Move to next waypoint and send it; or finish if we are at the end."""
        if not self.trajectory_poses:
            return

        if self.current_index < len(self.trajectory_poses) - 1:
            self.current_index += 1
            self._send_current_waypoint()
        else:
            self.get_logger().info("Final waypoint reached. Trajectory complete.")
            self.trajectory_poses = []
            self.current_index = 0
            self.last_waypoint_time = None
            self.last_segment_length = None

    def _timer_callback(self):
        """
        Every timer_dt seconds:
          - Check distance from EKF pose to current waypoint.
          - If within tolerance -> send next waypoint.
          - If too long without progress -> replan.
        """
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

            if self.last_segment_length is not None:
                expected_time = self.last_segment_length / self.speed
            else:
                expected_time = self.timer_dt

            timeout = self.segment_timeout_factor * expected_time

            if elapsed > timeout:
                self.get_logger().warn(
                    f"Timeout on waypoint (elapsed={elapsed:.1f}s > {timeout:.1f}s). "
                    f"Replanning from current pose."
                )
                self._plan_trajectory_from_current_pose()

    def _rpy_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """
        Convert roll, pitch, yaw (in radians) to a quaternion (x, y, z, w).
        """
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
