#!/usr/bin/env python3
"""Mission executor.

Dispatches `/path` (`nautilus_msgs/MissionCommand`) through the mission
factory and pumps the resulting profile's setpoint onto `POSITION_TARGET`
at 10 Hz. `/command` (start/stop/abort) drives the state machine.

`POSITION_TARGET.position.z` is gauge Pa (depth_node's contract);
`orientation` carries roll/pitch for acu_node.
"""

import rclpy
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import MissionCommand
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from ..physics import gauge_pressure_pa
from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from .missions import MissionProfile, MissionState, create_mission

REFERENCE_RATE_HZ = 10.0


class PathfindingNode(Node):
    def __init__(self):
        super().__init__("pathfinding_node")

        self._mission: MissionProfile | None = None
        self._mission_cmd: MissionCommand | None = None
        self._mode: str = "IDLE"  # IDLE | LOADED | RUNNING | STOPPED
        self._mission_t0_s: float | None = None

        self._current_pressure_pa: float | None = None
        self._current_pose: Pose | None = None

        create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self._on_pose
        )
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.COMMAND, self._on_command)
        create_subscription_for_topic(self, UUVTopics.PATH, self._on_path)

        self._target_pub = create_publisher_for_topic(self, UUVTopics.POSITION_TARGET)
        self.create_timer(1.0 / REFERENCE_RATE_HZ, self._tick)

        self.get_logger().info("pathfinding_node started (mission-id dispatch).")

    def _on_pose(self, msg: Pose) -> None:
        self._current_pose = msg

    def _on_pressure(self, msg) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the depth stack works in gauge.
        self._current_pressure_pa = gauge_pressure_pa(float(msg.data))

    def _on_path(self, msg: MissionCommand) -> None:
        try:
            self._mission = create_mission(msg.mission_id)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        self._mission_cmd = msg
        self._mode = "LOADED"
        self._mission_t0_s = None
        self.get_logger().info(f"Loaded mission_id={msg.mission_id}.")

    def _on_command(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"Received /command: '{cmd}'")
        if cmd == "start":
            self._handle_start()
        elif cmd == "stop":
            self._mode = "STOPPED"
            self.get_logger().info("Mode STOPPED (holding last setpoint).")
        elif cmd == "abort":
            self._handle_abort()
        else:
            self.get_logger().warn(f"Unknown /command: {cmd!r}")

    def _handle_start(self) -> None:
        if self._mission is None or self._mission_cmd is None:
            self.get_logger().warn("Cannot start: no mission loaded on /path.")
            return
        if self._current_pressure_pa is None:
            self.get_logger().warn("Cannot start: no pressure ingress yet.")
            return
        self._mission.start(
            MissionState(
                pose=self._current_pose,
                target_pressure_pa=float(self._mission_cmd.target_pressure_pa),
                angle_rad=float(self._mission_cmd.angle_rad),
                n_resurfaces=int(self._mission_cmd.n_resurfaces),
            )
        )
        self._mission_t0_s = self.get_clock().now().nanoseconds / 1e9
        self._mode = "RUNNING"
        self.get_logger().info("Mode RUNNING.")

    def _handle_abort(self) -> None:
        # Abort -> command resurface: depth_node will drive BCU to push the
        # glider up; ACU is left at neutral attitude.
        self._mission = None
        self._mission_cmd = None
        self._mode = "IDLE"
        self._mission_t0_s = None
        pose = Pose()
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        self._target_pub.publish(pose)
        self.get_logger().info("Mode ABORTED (resurfacing).")

    def _tick(self) -> None:
        if (
            self._mode != "RUNNING"
            or self._mission is None
            or self._mission_t0_s is None
            or self._current_pressure_pa is None
        ):
            return
        mission_t = self.get_clock().now().nanoseconds / 1e9 - self._mission_t0_s
        self._mission.update(self._current_pressure_pa)
        if self._mission.is_done(mission_t):
            self.get_logger().info("Mission complete.")
            self._mode = "IDLE"
            self._mission = None
            self._mission_cmd = None
            self._mission_t0_s = None
            return
        self._target_pub.publish(self._mission.reference(mission_t))


def main(args=None):
    # Catch SIGINT/SIGTERM so the process exits 0 instead of 1 on Ctrl-C —
    # otherwise launch_testing's exit-code check intermittently fails.
    rclpy.init(args=args)
    node = PathfindingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
