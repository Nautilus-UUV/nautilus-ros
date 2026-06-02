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
from rclpy.node import Node
from std_msgs.msg import Empty, String

from ..physics import gauge_pressure_pa
from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
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
        # `start` may arrive before `/path` or before pressure ingress;
        # buffer the intent and drain it once preconditions hold.
        self._start_pending: bool = False

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
        self._reset_pub = create_publisher_for_topic(self, UUVTopics.CONTROL_RESET)
        self.create_timer(1.0 / REFERENCE_RATE_HZ, self._tick)

        self.get_logger().info("pathfinding_node started (mission-id dispatch).")

    def _on_pose(self, msg: Pose) -> None:
        self._current_pose = msg

    def _on_pressure(self, msg) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the depth stack works in gauge.
        self._current_pressure_pa = gauge_pressure_pa(float(msg.data))
        if self._start_pending:
            self._handle_start()

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
        if self._start_pending:
            self._handle_start()

    def _on_command(self, msg: String) -> None:
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"Received /command: '{cmd}'")
        if cmd == "start":
            self._handle_start()
        elif cmd == "stop":
            self._start_pending = False
            self._mode = "STOPPED"
            self.get_logger().info("Mode STOPPED (holding last setpoint).")
        elif cmd == "abort":
            self._handle_abort()
        else:
            self.get_logger().warn(f"Unknown /command: {cmd!r}")

    def _handle_start(self) -> None:
        if (
            self._mission is None
            or self._mission_cmd is None
            or self._current_pressure_pa is None
        ):
            if not self._start_pending:
                self.get_logger().info(
                    "Start queued; waiting for /path and pressure ingress."
                )
            self._start_pending = True
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
        self._start_pending = False
        # Some missions (Do-Nothing) want the controllers back at their fresh,
        # no-mission state before they go quiet -- they advertise it via a
        # `resets_control_on_start` attribute. Emit CONTROL_RESET so depth_node
        # / acu_node drop any held target and wipe controller state.
        if getattr(self._mission, "resets_control_on_start", False):
            self._reset_pub.publish(Empty())
            self.get_logger().info("Emitted CONTROL_RESET (mission requested fresh state).")
        self.get_logger().info("Mode RUNNING.")

    def _handle_abort(self) -> None:
        # Abort -> command resurface: depth_node will drive BCU to push the
        # glider up; ACU is left at neutral attitude.
        self._mission = None
        self._mission_cmd = None
        self._mode = "IDLE"
        self._mission_t0_s = None
        self._start_pending = False
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
        # A mission may decline to command this tick (reference -> None); the
        # Do-Nothing mission always does. Publish nothing so the controllers
        # stay in their no-target hold rather than tracking a stale setpoint.
        ref = self._mission.reference(mission_t)
        if ref is not None:
            self._target_pub.publish(ref)


def main(args=None):
    rclpy.init(args=args)
    node = PathfindingNode()
    spin_node(node)


if __name__ == "__main__":
    main()
