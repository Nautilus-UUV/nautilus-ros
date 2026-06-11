#!/usr/bin/env python3
"""Mission executor.

Dispatches `/path` (`nautilus_msgs/MissionCommand`) through the mission
factory and pumps the resulting profile's setpoint onto `POSITION_TARGET`
at 10 Hz. `/command` (`std_msgs/Bool`: true=start, false=stop) latches the
operator's run intent. A stop clears the loaded mission; the controllers
reset themselves off the same `/command`=false (they subscribe to it
directly), so no separate reset signal is emitted here.

There is no explicit state enum -- the node's situation is read straight
off three fields: `_mission` (a mission loaded?), `_mission_t0_s` (has it
started running?), and `_run_requested` (does the operator want it
running?). `_tick` owns the load->run transition: once all three line up
(and pressure is in), it calls `mission.start()` once and stamps t0.

`POSITION_TARGET.position.z` is gauge Pa (depth_node's contract);
`orientation` carries roll/pitch for acu_node.
"""

import rclpy
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import MissionCommand
from rclpy.node import Node
from std_msgs.msg import Bool

from ..physics import SurfaceReference
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
        # `None` until the mission starts; set => running.
        self._mission_t0_s: float | None = None
        # Latched `/command`. `start` may arrive before `/path` or before
        # pressure ingress; we just remember the intent and let `_tick` fire
        # the mission once the preconditions line up.
        self._run_requested: bool = False

        self._current_pressure_pa: float | None = None
        self._current_pose: Pose | None = None

        # Gauge reference: standard atmosphere until the operator's
        # pre-dive Initialize registers the real surface pressure
        # (DIVE_INIT). Matters here more than anywhere -- "surfaced" is
        # defined as ~0.5 m of water (SURFACE_THRESHOLD_PA), well inside
        # what weather alone moves the surface pressure by.
        self._surface_ref = SurfaceReference()

        create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self._on_pose
        )
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.COMMAND, self._on_command)
        create_subscription_for_topic(self, UUVTopics.PATH, self._on_path)
        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)

        self._target_pub = create_publisher_for_topic(self, UUVTopics.POSITION_TARGET)
        self.create_timer(1.0 / REFERENCE_RATE_HZ, self._tick)

        self.get_logger().info("pathfinding_node started (mission-id dispatch).")

    def _on_pose(self, msg: Pose) -> None:
        self._current_pose = msg

    def _on_pressure(self, msg) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the depth stack works in gauge,
        # referenced to the registered surface pressure once it's in.
        self._current_pressure_pa = self._surface_ref.gauge(float(msg.data))

    def _on_dive_init(self, msg) -> None:
        self._surface_ref.register_logged(
            float(msg.surface_pressure_pa), self.get_logger()
        )

    def _on_path(self, msg: MissionCommand) -> None:
        # `/path` is latched (transient-local), so the same MissionCommand can be
        # redelivered on discovery re-matching. Reloading unconditionally would
        # null `_mission_t0_s` on a running mission, which then strands `_tick`
        # (it only publishes once running) -- the glider never gets a setpoint
        # and just drifts. Ignore a redelivery of the mission we're already on.
        if (
            self._mission_cmd is not None
            and msg.mission_id == self._mission_cmd.mission_id
        ):
            return
        try:
            self._mission = create_mission(msg.mission_id)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        self._mission_cmd = msg
        self._mission_t0_s = None
        self.get_logger().info(f"Loaded mission_id={msg.mission_id}.")

    def _on_command(self, msg: Bool) -> None:
        self._run_requested = bool(msg.data)
        self.get_logger().info(
            f"Received /command: {'start' if self._run_requested else 'stop'}"
        )
        if not self._run_requested:
            # Stop -> clean initial state. The controllers reset themselves off
            # this same /command=false (they subscribe to it directly), emitting
            # one safe-stop then going silent, so we publish nothing here.
            self._reset()
            self.get_logger().info("Mission stopped; stack reset to initial state.")

    def _reset(self) -> None:
        """Drop the loaded mission and the run intent -> nothing loaded."""
        self._mission = None
        self._mission_cmd = None
        self._mission_t0_s = None
        self._run_requested = False

    def _tick(self) -> None:
        # Load->run transition: fire the mission once the operator has asked for
        # it, a mission is loaded, it isn't already running, and pressure is in.
        # The `_mission_t0_s is None` guard keeps this idempotent -- a redelivered
        # /command can't re-`start()` and reset the mission clock mid-run.
        if (
            self._run_requested
            and self._mission is not None
            and self._mission_t0_s is None
            and self._current_pressure_pa is not None
        ):
            self._mission.start(
                MissionState(
                    pose=self._current_pose,
                    target_pressure_pa=float(self._mission_cmd.target_pressure_pa),
                    angle_rad=float(self._mission_cmd.angle_rad),
                    n_resurfaces=int(self._mission_cmd.n_resurfaces),
                )
            )
            self._mission_t0_s = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info("Mission running.")

        if self._mission_t0_s is None or self._current_pressure_pa is None:
            return
        mission_t = self.get_clock().now().nanoseconds / 1e9 - self._mission_t0_s
        self._mission.update(self._current_pressure_pa)
        if self._mission.is_done(mission_t):
            self.get_logger().info("Mission complete.")
            self._reset()
            return
        # A mission may decline to command this tick (reference -> None), e.g.
        # SURFACE/SAWTOOTH between phases. Publish nothing so the controllers
        # hold their last target rather than tracking a stale setpoint.
        ref = self._mission.reference(mission_t)
        if ref is not None:
            self._target_pub.publish(ref)


def main(args=None):
    rclpy.init(args=args)
    node = PathfindingNode()
    spin_node(node)


if __name__ == "__main__":
    main()
