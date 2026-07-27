#!/usr/bin/env python3
"""Mission executor.

Turns an operator's mission request into a stream of depth/attitude
setpoints for the controllers downstream.

Inputs (what this node listens to):
  - /path    (MissionCommand) -- which mission to run, plus its parameters
               (deep/shallow pressures, glide angle, how many oscillations).
  - /command (Bool)           -- the operator's run intent: true=start,
               false=stop.
  - position estimate (Pose)  -- current vehicle state. position.z is the
               depth expressed as a gauge pressure (0 at the surface);
               orientation is the current attitude.

Outputs:
  - POSITION_TARGET (Pose)    -- the setpoint the controllers track.
               position.z is gauge Pa (bcu_node's contract); orientation
               carries the roll/pitch targets for acu_node.
  - /mission/complete (Bool)  -- latched true, once, when the running
               mission finishes. See "how a run ends" below.

The node's situation is just read off three fields each tick:
  - _mission        -- is a mission loaded?
  - _mission_t0_s   -- has it started running? (None until start, then the
                       wall-clock time it began -- so a value means running.)
  - _run_requested  -- does the operator currently want it running?

_tick handles the one transition that matters: once a mission is loaded,
the operator has asked to run, and we have a pressure reading, it calls
mission.start() once and stamps t0.

How a run ends -- two causes, two topics, each published by whoever owns it:
  - an OPERATOR stop arrives on /command=false. This node resets; the
    controllers hear that same message and safe-stop themselves.
  - COMPLETION is ours to announce, and only ours: a latched Bool(true) on
    /mission/complete. bcu_node and acu_node subscribe to it and run the
    same safe-stop path they run on an operator stop, so neither holds its
    last setpoint after the mission ends.

This node never publishes /command. That channel carries the operator's
intent and nothing else, which is what keeps "finished" distinguishable from
"aborted" for every subscriber -- including a node that joins late and reads
the topic's TRANSIENT_LOCAL latch.
"""

import rclpy
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import MissionCommand
from rclpy.node import Node
from std_msgs.msg import Bool

from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    now_s,
    spin_node,
)
from .missions import MissionProfile, MissionState, create_mission

REFERENCE_RATE_HZ = 10.0


class PathfindingNode(Node):
    def __init__(self):
        super().__init__("pathfinding_node")

        self._mission: MissionProfile | None = None
        self._mission_cmd: MissionCommand | None = None
        # None until the mission starts; once set, it's the start time (=> running).
        self._mission_t0_s: float | None = None
        self._run_requested: bool = False

        self._current_pressure_pa: float | None = None
        self._current_pose: Pose | None = None

        create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self._on_pose
        )
        create_subscription_for_topic(self, UUVTopics.COMMAND, self._on_command)
        create_subscription_for_topic(self, UUVTopics.PATH, self._on_path)

        self._target_pub = create_publisher_for_topic(self, UUVTopics.POSITION_TARGET)
        # Latched completion event (one per finished mission), and the only
        # thing that ends a run other than the operator. An operator stop
        # (/command=false) is NOT a completion and never publishes here.
        self._complete_pub = create_publisher_for_topic(
            self, UUVTopics.MISSION_COMPLETE
        )
        self.create_timer(1.0 / REFERENCE_RATE_HZ, self._tick)

        self.get_logger().info("pathfinding_node started (mission-id dispatch).")

    def _on_pose(self, msg: Pose) -> None:
        self._current_pose = msg
        self._current_pressure_pa = float(msg.position.z)

    def _on_path(self, msg: MissionCommand) -> None:
        # /path is a latched topic: its last message is kept and re-sent to any
        # subscriber that (re)connects later, so we can receive the same
        # mission more than once. We ignore a command we're already running so we
        # don't reset the timestamp.
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
            # Stop: forget the mission and go back to the initial state. We
            # don't publish anything -- the controllers hear this same
            # /command=false themselves and safe-stop on it.
            self._reset()
            self.get_logger().info("Mission stopped; stack reset to initial state.")

    def _reset(self) -> None:
        """Drop the loaded mission and the run intent -> nothing loaded."""
        self._mission = None
        self._mission_cmd = None
        self._mission_t0_s = None
        self._run_requested = False

    def _tick(self) -> None:
        # Start the mission once everything lines up: the operator asked to run,
        # a mission is loaded, it isn't already running, and we have a pressure
        # reading. The "not already running" check (_mission_t0_s is None) means
        # a repeated /command can't restart a mission that's already underway.
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
                    shallow_pressure_pa=float(self._mission_cmd.shallow_pressure_pa),
                    angle_rad=float(self._mission_cmd.angle_rad),
                    n_resurfaces=int(self._mission_cmd.n_resurfaces),
                    n_steps=int(self._mission_cmd.n_steps),
                )
            )
            self._mission_t0_s = now_s(self)
            self.get_logger().info("Mission running.")

        if self._mission_t0_s is None or self._current_pressure_pa is None:
            return
        mission_t = now_s(self) - self._mission_t0_s
        self._mission.update(self._current_pressure_pa)
        if self._mission.is_done(mission_t):
            self.get_logger().info("Mission complete.")
            # The one announcement this event gets. bcu_node and acu_node
            # safe-stop off it; run_watchdog concludes on it. Nothing here
            # touches /command -- see the module docstring.
            done = Bool()
            done.data = True
            self._complete_pub.publish(done)
            self._reset()
            return

        self._target_pub.publish(self._mission.reference(mission_t))


def main(args=None):
    rclpy.init(args=args)
    node = PathfindingNode()
    spin_node(node)


if __name__ == "__main__":
    main()
