#!/usr/bin/env python3
"""Mission executor.

Turns an operator's mission request into a stream of depth/attitude
setpoints for the controllers downstream.

Inputs (what this node listens to):
  - /path    (MissionCommand) -- which mission to run, plus its parameters
               (target pressure, glide angle, how many times to resurface).
  - /command (Bool)           -- the operator's run intent: true=start,
               false=stop.
  - position estimate (Pose)  -- current vehicle state. position.z is the
               depth expressed as a gauge pressure (0 at the surface);
               orientation is the current attitude.

Output:
  - POSITION_TARGET (Pose)    -- the setpoint the controllers track.
               position.z is gauge Pa (bcu_node's contract); orientation
               carries the roll/pitch targets for acu_node.

The node's situation is just read off three fields each tick:
  - _mission        -- is a mission loaded?
  - _mission_t0_s   -- has it started running? (None until start, then the
                       wall-clock time it began -- so a value means running.)
  - _run_requested  -- does the operator currently want it running?

_tick handles the one transition that matters: once a mission is loaded,
the operator has asked to run, and we have a pressure reading, it calls
mission.start() once and stamps t0. A stop clears everything; the
controllers reset themselves off the same /command=false (they subscribe
to it directly), so this node doesn't have to tell them.
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
        # Latched completion event (one per finished mission). An operator
        # stop (/command=false) is NOT a completion and never publishes here.
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
            # don't publish anything -- the controllers see this same
            # /command=false themselves and reset, so there's nothing to send.
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
                    angle_rad=float(self._mission_cmd.angle_rad),
                    n_resurfaces=int(self._mission_cmd.n_resurfaces),
                    dwell_s=float(self._mission_cmd.dwell_s),
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
            done = Bool()
            done.data = True
            self._complete_pub.publish(done)
            self._reset()
            return
        # A mission can choose not to issue a setpoint this tick (reference
        # returns None) -- e.g. SURFACE/SAWTOOTH while between phases. When that
        # happens we publish nothing, so the controllers just hold their last
        # target instead of chasing a stale one.
        ref = self._mission.reference(mission_t)
        if ref is not None:
            self._target_pub.publish(ref)


def main(args=None):
    rclpy.init(args=args)
    node = PathfindingNode()
    spin_node(node)


if __name__ == "__main__":
    main()
