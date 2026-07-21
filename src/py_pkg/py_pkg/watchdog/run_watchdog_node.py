"""Sim run watchdog node: end a sweep run as soon as its outcome is decided.

Watches the latched mission-completion flag and the ground-truth odometry
stream and concludes the run on whichever verdict lands first:

- ``mission_complete`` (exit 0) -- pathfinding latched ``/mission/complete``;
- ``abort_floater`` (exit 2) / ``abort_sinker`` (exit 3) -- the pure
  ``RunPlausibility`` tracker decided the run can never become viable
  (never left the surface / sank and stopped climbing).

Exiting *is* the termination mechanism: ``run_watchdog.launch.py`` registers
an ``OnProcessExit`` handler on this node that turns the exit into a full
launch ``Shutdown``, and the sweep runner reaps the slot on process exit.
Each conclude first writes an optional ``run_verdict.json`` for the sweep
analysis, then holds a grace period so the 1 Hz record throttle captures the
tail of the run before teardown.

The plausibility clock arms only once ``/command`` goes true (mission
started), so bringup idling at the spawn depth can never read as a floater.
"""

import json
import sys
import time
from pathlib import Path
from typing import Callable

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

from py_pkg.uuv_ros_core import (
    UUVQoS,
    UUVTopics,
    create_subscription_for_topic,
    now_s,
    spin_node,
)
from py_pkg.watchdog.plausibility import PlausibilityConfig, RunPlausibility

# Per-verdict process exit codes the sweep runner sees on reap.
EXIT_MISSION_COMPLETE = 0
EXIT_ABORT_FLOATER = 2
EXIT_ABORT_SINKER = 3


class RunWatchdogNode(Node):
    def __init__(self, *, exit_fn: Callable[[int], None] = sys.exit, **kwargs) -> None:
        # **kwargs forwards rclpy Node options (e.g. parameter_overrides) so
        # tests can shrink the deadlines without a launch wrapper. exit_fn is
        # injectable for the same reason: the production default sys.exit
        # raises SystemExit through the executor (rclpy teardown then rides
        # spin_node's finally), which would kill pytest -- Tier 2 tests pass
        # a recorder instead and observe the conclude path in-process.
        super().__init__("run_watchdog", **kwargs)

        # Default matches bridge.launch.py's ground-truth recording topic,
        # f"/model/{model_name}/odometry", at the canonical rig model_name.
        # Model-scoped sim topic, deliberately not in the registry --
        # parameterized instead.
        self.declare_parameter("odom_topic", "/model/glider_nautilus/odometry")
        self.declare_parameter("dive_deadline_s", 120.0)
        self.declare_parameter("stall_grace_s", 300.0)
        # "" = write no verdict file.
        self.declare_parameter("verdict_path", "")
        self.declare_parameter("complete_grace_s", 12.0)
        self.declare_parameter("abort_grace_s", 5.0)

        odom_topic = self.get_parameter("odom_topic").value
        self._verdict_path = self.get_parameter("verdict_path").value
        self._complete_grace_s = self.get_parameter("complete_grace_s").value
        self._abort_grace_s = self.get_parameter("abort_grace_s").value

        self._plausibility = RunPlausibility(
            PlausibilityConfig(
                dive_deadline_s=self.get_parameter("dive_deadline_s").value,
                stall_grace_s=self.get_parameter("stall_grace_s").value,
            )
        )
        self._armed = False
        self._concluded: str | None = None
        self._grace_timer = None
        self._exit_fn = exit_fn

        create_subscription_for_topic(
            self, UUVTopics.MISSION_COMPLETE, self._on_mission_complete
        )
        create_subscription_for_topic(self, UUVTopics.COMMAND, self._on_command)
        self.create_subscription(
            Odometry, odom_topic, self._on_odom, UUVQoS.SENSOR_STREAM
        )

        self.get_logger().info(f"run_watchdog: watching {odom_topic}")

    def _on_command(self, msg: Bool) -> None:
        # Arm on the first start; a later /command=false mid-run is an
        # operator stop, not a verdict, so we never disarm.
        if msg.data and not self._armed:
            self._armed = True
            self.get_logger().info("run_watchdog: mission started, plausibility armed")

    def _on_mission_complete(self, msg: Bool) -> None:
        if msg.data:
            self._conclude(
                "mission_complete", EXIT_MISSION_COMPLETE, self._complete_grace_s
            )

    def _on_odom(self, msg: Odometry) -> None:
        if not self._armed or self._concluded is not None:
            return
        # classify_run's axis convention: raw odometry z, negative-down.
        verdict = self._plausibility.feed(now_s(self), msg.pose.pose.position.z)
        if verdict == "floater":
            self._conclude("abort_floater", EXIT_ABORT_FLOATER, self._abort_grace_s)
        elif verdict == "sinker":
            self._conclude("abort_sinker", EXIT_ABORT_SINKER, self._abort_grace_s)

    def _conclude(self, verdict: str, exit_code: int, grace_s: float) -> None:
        """Record the run verdict and schedule the exit. Idempotent -- the
        first verdict wins, so a completion that lands before an abort (or
        vice versa) is the one written and exited with."""
        if self._concluded is not None:
            return
        self._concluded = verdict

        record = {
            "verdict": verdict,
            "t_wall_s": time.time(),
            "dive_m": self._plausibility.dive_m,
            "drawup_m": self._plausibility.drawup_m,
        }
        if self._verdict_path:
            path = Path(self._verdict_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record) + "\n")
        # The grace lets the 1 Hz record throttle capture the tail of the
        # run (e.g. the surfaced samples) before the exit tears the launch
        # down.
        self.get_logger().info(
            f"run_watchdog: {record} -> exit {exit_code} after {grace_s}s grace"
        )
        self._grace_timer = self.create_timer(
            grace_s, lambda: self._finish_after_grace(exit_code)
        )

    def _finish_after_grace(self, exit_code: int) -> None:
        # rclpy timers repeat; cancel first so a non-raising injected
        # exit_fn (tests) can't be called twice.
        self._grace_timer.cancel()
        self._exit_fn(exit_code)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RunWatchdogNode()
    # The default exit_fn raises SystemExit out of the executor; spin_node's
    # finally still runs (destroy_node + rclpy.try_shutdown), so the process
    # tears rclpy down and exits with the verdict code.
    spin_node(node)


if __name__ == "__main__":
    main()
