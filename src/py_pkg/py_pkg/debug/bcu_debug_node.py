"""Manual BCU pump knob -- bypasses pathfinding + depth PID.

Subscribes to ``DEBUG_BCU_RPM`` (``nautilus_msgs/BcuPumpCommand``) and
drives ``BCU_RPM`` (``std_msgs/Int16``) at the requested rpm for the
requested duration, then publishes 0 to stop the motor. A new command
that arrives mid-pump cancels the previous stop and starts a fresh
window -- the motor never sees a spurious 0 between two manual commands.

The duration is clamped to ``MAX_PUMP_S`` so a typo'd UI input (e.g.
3000 instead of 3) can't leave the motor running for half an hour.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Int16

from nautilus_msgs.msg import BcuPumpCommand
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


MAX_PUMP_S = 30.0


def _clamp_duration(duration_s: float, max_s: float = MAX_PUMP_S) -> float:
    """Clamp the user-requested pump duration to [0, max_s]."""
    if duration_s <= 0.0:
        return 0.0
    if duration_s > max_s:
        return float(max_s)
    return float(duration_s)


class BcuDebugNode(Node):
    """Forwards manual pump commands to /bcu/rpm with a bounded duration."""

    def __init__(self) -> None:
        super().__init__("bcu_debug")
        self._pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self._override_pub = create_publisher_for_topic(
            self, UUVTopics.CONTROL_MANUAL_OVERRIDE
        )
        create_subscription_for_topic(
            self, UUVTopics.DEBUG_BCU_RPM, self._on_cmd
        )
        self._stop_timer = None

    def _on_cmd(self, msg: BcuPumpCommand) -> None:
        # A fresh command supersedes any pending stop -- otherwise an
        # in-flight stop timer would publish 0 mid-pump and the motor
        # would briefly idle between the two commands.
        self._cancel_stop_timer()

        duration = _clamp_duration(float(msg.duration_s))
        if duration == 0.0:
            # Explicit stop: drop the override first so depth_node resumes
            # control on the very next tick, then emit the zero RPM.
            self._publish_override(False)
            self._publish_rpm(0)
            return

        # Raise the override BEFORE the rpm so depth_node's next control
        # tick sees the flag and stays silent -- otherwise its periodic
        # zero-publish races and clobbers our setpoint.
        self._publish_override(True)
        self._publish_rpm(int(msg.rpm))
        self._stop_timer = self.create_timer(duration, self._on_stop)

    def _on_stop(self) -> None:
        self._cancel_stop_timer()
        # Stop the motor first, then release the override so depth_node
        # resumes from a known-zero state rather than fighting our last
        # nonzero RPM the moment it wakes back up.
        self._publish_rpm(0)
        self._publish_override(False)

    def _publish_rpm(self, rpm: int) -> None:
        out = Int16()
        out.data = int(rpm)
        self._pub.publish(out)

    def _publish_override(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self._override_pub.publish(msg)

    def _cancel_stop_timer(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    def destroy_node(self) -> bool:
        self._cancel_stop_timer()
        # Best-effort release of the override on shutdown so depth_node
        # doesn't stay silenced if we get killed mid-pump.
        try:
            self._publish_override(False)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BcuDebugNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
