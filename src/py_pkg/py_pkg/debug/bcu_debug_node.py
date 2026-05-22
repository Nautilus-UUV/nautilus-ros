"""Manual BCU pump knob -- bypasses pathfinding + depth PID.

Subscribes to ``DEBUG_BCU_RPM`` (``nautilus_msgs/BcuPumpCommand``) and
drives ``BCU_RPM`` (``std_msgs/Int16``) at the requested rpm for the
requested duration, then publishes 0 to stop the motor. A new command
that arrives mid-pump cancels the previous window and starts a fresh
one -- the motor never sees a spurious 0 between two manual commands.

During the active window the node *re-publishes* the held RPM on
``BCU_RPM`` at ``PUBLISH_PERIOD_S`` (10 Hz). The MQTT bridge's egress
side is a rate-limit-and-drop throttle (not a periodic re-emitter):
with a single-shot publish, the lone debug sample raced depth_node's
10 Hz zero stream against the bridge's 100 ms throttle window and got
dropped silently most of the time -- the operator UI would see the
pump command "sometimes" and never knew why. A 10 Hz heartbeat from
this node keeps the topic alive on the wire for as long as the
operator asked it to pump, so the egress always has a fresh sample
to forward.

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
# Re-publish cadence while a pump window is active. 10 Hz matches
# depth_node's control loop and the MQTT egress throttle, so each tick
# refreshes the bridge's per-topic _last_emit clock and the next sample
# is always allowed through.
PUBLISH_PERIOD_S = 0.1


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
        # Periodic timer that re-publishes _held_rpm until the wall-clock
        # _pump_deadline_s is reached. None when idle.
        self._pump_timer = None
        self._pump_deadline_s: float | None = None
        self._held_rpm: int = 0

    def _on_cmd(self, msg: BcuPumpCommand) -> None:
        # A fresh command supersedes any in-flight pump window -- otherwise
        # the leftover periodic publisher would either clobber the new
        # setpoint or fire its stop ahead of the new deadline.
        self._cancel_pump_timer()

        duration = _clamp_duration(float(msg.duration_s))
        if duration == 0.0:
            # Explicit stop: drop the override first so depth_node resumes
            # control on the very next tick, then emit the zero RPM.
            self._publish_override(False)
            self._publish_rpm(0)
            self._held_rpm = 0
            self._pump_deadline_s = None
            return

        # Raise the override BEFORE the rpm so depth_node's next control
        # tick sees the flag and stays silent -- otherwise its periodic
        # zero-publish races and clobbers our setpoint.
        self._publish_override(True)
        self._held_rpm = int(msg.rpm)
        # Send the first sample immediately rather than waiting one tick;
        # the periodic timer takes over from there.
        self._publish_rpm(self._held_rpm)
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._pump_deadline_s = now_s + duration
        self._pump_timer = self.create_timer(PUBLISH_PERIOD_S, self._on_tick)

    def _on_tick(self) -> None:
        if self._pump_deadline_s is None:
            # Defensive: deadline cleared from under us (e.g. shutdown
            # racing the timer). Stop quietly.
            self._cancel_pump_timer()
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s >= self._pump_deadline_s:
            # Duration elapsed. Stop the motor first, then release the
            # override so depth_node resumes from a known-zero state
            # rather than fighting our last nonzero RPM the moment it
            # wakes back up.
            self._cancel_pump_timer()
            self._publish_rpm(0)
            self._publish_override(False)
            self._held_rpm = 0
            self._pump_deadline_s = None
            return
        self._publish_rpm(self._held_rpm)

    def _publish_rpm(self, rpm: int) -> None:
        out = Int16()
        out.data = int(rpm)
        self._pub.publish(out)

    def _publish_override(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self._override_pub.publish(msg)

    def _cancel_pump_timer(self) -> None:
        if self._pump_timer is not None:
            self._pump_timer.cancel()
            self._pump_timer = None

    def destroy_node(self) -> bool:
        self._cancel_pump_timer()
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
