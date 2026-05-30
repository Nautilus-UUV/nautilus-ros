"""Per-subsystem liveness watchdog node.

Subscribes to the steady glider-side feedback/sensor streams and publishes one
``diagnostic_msgs/DiagnosticArray`` on ``/status/liveness`` summarizing which
subsystems are reporting. A subsystem reads "online" while its source is fresh
and "offline" once it goes stale (see ``py_pkg.liveness.watchdog``).

The MQTT bridge forwards the array to the operator UI. The UI keys off each
``DiagnosticStatus.message`` ("online"/"offline"), *not* the byte ``level`` --
``level`` is a ROS ``byte`` that the egress JSON path turns into a control-char
string.
``level`` is still set correctly here for native ROS consumers (``ros2 topic
echo``, ``rqt_robot_monitor``).
"""

from dataclasses import dataclass

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from py_pkg.liveness.watchdog import LivenessWatchdog
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


@dataclass(frozen=True)
class LivenessSource:
    """One steady stream and the subsystem rows its freshness proves alive."""

    topic: str
    subsystems: tuple[str, ...]


# Ordered: this is also the row order in the published array. The valve bitmask
# is the one source feeding two rows -- in sim the two valves share a single
# byte, so they're liveness-correlated (can't fail independently).
LIVENESS_SOURCES: tuple[LivenessSource, ...] = (
    LivenessSource(UUVTopics.ACU_FEEDBACK_OFFSET, ("acu_pitch",)),
    LivenessSource(UUVTopics.ACU_FEEDBACK_ANGLE, ("acu_roll",)),
    LivenessSource(UUVTopics.BCU_FEEDBACK_RPM, ("bcu_pump",)),
    LivenessSource(UUVTopics.BCU_FEEDBACK_VALVES, ("bcu_valve_1", "bcu_valve_2")),
    LivenessSource(UUVTopics.IMU_LEFT, ("imu_left",)),
    LivenessSource(UUVTopics.IMU_RIGHT, ("imu_right",)),
    LivenessSource(UUVTopics.EXTERNAL_PRESSURE, ("external_pressure",)),
    LivenessSource(UUVTopics.BCU_PRESSURE, ("tank_pressure",)),
)

SUBSYSTEMS: tuple[str, ...] = tuple(
    name for src in LIVENESS_SOURCES for name in src.subsystems
)

# subsystem -> source topic, reported as the DiagnosticStatus hardware_id.
SUBSYSTEM_TOPIC: dict[str, str] = {
    name: src.topic for src in LIVENESS_SOURCES for name in src.subsystems
}


class LivenessNode(Node):
    def __init__(self, **kwargs) -> None:
        # **kwargs forwards rclpy Node options (e.g. parameter_overrides) so
        # tests can inject a short staleness window without a launch wrapper.
        super().__init__("liveness_node", **kwargs)

        self.declare_parameter("staleness_timeout_s", 2.0)
        self.declare_parameter("publish_rate_hz", 2.0)
        timeout = self.get_parameter("staleness_timeout_s").value
        rate = self.get_parameter("publish_rate_hz").value

        self._watchdog = LivenessWatchdog(SUBSYSTEMS, timeout)

        for src in LIVENESS_SOURCES:
            # default-arg binds src.subsystems per iteration (avoid late binding)
            create_subscription_for_topic(
                self,
                src.topic,
                lambda _msg, names=src.subsystems: self._watchdog.mark_seen(
                    names, self._now()
                ),
            )

        self._pub = create_publisher_for_topic(self, UUVTopics.STATUS_LIVENESS)
        self.create_timer(1.0 / rate, self._publish_liveness)

        self.get_logger().info(
            f"liveness_node: watching {len(SUBSYSTEMS)} subsystems, "
            f"staleness={timeout}s"
        )

    def _now(self) -> float:
        # Same clock the timer runs on, consistent under both wall time and sim time.
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_liveness(self) -> None:
        array = DiagnosticArray()
        # header.stamp left at zero on purpose: keeps the serialized payload
        # byte-stable while nothing changes, so the MQTT egress on_change dedup
        # forwards only real transitions across the tether.
        now = self._now()
        for name, online in self._watchdog.snapshot(now):
            status = DiagnosticStatus()
            status.name = name
            status.hardware_id = SUBSYSTEM_TOPIC[name]
            status.level = DiagnosticStatus.OK if online else DiagnosticStatus.STALE
            status.message = "online" if online else "offline"
            array.status.append(status)
        self._pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LivenessNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
