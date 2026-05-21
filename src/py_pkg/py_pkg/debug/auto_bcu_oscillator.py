"""Boot-time BCU motor-direction smoke test.

Self-driven oscillator: publishes ``BCU_RPM`` (``std_msgs/Int16``) flipping
between ``+rpm`` and ``-rpm`` every ``period_s`` seconds. Intended to be
spawned by ``stm_debug_oscillator.launch.py`` alongside ``stm_com_node``
and ``mqtt_bridge_node`` so a sealed-Pi build can confirm the BCU motor
reverses on cue without any controller, EKF, or mission running.

We raise ``CONTROL_MANUAL_OVERRIDE`` on startup -- not strictly needed in
the debug launch (depth_node isn't running) but it costs nothing and
makes this node safe to drop into a full ``control_stack`` if someone
ever wants to oscillate the BCU while the rest of the stack is alive.
The override is dropped on shutdown so depth_node would resume cleanly.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Int16

from py_pkg.uuv_ros_core import UUVTopics, create_publisher_for_topic


class AutoBcuOscillator(Node):
    """Publishes alternating ±rpm on /bcu/rpm at a fixed cadence."""

    def __init__(self) -> None:
        super().__init__("auto_bcu_oscillator")

        self.declare_parameter("rpm", 10)
        self.declare_parameter("period_s", 5.0)

        rpm = self.get_parameter("rpm").get_parameter_value().integer_value
        period_s = self.get_parameter("period_s").get_parameter_value().double_value

        self._rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self._override_pub = create_publisher_for_topic(
            self, UUVTopics.CONTROL_MANUAL_OVERRIDE
        )

        # Raise override BEFORE the first RPM publish so a co-running
        # depth_node sees the flag on its next tick and stays silent.
        self._publish_override(True)

        self._current: int = int(rpm)
        self._publish_rpm(self._current)
        self.create_timer(period_s, self._tick)

        self.get_logger().info(
            f"auto_bcu_oscillator: ±{rpm} RPM every {period_s:.1f} s"
        )

    def _tick(self) -> None:
        self._current = -self._current
        self._publish_rpm(self._current)

    def _publish_rpm(self, rpm: int) -> None:
        msg = Int16()
        msg.data = int(rpm)
        self._rpm_pub.publish(msg)

    def _publish_override(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self._override_pub.publish(msg)

    def destroy_node(self) -> bool:
        # Best-effort release on shutdown so depth_node would resume from
        # a known-zero state rather than fighting our last published RPM.
        try:
            self._publish_rpm(0)
            self._publish_override(False)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoBcuOscillator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
