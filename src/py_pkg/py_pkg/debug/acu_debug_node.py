"""Manual ACU driver -- bypasses the attitude controller.

Drives ``/acu/pitch`` + ``/acu/roll`` by hand so an operator can park the
pitch mass / roll ring at a position. Whether it's allowed to touch the wire
is owned by the operator's manual-override slider: the mission UI raises
``CONTROL_ACU_OVERRIDE`` (through the MQTT bridge), acu_node stands down while
it's True, and this node only relays / heartbeats its held setpoints during
that window. Dropping the slider clears the held setpoints, so the heartbeat
goes silent and acu_node resumes -- turning the slider off *is* the release.

While a setpoint is held it's the only thing on the wire, so we re-publish it
at 10 Hz (same egress-throttle reasoning as bcu_debug -- a lone publish can
lose the race against the bridge throttle).

Wire formats match the actuator topics: pitch is ``Int16`` millimetres,
roll is ``Int16`` centidegrees (degrees * 100).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int16

from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
)


PUBLISH_PERIOD_S = 0.1


class AcuDebugNode(Node):
    """Holds operator-set ACU pitch/roll positions while in manual mode."""

    def __init__(self) -> None:
        super().__init__("acu_debug")
        self._pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)
        self._roll_pub = create_publisher_for_topic(self, UUVTopics.ACU_ROLL)

        create_subscription_for_topic(
            self, UUVTopics.DEBUG_ACU_PITCH, self._on_pitch_cmd
        )
        create_subscription_for_topic(
            self, UUVTopics.DEBUG_ACU_ROLL, self._on_roll_cmd
        )
        # The operator's manual-override slider gates whether we drive the ACU.
        create_subscription_for_topic(
            self, UUVTopics.CONTROL_ACU_OVERRIDE, self._on_override
        )

        # Held setpoints. None means "not commanding this axis" -- the axis
        # is left wherever acu_node last put it (we don't fabricate a zero).
        self._pitch_mm: int | None = None
        self._roll_cdeg: int | None = None
        # Whether the operator slider currently has us in manual mode.
        self._manual_override = False

        self._timer = self.create_timer(PUBLISH_PERIOD_S, self._on_tick)

    # --- command callbacks ----------------------------------------------

    def _on_override(self, msg: Bool) -> None:
        active = bool(msg.data)
        if active == self._manual_override:
            return
        self._manual_override = active
        if not active:
            # Slider off -> drop the held setpoints so the heartbeat goes
            # quiet and acu_node owns /acu/pitch + /acu/roll again.
            self._pitch_mm = None
            self._roll_cdeg = None
        self.get_logger().info(
            f"ACU override {'engaged' if active else 'released'} -- "
            f"acu_debug {'driving' if active else 'silent'}"
        )

    def _on_pitch_cmd(self, msg: Int16) -> None:
        if not self._manual_override:
            self.get_logger().warn(
                "pitch command ignored -- ACU override is off (acu_node owns the ACU)"
            )
            return
        self._pitch_mm = int(msg.data)
        self._publish_pitch(self._pitch_mm)

    def _on_roll_cmd(self, msg: Int16) -> None:
        if not self._manual_override:
            self.get_logger().warn(
                "roll command ignored -- ACU override is off (acu_node owns the ACU)"
            )
            return
        self._roll_cdeg = int(msg.data)
        self._publish_roll(self._roll_cdeg)

    # --- periodic heartbeat ---------------------------------------------

    def _on_tick(self) -> None:
        if not self._manual_override:
            return
        if self._pitch_mm is not None:
            self._publish_pitch(self._pitch_mm)
        if self._roll_cdeg is not None:
            self._publish_roll(self._roll_cdeg)

    # --- helpers --------------------------------------------------------

    def _publish_pitch(self, mm: int) -> None:
        out = Int16()
        out.data = int(mm)
        self._pitch_pub.publish(out)

    def _publish_roll(self, cdeg: int) -> None:
        out = Int16()
        out.data = int(cdeg)
        self._roll_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AcuDebugNode()
    spin_node(node)


if __name__ == "__main__":
    main()
