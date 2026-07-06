"""Manual ACU driver -- bypasses the attitude controller.

Drives ``/acu/pitch`` + ``/acu/roll`` by hand so an operator can park the
pitch mass / roll ring at a position. There is no override gate: receiving a
command means "hold this position." Contention with acu_node is avoided by the
mission being stopped first -- the UI publishes ``/command``=false when the
operator engages a manual command, which gates acu_node off -- so this node is
free to own the ACU. ``DEBUG_RESET`` releases both axes: command neutral (0/0)
once, then go silent so acu_node can reclaim the wire on the next mission.

While a setpoint is held it's the only thing on the wire, so we re-publish it
at 10 Hz (same egress-throttle reasoning as bcu_debug -- a lone publish can
lose the race against the bridge throttle); a short trailing flush carries the
neutral out after a reset.

Wire formats match the actuator topics: pitch is ``Int16`` millimetres,
roll is ``Int16`` centidegrees (degrees * 100).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, Int16

from py_pkg.debug import FLUSH_TICKS, PUBLISH_PERIOD_S
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
)


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
        # Red all-stop from the operator UI.
        create_subscription_for_topic(self, UUVTopics.DEBUG_RESET, self._on_reset)

        # Held setpoints. None means "not commanding this axis" -- the axis
        # is left wherever acu_node last put it (we don't fabricate a zero).
        self._pitch_mm: int | None = None
        self._roll_cdeg: int | None = None
        # Trailing-neutral flush countdown after a reset.
        self._flush_ticks: int = 0

        self._timer = self.create_timer(PUBLISH_PERIOD_S, self._on_tick)

    # --- command callbacks ----------------------------------------------

    def _on_reset(self, _msg: Empty) -> None:
        # Release both axes: drop the held setpoints, command neutral once, and
        # arm the trailing flush so the 0/0 reliably lands on the throttled
        # egress before we fall silent.
        self._pitch_mm = None
        self._roll_cdeg = None
        self._publish_pitch(0)
        self._publish_roll(0)
        self._flush_ticks = FLUSH_TICKS
        self.get_logger().info("debug reset -- ACU released to neutral (0/0)")

    def _on_pitch_cmd(self, msg: Int16) -> None:
        self._pitch_mm = int(msg.data)
        self._publish_pitch(self._pitch_mm)

    def _on_roll_cmd(self, msg: Int16) -> None:
        self._roll_cdeg = int(msg.data)
        self._publish_roll(self._roll_cdeg)

    # --- periodic heartbeat ---------------------------------------------

    def _on_tick(self) -> None:
        pitch_held = self._pitch_mm is not None
        roll_held = self._roll_cdeg is not None
        if pitch_held:
            self._publish_pitch(self._pitch_mm)
        if roll_held:
            self._publish_roll(self._roll_cdeg)
        if not pitch_held and not roll_held and self._flush_ticks > 0:
            # A reset just neutralized both axes: carry the 0/0 out for a few
            # ticks so it lands on the throttled egress, then go silent.
            self._flush_ticks -= 1
            self._publish_pitch(0)
            self._publish_roll(0)

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
