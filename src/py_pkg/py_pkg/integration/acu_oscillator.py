import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class ACUOscillator(Node):
    """
    Time-based ACU pitch oscillator.
    Units are: mm, radians, seconds
    Sequence:
    1. Start at center (0)
    2. Move to front (max pitch)
    3. Hold 3s
    4. Move to back (min pitch)
    5. Hold 30s
    6. Return to center
    7. Repeat
    """

    def __init__(self):
        super().__init__("acu_oscillator")

        # Parameters
        self.declare_parameter("front_pitch", 150.0)
        self.declare_parameter("back_pitch", -150.0)
        self.declare_parameter("center_pitch", 0.0)

        self.front_pitch = self.get_parameter("front_pitch").value
        self.back_pitch = self.get_parameter("back_pitch").value
        self.center_pitch = self.get_parameter("center_pitch").value

        # State machine
        self.state = "INIT"
        self.state_start_time = time.time()

        # Publisher
        self.pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)

        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("ACU Pitch Oscillator started")

    def _set_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f"Transition -> {self.state}")

    def _elapsed(self):
        return time.time() - self.state_start_time

    def _publish_pitch(self, value):
        msg = Float32()
        msg.data = value
        self.pitch_pub.publish(msg)

    def _control_loop(self):
        # Log every ~2 seconds
        if self.get_clock().now().nanoseconds % 2000000000 < 200000000:
            self.get_logger().info(f"STATE: {self.state}")

        if self.state == "INIT":
            self._publish_pitch(self.center_pitch)
            self._set_state("MOVE_FRONT")

        elif self.state == "MOVE_FRONT":
            self._publish_pitch(self.front_pitch)
            self._set_state("HOLD_FRONT")

        elif self.state == "HOLD_FRONT":
            self._publish_pitch(self.front_pitch)
            if self._elapsed() >= 3.0:
                self._set_state("MOVE_BACK")

        elif self.state == "MOVE_BACK":
            self._publish_pitch(self.back_pitch)
            self._set_state("HOLD_BACK")

        elif self.state == "HOLD_BACK":
            self._publish_pitch(self.back_pitch)
            if self._elapsed() >= 30.0:
                self._set_state("RETURN_CENTER")

        elif self.state == "RETURN_CENTER":
            self._publish_pitch(self.center_pitch)
            self._set_state("HOLD_CENTER")

        elif self.state == "HOLD_CENTER":
            self._publish_pitch(self.center_pitch)
            if self._elapsed() >= 5.0:
                self._set_state("MOVE_FRONT")


def main(args=None):
    rclpy.init(args=args)
    node = ACUOscillator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
