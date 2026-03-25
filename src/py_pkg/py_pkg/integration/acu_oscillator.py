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
    Time-based ACU tilt oscillator.
    Units are: mm, radians, seconds
    Sequence:
    1. Start at center (0)
    2. Move to front (max tilt)
    3. Hold 30s
    4. Move to back (min tilt)
    5. Hold 60s
    6. Return to center
    7. Repeat
    """

    def __init__(self):
        super().__init__("acu_oscillator")

        # Parameters
        self.declare_parameter("front_tilt", 150.0)
        self.declare_parameter("back_tilt", -150.0)
        self.declare_parameter("center_tilt", 0.0)

        self.front_tilt = self.get_parameter("front_tilt").value
        self.back_tilt = self.get_parameter("back_tilt").value
        self.center_tilt = self.get_parameter("center_tilt").value

        # State machine
        self.state = "INIT"
        self.state_start_time = time.time()

        # Publisher
        self.tilt_pub = create_publisher_for_topic(self, UUVTopics.ACU_TILT)

        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("ACU Tilt Oscillator started")

    def _set_state(self, new_state):
        self.state = new_state
        self.state_start_time = time.time()
        self.get_logger().info(f"Transition -> {self.state}")

    def _elapsed(self):
        return time.time() - self.state_start_time

    def _publish_tilt(self, value):
        msg = Float32()
        msg.data = value
        self.tilt_pub.publish(msg)

    def _control_loop(self):
        # Log every ~2 seconds
        if self.get_clock().now().nanoseconds % 2000000000 < 200000000:
            self.get_logger().info(f"STATE: {self.state}")

        if self.state == "INIT":
            self._publish_tilt(self.center_tilt)
            self._set_state("MOVE_FRONT")

        elif self.state == "MOVE_FRONT":
            self._publish_tilt(self.front_tilt)
            self._set_state("HOLD_FRONT")

        elif self.state == "HOLD_FRONT":
            self._publish_tilt(self.front_tilt)
            if self._elapsed() >= 30.0:
                self._set_state("MOVE_BACK")

        elif self.state == "MOVE_BACK":
            self._publish_tilt(self.back_tilt)
            self._set_state("HOLD_BACK")

        elif self.state == "HOLD_BACK":
            self._publish_tilt(self.back_tilt)
            if self._elapsed() >= 60.0:
                self._set_state("RETURN_CENTER")

        elif self.state == "RETURN_CENTER":
            self._publish_tilt(self.center_tilt)
            self._set_state("HOLD_CENTER")

        elif self.state == "HOLD_CENTER":
            self._publish_tilt(self.center_tilt)
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
