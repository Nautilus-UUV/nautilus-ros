import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class BCUOscillator(Node):
    """
    Robust oscillator with better feedback and logging to ensure the glider
    reaches the surface and reverses.
    """

    def __init__(self):
        super().__init__("bcu_oscillator")

        # Configuration
        self.declare_parameter("target_rpm", 1500)
        self.declare_parameter("min_vol_ml", 300)
        self.declare_parameter("max_vol_ml", 2400)
        self.declare_parameter("dive_depth_cm", 200)
        self.declare_parameter("surface_depth_cm", 15)

        self.target_rpm = self.get_parameter("target_rpm").value
        self.min_vol = self.get_parameter("min_vol_ml").value
        self.max_vol = self.get_parameter("max_vol_ml").value
        self.depth_limit = self.get_parameter("dive_depth_cm").value
        self.surface_limit = self.get_parameter("surface_depth_cm").value

        # State machine
        self.state = "ASCENDING"
        self.current_vol_ml = 0
        self.current_depth_cm = 0

        # Publishers
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)

        # Subscriptions
        self.vol_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._vol_callback
        )
        self.depth_sub = create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._depth_callback
        )

        # Control loop (5Hz)
        self.timer = self.create_timer(0.2, self._control_loop)

        self.get_logger().info(
            f"BCU Safety Oscillator: Safe Range [{self.min_vol}, {self.max_vol}] mL. Surface: < {self.surface_limit} cm"
        )

    def _vol_callback(self, msg):
        self.current_vol_ml = msg.data

    def _depth_callback(self, msg):
        self.current_depth_cm = msg.data

    def _control_loop(self):
        rpm_cmd = 0

        # Log every 2 seconds
        if self.get_clock().now().nanoseconds % 2000000000 < 200000000:
            self.get_logger().info(
                f"STATUS: State={self.state}, Depth={self.current_depth_cm}cm, Vol={self.current_vol_ml}mL"
            )

        if self.state == "ASCENDING":
            if self.current_depth_cm <= self.surface_limit:
                self.get_logger().info(f"Reached surface at {self.current_depth_cm}cm.")
                self.state = "DESCENDING"

            elif self.current_vol_ml >= self.max_vol:
                rpm_cmd = 0

            else:
                rpm_cmd = self.target_rpm

        elif self.state == "DESCENDING":
            if self.current_depth_cm >= self.depth_limit:
                self.get_logger().info(
                    f"Reached target dive depth at {self.current_depth_cm}cm."
                )
                self.state = "ASCENDING"

            elif self.current_vol_ml <= self.min_vol:
                rpm_cmd = 0

            else:
                rpm_cmd = -self.target_rpm

        # Publish command
        msg = Int32()
        msg.data = rpm_cmd
        self.rpm_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BCUOscillator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
