import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32

from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class Oscillator(Node):
    """
    Robust oscillator with BCU (buoyancy) and ACU (tilt) control.
    - BCU: better feedback and logging to ensure the glider reaches the surface and reverses
    - ACU: tilts front when descending, back when ascending
    """

    def __init__(self):
        super().__init__("oscillator")

        # Configuration
        self.declare_parameter(
            "target_rpm", 4100
        )  # taken max from: https://aris-space.atlassian.net/wiki/spaces/Nautilus/pages/306839555/ACU+and+BCU+Motors
        self.declare_parameter("min_vol_ml", 300)
        self.declare_parameter("max_vol_ml", 2400)
        self.declare_parameter("dive_depth_cm", 200)
        self.declare_parameter("surface_depth_cm", 15)
        self.declare_parameter("acu_front_tilt", 150.0) #need to change
        self.declare_parameter("acu_back_tilt", -150.0) #need to change to real vals

        self.target_rpm = self.get_parameter("target_rpm").value
        self.min_vol = self.get_parameter("min_vol_ml").value
        self.max_vol = self.get_parameter("max_vol_ml").value
        self.depth_limit = self.get_parameter("dive_depth_cm").value
        self.surface_limit = self.get_parameter("surface_depth_cm").value
        self.acu_front_tilt = self.get_parameter("acu_front_tilt").value
        self.acu_back_tilt = self.get_parameter("acu_back_tilt").value

        # State machine
        self.state = "ASCENDING"
        self.current_vol_ml = 0
        self.current_depth_cm = 0

        # Publishers
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.tilt_pub = create_publisher_for_topic(self, UUVTopics.ACU_TILT)

        # Subscriptions
        self.vol_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._vol_callback
        )
        self.depth_sub = create_subscription_for_topic(
            self, UUVTopics.TEST_EXTERNAL_DEPTH, self._depth_callback
        )

        # Control loop (10Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f"BCU Safety Oscillator: Safe Range [{self.min_vol}, {self.max_vol}] mL. Surface: < {self.surface_limit} cm"
        )
        self.get_logger().info(
            f"ACU Depth-based Oscillator: Front tilt={self.acu_front_tilt}, Back tilt={self.acu_back_tilt}"
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
    node = Oscillator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
