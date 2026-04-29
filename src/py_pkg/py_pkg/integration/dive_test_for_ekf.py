import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float32

from ..physics import pressure_to_depth
from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class Oscillator(Node):
    """
    Robust oscillator with BCU (buoyancy) and ACU (pitch) control.
    - BCU: better feedback and logging to ensure the glider reaches the surface and reverses
    - ACU: pitchs front when descending, back when ascending
    """

    def __init__(self):
        super().__init__("oscillator")

        # Configuration
        self.declare_parameter(
            "target_rpm", 4100
        )  # taken max from: https://aris-space.atlassian.net/wiki/spaces/Nautilus/pages/306839555/ACU+and+BCU+Motors
        self.declare_parameter("min_vol_ml", 300)
        self.declare_parameter("max_vol_ml", 2400)
        self.declare_parameter("dive_depth_m", 2.0)
        self.declare_parameter("surface_depth_m", 0.15)
        self.declare_parameter("acu_front_pitch", 150.0) #need to change
        self.declare_parameter("acu_back_pitch", -150.0) #need to change to real vals

        self.target_rpm = self.get_parameter("target_rpm").value
        self.min_vol = self.get_parameter("min_vol_ml").value
        self.max_vol = self.get_parameter("max_vol_ml").value
        self.depth_limit = self.get_parameter("dive_depth_m").value
        self.surface_limit = self.get_parameter("surface_depth_m").value
        self.acu_front_pitch = self.get_parameter("acu_front_pitch").value
        self.acu_back_pitch = self.get_parameter("acu_back_pitch").value

        # State machine
        self.state = "ASCENDING"
        self.current_vol_ml = 0
        self.current_depth_m = 0.0

        # Publishers
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)

        # Subscriptions
        self.vol_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._vol_callback
        )
        self.pressure_sub = create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._pressure_callback
        )

        # Control loop (10Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f"BCU Safety Oscillator: Safe Range [{self.min_vol}, {self.max_vol}] mL. Surface: < {self.surface_limit} m"
        )
        self.get_logger().info(
            f"ACU Depth-based Oscillator: Front pitch={self.acu_front_pitch}, Back pitch={self.acu_back_pitch}"
        )

    def _vol_callback(self, msg):
        self.current_vol_ml = msg.data

    def _pressure_callback(self, msg):
        self.current_depth_m = pressure_to_depth(float(msg.data))

    def _control_loop(self):
        rpm_cmd = 0

        # Log every 2 seconds
        if self.get_clock().now().nanoseconds % 2000000000 < 200000000:
            self.get_logger().info(
                f"STATUS: State={self.state}, Depth={self.current_depth_m:.2f}m, Vol={self.current_vol_ml}mL"
            )

        if self.state == "ASCENDING":
            if self.current_depth_m <= self.surface_limit:
                self.get_logger().info(
                    f"Reached surface at {self.current_depth_m:.2f}m."
                )
                self.state = "DESCENDING"

            elif self.current_vol_ml >= self.max_vol:
                rpm_cmd = 0

            else:
                rpm_cmd = self.target_rpm

        elif self.state == "DESCENDING":
            if self.current_depth_m >= self.depth_limit:
                self.get_logger().info(
                    f"Reached target dive depth at {self.current_depth_m:.2f}m."
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
