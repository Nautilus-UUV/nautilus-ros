import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16

from ..physics import (
    depth_to_pressure_pa,
    gauge_pressure_pa,
    pressure_to_depth,
)
from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class BCUOscillator(Node):
    """
    Robust oscillator with better feedback and logging to ensure the glider
    reaches the surface and reverses.

    State machine triggers run on gauge pressure (Pa); depth in metres
    is shown only in human-facing log lines.
    """

    def __init__(self):
        super().__init__("bcu_oscillator")

        # Configuration
        self.declare_parameter(
            "target_rpm", 4100
        )  # taken max from: https://aris-space.atlassian.net/wiki/spaces/Nautilus/pages/306839555/ACU+and+BCU+Motors
        self.declare_parameter("min_vol_ml", 300)
        self.declare_parameter("max_vol_ml", 2400)
        # Setpoints declared in Pa (gauge); defaults match the previous
        # 2.0 m / 0.3 m thresholds at salt-water density.
        self.declare_parameter(
            "dive_pressure_pa",
            float(gauge_pressure_pa(depth_to_pressure_pa(2.0))),
        )
        self.declare_parameter(
            "surface_pressure_pa",
            float(gauge_pressure_pa(depth_to_pressure_pa(0.3))),
        )

        self.target_rpm = self.get_parameter("target_rpm").value
        self.min_vol = self.get_parameter("min_vol_ml").value
        self.max_vol = self.get_parameter("max_vol_ml").value
        self.dive_pressure_pa = float(self.get_parameter("dive_pressure_pa").value)
        self.surface_pressure_pa = float(
            self.get_parameter("surface_pressure_pa").value
        )

        # State machine
        self.state = "ASCENDING"
        self.current_vol_ml = 0
        self.current_pressure_pa = 0.0

        # Publishers
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)

        # Subscriptions
        self.vol_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_VOLUME, self._vol_callback
        )
        self.pressure_sub = create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._pressure_callback
        )

        # Control loop (10Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f"BCU Safety Oscillator: Safe Range [{self.min_vol}, {self.max_vol}] mL. "
            f"Surface threshold: < {self.surface_pressure_pa:.0f} Pa "
            f"(~{pressure_to_depth(self.surface_pressure_pa + 101_325):.2f} m)"
        )

    def _vol_callback(self, msg):
        self.current_vol_ml = msg.data

    def _pressure_callback(self, msg):
        # EXTERNAL_PRESSURE is absolute Pa; convert to gauge once at ingress.
        self.current_pressure_pa = gauge_pressure_pa(float(msg.data))

    def _control_loop(self):
        rpm_cmd = 0

        depth_m_for_log = self.current_pressure_pa / 10_050.65  # display only
        self.get_logger().info(
            f"STATUS: State={self.state}, "
            f"P={self.current_pressure_pa:.0f}Pa (~{depth_m_for_log:.2f}m), "
            f"Vol={self.current_vol_ml}mL",
            throttle_duration_sec=2.0,
        )

        if self.state == "ASCENDING":
            if self.current_pressure_pa <= self.surface_pressure_pa:
                self.get_logger().info(
                    f"Reached surface at {self.current_pressure_pa:.0f} Pa."
                )
                self.state = "DESCENDING"

            elif self.current_vol_ml >= self.max_vol:
                rpm_cmd = 0

            else:
                rpm_cmd = self.target_rpm

        elif self.state == "DESCENDING":
            if self.current_pressure_pa >= self.dive_pressure_pa:
                self.get_logger().info(
                    f"Reached target dive pressure at {self.current_pressure_pa:.0f} Pa."
                )
                self.state = "ASCENDING"

            elif self.current_vol_ml <= self.min_vol:
                rpm_cmd = 0

            else:
                rpm_cmd = -self.target_rpm

        # Publish command
        msg = Int16()
        msg.data = int(rpm_cmd)
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
