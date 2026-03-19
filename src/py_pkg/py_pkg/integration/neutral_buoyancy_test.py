import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class NeutralBuoyancyTest(Node):
    """
    Integration test that fuses Depth and IMU data to find neutral buoyancy.
    The glider is considered stationary when:
    1. Vertical velocity (from depth) is < tolerance.
    2. Angular velocity (from IMU) is < tolerance.
    3. Linear acceleration (from IMU) is stable (gravity only).
    """

    def __init__(self):
        super().__init__("neutral_buoyancy_test")

        # Control Parameters
        self.declare_parameter("k_p", 1200.0)  # Gain: RPM per cm/s of error
        self.declare_parameter("max_rpm", 1500)
        self.declare_parameter("velocity_tolerance", 0.15)  # cm/s
        self.declare_parameter("gyro_tolerance", 0.02)  # rad/s
        self.declare_parameter("accel_tolerance", 0.05)  # m/s^2 (deviation from mean)
        self.declare_parameter("settle_time", 10.0)  # seconds

        self.k_p = self.get_parameter("k_p").value
        self.max_rpm = self.get_parameter("max_rpm").value
        self.v_tol = self.get_parameter("velocity_tolerance").value
        self.g_tol = self.get_parameter("gyro_tolerance").value
        self.a_tol = self.get_parameter("accel_tolerance").value
        self.settle_time = self.get_parameter("settle_time").value

        # State Variables
        self.current_depth = None
        self.last_depth = None
        self.last_time = None
        self.v_vert = 0.0  # Vertical velocity (cm/s)
        self.imu_msg = None
        self.accel_history = []  # For checking acceleration stability

        self.stationary_start_time = None
        self.is_finished = False

        # Comms
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)

        self.depth_sub = create_subscription_for_topic(
            self, UUVTopics.TEST_EXTERNAL_DEPTH, self._depth_callback
        )
        self.imu_sub = create_subscription_for_topic(
            self, UUVTopics.IMU_LEFT, self._imu_callback
        )

        # Main Loop (10Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("Neutral Buoyancy Test: Initialized.")

    def _depth_callback(self, msg):
        now = self.get_clock().now()
        depth = float(msg.data)

        if self.last_depth is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0:
                # sinking is positive depth
                instant_v = (depth - self.last_depth) / dt
                self.v_vert = 0.8 * self.v_vert + 0.2 * instant_v

        self.last_depth = depth
        self.last_time = now
        self.current_depth = depth

    def _imu_callback(self, msg):
        self.imu_msg = msg
        # Maintain a short window of acceleration magnitudes to check for "shaking"
        accel = msg.linear_acceleration
        mag = np.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        self.accel_history.append(mag)
        if len(self.accel_history) > 50:
            self.accel_history.pop(0)

    def _control_loop(self):
        if self.current_depth is None or self.imu_msg is None or self.is_finished:
            return

        # decide on RPM depending on the veritcal speed
        # we try to get to 0
        rpm_cmd = int(self.v_vert * self.k_p)
        rpm_cmd = max(-self.max_rpm, min(self.max_rpm, rpm_cmd))

        # (I) check vertical velocity
        v_ok = abs(self.v_vert) < self.v_tol

        # (II) check angular velocity
        gyro = self.imu_msg.angular_velocity
        g_mag = np.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2)
        g_ok = g_mag < self.g_tol

        # (III): Check external forces such as acceleration
        a_ok = False
        if len(self.accel_history) >= 20:
            a_std = np.std(self.accel_history)
            a_ok = a_std < self.a_tol

        # maintain stability for 10 seconds
        if v_ok and g_ok and a_ok:
            if self.stationary_start_time is None:
                self.stationary_start_time = self.get_clock().now()
                self.get_logger().info("Criteria met. Settling...")
            else:
                elapsed = (
                    self.get_clock().now() - self.stationary_start_time
                ).nanoseconds / 1e9
                if elapsed >= self.settle_time:
                    self.get_logger().info("--- TEST SUCCESSFUL ---")
                    self.get_logger().info(
                        f"Stationary at depth: {self.current_depth:.1f} cm"
                    )
                    self.get_logger().info(f"Final V_vert: {self.v_vert:.3f} cm/s")
                    self.get_logger().info(f"Final Gyro: {g_mag:.4f} rad/s")
                    self.is_finished = True
                    rpm_cmd = 0
        else:
            if self.stationary_start_time is not None:
                self.get_logger().warn(f"Lost stability! V:{v_ok} G:{g_ok} A:{a_ok}")
            self.stationary_start_time = None

        # publish rpm to BCU
        cmd_msg = Int32()
        cmd_msg.data = rpm_cmd
        self.rpm_pub.publish(cmd_msg)

        if self.get_clock().now().nanoseconds % 1000000000 < 100000000:
            self.get_logger().info(
                f"D:{self.current_depth:5.1f}cm | V:{self.v_vert:+6.2f} | G:{g_mag:5.3f} | RPM:{rpm_cmd:5d}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = NeutralBuoyancyTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
