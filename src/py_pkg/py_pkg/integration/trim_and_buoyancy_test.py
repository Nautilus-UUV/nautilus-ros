import math

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

from ..physics import pressure_to_depth
from ..uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class AutoTrimAndBuoyancyTest(Node):
    """
    Integration test that fuses Depth and IMU data to find neutral buoyancy
    AND horizontal trim.
    The glider is considered stationary and trimmed when:
    1. Vertical velocity is < tolerance.
    2. Pitch angle is < tolerance.
    3. Angular velocity is < tolerance.
    4. Linear acceleration is stable.
    """

    def __init__(self):
        super().__init__("auto_trim_and_buoyancy")

        # BCU Control Parameters
        self.declare_parameter("bcu_k_p", 120000.0)  # Gain: RPM per m/s of error
        self.declare_parameter("max_rpm", 4100)
        self.declare_parameter("velocity_tolerance", 0.0015)  # m/s

        # ACU Control Parameters
        self.declare_parameter(
            "acu_k_p", 500.0
        )  # Gain: mm of mass translation per radian of pitch
        self.declare_parameter("max_pitch", 300.0)  # Maximum mass translation in mm
        self.declare_parameter("pitch_tolerance", 0.05)  # radians (~2.8 degrees)

        # General Stability Parameters
        self.declare_parameter("gyro_tolerance", 0.02)  # rad/s
        self.declare_parameter("accel_tolerance", 0.05)  # m/s^2 (deviation from mean)
        self.declare_parameter("settle_time", 10.0)  # seconds

        self.bcu_k_p = self.get_parameter("bcu_k_p").value
        self.max_rpm = self.get_parameter("max_rpm").value
        self.v_tol = self.get_parameter("velocity_tolerance").value

        self.acu_k_p = self.get_parameter("acu_k_p").value
        self.max_pitch = self.get_parameter("max_pitch").value
        self.p_tol = self.get_parameter("pitch_tolerance").value

        self.g_tol = self.get_parameter("gyro_tolerance").value
        self.a_tol = self.get_parameter("accel_tolerance").value
        self.settle_time = self.get_parameter("settle_time").value

        # State Variables (depths in metres, velocities in m/s)
        self.current_depth = None
        self.last_depth = None
        self.last_time = None
        self.v_vert = 0.0

        self.imu_msg = None
        self.current_pitch = 0.0
        self.accel_history = []

        self.stationary_start_time = None
        self.is_finished = False

        # Comms: Publishers
        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)

        # Comms: Subscriptions
        self.pressure_sub = create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._pressure_callback
        )
        self.imu_sub = create_subscription_for_topic(
            self, UUVTopics.IMU_LEFT, self._imu_callback
        )

        # Main Loop (10Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("Auto Trim & Neutral Buoyancy Test: Initialized.")

    def _euler_from_quaternion(self, q):
        """
        Convert quaternion into euler angles (roll, pitch, yaw)
        Pitch is rotation around Y-axis.
        """
        # Pitch (y-axis rotation)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        # Avoid out of bounds errors from floating point precision
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        return pitch

    def _pressure_callback(self, msg):
        now = self.get_clock().now()
        depth = pressure_to_depth(float(msg.data))

        if self.last_depth is not None:
            dt = (now - self.last_time).nanoseconds / 1e9
            if dt > 0:
                instant_v = (depth - self.last_depth) / dt
                self.v_vert = 0.8 * self.v_vert + 0.2 * instant_v

        self.last_depth = depth
        self.last_time = now
        self.current_depth = depth

    def _imu_callback(self, msg):
        self.imu_msg = msg

        # Calculate pitch from orientation quaternion
        q = msg.orientation
        self.current_pitch = self._euler_from_quaternion(q)

        # Maintain acceleration magnitude window
        accel = msg.linear_acceleration
        mag = np.sqrt(accel.x**2 + accel.y**2 + accel.z**2)
        self.accel_history.append(mag)
        if len(self.accel_history) > 50:
            self.accel_history.pop(0)

    def _control_loop(self):
        if self.current_depth is None or self.imu_msg is None or self.is_finished:
            return

        # --- 1. Buoyancy Control (BCU) ---
        rpm_cmd = int(self.v_vert * self.bcu_k_p)
        rpm_cmd = max(-self.max_rpm, min(self.max_rpm, rpm_cmd))

        # --- 2. Trim Control (ACU) ---
        # If pitch is positive (nose down), move mass backward (negative pitch)
        # Check your specific frame conventions; you may need to invert the sign here.
        pitch_cmd = -1.0 * (self.current_pitch * self.acu_k_p)
        pitch_cmd = max(-self.max_pitch, min(self.max_pitch, float(pitch_cmd)))

        # --- 3. Check Stability Criteria ---
        v_ok = abs(self.v_vert) < self.v_tol
        p_ok = abs(self.current_pitch) < self.p_tol

        gyro = self.imu_msg.angular_velocity
        g_mag = np.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2)
        g_ok = g_mag < self.g_tol

        a_ok = False
        if len(self.accel_history) >= 20:
            a_std = np.std(self.accel_history)
            a_ok = a_std < self.a_tol

        # --- 4. State Machine ---
        if v_ok and p_ok and g_ok and a_ok:
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
                        f"Stationary at depth: {self.current_depth:.2f} m"
                    )
                    self.get_logger().info(f"Final V_vert: {self.v_vert:.4f} m/s")
                    self.get_logger().info(f"Final Pitch: {self.current_pitch:.4f} rad")
                    self.get_logger().info(f"Final Gyro: {g_mag:.4f} rad/s")
                    self.is_finished = True
                    rpm_cmd = 0
                    # Note: We do not zero out pitch_cmd, as the mass needs to stay
                    # in its current position to maintain horizontal trim.
        else:
            if self.stationary_start_time is not None:
                self.get_logger().warn(
                    f"Lost stability! V:{v_ok} P:{p_ok} G:{g_ok} A:{a_ok}"
                )
            self.stationary_start_time = None

        # --- 5. Publish Commands ---
        rpm_msg = Int32()
        rpm_msg.data = rpm_cmd
        self.rpm_pub.publish(rpm_msg)

        pitch_msg = Float32()
        pitch_msg.data = pitch_cmd
        self.pitch_pub.publish(pitch_msg)

        # Logging
        if self.get_clock().now().nanoseconds % 1000000000 < 100000000:
            self.get_logger().info(
                f"D:{self.current_depth:5.2f}m | V:{self.v_vert:+7.4f} | "
                f"P:{self.current_pitch:+5.3f}rad | ACU:{pitch_cmd:+6.1f}mm | RPM:{rpm_cmd:5d}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = AutoTrimAndBuoyancyTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
