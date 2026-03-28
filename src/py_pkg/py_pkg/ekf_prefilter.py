#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class EkfPrefilter(Node):
    """
    Exponential Moving Average (EMA) prefilter for IMU data before EKF.
    - Subscribes:  /imu1                (sensor_msgs/Imu)
    - Publishes:   /filtered_imu_data   (sensor_msgs/Imu)

    Purpose:
        Smooth out high-frequency noise from raw IMU data (acceleration
        and angular velocity) before passing it to an EKF node.
    """

    def __init__(self):
        super().__init__('ekf_prefilter')

        # Filter coefficient: 0 = very smooth, 1 = no filtering
        self.alpha = 0.5

        # Previous filtered values (initialized on first message)
        self.prev_ax = None
        self.prev_ay = None
        self.prev_az = None
        self.prev_wx = None
        self.prev_wy = None
        self.prev_wz = None

        # Subscriber and publisher
        # best_effort QoS matches the standard reliability of IMU sensor publishers
        imu_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            Imu,
            '/imu1',
            self.imu_callback,
            imu_qos
        )

        self.pub = self.create_publisher(
            Imu,
            '/filtered_imu_data',
            10
        )

        self.get_logger().info(f'EKF prefilter started (alpha={self.alpha})')

    def ema(self, x_new, x_prev):
        """Exponential Moving Average step; on first sample return x_new directly."""
        if x_prev is None:
            return x_new
        a = self.alpha
        return a * x_new + (1.0 - a) * x_prev

    def imu_callback(self, msg_in: Imu):
        # Filter linear acceleration
        fax = self.ema(msg_in.linear_acceleration.x, self.prev_ax)
        fay = self.ema(msg_in.linear_acceleration.y, self.prev_ay)
        faz = self.ema(msg_in.linear_acceleration.z, self.prev_az)

        # Filter angular velocity
        fwx = self.ema(msg_in.angular_velocity.x, self.prev_wx)
        fwy = self.ema(msg_in.angular_velocity.y, self.prev_wy)
        fwz = self.ema(msg_in.angular_velocity.z, self.prev_wz)

        # Update previous values for next iteration
        self.prev_ax, self.prev_ay, self.prev_az = fax, fay, faz
        self.prev_wx, self.prev_wy, self.prev_wz = fwx, fwy, fwz

        # Prepare output message
        msg_out = Imu()
        msg_out.header = msg_in.header

        # Pass orientation through from the raw IMU (not filtered — already computed by sensor)
        msg_out.orientation = msg_in.orientation
        msg_out.orientation_covariance = msg_in.orientation_covariance

        # Copy covariance and fill filtered data
        msg_out.angular_velocity.x = fwx
        msg_out.angular_velocity.y = fwy
        msg_out.angular_velocity.z = fwz
        msg_out.angular_velocity_covariance = msg_in.angular_velocity_covariance

        msg_out.linear_acceleration.x = fax
        msg_out.linear_acceleration.y = fay
        msg_out.linear_acceleration.z = faz
        msg_out.linear_acceleration_covariance = msg_in.linear_acceleration_covariance

        # Publish the filtered IMU message
        self.pub.publish(msg_out)


def main():
    rclpy.init()
    node = EkfPrefilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
