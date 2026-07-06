#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from py_pkg.uuv_ros_core.node_factory import (
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from py_pkg.uuv_ros_core.node_runtime import spin_node
from py_pkg.uuv_ros_core.topics import UUVTopics


class ImuPrefilter(Node):
    """
    Exponential Moving Average (EMA) prefilter for the IMU stream.

    - Subscribes:  UUVTopics.IMU          (sensor_msgs/Imu)
    - Publishes:   UUVTopics.IMU_FILTERED (sensor_msgs/Imu)

    Purpose:
        Smooth out high-frequency noise from the raw IMU (acceleration and
        angular velocity) before the attitude estimator reads the gravity
        vector off it.
    """

    def __init__(self):
        super().__init__("imu_prefilter")

        # EMA coefficient (0 = very smooth, 1 = no filtering). The filtered
        # stream feeds two consumers with opposite biases: the attitude
        # estimator runs on it at the full IMU rate and wants low lag, while
        # the MQTT egress to the operator UI is decimated to 10 Hz by plain
        # sample-dropping, so it wants the content band-limited below ~5 Hz or
        # the thinning aliases high-frequency noise into the display.
        #
        # alpha=0.15 puts the -3 dB cutoff at ~5 Hz for a 200 Hz IMU
        #   (alpha = 1 - exp(-2*pi*fc/fs)) with ~28 ms of group delay
        #   (~(1-alpha)/alpha samples).
        self.alpha = 0.15

        # Previous filtered values (initialized on first message)
        self.prev_ax = None
        self.prev_ay = None
        self.prev_az = None
        self.prev_wx = None
        self.prev_wy = None
        self.prev_wz = None

        self.sub = create_subscription_for_topic(
            self, UUVTopics.IMU, self.imu_callback
        )
        self.pub = create_publisher_for_topic(self, UUVTopics.IMU_FILTERED)

        self.get_logger().info(f"IMU prefilter started (alpha={self.alpha})")

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
    node = ImuPrefilter()
    spin_node(node)


if __name__ == "__main__":
    main()
