#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from py_pkg.scenarios.compile import imu_prefilter_spec_from_node
from py_pkg.uuv_ros_core.node_factory import (
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from py_pkg.uuv_ros_core.node_runtime import spin_node
from py_pkg.uuv_ros_core.topics import UUVTopics


def ema(alpha: float, x_new: float, x_prev: float | None) -> float:
    """One EMA step: ``alpha*x_new + (1-alpha)*x_prev``.

    On the first sample (``x_prev`` is None) the input passes through —
    the filter must not wind up from an implicit zero.
    """
    if x_prev is None:
        return x_new
    return alpha * x_new + (1.0 - alpha) * x_prev


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

        # EMA coefficient, from the scenario (ImuPrefilterSpec.alpha, which
        # carries the cutoff/group-delay derivation). The filtered stream feeds
        # two consumers with opposite biases: the attitude estimator runs on it
        # at the full IMU rate and wants low lag, while the MQTT egress to the
        # operator UI is decimated to 10 Hz by plain sample-dropping, so it
        # wants the content band-limited below ~5 Hz or the thinning aliases
        # high-frequency noise into the display.
        self.alpha = imu_prefilter_spec_from_node(self).alpha

        # Previous filtered (x, y, z) tuples, initialized on first message.
        self._prev_accel = None
        self._prev_gyro = None

        self.sub = create_subscription_for_topic(
            self, UUVTopics.IMU, self.imu_callback
        )
        self.pub = create_publisher_for_topic(self, UUVTopics.IMU_FILTERED)

        self.get_logger().info(f"IMU prefilter started (alpha={self.alpha})")

    def _ema3(self, vec, prev):
        """EMA-filter a Vector3 in place; on first sample it passes through.

        Returns the new (x, y, z) tuple to store as the next `prev`.
        """
        if prev is None:
            return (vec.x, vec.y, vec.z)
        vec.x, vec.y, vec.z = new = tuple(
            ema(self.alpha, n, p) for n, p in zip((vec.x, vec.y, vec.z), prev)
        )
        return new

    def imu_callback(self, msg: Imu):
        # Header, orientation, and all covariances pass through untouched; only
        # the six accel/gyro components are filtered — in place on the incoming
        # message (the callback owns it), so no per-sample Imu rebuild at the
        # raw IMU rate.
        self._prev_accel = self._ema3(msg.linear_acceleration, self._prev_accel)
        self._prev_gyro = self._ema3(msg.angular_velocity, self._prev_gyro)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ImuPrefilter()
    spin_node(node)


if __name__ == "__main__":
    main()
