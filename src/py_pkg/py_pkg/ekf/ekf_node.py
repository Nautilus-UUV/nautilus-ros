import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu

from py_pkg.uuv_ros_core.node_factory import (
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from py_pkg.uuv_ros_core.topics import UUVTopics

# Import EKF from the same package; relative import avoids module resolution issues
from .ekf_filter import EKFFilter


class EKFNode(Node):
    """
    ROS2 Node that runs an Extended Kalman Filter (EKF) for state estimation using IMU data.

    Subscribes to: UUVTopics.IMU_FILTERED_LEFT (sensor_msgs/Imu)
    Publishes to:  UUVTopics.POSITION_ESTIMATION (geometry_msgs/Pose)
    """

    def __init__(self):
        super().__init__("ekf_node")

        self.declare_parameter("dt", 0.01)
        dt = self.get_parameter("dt").get_parameter_value().double_value

        self.ekf = EKFFilter(dt)
        self._last_stamp = None  # tracks previous message timestamp
        self._initialized = False  # whether initial orientation has been set

        # Log throttle: emit the position line every Nth callback so the
        # ~50 Hz IMU stream doesn't flood the console (and CPU on the Pi).
        # 50 ≈ 1 Hz given the SDF's IMU `<update_rate>50.0</update_rate>`.
        self._log_every_n = 50
        self._cb_count = 0

        self.sub = create_subscription_for_topic(
            self, UUVTopics.IMU_FILTERED_LEFT, self.imu_callback
        )
        self.pub = create_publisher_for_topic(self, UUVTopics.POSITION_ESTIMATION)

        self.get_logger().info(
            "EKF Node started, subscribed to "
            f"{UUVTopics.IMU_FILTERED_LEFT} and publishing to "
            f"{UUVTopics.POSITION_ESTIMATION}"
        )

    def imu_callback(self, msg_in: Imu):
        """Run the EKF prediction and update step for incoming IMU data."""
        measured_accel = np.array(
            [
                msg_in.linear_acceleration.x,
                msg_in.linear_acceleration.y,
                msg_in.linear_acceleration.z,
            ]
        )
        measured_gyro = np.array(
            [
                msg_in.angular_velocity.x,
                msg_in.angular_velocity.y,
                msg_in.angular_velocity.z,
            ]
        )

        # On the first message, seed orientation from the IMU if available
        if not self._initialized:
            if msg_in.orientation_covariance[0] != -1.0:
                self.ekf.x[6] = msg_in.orientation.x
                self.ekf.x[7] = msg_in.orientation.y
                self.ekf.x[8] = msg_in.orientation.z
                self.ekf.x[9] = msg_in.orientation.w
            self._initialized = True

        # Compute dt from message timestamps; fall back to the parameter on the first message
        stamp = msg_in.header.stamp
        current_time = stamp.sec + stamp.nanosec * 1e-9
        if self._last_stamp is None:
            dt = None  # use self.ekf.dt (the parameter default)
        else:
            dt = current_time - self._last_stamp
            if dt <= 0.0:
                dt = None  # bad timestamp — fall back to default
        self._last_stamp = current_time

        # EKF Prediction step
        self.ekf.predict(measured_accel, measured_gyro, dt)
        # EKF Update step
        # Use accelerometer as attitude measurement; gyro is used as input in predict()
        self.ekf.update(measured_accel)

        # Publish position + orientation as a single Pose
        est_state = self.ekf.x
        pose_msg = Pose()
        pose_msg.position.x = est_state[0]
        pose_msg.position.y = est_state[1]
        pose_msg.position.z = est_state[2]
        pose_msg.orientation.x = est_state[6]
        pose_msg.orientation.y = est_state[7]
        pose_msg.orientation.z = est_state[8]
        pose_msg.orientation.w = est_state[9]
        self.pub.publish(pose_msg)

        self._cb_count += 1
        if self._cb_count % self._log_every_n == 0:
            self.get_logger().info(
                f"Current position [x,y,z]: "
                f"[{est_state[0]:.3f}, {est_state[1]:.3f}, {est_state[2]:.3f}] m"
            )


def main(args=None):
    # Catch SIGINT/SIGTERM so the process exits 0 instead of 1 on Ctrl-C —
    # otherwise launch_testing's exit-code check intermittently fails.
    rclpy.init(args=args)
    ekf_node = EKFNode()
    try:
        rclpy.spin(ekf_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ekf_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
