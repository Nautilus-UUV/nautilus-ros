import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, Quaternion
import numpy as np

from .ekf_filter import EKF_Filter


class EkfPredictNode(Node):
    """
    Predict-only EKF node for testing purposes.
    Identical to ekf_node but the update (measurement correction) step is disabled.
    This gives a baseline of pure dead-reckoning to compare against the full EKF.

    Subscribes to: /filtered_imu_data (sensor_msgs/Imu)
    Publishes to:  /ekf_predict_position    (geometry_msgs/Point)
                   /ekf_predict_orientation (geometry_msgs/Quaternion)
    """

    def __init__(self):
        super().__init__('ekf_predict_node')

        self.declare_parameter('dt', 0.01)
        dt = self.get_parameter('dt').get_parameter_value().double_value

        self.ekf = EKF_Filter(dt)
        self._last_stamp = None
        self._initialized = False

        self.sub = self.create_subscription(
            Imu,
            '/filtered_imu_data',
            self.imu_callback,
            10
        )

        self.pub_position = self.create_publisher(
            Point,
            '/ekf_predict_position',
            10
        )

        self.pub_orientation = self.create_publisher(
            Quaternion,
            '/ekf_predict_orientation',
            10
        )

        self.get_logger().info(
            'EKF Predict Node started (no update step), publishing to '
            '/ekf_predict_position and /ekf_predict_orientation')

    def imu_callback(self, msg_in: Imu):
        """Callback for incoming IMU data — prediction step only, no update."""

        measured_accel = np.array([
            msg_in.linear_acceleration.x,
            msg_in.linear_acceleration.y,
            msg_in.linear_acceleration.z
        ])
        measured_gyro = np.array([
            msg_in.angular_velocity.x,
            msg_in.angular_velocity.y,
            msg_in.angular_velocity.z
        ])

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
            dt = None
        else:
            dt = current_time - self._last_stamp
            if dt <= 0.0:
                dt = None
        self._last_stamp = current_time

        # Prediction step only — no update
        self.ekf.predict(measured_accel, measured_gyro, dt)

        est_state = self.ekf.x

        position_msg = Point()
        position_msg.x = est_state[0]
        position_msg.y = est_state[1]
        position_msg.z = est_state[2]
        self.pub_position.publish(position_msg)

        orientation_msg = Quaternion()
        orientation_msg.x = est_state[6]
        orientation_msg.y = est_state[7]
        orientation_msg.z = est_state[8]
        orientation_msg.w = est_state[9]
        self.pub_orientation.publish(orientation_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EkfPredictNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
