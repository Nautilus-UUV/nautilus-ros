import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
# Message type for position output (x, y, z)
from geometry_msgs.msg import Point, Quaternion
import numpy as np

# Import EKF from the same package; relative import avoids module resolution issues
from .ekf_filter import EKFFilter


class EKFNode(Node):
    """
    ROS2 Node that runs an Extended Kalman Filter (EKF) for state estimation using IMU data.

    Subscribes to: /filtered_imu_data  (sensor_msgs/Imu)
    Publishes to:  /ekf_position       (geometry_msgs/Point)
                   /ekf_orientation    (geometry_msgs/Quaternion)
    """

    def __init__(self):
        super().__init__('ekf_node')

        # Declare parameters
        self.declare_parameter('dt', 0.01)

        # Get parameters
        dt = self.get_parameter('dt').get_parameter_value().double_value

        # EKF filter instance
        self.ekf = EKFFilter(dt)
        self._last_stamp = None    # tracks previous message timestamp
        self._initialized = False  # whether initial orientation has been set

        # Subscriber to filtered IMU data
        self.sub = self.create_subscription(
            Imu,
            '/filtered_imu_data',
            self.imu_callback,
            10
        )

        # Publisher for estimated position
        self.pub = self.create_publisher(
            Point,
            '/ekf_position',
            10
        )

        # Publisher for estimated orientation
        self.pub_orientation = self.create_publisher(
            Quaternion,
            '/ekf_orientation',
            10
        )

        self.get_logger().info(
            'EKF Node started, subscribed to /filtered_imu_data and '
            'publishing to /ekf_position and /ekf_orientation'
        )

    def imu_callback(self, msg_in: Imu):
        """Run the EKF prediction and update step for incoming IMU data."""
        # Extract linear acceleration and angular velocity from the IMU message
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

        # Publish position as geometry_msgs/Point
        est_state = self.ekf.x
        position_msg = Point()
        position_msg.x = est_state[0]
        position_msg.y = est_state[1]
        position_msg.z = est_state[2]
        self.pub.publish(position_msg)

        # Publish orientation as geometry_msgs/Quaternion [x, y, z, w]
        orientation_msg = Quaternion()
        orientation_msg.x = est_state[6]
        orientation_msg.y = est_state[7]
        orientation_msg.z = est_state[8]
        orientation_msg.w = est_state[9]
        self.pub_orientation.publish(orientation_msg)

        self.get_logger().info(
            f'Current position [x,y,z]: '
            f'[{est_state[0]:.3f}, {est_state[1]:.3f}, {est_state[2]:.3f}] m'
        )


def main(args=None):
    rclpy.init(args=args)
    ekf_node = EKFNode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
