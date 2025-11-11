import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
# Message type for position output (x, y, z)
from geometry_msgs.msg import Point
import numpy as np

# Import EKF from the same package; relative import avoids module resolution issues
from .ekf_filter import EKF_Filter


class Ekf_Node(Node):
    """
    ROS2 Node that runs an Extended Kalman Filter (EKF) for state estimation using IMU data.
    Subscribes to: /filtered_imu_data (sensor_msgs/Imu)
    Publishes to:
    """

    def __init__(self):
        super().__init__('ekf_node')

        # Declare parameters
        self.declare_parameter('dt', 0.01)

        # Get parameters
        dt = self.get_parameter('dt').get_parameter_value().double_value

        # EKF filter instance
        self.ekf = EKF_Filter(dt)

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

        self.get_logger().info(
            'EKF Node started, subscribed to /filtered_imu_data and publishing to /ekf_position')

    def imu_callback(self, msg_in: Imu):
        """Callback for incoming IMU data to perform EKF prediction and update."""

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

        # EKF Prediction step
        self.ekf.predict(measured_accel, measured_gyro)
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

        self.get_logger().info(
            f'Current position [x,y,z]: [{est_state[0]:.3f}, {est_state[1]:.3f}, {est_state[2]:.3f}] m')


def main(args=None):
    rclpy.init(args=args)
    ekf_node = Ekf_Node()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
