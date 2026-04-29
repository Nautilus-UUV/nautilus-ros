"""
Author: Lisa Lustenberger
Date: November 2025
Description:    Node that controls roll and pitch. Part of Control Node -> possibly combine with BCU control node
                Input: UUVTopics.POSITION_ESTIMATION, UUVTopics.POSITION_TARGET -> current and target pose of the UUV.
                Output: UUVTopics.ACU_ROLL, UUVTopics.ACU_PITCH -> target motor positions for roll (angle) and pitch (mm) for the ACU motors.
Background:     Based on depth_control_node from Divetest 2025, modified for pitch and roll control.
"""

#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose

# from sensor_msgs.msg import Imu
# from StatePackage.msg import StateVector
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Float32, Float64

from py_pkg.pid.acu_control_system import ACUController as ControlSystem
from py_pkg.math_utils import euler_to_direction
from py_pkg.uuv_ros_core.node_factory import (
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from py_pkg.uuv_ros_core.topics import UUVTopics


class ACUControlNode(Node):
    def __init__(self):
        super().__init__("acu_control_node")

        # Create a reentrant callback group to allow concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        self.control_system = ControlSystem()
        self.current_time = self.get_clock().now().nanoseconds / 1e9

        self.control_output = 0.0  # Control output for buoyancy engine

        # Publisher for the target depth
        self.target_pitch_publisher = create_publisher_for_topic(Float64, "acu/pitch", 10)
        self.target_roll_publisher = create_publisher_for_topic(Float64, "acu/roll", 10)

        # Subscriber for the current depth
        self.current_pose = create_subscription_for_topic(
            UUVTopics.POSITION_ESTIMATION,
            self.current_pose_callback,
            10,
            callback_group=self.callback_group,
        )
        self.target_pose = create_subscription_for_topic(
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            10,
            callback_group=self.callback_group,
        )

        # Subscriber for SIMULATION
        self.imu_simulated = create_subscription_for_topic(
            UUVTopics.IMU, self.imu_callback, 10, callback_group=self.callback_group
        )

        # Timer to periodically run the control loop
        self.control_timer = self.create_timer(
            1.0 / 10.0,  # 10 Hz control frequency (adjust as needed)
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info("ACU control node started.")

    def target_pose_callback(self, msg: Pose):
        self.target_pose = msg
        q = self.target_pose.orientation
        roll, pitch, _ = euler_to_direction([q.x, q.y, q.z, q.w])
        self.target_roll = roll
        self.target_pitch = pitch
        # self.control_system.target_pose = self.target_pose
        self.get_logger().info(f"Updated target pose: {self.target_pose}")

    def current_pose_callback(self, msg: Pose):
        self.current_pose = msg
        q = self.current_pose.orientation
        roll, pitch, _ = euler_to_direction([q.x, q.y, q.z, q.w])
        self.current_roll = roll
        self.current_pitch = pitch
        self.get_logger().debug(f"Received current pose: {self.current_pose}")

    def control_loop(self):
        """Control loop to update ACU commands based on current and target poses. -> calls on ControlSystem, which currently uses P-controller"""
        self.current_time = self.get_clock().now().nanoseconds / 1e9
        control_output = self.control_system.update(
            self.target_roll, self.target_pitch, self.current_roll, self.current_pitch
        )
        if control_output is not None:
            self.control_output = control_output
            msg_pitch = Float32()
            msg_roll = Float32()
            pitch_cmd = self.control_output["pitch"]
            roll_cmd = self.control_output["roll"]
            msg_pitch.data = float(pitch_cmd)
            msg_roll.data = float(roll_cmd)
            self.target_pitch_publisher.publish(msg_pitch)
            self.target_roll_publisher.publish(msg_roll)
            self.get_logger().debug(
                f"Fraction of bladder volume filled per second in Hz: {self.control_output}"
            )
            self.get_logger().debug(f"Command to motor in RPM: {self.control_output}")


class TopicCheckerNode(Node):
    def __init__(self, topic_name):
        super().__init__("topic_checker_node")
        self.topic_name = topic_name
        self.message_received = False
        self.subscription = self.create_subscription(
            Float64,  # Replace with the appropriate message type
            topic_name,
            self.topic_callback,
            10,
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f"Waiting for message on topic {self.topic_name}")

    def topic_callback(self, msg):
        self.message_received = True
        self.get_logger().info(f"Message received on topic {self.topic_name}")


def main(args=None):
    rclpy.init(args=args)

    topic_name = "target_depth"
    topic_checker_node = TopicCheckerNode(topic_name)

    # Wait for a message to be received on the topic
    while not topic_checker_node.message_received:
        rclpy.spin_once(topic_checker_node, timeout_sec=1.0)

    # If a message is received, start the ACUControlNode
    acu_control_node = ACUControlNode()
    rclpy.spin(acu_control_node)

    acu_control_node.destroy_node()
    topic_checker_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
