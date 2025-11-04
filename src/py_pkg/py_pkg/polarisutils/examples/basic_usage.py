#!/usr/bin/env python3
"""
Basic usage example for polarisutils.

Shows different ways to use the package - from simple constants
to optional convenience functions.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

# Import the essentials
from polarisutils import PolarisTopics, PolarisQoS, TOPIC_MESSAGE_MAP
from polarisutils import create_publisher_for_topic


class ExampleNode(Node):
    def __init__(self):
        super().__init__("example_node")

        # Method 1: Use constants with standard ROS2 (most explicit)
        self.leak_sub = self.create_subscription(
            TOPIC_MESSAGE_MAP[PolarisTopics.INTERNAL_LEAK],
            PolarisTopics.INTERNAL_LEAK,
            self.leak_callback,
            PolarisQoS.SAFETY_CRITICAL
        )

        # Method 2: Mix constants and manual types/QoS (flexible)
        self.flow_pub = self.create_publisher(
            Float32,
            PolarisTopics.BCU_FLOW_RATE,
            PolarisQoS.CONTROL
        )

        # Method 3: Use convenience function (simplest)
        self.pressure_pub = create_publisher_for_topic(
            self, PolarisTopics.EXTERNAL_PRESSURE
        )

        # Method 4: Still works - pure standard ROS2
        self.custom_pub = self.create_publisher(
            Int32, "/my/custom/topic", 10
        )

        self.get_logger().info("Example node started with multiple approaches")

    def leak_callback(self, msg):
        self.get_logger().warn(f"Leak detection: {msg}")


def main(args=None):
    rclpy.init(args=args)
    node = ExampleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()