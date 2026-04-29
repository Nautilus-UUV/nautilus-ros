# Copied from divetest files: depth_control_node.py

#!/usr/bin/env python3

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Int32

from py_pkg import math_utils as SimMath
from py_pkg.physics import pressure_to_depth, q_to_rpm
from py_pkg.pid import depth_control_system as ControlSystem
from py_pkg.pid.depth_config import (
    init_buoyancy_engine,
    init_control,
    init_motor,
    init_pos,
)
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class DepthControlNode(Node):
    def __init__(self):
        super().__init__("depth_control_node")

        # Reentrant callback group so subscriptions and the control timer
        # can run concurrently.
        self.callback_group = ReentrantCallbackGroup()

        self.control_system = ControlSystem.DepthControlSystem(init_control)
        self.current_position = SimMath.Vector(
            init_pos.get("x"), init_pos.get("y"), init_pos.get("z")
        )
        self.current_bladder_level = init_buoyancy_engine.get(
            "initial_proportion_full"
        )
        self.bladder_volume = init_buoyancy_engine.get("tank_volume")
        self.current_time = self.get_clock().now().nanoseconds / 1e9

        self.control_output = 0.0
        self.motor_rpm = 0.0
        self.target_depth = 0.0
        self.current_depth = 0.0

        self.bcu_controller_rpm_publisher = create_publisher_for_topic(
            self, UUVTopics.BCU_RPM, callback_group=self.callback_group
        )

        self.target_depth_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.TARGET_DEPTH,
            self.target_depth_callback,
            callback_group=self.callback_group,
        )

        self.pressure_external_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.EXTERNAL_PRESSURE,
            self.current_depth_callback,
            callback_group=self.callback_group,
        )

        self.control_timer = self.create_timer(
            1.0 / 10.0,  # 10 Hz control frequency
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Depth control node started.")

    def target_depth_callback(self, msg):
        self.target_depth = float(msg.data)
        self.control_system.target_depth = self.target_depth
        self.get_logger().info(f"Updated target depth: {self.target_depth}")

    def current_depth_callback(self, msg):
        self.current_depth = pressure_to_depth(float(msg.data))
        self.get_logger().debug(f"Received current depth: {self.current_depth}")

    def control_loop(self):
        self.current_time = self.get_clock().now().nanoseconds / 1e9
        current_position = SimMath.Vector(0.0, 0.0, self.current_depth)
        self.control_output = self.control_system.calc_acc(
            current_position,
            0.0,
            self.current_time,
        )

        msg = Int32()
        self.motor_rpm = q_to_rpm(self.control_output, self.bladder_volume)
        min_rpm = init_motor.get("min_rpm")
        max_rpm = init_motor.get("max_rpm")
        if abs(self.motor_rpm) < min_rpm:
            # Pump cannot run reliably below min_rpm; suppress small commands
            # instead of slamming to +/-min_rpm (which would limit-cycle the setpoint).
            self.motor_rpm = 0.0
        elif self.motor_rpm > 0:
            self.motor_rpm = SimMath.clamp(self.motor_rpm, min_rpm, max_rpm)
        else:
            self.motor_rpm = SimMath.clamp(self.motor_rpm, -max_rpm, -min_rpm)
        # Pump wiring inverts direction: positive q (fill bladder → sink) is
        # delivered as a negative RPM command on the BCU bus.
        msg.data = int(-1 * self.motor_rpm)
        self.bcu_controller_rpm_publisher.publish(msg)
        self.get_logger().debug(
            f"Fraction of bladder volume filled per second in Hz: {self.control_output}"
        )
        self.get_logger().debug(f"Command to motor in RPM: {self.motor_rpm}")


def main(args=None):
    rclpy.init(args=args)
    depth_control_node = DepthControlNode()
    try:
        rclpy.spin(depth_control_node)
    finally:
        depth_control_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
