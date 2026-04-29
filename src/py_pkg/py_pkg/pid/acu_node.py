#!/usr/bin/env python3
"""ACU control node.

Owns a per-axis controller for roll and pitch. Roll uses AxisController
(degrees-in/degrees-out, slews current toward target). Pitch uses
MassShifterController (degrees-in/metres-out, commands mass-shifter
stroke directly from pitch error). Subscribes to POSITION_ESTIMATION
(current pose) and POSITION_TARGET (desired pose), extracts roll/pitch
from the quaternion, and publishes Int32 motor-step commands on
ACU_ROLL_STEPS / ACU_PITCH_STEPS.
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Int32

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.pid.acu_axis_controller import AxisController, MassShifterController
from py_pkg.pid.acu_pitch_config import init_acu_pitch
from py_pkg.pid.acu_roll_config import init_acu_roll
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


def _build_axis(cls, cfg):
    return cls(
        name=cfg["name"],
        Kp=cfg["Kp"],
        Ki=cfg["Ki"],
        Kd=cfg["Kd"],
        position_tolerance=cfg["position_tolerance"],
        command_tolerance=cfg["command_tolerance"],
        integral_limits=cfg["integral_limits"],
        output_limits=cfg["output_limits"],
        derivative_filter=cfg["derivative_filter"],
    )


class ACUControlNode(Node):
    def __init__(self):
        super().__init__("acu_control_node")

        self.callback_group = ReentrantCallbackGroup()

        self.pitch_axis = _build_axis(MassShifterController, init_acu_pitch)
        self.roll_axis = _build_axis(AxisController, init_acu_roll)
        self.pitch_steps_per_unit = init_acu_pitch["motor_steps_per_unit"]
        self.roll_steps_per_unit = init_acu_roll["motor_steps_per_unit"]

        self.target_roll_deg = 0.0
        self.target_pitch_deg = 0.0
        self.current_roll_deg = 0.0
        self.current_pitch_deg = 0.0

        # Motor-step topics (Int32). UUV-frame controller output is converted
        # to integer steps via the per-axis steps_per_unit constant.
        self.pitch_pub = create_publisher_for_topic(
            self, UUVTopics.ACU_PITCH_STEPS, callback_group=self.callback_group
        )
        self.roll_pub = create_publisher_for_topic(
            self, UUVTopics.ACU_ROLL_STEPS, callback_group=self.callback_group
        )

        create_subscription_for_topic(
            self,
            UUVTopics.POSITION_ESTIMATION,
            self.current_pose_callback,
            callback_group=self.callback_group,
        )
        create_subscription_for_topic(
            self,
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            callback_group=self.callback_group,
        )

        self.control_timer = self.create_timer(
            1.0 / 10.0,
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info("ACU control node started.")

    def target_pose_callback(self, msg: Pose):
        q = msg.orientation
        roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.target_roll_deg = math.degrees(roll)
        self.target_pitch_deg = math.degrees(pitch)
        self.get_logger().info(
            f"Updated target: roll={self.target_roll_deg:.2f}°, "
            f"pitch={self.target_pitch_deg:.2f}°"
        )

    def current_pose_callback(self, msg: Pose):
        q = msg.orientation
        roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.current_roll_deg = math.degrees(roll)
        self.current_pitch_deg = math.degrees(pitch)

    def control_loop(self):
        self.pitch_axis.update_sensor(self.current_pitch_deg)
        self.roll_axis.update_sensor(self.current_roll_deg)

        pitch_cmd = self.pitch_axis.update(self.target_pitch_deg)
        roll_cmd = self.roll_axis.update(self.target_roll_deg)

        if pitch_cmd is not None:
            msg = Int32()
            msg.data = int(round(pitch_cmd * self.pitch_steps_per_unit))
            self.pitch_pub.publish(msg)
            self.get_logger().debug(f"Pitch motor steps: {msg.data}")

        if roll_cmd is not None:
            msg = Int32()
            msg.data = int(round(roll_cmd * self.roll_steps_per_unit))
            self.roll_pub.publish(msg)
            self.get_logger().debug(f"Roll motor steps: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = ACUControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
