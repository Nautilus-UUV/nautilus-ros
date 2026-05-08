#!/usr/bin/env python3
"""Outer-loop attitude control node — turns desired roll/pitch into motor commands.

This is the ACU's outer loop. It looks at the desired vehicle attitude
(roll and pitch, extracted from the quaternion on POSITION_TARGET) and
the current attitude (same idea, from POSITION_ESTIMATION), runs each
axis through its own controller, and publishes the resulting motor
commands at 10 Hz.

Each axis has a different actuator, so each gets a different controller:

- Roll uses an `AxisController`: rotation in, rotation out. The roll
  motor is itself an angular position, so it makes sense to keep
  everything in the same unit (degrees) and treat the controller's
  output as an incremental adjustment.
- Pitch uses a `MassShifterController`: pitch error in, mass-shifter
  stroke out. Pitch is changed by sliding a weight forward or
  backward, so the controller works in metres at the output even
  though the error is in degrees.

Wire formats on the way out are integers in fixed units the HAL bridge
and EPOS driver already understand:

- `ACU_ROLL` is an `Int16` in centidegrees (degrees * 100, via
  `ACU_ROLL_CDEG_PER_DEG`).
- `ACU_PITCH` is an `Int16` in millimetres.

We deliberately stop at "degrees and millimetres" here. Converting
those into the raw encoder counts the motors actually take is the EPOS
driver's job, not ours — keeping that separation means tuning gains in
this file is done in human-meaningful units.
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int16

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.pid.acu_axis_controller import AxisController, MassShifterController
from py_pkg.pid.acu_pitch_config import init_acu_pitch
from py_pkg.pid.acu_roll_config import init_acu_roll
from py_pkg.robot_specs import ACU_ROLL_CDEG_PER_DEG
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)

# Pitch PID runs in metres (mass-shifter stroke); ACU_PITCH topic is mm.
_M_TO_MM = 1000.0


def _build_axis(cls, cfg):
    return cls(
        name=cfg["name"],
        Kp=cfg["Kp"],
        Ki=cfg["Ki"],
        Kd=cfg["Kd"],
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

        self.target_roll_deg = 0.0
        self.target_pitch_deg = 0.0
        self.current_roll_deg = 0.0
        self.current_pitch_deg = 0.0

        # Log throttle
        self._target_log_every_n = 10
        self._target_cb_count = 0

        self.pitch_pub = create_publisher_for_topic(
            self, UUVTopics.ACU_PITCH, callback_group=self.callback_group
        )
        self.roll_pub = create_publisher_for_topic(
            self, UUVTopics.ACU_ROLL, callback_group=self.callback_group
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
        self._target_cb_count += 1
        if self._target_cb_count % self._target_log_every_n == 0:
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

        # Real wallclock seconds: Ki/Kd stay in per-second units regardless
        # of loop rate or executor jitter.
        now = self.get_clock().now().nanoseconds / 1e9
        pitch_cmd = self.pitch_axis.update(self.target_pitch_deg, now)
        roll_cmd = self.roll_axis.update(self.target_roll_deg, now)

        if pitch_cmd is not None:
            msg = Int16()
            msg.data = int(round(pitch_cmd * _M_TO_MM))
            self.pitch_pub.publish(msg)
            self.get_logger().debug(f"Pitch position (mm): {msg.data}")

        if roll_cmd is not None:
            msg = Int16()
            msg.data = int(round(roll_cmd * ACU_ROLL_CDEG_PER_DEG))
            self.roll_pub.publish(msg)
            self.get_logger().debug(f"Roll position (cdeg): {msg.data}")


def main(args=None):
    # Catch SIGINT/SIGTERM so the process exits 0 instead of 1 on Ctrl-C —
    # otherwise launch_testing's exit-code check intermittently fails.
    rclpy.init(args=args)
    node = ACUControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()