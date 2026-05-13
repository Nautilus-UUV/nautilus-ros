#!/usr/bin/env python3
"""Outer-loop attitude control node — bang-bang on pitch, PID on roll.

The two axes are doing very different jobs and the controllers reflect
that.

Pitch is bang-bang on pressure error. We don't ask the EKF where the
nose is pointing; we just look at the external pressure sensor versus
the pressure setpoint pathfinding put on POSITION_TARGET.position.z. If
we are shallower than the setpoint we are diving, so we throw the
pitch mass-shifter all the way back. If we are deeper we are climbing,
so we throw it all the way front. That's the whole loop. Two values on
the wire, one comparison per tick. The HAL bridge handles the EPOS-side
slew rate.

Roll keeps the existing PID. Roll dynamics are well-behaved (the ring
motor is itself an angular position so the controller is unit-clean),
the gains were tuned conservatively against EKF noise, and there's no
analogue of pitch's "just rail it" simplification here — we want a
quiet ring sitting at zero unless the pose actually rolls off.

Wire formats are unchanged:

- `ACU_PITCH` is `Int16` in millimetres (one of two extremes from
  the pitch axis's `output_limits` in `AcuPitchSpec`).
- `ACU_ROLL` is `Int16` in centidegrees (degrees * `ACU_ROLL_CDEG_PER_DEG`).
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int16, Int32

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.physics import gauge_pressure_pa
from py_pkg.pid.acu_axis_controller import AxisController
from py_pkg.robot_specs import ACU_ROLL_CDEG_PER_DEG
from py_pkg.scenarios.compile import acu_pitch_spec_from_node, acu_roll_spec_from_node
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


class ACUControlNode(Node):
    def __init__(self):
        super().__init__("acu_control_node")

        self.callback_group = ReentrantCallbackGroup()

        # Bang-bang extremes on the wire. The pitch spec gives us the
        # operational soft-saturation in metres; the wire format is mm.
        # "Front" is the most negative end of the stroke (mass forward),
        # "back" is the least negative (mass aft).
        pitch_cfg = acu_pitch_spec_from_node(self)
        self._acu_pitch_back_mm = int(round(pitch_cfg.output_limits[0] * 1000.0))
        self._acu_pitch_front_mm = int(round(pitch_cfg.output_limits[1] * 1000.0))

        roll_cfg = acu_roll_spec_from_node(self)
        self.roll_axis = AxisController(
            name=roll_cfg.name,
            kp=roll_cfg.kp,
            ki=roll_cfg.ki,
            kd=roll_cfg.kd,
            command_tolerance=roll_cfg.command_tolerance,
            integral_limits=roll_cfg.integral_limits,
            output_limits=roll_cfg.output_limits,
            derivative_filter=roll_cfg.derivative_filter,
        )

        # Bang-bang state: gate the first command on having both a
        # pressure reading and a setpoint, so we don't pick a side from
        # uninitialized zeros.
        self.current_pressure_pa: float | None = None
        self.target_pressure_pa: float | None = None

        # Roll PID state.
        self.target_roll_deg = 0.0
        self.current_roll_deg = 0.0

        self.pitch_pub = create_publisher_for_topic(
            self, UUVTopics.ACU_PITCH, callback_group=self.callback_group
        )
        self.roll_pub = create_publisher_for_topic(
            self, UUVTopics.ACU_ROLL, callback_group=self.callback_group
        )

        create_subscription_for_topic(
            self,
            UUVTopics.EXTERNAL_PRESSURE,
            self.pressure_callback,
            callback_group=self.callback_group,
        )
        create_subscription_for_topic(
            self,
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            callback_group=self.callback_group,
        )
        create_subscription_for_topic(
            self,
            UUVTopics.POSITION_ESTIMATION,
            self.current_pose_callback,
            callback_group=self.callback_group,
        )

        self.control_timer = self.create_timer(
            1.0 / roll_cfg.frequency_hz,
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info("ACU control node started (bang-bang pitch, PID roll).")

    def pressure_callback(self, msg: Int32):
        # EXTERNAL_PRESSURE is absolute Pa; POSITION_TARGET.position.z is
        # gauge Pa (pathfinding's mission convention). Same conversion
        # depth_node.py and pathfinding.py apply at ingress -- without it,
        # absolute (~101 kPa at the surface) is always above any realistic
        # gauge target and the bang-bang never flips legs.
        self.current_pressure_pa = gauge_pressure_pa(float(msg.data))

    def target_pose_callback(self, msg: Pose):
        # POSITION_TARGET.position.z carries the target pressure in Pa
        # (pathfinding's TRIM convention). Roll comes off the quaternion
        # the same way as before.
        self.target_pressure_pa = float(msg.position.z)
        q = msg.orientation
        roll, _ = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.target_roll_deg = math.degrees(roll)

    def current_pose_callback(self, msg: Pose):
        # Used by the roll PID only — pitch deliberately ignores the
        # pose estimate.
        q = msg.orientation
        roll, _ = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.current_roll_deg = math.degrees(roll)

    def control_loop(self):
        self._update_pitch()
        self._update_roll()

    def _update_pitch(self):
        if self.current_pressure_pa is None or self.target_pressure_pa is None:
            return
        diving = self.current_pressure_pa < self.target_pressure_pa
        msg = Int16()
        msg.data = self._acu_pitch_back_mm if diving else self._acu_pitch_front_mm
        self.pitch_pub.publish(msg)

    def _update_roll(self):
        self.roll_axis.update_sensor(self.current_roll_deg)
        now = self.get_clock().now().nanoseconds / 1e9
        roll_cmd = self.roll_axis.update(self.target_roll_deg, now)
        if roll_cmd is not None:
            msg = Int16()
            msg.data = int(round(roll_cmd * ACU_ROLL_CDEG_PER_DEG))
            self.roll_pub.publish(msg)


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
