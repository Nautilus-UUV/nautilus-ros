#!/usr/bin/env python3
"""

DEPRECATED; It is not implemented, ingore for now
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Bool, Int16

from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.pid.acu_axis_controller import AxisController
from py_pkg.robot_specs import ACU_ROLL_CDEG_PER_DEG
from py_pkg.scenarios.compile import acu_pitch_spec_from_node, acu_roll_spec_from_node
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
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
        # depth reading and a setpoint, so we don't pick a side from
        # uninitialized zeros. Depth (gauge Pa) arrives on
        # POSITION_ESTIMATION.position.z, already gauged by attitude_node.
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
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            callback_group=self.callback_group,
        )
        # Single vehicle-state input: roll (off the quaternion) for the roll PID
        # and gauge depth (position.z) for the bang-bang pitch leg select.
        create_subscription_for_topic(
            self,
            UUVTopics.POSITION_ESTIMATION,
            self.current_pose_callback,
            callback_group=self.callback_group,
        )

        # Mission run/stop. /command=false drops the target, neutralizes the
        # ACU once, and gates control_loop off (silent) so acu_debug can own
        # the wire; /command=true is a no-op (we wait for POSITION_TARGET).
        create_subscription_for_topic(
            self,
            UUVTopics.COMMAND,
            self._on_command,
            callback_group=self.callback_group,
        )

        self.control_timer = self.create_timer(
            1.0 / roll_cfg.frequency_hz,
            self.control_loop,
            callback_group=self.callback_group,
        )

        # Leave the ACU wire at neutral at boot, before any target arrives.
        self._publish_acu_neutral()

        self.get_logger().info("ACU control node started (bang-bang pitch, PID roll).")

    def target_pose_callback(self, msg: Pose):
        # POSITION_TARGET.position.z carries the target pressure in Pa
        # (pathfinding's TRIM convention). Roll comes off the quaternion
        # the same way as before.
        self.target_pressure_pa = float(msg.position.z)
        q = msg.orientation
        roll, _ = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.target_roll_deg = math.degrees(roll)

    def current_pose_callback(self, msg: Pose):
        # Roll feeds the roll PID; gauge depth (position.z) feeds the bang-bang
        # pitch leg select. Pitch off the pose is deliberately ignored.
        q = msg.orientation
        roll, _ = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.current_roll_deg = math.degrees(roll)
        self.current_pressure_pa = float(msg.position.z)

    def _publish_acu_neutral(self) -> None:
        pitch = Int16()
        pitch.data = 0
        self.pitch_pub.publish(pitch)
        roll = Int16()
        roll.data = 0
        self.roll_pub.publish(roll)

    def _on_command(self, msg: Bool) -> None:
        # /command=true (start) is a no-op -- we wait for POSITION_TARGET.
        # /command=false (stop) drops the target, wipes the roll PID, gates the
        # loop off, and commands pitch + roll to neutral ONCE before going
        # silent. The single neutral matters because the STM re-ships the last
        # value forever (no staleness watchdog), so silence alone would leave
        # the last mission attitude latched on the wire.
        if bool(msg.data):
            return
        self.roll_axis.reset()
        self.target_pressure_pa = None
        self.target_roll_deg = 0.0
        self.current_roll_deg = 0.0
        self._publish_acu_neutral()
        self.get_logger().info(
            "stop -> neutral emitted, ACU going silent (fresh state)."
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No active mission target -> stay off /acu/pitch and /acu/roll
            # entirely (roll PID included) so acu_debug can own the wire after a
            # stop. The one neutral sample was already emitted on stop/boot.
            # (Mirrors bcu_node's no-target gate -- same None sentinel.)
            return
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
    rclpy.init(args=args)
    node = ACUControlNode()
    spin_node(node)


if __name__ == "__main__":
    main()
