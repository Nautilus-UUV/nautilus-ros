# Copied from divetest files: depth_control_node.py

#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int16, UInt8

from py_pkg import math_utils as SimMath
from py_pkg.physics import gauge_pressure_pa, q_to_rpm
from py_pkg.pid import depth_control_system as ControlSystem
from py_pkg.pid.depth_config import (
    init_buoyancy_engine,
    init_control,
    init_motor,
    init_pos,
)
from py_pkg.robot_specs import BCU_DEEP_THRESHOLD_PA
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


def select_pump_and_valves(
    current_pressure_pa: float,
    q: float,
    pump_rpm: int,
    deep_threshold_pa: float,
) -> tuple[int, int, int]:
    """Decide pump RPM and valve bitmask from gauge pressure + descent intent.

    Below ``deep_threshold_pa`` (Z-positive-down: ``current_pressure_pa
    > threshold``), a descent intent (q > 0) is satisfied passively:
    the pump is forced off and valve 2 vents the bladder. Otherwise
    valve 1 carries the pumped flow when the command is non-zero;
    both valves stay closed when the pump is idle.

    Returns ``(pump_rpm, valve1_open, valve2_open)``.
    """
    deep = current_pressure_pa > deep_threshold_pa
    wants_to_descend = q > 0
    if deep and wants_to_descend:
        return 0, 0, 1
    if pump_rpm != 0:
        return pump_rpm, 1, 0
    return pump_rpm, 0, 0


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
        self.current_bladder_level = init_buoyancy_engine.get("initial_proportion_full")
        self.bladder_volume = init_buoyancy_engine.get("tank_volume")
        self.current_time = self.get_clock().now().nanoseconds / 1e9

        self.control_output = 0.0
        self.motor_rpm = 0.0
        # Held None until the first POSITION_TARGET arrives
        self.target_pressure_pa: float | None = None
        self.current_pressure_pa = 0.0

        self.bcu_controller_rpm_publisher = create_publisher_for_topic(
            self, UUVTopics.BCU_RPM, callback_group=self.callback_group
        )

        self.bcu_valves_publisher = create_publisher_for_topic(
            self, UUVTopics.BCU_VALVES, callback_group=self.callback_group
        )

        # Pressure setpoint is `position.z` of POSITION_TARGET, in gauge
        # Pa (Z-positive-down: deeper = higher gauge pressure). Pose's
        # position.z is reused as a pressure channel — the depth
        # controller's contract; pathfinding/EKF emit accordingly.
        self.target_pose_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.POSITION_TARGET,
            self.target_pose_callback,
            callback_group=self.callback_group,
        )

        self.pressure_external_subscriber = create_subscription_for_topic(
            self,
            UUVTopics.EXTERNAL_PRESSURE,
            self.current_pressure_callback,
            callback_group=self.callback_group,
        )

        self.control_timer = self.create_timer(
            1.0 / 10.0,  # 10 Hz control frequency
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Depth control node started.")

    def target_pose_callback(self, msg: Pose):
        self.target_pressure_pa = float(msg.position.z)
        self.control_system.target_pressure_pa = self.target_pressure_pa
        self.get_logger().info(f"Updated target pressure: {self.target_pressure_pa} Pa")

    def current_pressure_callback(self, msg):
        # EXTERNAL_PRESSURE is absolute Pa; the controller works in gauge.
        self.current_pressure_pa = gauge_pressure_pa(float(msg.data))
        self.get_logger().debug(
            f"Received current pressure: {self.current_pressure_pa} Pa"
        )

    def control_loop(self):
        if self.target_pressure_pa is None:
            # No target yet — emit a zero-RPM hold so the BCU bridge
            # doesn't drift, and skip the cascaded controller work.
            zero_msg = Int16()
            zero_msg.data = 0
            self.bcu_controller_rpm_publisher.publish(zero_msg)
            valves_off = UInt8()
            valves_off.data = 0
            self.bcu_valves_publisher.publish(valves_off)
            return

        self.current_time = self.get_clock().now().nanoseconds / 1e9
        current_position = SimMath.Vector(0.0, 0.0, self.current_pressure_pa)
        self.control_output = self.control_system.calc_acc(
            current_position,
            0.0,
            self.current_time,
        )

        msg = Int16()
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
        pump_rpm = int(-1 * self.motor_rpm)

        pump_rpm, valve1_open, valve2_open = select_pump_and_valves(
            self.current_pressure_pa,
            self.control_output,
            pump_rpm,
            BCU_DEEP_THRESHOLD_PA,
        )
        msg.data = pump_rpm
        valves_msg = UInt8()
        valves_msg.data = valve1_open | (valve2_open << 1)

        # RPM very shortly before valves: same callback, no sleep — the
        # publish ordering on the wire follows the call order here.
        self.bcu_controller_rpm_publisher.publish(msg)
        self.bcu_valves_publisher.publish(valves_msg)
        self.get_logger().debug(
            f"Fraction of bladder volume filled per second in Hz: {self.control_output}"
        )
        self.get_logger().debug(f"Command to motor in RPM: {self.motor_rpm}")
        self.get_logger().debug(
            f"Valves bitmask (bit0=v1, bit1=v2): {valves_msg.data:#04b}"
        )


def main(args=None):
    # Catch SIGINT/SIGTERM so the process exits 0 instead of 1 on Ctrl-C —
    # otherwise launch_testing's exit-code check intermittently fails.
    rclpy.init(args=args)
    depth_control_node = DepthControlNode()
    try:
        rclpy.spin(depth_control_node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        depth_control_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
