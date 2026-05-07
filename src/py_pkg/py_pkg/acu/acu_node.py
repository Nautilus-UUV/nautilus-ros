"""Inner-loop ACU node — drives the physical roll and pitch motors.

This is the half of the attitude stack that actually talks to the
hardware. It listens for target positions on ACU_ROLL (angle) and
ACU_PITCH (linear stroke, mm) and forwards them to two EPOS
``MotorController`` instances over USB. The outer-loop PID that
decides what those target positions should be lives in
``pid/acu_node.py``.

Note: this file was copied from the BCU controller node and adapted
for two axes. There's still some BCU-shaped scaffolding left over —
the ``_tankPressure`` field and the velocity-mode safety check in
``timer_callback`` were inherited from the BCU and don't quite fit
the ACU yet.
"""

import rclpy
from rclpy.node import Node

# from simple_pid import PID
from py_pkg.motor import MotorController
from py_pkg.uuv_ros_core.node_factory import create_subscription_for_topic
from py_pkg.uuv_ros_core.topics import UUVTopics


class ACUControllerNode(Node):
    # Node that sends ACU changes (roll and pitch) to ACU node.

    _state = "STARTUP"
    _mode: int = 0
    _rpm = 0
    _timer: rclpy.timer = None

    _tankPressure: float = 0.0

    def __init__(self):
        super().__init__("acu_control_node")

        # self.create_subscription(Int32, "/ACU_controller/RPM", self.rpm_callback, 10) -> was uin BCU control node for divetest
        create_subscription_for_topic(
            UUVTopics.ACU_ROLL,
            self.roll_callback,
            10,
            callback_group=self.callback_group,
        )
        create_subscription_for_topic(
            UUVTopics.ACU_PITCH,
            self.pitch_callback,
            10,
            callback_group=self.callback_group,
        )

        self._motor_pitch = MotorController(
            "libEposCmd.so.6.8.1.0"
        )  # FIND AND ADD MOTOR NAME -> electronics
        self._motor_roll = MotorController(
            "libEposCmd.so.6.8.1.0"
        )  # FIND AND ADD MOTOR NAME -> electronics

        self._motor_pitch.initialize_motor(mode=1)  # Position profile mode
        self._motor_roll.initialize_motor(mode=1)  # Position profile mode

        self.get_logger().info("ACU controller node has started.")

        self._timer = self.create_timer(0.1, self.timer_callback)

    def roll_callback(self, msg):
        # store imu data in dict, call for ex. as follows: self._imu_data['linear_acceleration'].z
        command = msg.data
        self._motor_roll.set_position(command)

    def pitch_callback(self, msg):
        command = msg.data
        self._motor_pitch.set_position(command)

    def timer_callback(self):
        # ADD SAFETY CHECKS FOR ACU MOTOR OPERATION LIMITS
        if self._tankPressure > 1.7 and self._rpm > 0:
            self._motor.set_velocity(0)
            return

        if self._tankPressure < 0.0 and self._rpm < 0:
            self._motor.set_velocity(0)
            return

        self._motor.set_velocity(self._rpm)


def main(args=None):
    rclpy.init(args=args)
    node = ACUControllerNode()
    rclpy.spin(node)

    node._motor_pitch.close_motor()
    node._motor_roll.close_motor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
