# Copied from divetest: BCU_controller_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int32

# from simple_pid import PID
from py_pkg.motor import MotorController


class BCUControllerNode(Node):
    _state = "STARTUP"
    _mode: int = 0
    _rpm = 0
    _timer: rclpy.timer = None

    _tankPressure: float = 0.0

    def __init__(self):
        super().__init__("bcu_control_node")

        self.create_subscription(
            Float64, "/sensor/pressure_tank", self.tank_pressure_callback, 10
        )
        self.create_subscription(Int32, "/BCU_controller/RPM", self.rpm_callback, 10)

        self._motor = MotorController("libEposCmd.so.6.8.1.0")

        self._motor.initialize_motor()

        self.get_logger().info("BCU controler node has started.")

        self._timer = self.create_timer(0.1, self.timer_callback)

    def tank_pressure_callback(self, msg):
        self._tankPressure = msg.data

    def rpm_callback(self, msg):
        rpm = msg.data
        if rpm != self._rpm and rpm <= 4000 and rpm >= -4000:
            self._rpm = msg.data
            self.get_logger().info(f"Received RPM: {self._rpm}")

    def timer_callback(self):
        if self._tankPressure > 1.7 and self._rpm > 0:
            self._motor.set_velocity(0)
            return

        if self._tankPressure < 0.0 and self._rpm < 0:
            self._motor.set_velocity(0)
            return

        self._motor.set_velocity(self._rpm)


def main(args=None):
    rclpy.init(args=args)
    node = BCUControllerNode()
    rclpy.spin(node)

    node._motor.close_motor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
