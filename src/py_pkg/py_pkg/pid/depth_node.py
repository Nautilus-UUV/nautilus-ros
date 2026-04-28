# Copied from divetest files: depth_control_node.py

#!/usr/bin/env python3

import rclpy

# from sensor_msgs.msg import Imu
# from StatePackage.msg import StateVector
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Float64, Int32

from py_pkg import SimMath
from py_pkg.pid import depth_control_system as ControlSystem
from py_pkg.pid.depth_config import (
    init_acc,
    init_buoyancy_engine,
    init_control,
    init_motor,
    init_pos,
    init_vel,
)


def pressure_to_depth(pressure: float, density=1e3, std_pressure=1e5) -> float:
    """
    Convert pressure in Pa to depth in meters.
    :param
    pressure: Pressure in Pa
    density: Density of the fluid in kg/m^3 (default is 1e3 kg/m^3 for water)
    std_pressure: Standard atmospheric pressure in Pa (default is 1e5 Pa)
    :return: Depth in meters
    """
    g = 9.806  # Zurich Area

    # Convert pressure to depth using the formula: depth = pressure / (density * g)
    depth = (pressure - std_pressure) / (density * g)
    return depth


def q_to_rpm(q: float, bladder_volume: float) -> float:
    """
    Convert flow rate ratio in 1/s to bladder level as a fraction of the bladder volume.
    :param q: Flow rate as a fraction of total volume per second in Hz
    :param bladder_volume: Bladder volume in m^3
    :return motor_rpm: Motor's speed in RPM
    """
    efficiency = 0.93  # efficiency between 1000 and 3000 RPM
    volume_per_rev = 0.32 * 1e-6  # m^3 / rev
    flow_rate = q * bladder_volume  # m^3/s
    min = 60  # seconds
    motor_rpm = min / (volume_per_rev * efficiency) * flow_rate  # rev/min
    return motor_rpm


def rpm_to_q(motor_rpm: float, bladder_volume: float) -> float:
    """
    Convert motor RPM to flow rate ratio in 1/s.
    :param motor_rpm: Motor's speed in RPM
    :param bladder_volume: Bladder volume in m^3
    :return q: Flow rate ratio is the flow rate as a fraction of total volume in 1/s
    """
    efficiency = 0.93
    volume_per_rev = 0.32 * 1e-6
    q = (motor_rpm * volume_per_rev * efficiency) / (60 * bladder_volume)
    return q


class DepthControlNode(Node):
    def __init__(self):
        super().__init__("depth_control_node")

        # Create a reentrant callback group to allow concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        self.control_system = ControlSystem.DepthControlSystem(init_control)
        self.current_position = SimMath.Vector(
            init_pos.get("x"), init_pos.get("y"), init_pos.get("z")
        )
        self.current_velocity = SimMath.Vector(
            init_vel.get("x"), init_vel.get("y"), init_vel.get("z")
        )  # Initialize velocity
        self.current_acceleration = SimMath.Vector(
            init_acc.get("x"), init_acc.get("y"), init_acc.get("z")
        )  # Initialize acceleration
        self.current_bladder_level = init_buoyancy_engine.get(
            "initial_proportion_full"
        )  # Initialize bladder level
        self.bladder_volume = init_buoyancy_engine.get(
            "tank_volume"
        )  # Bladder volume in m^3
        self.current_time = self.get_clock().now().nanoseconds / 1e9

        self.control_output = 0.0  # Control output for buoyancy engine
        self.motor_rpm = 0.0  # Motor RPM
        self.target_depth = 0.0  # Target depth in meters
        self.current_depth = 0.0  # Current depth in meterscontrol_action

        # Publisher for the control output (e.g., to a buoyancy control unit)
        self.bcu_controller_rpm_publisher = self.create_publisher(
            Int32, "BCU_controller/RPM", 10, callback_group=self.callback_group
        )

        # Subscriber for the target depth
        self.target_depth_subscriber = self.create_subscription(
            Float64,
            "target_depth",
            self.target_depth_callback,
            10,
            callback_group=self.callback_group,
        )

        # Subscriber for the current depth
        self.pressure_external_subscriber = self.create_subscription(
            Float64,
            "/sensor/pressure_ext",
            self.current_depth_callback,
            10,
            callback_group=self.callback_group,
        )

        # Subscriber for the current velocity (assuming you have this topic)
        # self.State_subscriber = self.create_subscription(
        #    StateVector, "/state", self.velocity_callback, 10, callback_group=self.callback_group
        # )

        # Subscriber for the current acceleration (assuming you have this topic)
        # self.IMU_subscriber = self.create_subscription(
        #     Imu, "/sensor/imu1", self.IMU_callback, 10, callback_group=self.callback_group
        # )

        # Timer to periodically run the control loop
        self.control_timer = self.create_timer(
            1.0 / 10.0,  # 10 Hz control frequency (adjust as needed)
            self.control_loop,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Depth control node started.")

    def target_depth_callback(self, msg: Float64):
        self.target_depth = msg.data
        self.control_system.target_depth = self.target_depth
        self.get_logger().info(f"Updated target depth: {self.target_depth}")

    def current_depth_callback(self, msg: Float64):
        self.current_depth = pressure_to_depth(msg.data)
        self.get_logger().debug(f"Received current depth: {self.current_depth}")

    # def State_callback(self, msg: Float64):
    #     self.current_velocity.x = msg.linear_velocity.x
    #     self.current_velocity.y = msg.linear_velocity.y
    #     self.current_velocity.z = msg.linear_velocity.z
    #     self.get_logger().debug(f"Received velocity: {self.current_velocity.z} m/s")

    # def IMU_callback(self, msg: Imu):
    #     self.current_acceleration.x = msg.linear_acceleration.x
    #     self.current_acceleration.y = msg.linear_acceleration.y
    #     self.current_acceleration.z = msg.linear_acceleration.z
    #     self.get_logger().debug(f"Received acceleration: {self.current_acceleration.z} m/s^2")

    def control_loop(self):
        self.current_time = self.get_clock().now().nanoseconds / 1e9
        current_position = SimMath.Vector(
            0.0, 0.0, self.current_depth
        )  # Assuming x and y are irrelevant for depth control
        self.control_output = self.control_system.calc_acc(
            current_position,
            self.current_velocity,
            self.current_acceleration,
            0.0,
            self.current_time,
        )  # vel and acc are estimated from depth with finite difference method

        msg = Int32()
        self.motor_rpm = q_to_rpm(self.control_output, self.bladder_volume)
        if self.motor_rpm > 0:
            self.motor_rpm = SimMath.clamp(
                self.motor_rpm, init_motor.get("min_rpm"), init_motor.get("max_rpm")
            )
        else:
            self.motor_rpm = SimMath.clamp(
                self.motor_rpm, -init_motor.get("max_rpm"), -init_motor.get("min_rpm")
            )
        msg.data = int(-1 * self.motor_rpm)
        self.bcu_controller_rpm_publisher.publish(msg)
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

    # If a message is received, start the DepthControlNode
    depth_control_node = DepthControlNode()
    rclpy.spin(depth_control_node)

    depth_control_node.destroy_node()
    topic_checker_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
