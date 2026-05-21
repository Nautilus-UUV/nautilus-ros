"""Boot-time BCU motor-direction smoke test for a sealed Pi.

Spawns three nodes -- nothing else. The whole purpose is to make the
glider audibly reverse its BCU pump on a fixed cadence on boot, while
still appearing alive to the topside frontend over MQTT:

    auto_bcu_oscillator  --(±rpm on /bcu/rpm)-->  stm_com_node  -->  UART -> STM32
    mqtt_bridge_node     <-->  mission laptop (link status + heartbeat)

No EKF, no depth PID, no pathfinding. The oscillator publishes
``/bcu/rpm`` so a second shell can ``ros2 topic echo /bcu/rpm`` and see
the alternating values; stm_com_node consumes that topic exactly the
way it would in the real control stack.

Wire this into the Pi's autostart unit's ExecStart=, e.g.::

    exec ros2 launch py_pkg stm_debug_oscillator.launch.py \\
        rpm:=20 period_s:=5.0 mqtt_broker_host:=192.168.10.20

All launch args are bare ``name:=value`` -- ``ros2 launch`` does not
accept ``--ros-args``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "port",
                default_value="/dev/serial0",
                description="Serial device the STM32 is on. Override for bench tests with socat.",
            ),
            DeclareLaunchArgument(
                "baud",
                default_value="115200",
                description="UART baud rate; must match the STM firmware.",
            ),
            DeclareLaunchArgument(
                "poll_period_s",
                default_value="0.01",
                description="stm_com ROS timer period driving serial drain + tx.",
            ),
            DeclareLaunchArgument(
                "rpm",
                default_value="10",
                description="Magnitude of the alternating RPM setpoint (sign flips every period).",
            ),
            DeclareLaunchArgument(
                "period_s",
                default_value="5.0",
                description="Seconds between sign flips on /bcu/rpm.",
            ),
            DeclareLaunchArgument(
                "mqtt_broker_host",
                default_value="127.0.0.1",
                description=(
                    "MQTT broker host the topside bridge connects to. Override "
                    "with the mission-laptop IP on the Pi (e.g. 192.168.10.20)."
                ),
            ),
            DeclareLaunchArgument(
                "mqtt_broker_port",
                default_value="1883",
                description="MQTT broker TCP port.",
            ),
            Node(
                package="py_pkg",
                executable="stm_com_node",
                name="stm_com",
                output="screen",
                parameters=[
                    {
                        "port": LaunchConfiguration("port"),
                        "baud": ParameterValue(
                            LaunchConfiguration("baud"), value_type=int
                        ),
                        "poll_period_s": ParameterValue(
                            LaunchConfiguration("poll_period_s"), value_type=float
                        ),
                    }
                ],
            ),
            Node(
                package="py_pkg",
                executable="auto_bcu_oscillator",
                name="auto_bcu_oscillator",
                output="screen",
                parameters=[
                    {
                        "rpm": ParameterValue(
                            LaunchConfiguration("rpm"), value_type=int
                        ),
                        "period_s": ParameterValue(
                            LaunchConfiguration("period_s"), value_type=float
                        ),
                    }
                ],
            ),
            Node(
                package="py_pkg",
                executable="mqtt_bridge_node",
                name="mqtt_bridge",
                output="screen",
                parameters=[
                    {
                        "broker_host": LaunchConfiguration("mqtt_broker_host"),
                        "broker_port": ParameterValue(
                            LaunchConfiguration("mqtt_broker_port"), value_type=int
                        ),
                    }
                ],
            ),
        ]
    )
