"""STM com boot-time debug oscillator: ±10 RPM every 5 s, alternating.

Brings up *only* stm_com_node, with the oscillator forced on. No EKF,
no depth PID, no pathfinding -- this launch's whole purpose is to drive
the BCU motor with a simple alternating pattern so a sealed Pi (the
CM is no longer reachable once mounted on the main board) can be
smoke-tested on boot.

Wire this into the Pi's autostart unit's ExecStart=, e.g.::

    ExecStart=/bin/bash -lc 'source /opt/ros/jazzy/setup.bash \\
        && source /home/<user>/dave_ws/install/setup.bash \\
        && ros2 launch py_pkg stm_debug_oscillator.launch.py'

Port / baud / poll period plus the oscillator magnitude and period are
all exposed as launch args, so the autostart shell script can tweak the
pattern without rebuilding. The one knob deliberately hardcoded is
``debug_oscillate_enabled`` -- that's what makes this the "debug
oscillator" launch in the first place.
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
                description="ROS timer period driving serial drain + tx.",
            ),
            DeclareLaunchArgument(
                "debug_oscillate_rpm",
                default_value="10",
                description="Magnitude of the alternating RPM setpoint (sign flips every period).",
            ),
            DeclareLaunchArgument(
                "debug_oscillate_period_s",
                default_value="5.0",
                description="Seconds between sign flips.",
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
                        "debug_oscillate_enabled": True,
                        "debug_oscillate_rpm": ParameterValue(
                            LaunchConfiguration("debug_oscillate_rpm"),
                            value_type=int,
                        ),
                        "debug_oscillate_period_s": ParameterValue(
                            LaunchConfiguration("debug_oscillate_period_s"),
                            value_type=float,
                        ),
                    }
                ],
            ),
        ]
    )
