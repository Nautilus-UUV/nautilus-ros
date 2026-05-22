"""Latched mission autostart, gated on ``mission_autostart``.

Brings up the ``auto_mission`` node ~8 s after launch (so the control stack and
sensor bridges have settled), which publishes one MissionCommand on ``/path``
and one ``start`` on ``/command`` and then holds both latched for the launch
lifetime. Replaces the old per-launch ``ros2 topic pub --once`` autostart whose
latched samples vanished the instant the publisher process exited -- a DDS
discovery race that intermittently left ``pathfinding_node`` stuck on
"waiting for /path". See ``py_pkg/debug/auto_mission.py`` for the full rationale.

Included by the ``*_sim`` HAL launches; each passes its own ``mission_id`` and
mission fields. Lives in py_pkg (the package that owns the control nodes) so the
HAL launches compose it via ``IncludeLaunchDescription`` rather than embedding a
``py_pkg`` node directly.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Match the historical autostart: hold off until bringup has settled.
_BRINGUP_DELAY_S = 8.0


def generate_launch_description() -> LaunchDescription:
    mission_autostart = LaunchConfiguration("mission_autostart")

    autostart_node = Node(
        package="py_pkg",
        executable="auto_mission",
        name="auto_mission",
        output="screen",
        condition=IfCondition(mission_autostart),
        parameters=[
            {
                "mission_id": ParameterValue(
                    LaunchConfiguration("mission_id"), value_type=int
                ),
                "target_pressure_pa": ParameterValue(
                    LaunchConfiguration("target_pressure_pa"), value_type=float
                ),
                "angle_rad": ParameterValue(
                    LaunchConfiguration("angle_rad"), value_type=float
                ),
                "n_resurfaces": ParameterValue(
                    LaunchConfiguration("n_resurfaces"), value_type=int
                ),
                "start_delay_s": ParameterValue(
                    LaunchConfiguration("start_delay_s"), value_type=float
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("mission_autostart", default_value="false"),
            DeclareLaunchArgument("mission_id", default_value="1"),
            DeclareLaunchArgument("target_pressure_pa", default_value="0.0"),
            DeclareLaunchArgument("angle_rad", default_value="0.0"),
            DeclareLaunchArgument("n_resurfaces", default_value="0"),
            DeclareLaunchArgument("start_delay_s", default_value="2.0"),
            TimerAction(period=_BRINGUP_DELAY_S, actions=[autostart_node]),
        ]
    )
