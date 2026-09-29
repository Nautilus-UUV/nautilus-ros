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

A non-empty ``scenario`` additionally arms bcu_node's tank-limit clamp: the
scenario's plant tank endpoints ride a latched DiveInit (the sim surrogate for
the operator UI's Initialize button -- see ``params_for_auto_mission``). Empty
(the default) publishes no DiveInit, exactly the pre-v2 behavior.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from py_pkg.debug.mission_fields import MISSION_FIELDS
from py_pkg.scenarios.compile import params_for_auto_mission
from py_pkg.scenarios.loader import load_scenario

# Match the historical autostart: hold off until bringup has settled.
_BRINGUP_DELAY_S = 8.0


def _autostart_setup(context):
    # Mission fields come off the shared MISSION_FIELDS table that auto_mission
    # declares its parameters from, so this file can't drift out of step with it.
    parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=cast)
        for name, _field, cast, _default in MISSION_FIELDS
    }
    # Sequencing knobs: node behaviour, not mission content.
    parameters["start_delay_s"] = ParameterValue(
        LaunchConfiguration("start_delay_s"), value_type=float
    )
    parameters["wait_for_sim_ready"] = ParameterValue(
        LaunchConfiguration("wait_for_sim_ready"), value_type=bool
    )
    scenario_path = LaunchConfiguration("scenario").perform(context)
    if scenario_path:
        parameters.update(params_for_auto_mission(load_scenario(scenario_path)))

    autostart_node = Node(
        package="py_pkg",
        executable="auto_mission",
        name="auto_mission",
        output="screen",
        condition=IfCondition(LaunchConfiguration("mission_autostart")),
        parameters=[parameters],
    )
    return [TimerAction(period=_BRINGUP_DELAY_S, actions=[autostart_node])]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("mission_autostart", default_value="false"),
            *[
                DeclareLaunchArgument(name, default_value=str(default))
                for name, _field, _cast, default in MISSION_FIELDS
            ],
            DeclareLaunchArgument("start_delay_s", default_value="2.0"),
            # True (the sim launches pass it) holds the mission until the
            # sim_ready_gate's latched /sim/ready; false keeps the legacy
            # fixed-timer autostart for gate-less compositions.
            DeclareLaunchArgument("wait_for_sim_ready", default_value="false"),
            DeclareLaunchArgument("scenario", default_value=""),
            OpaqueFunction(function=_autostart_setup),
        ]
    )
