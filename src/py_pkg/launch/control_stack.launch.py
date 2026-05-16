"""Control stack: estimation + cascaded depth + per-axis ACU + mission dispatch.

Sim-agnostic: brings up only the ROS-side controllers and estimators that
must run in *both* sim and on the bench. Wrap this from
``nautilus_hal/launch/trim_sim.launch.py`` (sim) or pair it with the real
STM bridges (hardware).

Parameterized by a single ``scenario:=`` launch arg. The scenario YAML's
``control:`` block is compiled into per-node parameter dicts via
``py_pkg.scenarios.compile``; the ``rig:`` block is never read here.

Composition (inputs -> outputs):
    /imu/left           -> ekf_prefilter -> /imu/filtered/left
    /imu/filtered/left  -> ekf_node      -> /position/estimation
    /position/target +  -> depth_node    -> /bcu/rpm + /bcu/valves
      /external/pressure
    /position/target +  -> acu_node      -> /acu/pitch + /acu/roll
      /position/estimation
    /path + /command    -> pathfinding_node -> /position/target
    MQTT nautilus/cmd/* -> mqtt_bridge   -> /command + /path
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_scenario_path() -> str:
    return os.path.join(
        get_package_share_directory("py_pkg"),
        "scenarios",
        "library",
        "nominal.yaml",
    )


def _wire_control_stack(context, *_args, **_kwargs):
    # OpaqueFunction so LaunchConfiguration is resolvable. Only the
    # `.control` half of the scenario is read here.
    from py_pkg.scenarios.compile import (
        params_for_acu_node,
        params_for_depth_node,
    )
    from py_pkg.scenarios.loader import load_scenario

    control = load_scenario(LaunchConfiguration("scenario").perform(context)).control
    mqtt_broker_host = LaunchConfiguration("mqtt_broker_host").perform(context)
    mqtt_broker_port = int(LaunchConfiguration("mqtt_broker_port").perform(context))
    return [
        Node(
            package="py_pkg",
            executable="ekf_prefilter",
            name="ekf_prefilter",
            output="screen",
        ),
        Node(
            package="py_pkg",
            executable="ekf_node",
            name="ekf_node",
            output="screen",
        ),
        Node(
            package="py_pkg",
            executable="depth_node",
            name="depth_control_node",
            output="screen",
            parameters=[params_for_depth_node(control)],
        ),
        Node(
            package="py_pkg",
            executable="acu_node",
            name="acu_control_node",
            output="screen",
            parameters=[params_for_acu_node(control)],
        ),
        Node(
            package="py_pkg",
            executable="pathfinding_node",
            name="pathfinding_node",
            output="screen",
        ),
        Node(
            package="py_pkg",
            executable="mqtt_bridge_node",
            name="mqtt_bridge",
            output="screen",
            parameters=[
                {
                    "broker_host": mqtt_broker_host,
                    "broker_port": mqtt_broker_port,
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario",
                default_value=_default_scenario_path(),
                description=(
                    "Path to a scenario YAML. Loaded via py_pkg.scenarios.load_scenario "
                    "to parameterize every controller; defaults to the installed nominal (fault-injection off)."
                ),
            ),
            DeclareLaunchArgument(
                "mqtt_broker_host",
                default_value="127.0.0.1",
                description=(
                    "MQTT broker host the topside bridge connects to. Default targets a "
                    "local mosquitto; override with the tether broker IP on the mission laptop."
                ),
            ),
            DeclareLaunchArgument(
                "mqtt_broker_port",
                default_value="1883",
                description="MQTT broker TCP port.",
            ),
            OpaqueFunction(function=_wire_control_stack),
        ]
    )
