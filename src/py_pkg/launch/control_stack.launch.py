"""Control stack: estimation + cascaded depth + per-axis ACU + mission dispatch.

Sim-agnostic: brings up only the ROS-side controllers and estimators that
must run in *both* sim and on the bench. Wrap this from
``nautilus_hal/launch/trim_sim.launch.py`` (sim) or pair it with the real
STM bridges (hardware).

Parameterized by a single ``scenario:=`` launch arg. The scenario YAML's
``control:`` block is compiled into per-node parameter dicts via
``py_pkg.scenarios.compile``; the ``rig:`` block is never read here.

Composition (inputs -> outputs):
    /imu                -> imu_prefilter -> /imu/filtered
    /imu/filtered +     -> attitude_node -> /position/estimation
      /external/pressure                    (roll/pitch from gravity; gauge
      + /init/dive                          depth in position.z)
    /position/target +  -> bcu_node    -> /bcu/rpm + /bcu/valves
      /position/estimation
    /position/target +  -> acu_node      -> /acu/pitch + /acu/roll
      /position/estimation
    /path + /command    -> pathfinding_node -> /position/target
      (/command is std_msgs/Bool: true=start, false=stop. bcu_node and
       acu_node also subscribe to /command and reset to a safe-silent state
       on false.)
    feedback + sensors  -> liveness_node -> /status/liveness (per-subsystem
                                            DiagnosticArray; freshness watchdog)
    MQTT nautilus/cmd/* -> mqtt_bridge   -> /command + /path + /debug/*
                                            + /debug/reset
    /debug/bcu/rpm +    -> bcu_debug     -> /bcu/rpm + /bcu/valves; drives the
      /debug/bcu/valves +                   wire whenever it holds a command
      /debug/emergency_surface              (emergency surface always acts).
    /debug/acu/pitch +  -> acu_debug     -> /acu/pitch + /acu/roll; drives the
      /debug/acu/roll                       wire whenever it holds a setpoint.
    /bcu/rpm            -> stm_com       -> UART (hardware only; gated by
                                            enable_stm_com:= launch arg)
    /bcu/* + /acu/*     -> can_com       -> CAN PDO 0x181 (hardware only;
                                            gated by enable_can_com:= launch arg)

There is no manual-override flag. A controller drives its actuator only while
it has an active mission target, and two events end that: the operator's
/command=false, and pathfinding's /mission/complete when the mission finishes
on its own. Either one makes the controller emit one safe-stop and go silent,
freeing the wire for a debug node. They stay separate topics so "aborted" and
"finished" remain distinguishable downstream. The operator's red Reset button
publishes /debug/reset to all-stop the debug nodes.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
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
        params_for_bcu_node,
        params_for_imu_prefilter,
    )
    from py_pkg.scenarios.loader import load_scenario

    control = load_scenario(LaunchConfiguration("scenario").perform(context)).control
    mqtt_broker_host = LaunchConfiguration("mqtt_broker_host").perform(context)
    mqtt_broker_port = int(LaunchConfiguration("mqtt_broker_port").perform(context))
    lifeguard_timeout_s = float(
        LaunchConfiguration("lifeguard_timeout_s").perform(context)
    )
    return [
        Node(
            package="py_pkg",
            executable="imu_prefilter",
            name="imu_prefilter",
            output="screen",
            parameters=[params_for_imu_prefilter(control)],
        ),
        Node(
            package="py_pkg",
            executable="attitude_node",
            name="attitude_node",
            output="screen",
        ),
        Node(
            package="py_pkg",
            executable="bcu_node",
            name="bcu_node",
            output="screen",
            parameters=[params_for_bcu_node(control)],
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
            executable="liveness_node",
            name="liveness_node",
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
                    "lifeguard_timeout_s": lifeguard_timeout_s,
                }
            ],
        ),
        Node(
            package="py_pkg",
            executable="bcu_debug_node",
            name="bcu_debug",
            output="screen",
        ),
        Node(
            package="py_pkg",
            executable="acu_debug_node",
            name="acu_debug",
            output="screen",
        ),
        Node(
            package="py_pkg",
            executable="stm_com_node",
            name="stm_com",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_stm_com")),
        ),
        Node(
            package="py_pkg",
            executable="can_com_node",
            name="can_com",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_can_com")),
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
            DeclareLaunchArgument(
                "lifeguard_timeout_s",
                default_value="15.0",
                description=(
                    "Dead-man window for the lifeguard failsafe: once armed "
                    "(nautilus/cmd/lifeguard), this many seconds without a "
                    "laptop heartbeat latches the emergency surface. Tests "
                    "shorten it further."
                ),
            ),
            DeclareLaunchArgument(
                "enable_stm_com",
                default_value="false",
                description=(
                    "Spawn stm_com_node, which opens /dev/serial0 to talk to "
                    "the STM32. Off by default so sim launches don't crash on "
                    "hosts without the UART device; set true on the Pi."
                ),
            ),
            DeclareLaunchArgument(
                "enable_can_com",
                default_value="false",
                description=(
                    "Spawn can_com_node, which binds a raw SocketCAN socket on "
                    "can0 and heartbeats the actuator PDO (id 0x181) to the CU "
                    "board. Off by default so sim/dev hosts without a CAN "
                    "interface don't crash; set true on the vehicle (requires "
                    "`ip link set can0 type can bitrate 125000 && ip link set "
                    "up can0` first)."
                ),
            ),
            OpaqueFunction(function=_wire_control_stack),
        ]
    )
