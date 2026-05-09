"""Control stack: estimation + cascaded depth + per-axis ACU + mission dispatch.

Sim-agnostic: brings up only the ROS-side controllers and estimators that
must run in *both* sim and on the bench. Wrap this from
``nautilus_hal/launch/trim_sim.launch.py`` (sim) or pair it with the real
STM bridges (hardware).

Composition (inputs → outputs):
    /imu/left           -> ekf_prefilter -> /imu/filtered/left
    /imu/filtered/left  -> ekf_node      -> /position/estimation
    /position/target +  -> depth_node    -> /bcu/rpm + /bcu/valves
      /external/pressure
    /position/target +  -> acu_node      -> /acu/pitch + /acu/roll
      /position/estimation
    /path + /command    -> pathfinding_node -> /position/target
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
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
            ),
            Node(
                package="py_pkg",
                executable="acu_node",
                name="acu_control_node",
                output="screen",
            ),
            Node(
                package="py_pkg",
                executable="pathfinding_node",
                name="pathfinding_node",
                output="screen",
            ),
        ]
    )
