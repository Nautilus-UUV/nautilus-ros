"""Sim run watchdog, gated on ``watchdog``.

Brings up the ``run_watchdog`` node (mission-complete / floater / sinker
early termination) and registers an ``OnProcessExit`` handler that turns the
node's verdict exit into a full launch ``Shutdown`` -- the sweep runner then
reaps the slot on process exit.

Included by the ``*_sim`` HAL launches; each forwards its own ``dwell_s`` /
``bag_path``. Lives in py_pkg (the package that owns the control nodes) so
the HAL launches compose it via ``IncludeLaunchDescription`` rather than
embedding a ``py_pkg`` node directly.

The sinker stall grace is derived, not declared: the stall clock restarts
at every deepening event, so what it must cover is the longest *single*
intended hold (``dwell_s``) plus a fixed margin for transit and controller
settling. The verdict JSON lands next to the run's bag directory
(``{bag_path}/../run_verdict.json``) where the sweep analysis picks it up;
no ``bag_path`` means no verdict file.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Added on top of one intended dwell to form the sinker stall grace: budget
# for transit legs and controller settling around the holds, so only a
# genuinely stuck vehicle trips the sinker rule.
_STALL_GRACE_MARGIN_S = 240.0


def _wire_watchdog(context, *_args, **_kwargs):
    # OpaqueFunction so the grace arithmetic and verdict-path derivation can
    # run on the resolved LaunchConfiguration values.
    dwell_s = float(LaunchConfiguration("dwell_s").perform(context))
    bag_path = LaunchConfiguration("bag_path").perform(context).strip()

    stall_grace_s = dwell_s + _STALL_GRACE_MARGIN_S
    verdict_path = str(Path(bag_path).parent / "run_verdict.json") if bag_path else ""

    watchdog_node = Node(
        package="py_pkg",
        executable="run_watchdog",
        name="run_watchdog",
        output="screen",
        parameters=[
            {
                "stall_grace_s": stall_grace_s,
                "verdict_path": verdict_path,
            }
        ],
    )
    return [
        watchdog_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=watchdog_node,
                on_exit=[EmitEvent(event=Shutdown(reason="run_watchdog verdict"))],
            )
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "watchdog",
                default_value="false",
                description=(
                    "If true, run the sim run watchdog: shut the launch down "
                    "early on mission completion or an implausibility verdict "
                    "(surface-floater / bottom-sinker)."
                ),
            ),
            DeclareLaunchArgument(
                "dwell_s",
                default_value="0.0",
                description=(
                    "The mission's per-hold dwell time (seconds); feeds the "
                    "sinker stall grace so an intended hold at depth is never "
                    "mistaken for a sinker."
                ),
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value="",
                description=(
                    "The run's bag output directory; run_verdict.json is "
                    "written next to it (its parent). Empty writes no verdict "
                    "file."
                ),
            ),
            OpaqueFunction(
                function=_wire_watchdog,
                condition=IfCondition(LaunchConfiguration("watchdog")),
            ),
        ]
    )
