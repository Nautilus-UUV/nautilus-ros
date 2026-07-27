"""Sim run watchdog, gated on ``watchdog``.

Brings up the ``run_watchdog`` node (mission-complete / floater / sinker
early termination) and registers an ``OnProcessExit`` handler that turns the
node's verdict exit into a full launch ``Shutdown`` -- the sweep runner then
reaps the slot on process exit.

Included by the ``*_sim`` HAL launches; each forwards its own ``bag_path``.
Lives in py_pkg (the package that owns the control nodes) so the HAL
launches compose it via ``IncludeLaunchDescription`` rather than embedding
a ``py_pkg`` node directly.

Both deadlines are fixed, and owned by ``watchdog/plausibility.py`` rather
than restated here. Missions no longer hold at depth -- a bang-bang leg
either reaches its turn or rails the tank and coasts to it -- so there is
no intended-hold duration left for them to scale with. The verdict JSON
lands next to the run's bag directory
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


def _wire_watchdog(context, *_args, **_kwargs):
    # OpaqueFunction so the verdict-path derivation can run on the resolved
    # LaunchConfiguration values.
    #
    # The floater deadline and sinker stall grace are deliberately NOT passed:
    # they are run-verdict policy and live on PlausibilityConfig, which the
    # node reads its parameter defaults from and scripts/analysis imports its
    # own thresholds from. Overriding them here would give sweeps one set of
    # numbers and the offline classifier another.
    bag_path = LaunchConfiguration("bag_path").perform(context).strip()

    verdict_path = str(Path(bag_path).parent / "run_verdict.json") if bag_path else ""

    watchdog_node = Node(
        package="py_pkg",
        executable="run_watchdog",
        name="run_watchdog",
        output="screen",
        parameters=[
            {
                "verdict_path": verdict_path,
                "max_start_depth_m": float(
                    LaunchConfiguration("max_start_depth_m").perform(context)
                ),
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
                "bag_path",
                default_value="",
                description=(
                    "The run's bag output directory; run_verdict.json is "
                    "written next to it (its parent). Empty writes no verdict "
                    "file."
                ),
            ),
            DeclareLaunchArgument(
                "max_start_depth_m",
                default_value="5.0",
                description=(
                    "Bad-start guard: abort (verdict abort_bad_start) if the "
                    "first armed odometry sample is already deeper than this "
                    "many metres — the vehicle fell during bringup and the "
                    "run can never be a valid surface-start recording. "
                    "0 disables the guard."
                ),
            ),
            OpaqueFunction(
                function=_wire_watchdog,
                condition=IfCondition(LaunchConfiguration("watchdog")),
            ),
        ]
    )
