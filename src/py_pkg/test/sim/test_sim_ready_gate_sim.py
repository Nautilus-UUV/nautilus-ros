"""Tier 3 sim test: the physics-liveness probe end-to-end against Gazebo.

Boots ``nautilus_hal/sawtooth_sim.launch.py`` with ``physics_probe:=true``
(the argument run_sweep injects into every campaign run) and nothing else
driving the plant: no mission autostart, no watchdog, no recorder. The
gate must then walk its full probe pipeline against the real stack —

    graph checks -> unpause -> stepping proof
      -> entry-servo suppress (gz-transport -> HeaveAugmentPlugin;
         the canonical model.sdf ships entry momentum ON, so a broken
         suppress drags the hull metres deep and fails this test)
      -> quiescence-gated float baseline (passive spawn settle waited out)
      -> commanded deflate, contested against the bcu bridge's 10 Hz
         spawn-set-point republish, until the hull demonstrably sinks
      -> silent restore (the bridge republish re-inflates the plugin)
      -> settle, unsuppress, /sim/ready latch

and the test asserts the observable contract: the latch arrives, the
hull actually dipped below its restored float beforehand (the liveness
evidence), and it is NOT descending after the latch (the entry-servo
suppress/release regression check — a mis-released servo fires on the
probe's sink and is still burying the hull when /sim/ready lands).

Spawn is the campaign's ``z:=-0.115`` so the settle matches sweep
bringup. Marker-gated ``@pytest.mark.sim``; ``SIM_GUI=1`` shows the GUI.
Don't run alongside any other sim launch on the host — both bind the
same gz transport bus.
"""

import os
import statistics
import time
import unittest

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from nav_msgs.msg import Odometry
from py_pkg.uuv_ros_core import UUVTopics, create_subscription_for_topic
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool

from ._sim_helpers import (
    GROUND_TRUTH_ODOM_TOPIC,
    reap_lingering_gz,
    sim_gui_enabled,
    spin_for,
    spin_until,
    window,
)

# Bringup (~20-40 s) + settle-to-quiescence + deflate/restore/settle at
# sweep-load RTFs well under 1. Generous — the probe failing its own
# deadlines shuts the launch down long before this expires.
READY_TIMEOUT_S = 300.0
POST_READY_WINDOW_S = 5.0
# The gate demands a 0.10 m sink below its own baseline and restores to
# within 0.10 m of it; measured against the post-ready float (not the
# gate's internal baseline) the guaranteed dip margin shrinks, hence 0.05.
MIN_DIP_BELOW_FLOAT_M = 0.05
# A released-but-refiring entry servo pulls ~0.2 m/s: 5 s of post-ready
# descent would show up as metres, not centimetres.
MAX_POST_READY_SINK_M = 0.15


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    # Reap MUST happen here, not in setUpClass.
    reap_lingering_gz()

    sawtooth_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "sawtooth_sim.launch.py",
                )
            ]
        ),
        launch_arguments={
            "headless": "false" if sim_gui_enabled() else "true",
            "mission_autostart": "false",
            "watchdog": "false",
            "record": "false",
            "physics_probe": "true",
            "angle_rad": "0.0",
            "z": "-0.115",
        }.items(),
    )
    return (
        LaunchDescription([sawtooth_launch, launch_testing.actions.ReadyToTest()]),
        {},
    )


class _GateProbeObserver(Node):
    """Watches /sim/ready and ground-truth z; commands nothing."""

    def __init__(self):
        super().__init__("sim_ready_gate_probe_observer")
        self.ready_at: float | None = None  # monotonic
        self.odom_samples: list[tuple[float, float]] = []  # (monotonic, z)

        # COMMAND QoS (TRANSIENT_LOCAL): a late-joining observer still
        # sees the latch, exactly like auto_mission.
        create_subscription_for_topic(self, UUVTopics.SIM_READY, self._on_ready)
        self.create_subscription(
            Odometry, GROUND_TRUTH_ODOM_TOPIC, self._on_odom, 10
        )

    def _on_ready(self, msg: Bool) -> None:
        if msg.data and self.ready_at is None:
            self.ready_at = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_samples.append((time.monotonic(), msg.pose.pose.position.z))


@pytest.mark.sim
class SimReadyGateProbeTest(unittest.TestCase):
    """Behavior: probe sinks and restores the hull, then latches ready."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.observer = _GateProbeObserver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.observer)

    def tearDown(self):
        self.executor.remove_node(self.observer)
        self.observer.destroy_node()
        self.executor.shutdown()

    def test_probe_dips_restores_and_latches_ready(self):
        latched = spin_until(
            self.executor,
            lambda: self.observer.ready_at is not None,
            timeout_s=READY_TIMEOUT_S,
        )
        self.assertTrue(
            latched,
            f"/sim/ready never latched within {READY_TIMEOUT_S:.0f}s — the "
            "probe failed (no hull response / no restore / no quiescent "
            "baseline) or bringup never completed. The gate's log line "
            "spells out which.",
        )

        # Keep observing past the latch for the restore/servo checks.
        spin_for(self.executor, POST_READY_WINDOW_S)

        ready_at = self.observer.ready_at
        pre = [z for t, z in self.observer.odom_samples if t < ready_at]
        post = window(self.observer.odom_samples, ready_at)
        self.assertGreater(len(pre), 50, "no pre-ready odometry captured")
        self.assertGreater(len(post), 50, "no post-ready odometry captured")

        z_float = statistics.median(post)

        # Liveness evidence: the hull demonstrably sank below the float it
        # was restored to. A gate that latched without a real response
        # (the v3 frozen mode) leaves no such dip.
        dip_m = z_float - min(pre)
        self.assertGreaterEqual(
            dip_m,
            MIN_DIP_BELOW_FLOAT_M,
            f"no probe dip in the pre-ready odometry (max excursion "
            f"{dip_m:.3f}m below the post-ready float {z_float:.3f}m) — "
            "did the probe actually command the deflate?",
        )

        # Servo regression: after the latch the hull must float, not
        # descend — a mis-suppressed/mis-released entry servo is still
        # dragging it down here.
        post_sink_m = z_float - min(post)
        self.assertLessEqual(
            post_sink_m,
            MAX_POST_READY_SINK_M,
            f"hull sinking after /sim/ready (min post-ready z is "
            f"{post_sink_m:.3f}m below the median float) — entry-momentum "
            "servo fired around the probe?",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class SimReadyGateProbePostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths. A gate
        # probe failure exits 5 and would surface here as well as in the
        # active test above.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
