"""Tier 3 sim test for the BCU control path.

Drives BCU_RPM and asserts the simulated bladder fills: BCU_FLOW_RATE
goes positive, BCU_VOLUME (mL) increases monotonically. Verifies the
RPM -> flow -> volume integration end-to-end through the HAL bridge and
Gazebo buoyancy plugin; depth-tracking is intentionally out of scope.

Composed lean (bridges + Gazebo + robot, no oscillator) so we don't race
``unified_sim.launch.py``'s ``bcu_oscillator`` for BCU_RPM. Marker-gated
``@pytest.mark.sim``; opt in with ``pytest -m sim test/sim/`` after
sourcing the workspace install. ``BCU_SIM_GUI=1`` shows the Gazebo GUI.
Don't run alongside any other sim/rclpy process on the host — the
production topic names overlap.
"""

import os
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
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int16, Int32

from ._sim_helpers import reap_lingering_gz

# ---------------------------------------------------------------------------
# Launch description — composed, NOT a wholesale include of unified_sim.
# ---------------------------------------------------------------------------


# launch_test: tells the launch_testing pytest plugin to run this file as
# a real launch test (without it, generate_test_description never fires
# and proc_info isn't injected). sim: keeps it out of default runs.
@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    """Lean BCU sim stack: bridges + Gazebo + robot, no oscillator."""
    # Reap MUST happen here, not in setUpClass — by then LaunchService has
    # already spawned this test's own gz sim, and the pkill regex would
    # kill it (causing world-name lookup to time out, model never spawns).
    reap_lingering_gz()

    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "bridge.launch.py",
                )
            ]
        )
    )

    # DAVE convention: `gui` is always "true"; `headless` controls the display.
    gui_enabled = os.environ.get("BCU_SIM_GUI", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("dave_demos").find("dave_demos"),
                    "launch",
                    "dave_robot.launch.py",
                )
            ]
        ),
        # Same spawn pose as unified_sim.launch.py.
        launch_arguments={
            "z": "-5",
            "roll": "3.141592653589793",
            "yaw": "1.5707963267948966",
            "namespace": "glider_nautilus",
            "world_name": "dave_ocean_waves",
            "paused": "false",
            "gui": "true",
            "headless": "false" if gui_enabled else "true",
        }.items(),
    )

    return (
        LaunchDescription(
            [
                bridge_launch,
                robot_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


# ---------------------------------------------------------------------------
# Test driver node
# ---------------------------------------------------------------------------


class _BCUTestDriver(Node):
    """Publishes BCU_RPM, captures BCU_FLOW_RATE / BCU_VOLUME / IMU_LEFT."""

    def __init__(self):
        super().__init__("bcu_sim_test_driver")
        self.received_flow: list[float] = []
        self.received_volume_ml: list[int] = []
        self.imu_msg_count: int = 0

        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        create_subscription_for_topic(self, UUVTopics.BCU_FLOW_RATE, self._on_flow)
        create_subscription_for_topic(self, UUVTopics.BCU_VOLUME, self._on_volume)
        # IMU_LEFT is the sim-readiness signal: imu_sim_bridge has no timer,
        # so any message proves Gazebo physics + plugins are alive.
        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)

    def _on_flow(self, msg: Float32) -> None:
        self.received_flow.append(float(msg.data))

    def _on_volume(self, msg: Int32) -> None:
        self.received_volume_ml.append(int(msg.data))

    def _on_imu(self, msg: Imu) -> None:
        self.imu_msg_count += 1

    def publish_rpm(self, rpm: int) -> None:
        msg = Int16()
        msg.data = int(rpm)
        self.rpm_pub.publish(msg)


# ---------------------------------------------------------------------------
# Test class — runs after ReadyToTest fires.
# ---------------------------------------------------------------------------


@pytest.mark.sim
class BCUSimTest(unittest.TestCase):
    """Behavior: a sustained positive RPM fills the bladder in sim."""

    @classmethod
    def setUpClass(cls):
        # Pre-launch gz reap lives in generate_test_description; can't move
        # it here without killing this test's own gz process.
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _BCUTestDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    # ---- helpers ----------------------------------------------------------

    def _spin_for(self, duration_s: float, slice_s: float = 0.05) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def _spin_until(self, predicate, timeout_s: float, slice_s: float = 0.05):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=slice_s)
        return predicate()

    # ---- the test ---------------------------------------------------------

    def test_positive_rpm_fills_bladder(self):
        """Sustained positive RPM -> positive flow + monotonic volume rise.

        Sequence: wait for IMU_LEFT (sim ready), settle, send RPM=0 a few
        times to fire the bridge's startup clamp to BLADDER_MIN_VOLUME_M3
        (otherwise the +RPM accumulation gets subtracted from the SDF's
        ~1250 mL initial volume and looks like a decrease), snapshot the
        starting volume, drive +RPM at 10 Hz, then assert direction and
        sign — not exact volumetric rate (that's Tier 1).
        """
        target_rpm = 2000  # within [BCU_MOTOR_MIN_RPM, MAX]
        # CPU-only software rendering can need 20-40s for the world.
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        sync_settle_s = 2.0
        drive_duration_s = 5.0
        drive_period_s = 0.1  # 10 Hz, mirrors the production control loop
        settle_s = 2.0

        # 1) Wait for sim. IMU_LEFT is the readiness signal — BCU_VOLUME
        # isn't (its timer publishes 0 immediately).
        sim_ready = self._spin_until(
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU_LEFT never arrived within {startup_timeout_s}s — "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        # 2) Let the buoyancy plugin finish loading.
        self._spin_for(post_ready_settle_s)

        # 3) Fire the bridge's startup clamp to BLADDER_MIN_VOLUME_M3 by
        #    publishing RPM=0, then let the volume roundtrip settle.
        for _ in range(3):
            self.driver.publish_rpm(0)
            self.executor.spin_once(timeout_sec=0.05)
        self._spin_for(sync_settle_s)

        self.assertGreater(
            len(self.driver.received_volume_ml),
            0,
            "BCU_VOLUME never arrived even after IMU_LEFT confirmed sim "
            "readiness — bcu_sim_bridge may be down.",
        )
        starting_volume_ml = self.driver.received_volume_ml[-1]

        # Drop the zero-flow samples from the RPM=0 prelude.
        self.driver.received_flow.clear()

        # 5) Drive +RPM at 10 Hz for drive_duration_s.
        deadline = time.monotonic() + drive_duration_s
        next_publish = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.driver.publish_rpm(target_rpm)
                next_publish = now + drive_period_s
            self.executor.spin_once(timeout_sec=0.02)

        # 6) Stop the pump and let queues drain.
        self.driver.publish_rpm(0)
        self._spin_for(settle_s)

        # 7a) Need >=1 positive flow sample (not all — the first can land
        # before the subscription handshake completes).
        positive_flow_samples = [f for f in self.driver.received_flow if f > 0.0]
        self.assertGreaterEqual(
            len(positive_flow_samples),
            1,
            f"expected >=1 positive BCU_FLOW_RATE sample while driving "
            f"+{target_rpm} RPM; saw {self.driver.received_flow!r}",
        )

        # A negative sample means the bridge sign convention flipped.
        negative_flow_samples = [f for f in self.driver.received_flow if f < 0.0]
        self.assertEqual(
            negative_flow_samples,
            [],
            f"unexpected negative flow samples while driving +RPM: "
            f"{negative_flow_samples!r}",
        )

        # 7b) Bladder volume must rise (we sync'd to the floor in step 3).
        ending_volume_ml = self.driver.received_volume_ml[-1]
        self.assertGreater(
            ending_volume_ml,
            starting_volume_ml,
            f"bladder volume did not increase under sustained +RPM: "
            f"start={starting_volume_ml} mL, end={ending_volume_ml} mL",
        )


# ---------------------------------------------------------------------------
# Post-shutdown tests — run after the launched processes are torn down.
# ---------------------------------------------------------------------------


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class BCUSimPostShutdown(unittest.TestCase):
    """Sanity-check + cleanup after launched processes are torn down."""

    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort, no assert."""
        reap_lingering_gz()
