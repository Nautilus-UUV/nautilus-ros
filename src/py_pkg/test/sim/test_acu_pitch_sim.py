"""Tier 3 sim test for the ACU pitch control path.

Drives ACU_PITCH (mm) and reads the bridge's /sim/.../pitch_position_m
feedback topic. Asserts the simulated acu_tilt_joint moves in the
commanded direction. Negative-mm is the only in-range motion from the
SDF resting pose; positive-mm would clamp to 0 and look static.

Assertion is on (final - initial) so the test survives SDF link/joint
pose changes. Composed lean (bridges + Gazebo + robot, no oscillator)
so we don't race acu_oscillator for ACU_PITCH.
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
from std_msgs.msg import Float32, Float64

from ._sim_helpers import reap_lingering_gz

# /sim/* feedback topic — intentionally outside uuv_ros_core (see
# SimDebugTopics docstring in nautilus_hal/constants.py).
_MODEL_NAME = "glider_nautilus"
_PITCH_POSITION_TOPIC = f"/sim/{_MODEL_NAME}/acu/pitch_position_m"


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    """Lean ACU pitch sim stack: bridges + Gazebo + robot, no oscillator."""
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

    gui_enabled = os.environ.get("ACU_SIM_GUI", "").lower() in (
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
        launch_arguments={
            "z": "-5",
            "roll": "3.141592653589793",
            "yaw": "1.5707963267948966",
            "namespace": _MODEL_NAME,
            "world_name": "dave_ocean_waves",
            "paused": "false",
            # DAVE: `gui` always "true"; `headless` controls the display.
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


class _ACUPitchTestDriver(Node):
    """Publishes ACU_PITCH (mm), records pitch joint position (m)."""

    def __init__(self):
        super().__init__("acu_pitch_sim_test_driver")
        self.received_pitch_pos: list[float] = []
        self.imu_msg_count: int = 0

        self.pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)
        # IMU_LEFT is the sim-readiness signal. Factory-create the
        # subscription so QoS auto-matches SENSOR_STREAM (BEST_EFFORT) —
        # a raw RELIABLE subscription would silently drop every message.
        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
        # /sim/* lives outside uuv_ros_core; depth=10 matches the bridge.
        self.create_subscription(
            Float64, _PITCH_POSITION_TOPIC, self._on_pitch_pos, 10
        )

    def _on_pitch_pos(self, msg: Float64) -> None:
        self.received_pitch_pos.append(float(msg.data))

    def _on_imu(self, msg: Imu) -> None:
        self.imu_msg_count += 1

    def publish_pitch_mm(self, mm: float) -> None:
        msg = Float32()
        msg.data = float(mm)
        self.pitch_pub.publish(msg)


@pytest.mark.sim
class ACUPitchSimTest(unittest.TestCase):
    """Behavior: a sustained negative-mm pitch command extends the tilt joint."""

    @classmethod
    def setUpClass(cls):
        # Pre-launch gz reap lives in generate_test_description; can't move
        # it here without killing this test's own gz process.
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _ACUPitchTestDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

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

    def test_negative_mm_command_extends_tilt_joint(self):
        """Sustained -50 mm command on ACU_PITCH -> tilt joint moves negative."""
        # 50 mm: well under the 119.5 mm saturation; joint vel limit is
        # 0.5 m/s so 50 mm settles in ~0.1 s, 4 s gives plenty of margin.
        target_mm = -50.0
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        drive_duration_s = 4.0
        drive_period_s = 0.1
        settle_s = 1.5

        sim_ready = self._spin_until(
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU_LEFT never arrived within {startup_timeout_s}s — "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        self._spin_for(post_ready_settle_s)

        # Snapshot resting position; (final - initial) instead of (final
        # vs target) survives future SDF link/joint pose changes.
        self._spin_until(
            lambda: len(self.driver.received_pitch_pos) >= 1,
            timeout_s=5.0,
        )
        self.assertGreaterEqual(
            len(self.driver.received_pitch_pos),
            1,
            "no pitch_position_m samples before drive — bridge republish "
            "path may be down",
        )
        initial_pos_m = self.driver.received_pitch_pos[-1]

        # Drive a constant -50 mm command at 10 Hz.
        deadline = time.monotonic() + drive_duration_s
        next_publish = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.driver.publish_pitch_mm(target_mm)
                next_publish = now + drive_period_s
            self.executor.spin_once(timeout_sec=0.02)

        self._spin_for(settle_s)

        # 1) Joint-state stream flowed during the drive.
        self.assertGreaterEqual(
            len(self.driver.received_pitch_pos),
            5,
            f"expected >=5 /sim/.../pitch_position_m samples; saw "
            f"{len(self.driver.received_pitch_pos)}",
        )

        final_pos_m = self.driver.received_pitch_pos[-1]
        delta_m = final_pos_m - initial_pos_m
        target_m = target_mm * 1e-3

        # 2) Moved at least 50% of the commanded magnitude.
        self.assertLessEqual(
            delta_m,
            target_m * 0.5,
            f"acu_tilt_joint did not extend enough under sustained "
            f"{target_mm} mm command: initial={initial_pos_m} m, "
            f"final={final_pos_m} m, delta={delta_m} m, "
            f"expected delta <= {target_m * 0.5} m",
        )

        # 3) Sign matches commanded sign (a flip means bridge sign or
        # wrong-end SDF limit).
        self.assertLess(
            delta_m,
            0.0,
            f"acu_tilt_joint moved in wrong direction: delta={delta_m} m "
            f"under {target_mm} mm command "
            f"(initial={initial_pos_m}, final={final_pos_m})",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class ACUPitchSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
