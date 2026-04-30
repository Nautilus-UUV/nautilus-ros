"""Tier 3 sim test for the ACU roll control path.

Drives ACU_ROLL (rad) and reads the bridge's /sim/.../roll_position_rad
feedback. Asserts the simulated acu_roll_joint rotates in the commanded
direction. Bridge passes rad through unchanged; SDF range is symmetric
(±0.5236 rad), we pick positive. Composed lean (no oscillator) so we
don't race acu_oscillator for ACU_ROLL.
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
_ROLL_POSITION_TOPIC = f"/sim/{_MODEL_NAME}/acu/roll_position_rad"


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    """Lean ACU roll sim stack: bridges + Gazebo + robot, no oscillator."""
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


class _ACURollTestDriver(Node):
    """Publishes ACU_ROLL (rad), records roll joint position (rad)."""

    def __init__(self):
        super().__init__("acu_roll_sim_test_driver")
        self.received_roll_pos: list[float] = []
        self.imu_msg_count: int = 0

        self.roll_pub = create_publisher_for_topic(self, UUVTopics.ACU_ROLL)
        # IMU_LEFT is the sim-readiness signal. Factory-create the
        # subscription so QoS auto-matches SENSOR_STREAM (BEST_EFFORT) —
        # a raw RELIABLE subscription would silently drop every message.
        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
        # /sim/* lives outside uuv_ros_core; depth=10 matches the bridge.
        self.create_subscription(
            Float64, _ROLL_POSITION_TOPIC, self._on_roll_pos, 10
        )

    def _on_roll_pos(self, msg: Float64) -> None:
        self.received_roll_pos.append(float(msg.data))

    def _on_imu(self, msg: Imu) -> None:
        self.imu_msg_count += 1

    def publish_roll_rad(self, rad: float) -> None:
        msg = Float32()
        msg.data = float(rad)
        self.roll_pub.publish(msg)


@pytest.mark.sim
class ACURollSimTest(unittest.TestCase):
    """Behavior: a sustained +0.2618 rad roll command rotates the roll joint."""

    @classmethod
    def setUpClass(cls):
        # Pre-launch gz reap lives in generate_test_description; can't move
        # it here without killing this test's own gz process.
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _ACURollTestDriver()
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

    def test_positive_rad_command_rotates_roll_joint(self):
        """Sustained +30deg command on ACU_ROLL -> roll joint rotates positive."""
        # Drive at the +30° SDF limit for 12 s: smaller angles or shorter
        # runs get swallowed by the kP=-13 roll damping and the body
        # rotation is invisible in the GUI.
        target_rad = 0.5236
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        drive_duration_s = 12.0
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
            lambda: len(self.driver.received_roll_pos) >= 1,
            timeout_s=5.0,
        )
        self.assertGreaterEqual(
            len(self.driver.received_roll_pos),
            1,
            "no roll_position_rad samples before drive — bridge republish "
            "path may be down",
        )
        initial_pos_rad = self.driver.received_roll_pos[-1]

        deadline = time.monotonic() + drive_duration_s
        next_publish = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.driver.publish_roll_rad(target_rad)
                next_publish = now + drive_period_s
            self.executor.spin_once(timeout_sec=0.02)

        self._spin_for(settle_s)

        # 1) Joint-state stream flowed during the drive.
        self.assertGreaterEqual(
            len(self.driver.received_roll_pos),
            5,
            f"expected >=5 /sim/.../roll_position_rad samples; saw "
            f"{len(self.driver.received_roll_pos)}",
        )

        final_pos_rad = self.driver.received_roll_pos[-1]
        delta_rad = final_pos_rad - initial_pos_rad

        # 2) Rotated at least 50% of the commanded magnitude.
        self.assertGreaterEqual(
            delta_rad,
            target_rad * 0.5,
            f"acu_roll_joint did not rotate enough under sustained "
            f"+{target_rad} rad command: initial={initial_pos_rad} rad, "
            f"final={final_pos_rad} rad, delta={delta_rad} rad, "
            f"expected delta >= {target_rad * 0.5} rad",
        )

        # 3) Sign matches: bridge passes rad through unchanged.
        self.assertGreater(
            delta_rad,
            0.0,
            f"acu_roll_joint moved in wrong direction: delta={delta_rad} rad "
            f"under +{target_rad} rad command "
            f"(initial={initial_pos_rad}, final={final_pos_rad})",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class ACURollSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
