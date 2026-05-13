"""Tier 3 sim test for the EKF prefilter -> EKF pipeline.

Pipeline under test:

    Gazebo IMU plugin -> /imu/left (raw, BEST_EFFORT)
                      -> ekf_prefilter (EMA smoothing)
                      -> /imu/filtered/left
                      -> ekf_node (predict + update)
                      -> /position/estimation

The robot floats passively — gravity alone drives the IMU. The headline
contract is "the EKF stack is wired up and produces well-formed pose
estimates"; we don't pin filter cutoff, drift rate, or absolute pose
(those are SDF / numerics dependent and belong in Tier 1).

Composed lean: bridges + Gazebo + robot + the two EKF nodes added
explicitly (no launch wrapper exists for the EKF stack). Marker-gated
``@pytest.mark.sim``; opt in with ``pytest -m sim test/sim/`` after
sourcing the workspace install. ``EKF_SIM_GUI=1`` shows the Gazebo GUI.
"""

import math
import os
import time
import unittest

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from geometry_msgs.msg import Pose
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node as LaunchNode
from launch_ros.substitutions import FindPackageShare
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu

from ._sim_helpers import reap_lingering_gz


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    """Lean EKF sim stack: bridges + Gazebo + robot + prefilter + ekf_node."""
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

    gui_enabled = os.environ.get("EKF_SIM_GUI", "").lower() in (
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
        # Same spawn pose as trim_sim.launch.py.
        launch_arguments={
            "z": "-5",
            "roll": "3.141592653589793",
            "yaw": "1.5707963267948966",
            "namespace": "glider_nautilus",
            "world_name": "dave_ocean_waves",
            "paused": "false",
            # DAVE: `gui` always "true"; `headless` controls the display.
            "gui": "true",
            "headless": "false" if gui_enabled else "true",
        }.items(),
    )

    prefilter_node = LaunchNode(
        package="py_pkg",
        executable="ekf_prefilter",
        name="ekf_prefilter",
        output="screen",
    )
    ekf_node = LaunchNode(
        package="py_pkg",
        executable="ekf_node",
        name="ekf_node",
        output="screen",
    )

    return (
        LaunchDescription(
            [
                bridge_launch,
                robot_launch,
                prefilter_node,
                ekf_node,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _EKFTestDriver(Node):
    """Captures raw IMU, filtered IMU, and EKF pose-estimation traffic."""

    def __init__(self):
        super().__init__("ekf_sim_test_driver")
        self.raw_imu_count: int = 0
        self.filtered_imu_count: int = 0
        self.received_poses: list[Pose] = []

        # Factories ensure QoS matches what each producer publishes:
        # IMU_LEFT is SENSOR_STREAM (BEST_EFFORT); IMU_FILTERED_LEFT and
        # POSITION_ESTIMATION are CONTROL (RELIABLE). A raw RELIABLE
        # subscription on IMU_LEFT would silently drop every message.
        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_raw_imu)
        create_subscription_for_topic(
            self, UUVTopics.IMU_FILTERED_LEFT, self._on_filtered_imu
        )
        create_subscription_for_topic(
            self, UUVTopics.POSITION_ESTIMATION, self._on_pose
        )

    def _on_raw_imu(self, msg: Imu) -> None:
        self.raw_imu_count += 1

    def _on_filtered_imu(self, msg: Imu) -> None:
        self.filtered_imu_count += 1

    def _on_pose(self, msg: Pose) -> None:
        self.received_poses.append(msg)


@pytest.mark.sim
class EKFPipelineSimTest(unittest.TestCase):
    """Behavior: the EKF stack produces well-formed pose estimates from sim IMU."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _EKFTestDriver()
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

    def test_pipeline_produces_well_formed_poses(self):
        """Raw IMU -> prefilter -> EKF -> finite poses with unit quaternions."""
        # CPU-only software rendering can need 20-40s for the world;
        # 60s leaves margin without slowing healthy runs.
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # 5 s at ~50-100 Hz IMU gives many samples to assert against.
        collection_s = 5.0

        # 1) Wait for raw IMU traffic — the readiness signal common to all
        #    sim tests.
        sim_ready = self._spin_until(
            lambda: self.driver.raw_imu_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU_LEFT never arrived within {startup_timeout_s}s — "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        # 2) Let the prefilter and EKF subscriptions handshake; their
        #    output topics are CONTROL/RELIABLE so an early sample loss
        #    here would not be recovered.
        self._spin_for(post_ready_settle_s)
        raw_baseline = self.driver.raw_imu_count
        filtered_baseline = self.driver.filtered_imu_count
        poses_baseline = len(self.driver.received_poses)

        # 3) Collect over a steady window, ignoring startup transients.
        self._spin_for(collection_s)
        raw_during = self.driver.raw_imu_count - raw_baseline
        filtered_during = self.driver.filtered_imu_count - filtered_baseline
        poses_collected = self.driver.received_poses[poses_baseline:]

        # 4a) Prefilter alive: filtered IMU must keep up with raw IMU
        #     within 20 %. A stuck prefilter (silently consuming the first
        #     msg and dying, or wrong subscription) would show 0 here.
        self.assertGreater(
            raw_during,
            0,
            "raw IMU stream stalled mid-test — Gazebo IMU plugin or imu_sim_bridge died",
        )
        self.assertGreater(
            filtered_during,
            0,
            "ekf_prefilter never republished — check IMU_LEFT subscription / QoS",
        )
        self.assertGreaterEqual(
            filtered_during,
            int(0.8 * raw_during),
            f"prefilter rate ({filtered_during}) lags raw rate "
            f"({raw_during}) by >20% — prefilter is dropping messages",
        )

        # 4b) EKF alive: at least 5 published poses over the window.
        #     Threshold low enough to survive a slow startup tick or two,
        #     high enough that "EKF callback fires once and dies" fails.
        self.assertGreaterEqual(
            len(poses_collected),
            5,
            f"ekf_node published only {len(poses_collected)} poses in "
            f"{collection_s}s — check IMU_FILTERED_LEFT subscription / QoS",
        )

        # 4c) Every pose has finite position. NaN/Inf here means the EKF
        #     diverged numerically (a known fragility — see the covariance
        #     hygiene caveats in test/unit/test_ekf_filter.py).
        for i, pose in enumerate(poses_collected):
            for axis_name, axis_value in (
                ("x", pose.position.x),
                ("y", pose.position.y),
                ("z", pose.position.z),
            ):
                self.assertTrue(
                    math.isfinite(axis_value),
                    f"pose[{i}].position.{axis_name} = {axis_value!r} "
                    "(EKF diverged to NaN/Inf)",
                )

        # 4d) Every quaternion is unit-norm. The EKF re-normalises after
        #     each update (ekf_filter.py:154), so any drift here is a
        #     post-publish corruption or a pre-normalisation bug.
        for i, pose in enumerate(poses_collected):
            q = pose.orientation
            norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
            self.assertAlmostEqual(
                norm_sq,
                1.0,
                delta=0.02,
                msg=(
                    f"pose[{i}].orientation has |q|^2 = {norm_sq:.6f}, "
                    f"expected ~1.0 (q = [{q.x}, {q.y}, {q.z}, {q.w}])"
                ),
            )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class EKFPipelineSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
