"""Tier 3 sim test: TRIM_AND_NEUTRAL with EKF replaced by ground-truth pose.

Mirror of ``test_trim_neutral_sim.py`` — same controllers, same mission,
same Gazebo composition — but the EKF stack is swapped for the sim-only
``gt_pose_bridge`` (``nautilus_hal``) which republishes Gazebo's
``/model/glider_nautilus/odometry`` onto ``POSITION_ESTIMATION``.

Why a separate test: the EKF currently diverges over long runs (see
``src/nautilus-ros/docs/ekf_node_issues.md`` — additive quaternion update,
fixed-dt assumptions). That makes the EKF-in-loop trim test sensitive to
attitude noise the ACU then chases. This test isolates the depth + ACU
controllers from EKF drift, so any failure here is unambiguously a
controller regression. Tightened thresholds reflect that.

Composition (no ``IncludeLaunchDescription`` of trim_sim.launch.py — we
need to swap nodes, not parameterise):

    nautilus_hal/bridge.launch.py    (HAL bridges)
    dave_demos/dave_robot.launch.py  (Gazebo + glider_nautilus)
    nautilus_hal gt_pose_bridge      (instead of ekf_prefilter + ekf_node)
    py_pkg depth_node
    py_pkg acu_node
    py_pkg pathfinding_node

Marker-gated ``@pytest.mark.sim``; opt in with
``pytest -m sim test/sim/`` after sourcing the workspace install.
``TRIM_GT_SIM_GUI=1`` shows the Gazebo GUI.
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
from nautilus_msgs.msg import MissionCommand
from nav_msgs.msg import Odometry
from py_pkg.path.missions.factory import MissionId
from py_pkg.physics import gauge_pressure_pa
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32, String

from ._sim_helpers import reap_lingering_gz

TARGET_PRESSURE_PA = 65332.0  # ~6.5 m of seawater (gauge); spawn is ~5 m
GROUND_TRUTH_TOPIC = "/model/glider_nautilus/odometry"


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    # Reap MUST happen here, not in setUpClass.
    reap_lingering_gz()

    gui_enabled = os.environ.get("TRIM_GT_SIM_GUI", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # Paired with test_trim_neutral_sim.py — same scenario, same plant
    # and control config (the memory rule on paired EKF/GT thresholds).
    test_scenario = os.path.join(
        os.path.dirname(__file__), "scenarios", "test_trim_neutral.yaml"
    )

    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "bridge.launch.py",
                )
            ]
        ),
        launch_arguments={"scenario": test_scenario}.items(),
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
        # Same spawn pose as test_trim_neutral_sim.py / trim_sim.launch.py.
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

    # gt_pose_bridge declares `model_name` itself (default mirrors
    # robot_specs / baseline scenario), so no per-bridge param overlay
    # is needed here unless the paired test wants to override it.
    gt_pose_bridge = LaunchNode(
        package="nautilus_hal",
        executable="gt_pose_bridge",
        name="nautilus_gt_pose_bridge",
        output="screen",
    )

    depth_node = LaunchNode(
        package="py_pkg",
        executable="depth_node",
        name="depth_control_node",
        output="screen",
    )
    acu_node = LaunchNode(
        package="py_pkg",
        executable="acu_node",
        name="acu_control_node",
        output="screen",
    )
    pathfinding_node = LaunchNode(
        package="py_pkg",
        executable="pathfinding_node",
        name="pathfinding_node",
        output="screen",
    )

    return (
        LaunchDescription(
            [
                bridge_launch,
                robot_launch,
                gt_pose_bridge,
                depth_node,
                acu_node,
                pathfinding_node,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _TrimNeutralGTTestDriver(Node):
    """Kicks the TRIM mission and captures pressure, ground-truth pose, setpoint."""

    def __init__(self):
        super().__init__("trim_neutral_sim_gt_test_driver")
        self.gauge_pressure_pa: list[tuple[float, float]] = []
        self.odom_samples: list[tuple[float, Odometry]] = []
        self.target_samples: list[tuple[float, Pose]] = []
        self.bcu_volume_ml: list[tuple[float, int]] = []
        self.bcu_rpm: list[tuple[float, int]] = []
        self.imu_msg_count: int = 0

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)
        create_subscription_for_topic(
            self, UUVTopics.BCU_VOLUME, self._on_volume
        )
        create_subscription_for_topic(self, UUVTopics.BCU_RPM, self._on_rpm)

        # Privileged sim-only ground truth, same as the EKF-in-loop variant.
        self.create_subscription(Odometry, GROUND_TRUTH_TOPIC, self._on_odom, 10)

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_pressure(self, msg: Int32) -> None:
        self.gauge_pressure_pa.append(
            (time.monotonic(), gauge_pressure_pa(float(msg.data)))
        )

    def _on_target(self, msg: Pose) -> None:
        self.target_samples.append((time.monotonic(), msg))

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_samples.append((time.monotonic(), msg))

    def _on_volume(self, msg: Int32) -> None:
        self.bcu_volume_ml.append((time.monotonic(), int(msg.data)))

    def _on_rpm(self, msg) -> None:
        self.bcu_rpm.append((time.monotonic(), int(msg.data)))

    def publish_mission(self, target_pa: float) -> None:
        cmd = MissionCommand()
        cmd.mission_id = int(MissionId.TRIM_AND_NEUTRAL_BUOYANCY)
        cmd.target_pressure_pa = float(target_pa)
        cmd.angle_rad = 0.0
        cmd.n_resurfaces = 0
        self.path_pub.publish(cmd)

    def publish_start(self) -> None:
        msg = String()
        msg.data = "start"
        self.command_pub.publish(msg)


def _window(
    samples: list[tuple[float, object]],
    window_start_t: float,
) -> list[object]:
    return [s for (t, s) in samples if t >= window_start_t]


@pytest.mark.sim
class TrimNeutralGTSimTest(unittest.TestCase):
    """Behavior: with perfect pose, TRIM holds depth and the glider stops moving.

    With the EKF replaced by ground truth, the only loops under evaluation
    are the depth controller (BCU PID) and the ACU controller — pathfinding
    is just a setpoint pump. Tightened thresholds vs. the EKF-in-loop
    test enforce that the controller actually tracks rather than only
    "doesn't diverge".
    """

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _TrimNeutralGTTestDriver()
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

    def test_trim_and_neutral_buoyancy_holds_at_target(self):
        """TRIM mission -> pressure converges to target, glider stops moving."""
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # 120 s budget: ~50 s for the bladder to drain at saturated pump
        # rate (1.25 L -> ~0.25 L, the SDF's effective hold-buoyancy
        # range), then ~30-40 s for the glider's velocity / integrator
        # transients to settle around target before the 5 s assertion
        # window samples a steady mean.
        mission_duration_s = 120.0
        drain_s = 2.0
        assert_window_s = 5.0

        # Tightened thresholds vs the EKF-in-loop test, but bounded by
        # what the BCU plant naturally delivers: the pump is rate-limited
        # (~21 mL/s) and the bladder swing for the full 1.5 m descent
        # is ~1 L, so the glider's momentum carries it ~0.3 m past
        # target before the bladder refills toward neutral. 3000 Pa
        # caps that overshoot. Tighten as the SDF mass/buoyancy balance
        # improves or the pump gets faster.
        pressure_tol_pa = 3000.0  # ~0.3 m
        v_linear_max = 0.05  # m/s
        omega_max = 0.05  # rad/s

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

        self.driver.publish_mission(TARGET_PRESSURE_PA)
        # Tiny gap so PATH lands before COMMAND.
        self._spin_for(0.5)
        self.driver.publish_start()

        mission_start_t = time.monotonic()
        self._spin_for(mission_duration_s)
        self._spin_for(drain_s)

        window_start_t = time.monotonic() - drain_s - assert_window_s
        window_pressure = _window(self.driver.gauge_pressure_pa, window_start_t)
        window_odom = _window(self.driver.odom_samples, window_start_t)

        targets_during = _window(self.driver.target_samples, mission_start_t)
        self.assertGreaterEqual(
            len(targets_during),
            100,
            f"pathfinding broadcast only {len(targets_during)} setpoints in "
            f"{mission_duration_s}s — expected ~{int(mission_duration_s * 10)}. "
            "Did `start` reach pathfinding_node?",
        )
        last_target = self.driver.target_samples[-1][1]
        self.assertEqual(
            last_target.position.z,
            TARGET_PRESSURE_PA,
            f"latest POSITION_TARGET.position.z = {last_target.position.z} Pa, "
            f"expected {TARGET_PRESSURE_PA} Pa (TRIM mission's hold-depth)",
        )

        self.assertGreaterEqual(
            len(window_pressure),
            5,
            f"only {len(window_pressure)} EXTERNAL_PRESSURE samples in the "
            f"last {assert_window_s}s — bridge or sensor stalled.",
        )
        mean_gauge = sum(window_pressure) / len(window_pressure)
        self.assertAlmostEqual(
            mean_gauge,
            TARGET_PRESSURE_PA,
            delta=pressure_tol_pa,
            msg=(
                f"mean gauge pressure over last {assert_window_s}s = "
                f"{mean_gauge:.0f} Pa, target {TARGET_PRESSURE_PA:.0f} Pa "
                f"(tol ±{pressure_tol_pa:.0f} Pa)."
            ),
        )

        self.assertGreaterEqual(
            len(window_odom),
            5,
            f"only {len(window_odom)} odometry samples in last "
            f"{assert_window_s}s — sim ground-truth bridge stalled.",
        )
        mean_v = sum(_speed(o) for o in window_odom) / len(window_odom)
        mean_w = sum(_omega(o) for o in window_odom) / len(window_odom)
        self.assertLess(
            mean_v,
            v_linear_max,
            f"mean |v_linear| over last {assert_window_s}s = {mean_v:.3f} m/s "
            f"(>= {v_linear_max} m/s). Glider hasn't come to rest.",
        )
        self.assertLess(
            mean_w,
            omega_max,
            f"mean |omega| over last {assert_window_s}s = {mean_w:.3f} rad/s "
            f"(>= {omega_max} rad/s). ACU never trimmed the attitude.",
        )

        # Sanity: every odom pose finite + quaternion unit-norm.
        for i, odom in enumerate(window_odom):
            p = odom.pose.pose.position
            for axis_name, axis_value in (("x", p.x), ("y", p.y), ("z", p.z)):
                self.assertTrue(
                    math.isfinite(axis_value),
                    f"odom[{i}].position.{axis_name} = {axis_value!r}",
                )
            q = odom.pose.pose.orientation
            norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
            self.assertAlmostEqual(
                norm_sq,
                1.0,
                delta=0.02,
                msg=f"odom[{i}].orientation |q|^2 = {norm_sq:.6f}",
            )


def _speed(odom: Odometry) -> float:
    v = odom.twist.twist.linear
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _omega(odom: Odometry) -> float:
    w = odom.twist.twist.angular
    return math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class TrimNeutralGTSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
