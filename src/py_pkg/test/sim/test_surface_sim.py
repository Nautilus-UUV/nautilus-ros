"""Tier 3 sim test: full SURFACE mission, end-to-end.

Pipeline under test (same as test_trim_neutral_sim, swapped mission):

    /path + /command -> pathfinding_node -> /position/target
                                          ^
                       /external/pressure -|
    /position/target +-> bcu_node       -> /bcu/rpm + /bcu/valves -> bridge -> Gazebo
    /position/target +-> acu_node         -> /acu/pitch + /acu/roll -> bridge -> Gazebo
                       /position/estimation
                          ^
                          attitude_node <- imu_prefilter <- /imu/left

The test loads ``MissionId.SURFACE = 2`` (no operator parameters) and
asserts:

  * the closed loop drives the glider up to gauge pressure
    <= SURFACE_THRESHOLD_PA (~5 kPa, ~0.5 m), and
  * the mission *self-terminates* after ~10 s of continuous surface
    dwell — pathfinding_node returns to IDLE and stops broadcasting
    POSITION_TARGET.

Tolerances are deliberately loose — the estimator is in the loop, and
any orientation drift it carries feeds the ACU. Only one Tier 3 test
for SURFACE for now; a GT-pose mirror can be added later if the
estimator-in-loop variant gets too flaky.

Composed via ``nautilus_hal/launch/surface_sim.launch.py``.
Marker-gated ``@pytest.mark.sim``; opt in with
``pytest -m sim test/sim/`` after sourcing the workspace install.
``SIM_GUI=1`` shows the Gazebo GUI.
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
from launch_ros.substitutions import FindPackageShare
from nautilus_msgs.msg import MissionCommand
from nav_msgs.msg import Odometry
from py_pkg.path.missions.factory import MissionId
from py_pkg.path.missions.sawtooth import SURFACE_THRESHOLD_PA
from py_pkg.physics import gauge_pressure_pa
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32

from ._sim_helpers import (
    omega,
    reap_lingering_gz,
    sim_gui_enabled,
    speed,
    spin_for,
    spin_until,
    window,
)

GROUND_TRUTH_TOPIC = "/model/glider_nautilus/odometry"


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    # Reap MUST happen here, not in setUpClass.
    reap_lingering_gz()

    gui_enabled = sim_gui_enabled()

    surface_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "surface_sim.launch.py",
                )
            ]
        ),
        # Test publishes the MissionCommand + start itself for deterministic
        # timing; launches default `mission_autostart` to false. The
        # baseline scenario covers all plant/control fields.
        launch_arguments={
            "headless": "false" if gui_enabled else "true",
        }.items(),
    )

    return (
        LaunchDescription(
            [
                surface_sim_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _SurfaceTestDriver(Node):
    """Kicks the SURFACE mission and captures pressure, ground-truth pose, setpoint."""

    def __init__(self):
        super().__init__("surface_sim_test_driver")
        self.gauge_pressure_pa: list[tuple[float, float]] = []
        self.odom_samples: list[tuple[float, Odometry]] = []
        self.target_samples: list[tuple[float, Pose]] = []
        self.imu_msg_count: int = 0

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        # IMU readiness signal (same convention as the other Tier 3 tests).
        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)

        # Privileged sim-only ground truth — same path as test_trim_neutral_sim.
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

    def publish_mission(self) -> None:
        cmd = MissionCommand()
        cmd.mission_id = int(MissionId.SURFACE)
        # SURFACE has no operator parameters; leave them all at zero.
        cmd.target_pressure_pa = 0.0
        cmd.angle_rad = 0.0
        cmd.n_resurfaces = 0
        self.path_pub.publish(cmd)

    def publish_start(self) -> None:
        msg = Bool()
        msg.data = True
        self.command_pub.publish(msg)


@pytest.mark.sim
class SurfaceSimTest(unittest.TestCase):
    """Behavior: SURFACE mission ascends to gauge ~0 and self-terminates."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _SurfaceTestDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    def test_surface_mission_ascends_and_self_terminates(self):
        """SURFACE mission -> gauge pressure ~0, then mission completes."""
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # ~4.2 m ascent (BCU-bladder rate-limited) + 10 s surface dwell +
        # margin. Spawn is z=-5 (~49 kPa gauge). On the lake-calibrated
        # plant the ascent runs ~0.035-0.04 m/s at the railed bladder;
        mission_duration_s = 300.0
        drain_s = 2.0
        # Last 5 s of the mission window — the dwell guarantees the glider
        # has been at the surface for at least 10 s by then.
        assert_window_s = 5.0
        # Mission self-terminates after a 10 s surface dwell. After that
        # pathfinding_node returns to IDLE and stops emitting POSITION_TARGET.
        # Allow one tick (0.1 s) of slop on top of that.
        no_target_window_s = 2.0

        # Loose tolerances — estimator is in the loop. Tighten as the estimator stabilises.
        v_linear_max = 0.10  # m/s
        omega_max = 0.15  # rad/s

        # 1) Wait for sim. IMU is the readiness signal.
        sim_ready = spin_until(
            self.executor,
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU never arrived within {startup_timeout_s}s — "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        # 2) Let subscriptions handshake.
        spin_for(self.executor, post_ready_settle_s)

        # 3) Kick the mission.
        self.driver.publish_mission()
        spin_for(self.executor, 0.5)
        self.driver.publish_start()

        # 4) Run the closed loop. Don't early-exit on ascent: we want the
        #    full duration so we can assert the no-emission window after
        #    self-termination.
        mission_start_t = time.monotonic()
        spin_for(self.executor, mission_duration_s)
        spin_for(self.executor, drain_s)

        # 5) Take the last `assert_window_s` of each stream.
        window_start_t = time.monotonic() - drain_s - assert_window_s
        window_pressure = window(self.driver.gauge_pressure_pa, window_start_t)
        window_odom = window(self.driver.odom_samples, window_start_t)

        # 6a) Pathfinding broadcast at ~10 Hz across the early portion of
        #     the mission (before self-termination).
        targets_during = window(self.driver.target_samples, mission_start_t)
        self.assertGreaterEqual(
            len(targets_during),
            50,
            f"pathfinding broadcast only {len(targets_during)} setpoints during "
            "the mission — did `start` reach pathfinding_node?",
        )
        # SURFACE setpoint is always (z=0, identity quaternion).
        first_target = targets_during[0]
        self.assertEqual(
            first_target.position.z,
            0.0,
            f"first SURFACE POSITION_TARGET.position.z = {first_target.position.z}, "
            "expected 0.0 (gauge Pa).",
        )
        self.assertAlmostEqual(
            first_target.orientation.w,
            1.0,
            delta=1e-6,
            msg="SURFACE setpoint should have identity orientation (w=1).",
        )

        # 6b) Self-termination: no POSITION_TARGET in the final
        #     `no_target_window_s` of the run. Once is_done() trips,
        #     pathfinding_node clears its mission and the 10 Hz tick
        #     becomes a no-op.
        no_target_start = time.monotonic() - drain_s - no_target_window_s
        late_targets = window(self.driver.target_samples, no_target_start)
        self.assertEqual(
            len(late_targets),
            0,
            f"expected 0 POSITION_TARGET messages in the final "
            f"{no_target_window_s}s, got {len(late_targets)}. SURFACE "
            "should have self-terminated by now (10 s surface dwell).",
        )

        # 6c) Pressure convergence. Mean gauge pressure in the assert window
        #     must be at or below the surface threshold.
        self.assertGreaterEqual(
            len(window_pressure),
            5,
            f"only {len(window_pressure)} EXTERNAL_PRESSURE samples in the "
            f"last {assert_window_s}s — bridge or sensor stalled.",
        )
        mean_gauge = sum(window_pressure) / len(window_pressure)
        self.assertLessEqual(
            mean_gauge,
            SURFACE_THRESHOLD_PA,
            msg=(
                f"mean gauge pressure over last {assert_window_s}s = "
                f"{mean_gauge:.0f} Pa, expected <= {SURFACE_THRESHOLD_PA:.0f} Pa "
                "(SURFACE_THRESHOLD_PA, ~0.8 m). Glider didn't fully surface."
            ),
        )

        # 6d) Trim: ground-truth velocities small at the surface (the BCU
        #     should converge to neutral buoyancy at z=0 and the ACU is
        #     commanded to identity attitude).
        self.assertGreaterEqual(
            len(window_odom),
            5,
            f"only {len(window_odom)} odometry samples in last "
            f"{assert_window_s}s — sim ground-truth bridge stalled.",
        )
        mean_v = sum(speed(o) for o in window_odom) / len(window_odom)
        mean_w = sum(omega(o) for o in window_odom) / len(window_odom)
        self.assertLess(
            mean_v,
            v_linear_max,
            f"mean |v_linear| over last {assert_window_s}s = {mean_v:.3f} m/s "
            f"(>= {v_linear_max} m/s). Glider hasn't settled at the surface.",
        )
        self.assertLess(
            mean_w,
            omega_max,
            f"mean |omega| over last {assert_window_s}s = {mean_w:.3f} rad/s "
            f"(>= {omega_max} rad/s). ACU/estimator combination keeps disturbing "
            "attitude.",
        )

        # 6e) Sanity: every odom pose finite + quaternion unit-norm.
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


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class SurfaceSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
