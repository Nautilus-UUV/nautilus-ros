"""Tier 3 sim test: full TRIM_AND_NEUTRAL mission, end-to-end.

Pipeline under test:

    /path + /command -> pathfinding_node -> /position/target
                                          ^
                       /external/pressure -|
    /position/target +-> depth_node       -> /bcu/rpm + /bcu/valves -> bridge -> Gazebo
    /position/target +-> acu_node         -> /acu/pitch + /acu/roll -> bridge -> Gazebo
                       /position/estimation
                          ^
                          ekf_node <- ekf_prefilter <- /imu/left

The test loads ``MissionId.TRIM_AND_NEUTRAL_BUOYANCY = 0`` with
``target_pressure_pa = 65332`` (~6.5 m of seawater) and asserts the
closed loop holds depth + comes to a stop. Spawn is at z=-5 (~5 m), so
the BCU has to descend ~1.5 m by draining the bladder. The target is
deliberately inside the depth envelope the SDF's mass + 0.0544 m^3
hull-collision balance allows the BuoyancyEngine plugin to reach with
the bladder clamp at ``rig.plant.bladder_min_m3``. Ground truth comes from
the model's already-bridged ``/model/glider_nautilus/odometry`` topic
— privileged sim-only info kept *out* of the Nautilus topic registry
so production controllers can't accidentally depend on it.

Tolerances here are deliberately looser than ``test_trim_neutral_sim_gt``
(the EKF-bypass mirror): the EKF is in the loop, and its known
orientation drift (``src/nautilus-ros/docs/ekf_node_issues.md``) feeds
the ACU. This test also acts as an EKF-stability smoke alongside
``test_ekf_pipeline_sim``. If only the |omega| assertion regresses,
add a narrow xfail there pointing at the EKF; don't blanket-skip the
test.

Composed via ``nautilus_hal/launch/trim_sim.launch.py`` (which
``IncludeLaunchDescription``s ``py_pkg/launch/control_stack.launch.py``).
Marker-gated ``@pytest.mark.sim``; opt in with
``pytest -m sim test/sim/`` after sourcing the workspace install.
``TRIM_SIM_GUI=1`` shows the Gazebo GUI.
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
    # Reap MUST happen here, not in setUpClass
    reap_lingering_gz()

    gui_enabled = os.environ.get("TRIM_SIM_GUI", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    test_scenario = os.path.join(
        os.path.dirname(__file__), "scenarios", "test_trim_neutral.yaml"
    )
    trim_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "trim_sim.launch.py",
                )
            ]
        ),
        # Test publishes the MissionCommand + start itself for deterministic
        # timing; the scenario YAML's autostart=false matches that intent
        # and also pins the target pressure for documentation parity with
        # TARGET_PRESSURE_PA below.
        launch_arguments={
            "scenario": test_scenario,
            "headless": "false" if gui_enabled else "true",
        }.items(),
    )

    return (
        LaunchDescription(
            [
                trim_sim_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _TrimNeutralTestDriver(Node):
    """Kicks the TRIM mission and captures pressure, ground-truth pose, setpoint."""

    def __init__(self):
        super().__init__("trim_neutral_sim_test_driver")
        # Capture timeseries of (monotonic_t, sample) so we can window on
        # "the last N seconds" rather than "the last N messages" (the rates
        # of the streams differ by ~10x).
        self.gauge_pressure_pa: list[tuple[float, float]] = []
        self.odom_samples: list[tuple[float, Odometry]] = []
        self.target_samples: list[tuple[float, Pose]] = []
        self.imu_msg_count: int = 0

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        # IMU_LEFT readiness signal (same convention as test_bcu_sim et al.).
        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
        # SAFETY_CRITICAL on the producer side; the factory matches QoS.
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        # Sanity: pathfinding must actually broadcast the setpoint at 10 Hz.
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)

        # Privileged sim-only ground truth: bridged out of Gazebo by
        # dave_robot_models/config/glider_nautilus/robot_config.py:16.
        # Default reliable QoS depth=10 — matches parameter_bridge defaults.
        self.create_subscription(Odometry, GROUND_TRUTH_TOPIC, self._on_odom, 10)

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_pressure(self, msg: Int32) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the controller works in gauge.
        self.gauge_pressure_pa.append(
            (time.monotonic(), gauge_pressure_pa(float(msg.data)))
        )

    def _on_target(self, msg: Pose) -> None:
        self.target_samples.append((time.monotonic(), msg))

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_samples.append((time.monotonic(), msg))

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
class TrimNeutralSimTest(unittest.TestCase):
    """Behavior: TRIM mission holds depth + the glider comes to rest."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _TrimNeutralTestDriver()
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
        # Mission timeline: spawn is at z=-5 (~51 kPa), target is ~65 kPa
        # (6.5 m). 120 s matches test_trim_neutral_sim_gt — see the
        # rationale there (saturated drain + bladder swing + momentum
        # bleed budget for the BCU plant).
        mission_duration_s = 120.0
        drain_s = 2.0
        # Last 5 s used for the convergence assertions
        assert_window_s = 5.0

        # Looser than test_trim_neutral_sim_gt — the EKF is in the loop
        # here, so attitude noise drives extra ACU activity that this
        # test has to absorb. Tighten as the EKF stabilises.
        pressure_tol_pa = 4000.0  # ~0.4 m
        v_linear_max = 0.10  # m/s
        omega_max = 0.15  # rad/s

        # 1) Wait for sim. IMU_LEFT is the readiness signal.
        sim_ready = self._spin_until(
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU_LEFT never arrived within {startup_timeout_s}s — "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        # 2) Let subscriptions handshake (PATH/COMMAND are TRANSIENT_LOCAL,
        #    but pathfinding must be up first or pose ingress lags).
        self._spin_for(post_ready_settle_s)

        # 3) Kick the mission. Two publishes; the start handler in
        #    pathfinding requires at least one EXTERNAL_PRESSURE message
        #    first, which the bridge has been delivering since IMU came up.
        self.driver.publish_mission(TARGET_PRESSURE_PA)
        # Tiny gap so PATH lands before COMMAND (both reliable, but the
        # state machine flips LOADED -> RUNNING only on `start` + a loaded
        # mission).
        self._spin_for(0.5)
        self.driver.publish_start()

        # 4) Run the closed loop.
        mission_start_t = time.monotonic()
        self._spin_for(mission_duration_s)
        self._spin_for(drain_s)

        # 5) Take the last `assert_window_s` of each stream.
        window_start_t = time.monotonic() - drain_s - assert_window_s
        window_pressure = _window(self.driver.gauge_pressure_pa, window_start_t)
        window_odom = _window(self.driver.odom_samples, window_start_t)

        # 6a) Setpoint actually broadcast at ~10 Hz across the full mission.
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

        # 6b) Pressure convergence. Mean over the window, not the last
        #     sample, to absorb 10 Hz quantization on EXTERNAL_PRESSURE.
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

        # 6c) Trim: ground-truth linear + angular velocity both small.
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
            f"(>= {omega_max} rad/s). ACU/EKF combination keeps disturbing "
            "attitude — see docs/ekf_node_issues.md.",
        )

        # 6d) Sanity: every odom pose is finite + quaternion unit-norm.
        #     Same defensive check as test_ekf_pipeline_sim.py.
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
class TrimNeutralSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
