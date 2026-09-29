"""Tier 3 sim test: full TRIM_AND_NEUTRAL mission, end-to-end.

Pipeline under test:

    /path + /command -> pathfinding_node -> /position/target
                                          ^
                       /external/pressure -|
    /position/target +-> bcu_node       -> /bcu/rpm + /bcu/valves -> bridge -> Gazebo
    /position/target +-> acu_node         -> /acu/pitch + /acu/roll -> bridge -> Gazebo
                       /position/estimation
                          ^
                          attitude_node <- imu_prefilter <- /imu/left

The test loads ``MissionId.TRIM_AND_NEUTRAL_BUOYANCY = 0`` with
``target_pressure_pa = 65332`` (~6.66 m of lake water at the sensor's
9806 Pa/m gradient). Spawn is at z=-5 (~5 m), so the BCU has to descend
~1.7 m by draining the bladder below the lake-calibrated neutral fill.

**What this asserts, and why it is not "holds at target".** Under the
bang-bang BCU there is no neutral-buoyancy hold to converge to: the
bladder sits at a tank rail for the whole descent, so the vehicle coasts
through the target and keeps sinking after the run ends. Arrival IS the
completion condition (see ``path/missions/trim_and_neutral.py``). The PID
-era version of this test asserted a terminal mean pressure at the target
and near-zero ground-truth velocity; both are unbuildable against a
controller with no idle equilibrium, and asserting them measured the
plant's momentum rather than the mission. What survives is the contract
the mission actually makes:

  1. the vehicle enters the ``NEAR_GOAL_PA`` band around the target
     (closest approach, not a terminal mean);
  2. ``/mission/complete`` latches, i.e. the mission self-terminates;
  3. ``POSITION_TARGET`` falls silent afterwards (pathfinding reset).

Ground truth still comes from the model's already-bridged
``/model/glider_nautilus/odometry`` topic — privileged sim-only info kept
*out* of the Nautilus topic registry so production controllers can't
accidentally depend on it — but it is now only used for the finite/unit-norm
sanity sweep, not for a rest assertion.

Composed via ``nautilus_hal/launch/trim_sim.launch.py`` (which
``IncludeLaunchDescription``s ``py_pkg/launch/control_stack.launch.py``).
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
from nav_msgs.msg import Odometry
from py_pkg.path.missions.factory import MissionId
from py_pkg.path.missions.trim_and_neutral import NEAR_GOAL_PA
from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M, gauge_pressure_pa
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
    mission_command,
    reap_lingering_gz,
    sim_gui_enabled,
    spin_for,
    spin_until,
    window,
)

TARGET_PRESSURE_PA = 65332.0  # ~6.66 m of lake water (gauge); spawn is ~5 m
GROUND_TRUTH_TOPIC = "/model/glider_nautilus/odometry"


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    # Reap MUST happen here, not in setUpClass
    reap_lingering_gz()

    gui_enabled = sim_gui_enabled()

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
        self.mission_complete: bool = False

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        # IMU readiness signal (same convention as test_bcu_sim et al.).
        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        # SAFETY_CRITICAL on the producer side; the factory matches QoS.
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        # Sanity: pathfinding must actually broadcast the setpoint at 10 Hz.
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)
        # Arrival latches this. Under bang-bang it IS the mission's terminal
        # condition, so it's both the wait signal and an assertion target.
        create_subscription_for_topic(
            self, UUVTopics.MISSION_COMPLETE, self._on_mission_complete
        )

        # Privileged sim-only ground truth: bridged out of Gazebo by
        # dave_robot_models/config/glider_nautilus/robot_config.py:16.
        # Default reliable QoS depth=10 — matches parameter_bridge defaults.
        self.create_subscription(Odometry, GROUND_TRUTH_TOPIC, self._on_odom, 10)

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_mission_complete(self, msg: Bool) -> None:
        self.mission_complete = self.mission_complete or bool(msg.data)

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
        # TRIM reads only the target; everything else stays at the msg default.
        self.path_pub.publish(
            mission_command(
                MissionId.TRIM_AND_NEUTRAL_BUOYANCY, target_pressure_pa=target_pa
            )
        )

    def publish_start(self) -> None:
        msg = Bool()
        msg.data = True
        self.command_pub.publish(msg)


@pytest.mark.sim
class TrimNeutralSimTest(unittest.TestCase):
    """Behavior: TRIM mission reaches the target band and self-terminates."""

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

    def test_trim_and_neutral_buoyancy_reaches_target_and_completes(self):
        """TRIM mission -> vehicle reaches the target band, mission self-ends."""
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # Spawn is at z=-5 (~51 kPa), target ~65 kPa (6.66 m): ~1.7 m of
        # saturated drain. Only an UPPER BOUND -- the run ends on
        # /mission/complete, so a healthy descent costs its real duration.
        mission_budget_s = 120.0
        drain_s = 2.0
        # Silence window after completion: pathfinding resets on is_done, so
        # POSITION_TARGET must stop. Sized well above the 10 Hz setpoint period.
        quiet_tail_s = 5.0

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

        # 2) Let subscriptions handshake (PATH/COMMAND are TRANSIENT_LOCAL,
        #    but pathfinding must be up first or pose ingress lags).
        spin_for(self.executor, post_ready_settle_s)

        # 3) Kick the mission. Two publishes; the start handler in
        #    pathfinding requires at least one EXTERNAL_PRESSURE message
        #    first, which the bridge has been delivering since IMU came up.
        self.driver.publish_mission(TARGET_PRESSURE_PA)
        # Tiny gap so PATH lands before COMMAND (both reliable, but the
        # state machine flips LOADED -> RUNNING only on `start` + a loaded
        # mission).
        spin_for(self.executor, 0.5)
        self.driver.publish_start()

        # 4) Run the closed loop until the mission ends on arrival.
        mission_start_t = time.monotonic()
        completed = spin_until(
            self.executor,
            lambda: self.driver.mission_complete,
            timeout_s=mission_budget_s,
        )
        mission_complete_t = time.monotonic()
        self.assertTrue(
            completed,
            f"TRIM never latched /mission/complete within {mission_budget_s}s. "
            "Under bang-bang the mission ends on ARRIVAL (within NEAR_GOAL_PA of "
            "the target), so this means the BCU never drove the vehicle into the "
            "band -- check the drain direction and the tank-limit guard.",
        )
        # Hold the window open so the POSITION_TARGET silence below is real.
        spin_for(self.executor, quiet_tail_s + drain_s)

        # 5) Streams over the mission proper (start -> completion).
        window_odom = window(self.driver.odom_samples, mission_start_t)

        # 6a) Setpoint actually broadcast at ~10 Hz while the mission ran.
        targets_during = window(self.driver.target_samples, mission_start_t)
        mission_elapsed_s = mission_complete_t - mission_start_t
        self.assertGreaterEqual(
            len(targets_during),
            100,
            f"pathfinding broadcast only {len(targets_during)} setpoints in "
            f"{mission_elapsed_s:.1f}s — expected ~{int(mission_elapsed_s * 10)}. "
            "Did `start` reach pathfinding_node?",
        )
        last_target = self.driver.target_samples[-1][1]
        self.assertEqual(
            last_target.position.z,
            TARGET_PRESSURE_PA,
            f"latest POSITION_TARGET.position.z = {last_target.position.z} Pa, "
            f"expected {TARGET_PRESSURE_PA} Pa (TRIM mission's arrival depth)",
        )

        # 6b) Arrival: the vehicle actually entered the target band at some
        #     point. This is the depth claim the bang-bang plant can make --
        #     it cannot HOLD the target (see the module docstring), so the
        #     assertion is on the closest approach, not on a terminal mean.
        gauges = window(self.driver.gauge_pressure_pa, mission_start_t)
        self.assertGreaterEqual(
            len(gauges),
            5,
            f"only {len(gauges)} EXTERNAL_PRESSURE samples during the mission "
            "— bridge or sensor stalled.",
        )
        closest_pa = min(abs(pa - TARGET_PRESSURE_PA) for pa in gauges)
        self.assertLessEqual(
            closest_pa,
            NEAR_GOAL_PA,
            f"closest approach to target was {closest_pa:.0f} Pa "
            f"({closest_pa / WATER_PRESSURE_GRADIENT_PA_PER_M:.2f} m) — never "
            f"entered the ±{NEAR_GOAL_PA:.0f} Pa arrival band around "
            f"{TARGET_PRESSURE_PA:.0f} Pa. The mission's own is_done uses this "
            "band, so completion without arrival would be a profile bug.",
        )

        # 6c) Self-termination: POSITION_TARGET falls silent after completion.
        #     pathfinding resets on is_done; bcu_node/acu_node then safe-stop.
        #     This replaces the old "glider comes to rest" check -- with the
        #     bladder parked at a tank rail the vehicle keeps coasting, so
        #     quiescence is a property of the SETPOINT stream, not the plant.
        late_targets = window(self.driver.target_samples, mission_complete_t + 1.0)
        self.assertEqual(
            len(late_targets),
            0,
            f"pathfinding published {len(late_targets)} setpoints more than 1 s "
            "after /mission/complete — it did not reset on is_done.",
        )

        # 6d) Sanity: every odom pose is finite + quaternion unit-norm.
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
