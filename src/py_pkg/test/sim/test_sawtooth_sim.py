"""Tier 3 sim test: full SAWTOOTH cycle, bang-bang ACU pitch endpoints.

Pipeline under test (mission state machine + cascaded controllers + EKF):

    /path + /command -> pathfinding_node -> /position/target
                                          ^
                       /external/pressure -|
    /position/target +-> depth_node       -> /bcu/rpm + /bcu/valves -> bridge -> Gazebo
    /position/target +-> acu_node         -> /acu/pitch + /acu/roll -> bridge -> Gazebo
                       /position/estimation
                          ^
                          ekf_node <- ekf_prefilter <- /imu/left

The ACU pitch axis is now a bang-bang controller on
``EXTERNAL_PRESSURE`` vs the pressure setpoint pathfinding stamps onto
``POSITION_TARGET.position.z`` -- it picks one of two extremes
(``ACU_PITCH_OUTPUT_LIMITS_M``, on the wire as Int16 mm) and sticks
there until the sign of the pressure error flips. So we don't measure a
body-frame pitch angle here; we just check that the right extreme
shows up on ``ACU_PITCH`` during each leg:

    descend leg (current_pa < target_pa)  -> back-stroke endpoint (mass aft)
    ascend  leg (current_pa > target=0)   -> front-stroke endpoint (mass forward)

That's a strict regression check on the wiring + state machine without
relying on the EKF pose, which means the EKF orientation drift
(``docs/ekf_node_issues.md``) doesn't flake this test the way it does
the GT-mirror tests for the closed-loop pitch attitude.

Composed via ``nautilus_hal/launch/sawtooth_sim.launch.py`` (which
``IncludeLaunchDescription``s ``py_pkg/launch/control_stack.launch.py``).
Marker-gated ``@pytest.mark.sim``; opt in with
``pytest -m sim test/sim/`` after sourcing the workspace install.
``SAWTOOTH_SIM_GUI=1`` shows the Gazebo GUI.
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
from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.path.missions.factory import MissionId
from py_pkg.robot_specs import ACU_PITCH_OUTPUT_LIMITS_M
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, String

from ._sim_helpers import reap_lingering_gz

TARGET_PRESSURE_PA = 73575.0  # ~7.5 m of seawater (gauge); spawn is ~5 m
PITCH_RAD = math.radians(30.0)  # SAWTOOTH glide magnitude
N_RESURFACES = 1

# Bang-bang ACU pitch endpoints on the wire (Int16 mm). Same derivation
# as in ``pid/acu_node.py``: the soft-saturation tuple is ordered
# (front, back) with "front" the most-negative end of stroke.
ACU_PITCH_FRONT_MM = int(round(ACU_PITCH_OUTPUT_LIMITS_M[0] * 1000.0))
ACU_PITCH_BACK_MM = int(round(ACU_PITCH_OUTPUT_LIMITS_M[1] * 1000.0))


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    reap_lingering_gz()

    gui_enabled = os.environ.get("SAWTOOTH_SIM_GUI", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    sawtooth_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("nautilus_hal").find("nautilus_hal"),
                    "launch",
                    "sawtooth_sim.launch.py",
                )
            ]
        ),
        # Test publishes the MissionCommand + start itself for deterministic
        # timing; the launch's autostart path is the CLI-only convenience.
        launch_arguments={
            "target_pressure_pa": str(TARGET_PRESSURE_PA),
            "angle_rad": str(PITCH_RAD),
            "n_resurfaces": str(N_RESURFACES),
            "headless": "false" if gui_enabled else "true",
            "mission_autostart": "false",
        }.items(),
    )

    return (
        LaunchDescription(
            [
                sawtooth_sim_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _SawtoothTestDriver(Node):
    """Kicks the SAWTOOTH mission and captures the streams we assert on."""

    def __init__(self):
        super().__init__("sawtooth_sim_test_driver")
        self.target_samples: list[tuple[float, Pose]] = []
        self.acu_pitch_samples: list[tuple[float, int]] = []
        self.imu_msg_count: int = 0

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)
        create_subscription_for_topic(self, UUVTopics.ACU_PITCH, self._on_acu_pitch)

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_target(self, msg: Pose) -> None:
        self.target_samples.append((time.monotonic(), msg))

    def _on_acu_pitch(self, msg: Int16) -> None:
        self.acu_pitch_samples.append((time.monotonic(), int(msg.data)))

    def publish_mission(self) -> None:
        cmd = MissionCommand()
        cmd.mission_id = int(MissionId.SAWTOOTH)
        cmd.target_pressure_pa = float(TARGET_PRESSURE_PA)
        cmd.angle_rad = float(PITCH_RAD)
        cmd.n_resurfaces = int(N_RESURFACES)
        self.path_pub.publish(cmd)

    def publish_start(self) -> None:
        msg = String()
        msg.data = "start"
        self.command_pub.publish(msg)


def _setpoint_pitch_rad(pose: Pose) -> float:
    q = pose.orientation
    _roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
    return float(pitch)


def _find_descend_to_ascend_flip(
    target_samples: list[tuple[float, Pose]],
) -> tuple[float, int] | None:
    """First (t, index) where setpoint pitch crosses from negative to positive.

    SAWTOOTH starts in the descending leg with ``pitch=-angle_rad`` and
    flips to ``+angle_rad`` once gauge pressure crosses
    ``target_pa - DESCEND_TOLERANCE_PA``. We watch for the sign flip on
    the published setpoint stream so the leg windows track the
    pathfinding state machine exactly, regardless of plant lag.
    """
    seen_negative = False
    for i, (t, pose) in enumerate(target_samples):
        pitch = _setpoint_pitch_rad(pose)
        if pitch < -1e-3:
            seen_negative = True
        elif pitch > 1e-3 and seen_negative:
            return (t, i)
    return None


@pytest.mark.sim
class SawtoothSimTest(unittest.TestCase):
    """Behavior: SAWTOOTH drives one full descend+ascend cycle and the
    ACU bang-bang pitch hits both stroke endpoints in the right legs."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _SawtoothTestDriver()
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

    def test_sawtooth_pitch_endpoints_per_leg(self):
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # 300 s budget for the full descend+ascend round trip at the
        # 7.5 m target. Spawn is at ~5 m, so descent is only ~2.5 m of
        # glide plus the saturated bladder drain; ascent then refills
        # and rises ~7.5 m back through the surface threshold. Budget
        # is unchanged from the 10 m profile -- no need to tighten it,
        # the shorter descent just buys extra margin.
        mission_duration_s = 300.0
        drain_s = 2.0

        sim_ready = self._spin_until(
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU_LEFT never arrived within {startup_timeout_s}s -- "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        self._spin_for(post_ready_settle_s)

        self.driver.publish_mission()
        self._spin_for(0.5)
        self.driver.publish_start()

        mission_start_t = time.monotonic()
        self._spin_for(mission_duration_s)
        self._spin_for(drain_s)
        mission_end_t = time.monotonic() - drain_s

        # 1) Setpoint stream alive across the mission window. ~10 Hz
        #    publish, so 300 s should give ~3000 samples; 200 is a
        #    crash-detector floor, not a rate check.
        targets_during = [
            (t, p)
            for (t, p) in self.driver.target_samples
            if mission_start_t <= t <= mission_end_t
        ]
        self.assertGreaterEqual(
            len(targets_during),
            200,
            f"pathfinding broadcast only {len(targets_during)} setpoints in "
            f"{mission_duration_s}s -- expected ~{int(mission_duration_s * 10)}. "
            "Did `start` reach pathfinding_node?",
        )

        # 2) Both legs encoded on the setpoint stream.
        z_values = {round(p.position.z, 3) for (_t, p) in targets_during}
        self.assertIn(
            float(TARGET_PRESSURE_PA),
            z_values,
            f"never saw a descend setpoint with position.z = "
            f"{TARGET_PRESSURE_PA} Pa; SAWTOOTH state machine never started",
        )
        self.assertIn(
            0.0,
            z_values,
            "never saw an ascend setpoint with position.z = 0.0; SAWTOOTH "
            "never flipped to the surfacing leg within the mission budget",
        )

        # 3) Leg flip detected on the setpoint pitch stream.
        flip = _find_descend_to_ascend_flip(targets_during)
        self.assertIsNotNone(
            flip,
            "no descend->ascend pitch sign flip seen in POSITION_TARGET; "
            "either pathfinding stayed in the descend leg or the orientation "
            "encoding regressed",
        )
        t_flip, _flip_idx = flip

        # 4) ACU pitch endpoint reached during the descend leg. The
        #    bang-bang controller picks ``ACU_PITCH_BACK_MM`` (mass aft,
        #    nose tips down) whenever the current pressure is below the
        #    setpoint pressure -- which is the entire descend leg, so a
        #    healthy controller emits this value many times. Only one
        #    matching sample is required to pass.
        descend_pitch_cmds = [
            v
            for (t, v) in self.driver.acu_pitch_samples
            if mission_start_t <= t <= t_flip
        ]
        self.assertGreaterEqual(
            len(descend_pitch_cmds),
            5,
            f"only {len(descend_pitch_cmds)} ACU_PITCH samples in the "
            f"descend leg ({t_flip - mission_start_t:.1f}s); ACU control "
            "node may not be alive or its publish QoS regressed.",
        )
        self.assertIn(
            ACU_PITCH_BACK_MM,
            descend_pitch_cmds,
            f"ACU never commanded the back-stroke endpoint "
            f"({ACU_PITCH_BACK_MM} mm) during the descend leg. "
            f"Saw values: {sorted(set(descend_pitch_cmds))}. "
            "Bang-bang sign convention or pressure ingress regression.",
        )

        # 5) ACU pitch endpoint reached during the ascend leg. After
        #    the leg flip the setpoint pressure drops to 0 Pa, so the
        #    current pressure is above target and the bang-bang
        #    controller picks ``ACU_PITCH_FRONT_MM`` (mass forward,
        #    nose tips up).
        ascend_pitch_cmds = [
            v
            for (t, v) in self.driver.acu_pitch_samples
            if t_flip <= t <= mission_end_t
        ]
        self.assertGreaterEqual(
            len(ascend_pitch_cmds),
            5,
            f"only {len(ascend_pitch_cmds)} ACU_PITCH samples in the "
            f"ascend leg ({mission_end_t - t_flip:.1f}s); mission may have "
            "terminated unexpectedly early.",
        )
        self.assertIn(
            ACU_PITCH_FRONT_MM,
            ascend_pitch_cmds,
            f"ACU never commanded the front-stroke endpoint "
            f"({ACU_PITCH_FRONT_MM} mm) during the ascend leg. "
            f"Saw values: {sorted(set(ascend_pitch_cmds))}. "
            "Bang-bang sign convention regression.",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class SawtoothSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
