"""Tier 3 sim test: full SAWTOOTH cycle, EKF in the loop.

Pipeline under test (mission state machine + cascaded controllers + EKF):

    /path + /command -> pathfinding_node -> /position/target
                                          ^
                       /external/pressure -|
    /position/target +-> depth_node       -> /bcu/rpm + /bcu/valves -> bridge -> Gazebo
    /position/target +-> acu_node         -> /acu/pitch + /acu/roll -> bridge -> Gazebo
                       /position/estimation
                          ^
                          ekf_node <- ekf_prefilter <- /imu/left

Composed via ``nautilus_hal/launch/sawtooth_sim.launch.py`` (which
``IncludeLaunchDescription``s ``py_pkg/launch/control_stack.launch.py``).

Thresholds are *identical* to ``test_sawtooth_sim_gt`` by design. The EKF
orientation update is currently non-multiplicative (see
``docs/ekf_node_issues.md``) and drifts over long runs, which feeds
attitude noise into the ACU; this test is allowed to fail until that's
fixed. Loosening the thresholds here would mask the drift the GT pair is
meant to expose, so don't -- if the GT variant passes and this one
fails, the failure is a real EKF regression and the right fix is in the
EKF, not in this test.

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
from nav_msgs.msg import Odometry
from py_pkg.math_utils import quaternion_to_roll_pitch
from py_pkg.path.missions.factory import MissionId
from py_pkg.path.missions.sawtooth import (
    DESCEND_TOLERANCE_PA,
    SURFACE_THRESHOLD_PA,
)
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

TARGET_PRESSURE_PA = 98100.0  # ~10 m of seawater (gauge); spawn is ~5 m
PITCH_RAD = math.radians(30.0)
N_RESURFACES = 1
# Identical to the GT variant; module-level so the driver can fail-fast
# on the first violating odom sample.
PITCH_OVERSHOOT_MAX_RAD = math.radians(40.0)
GROUND_TRUTH_TOPIC = "/model/glider_nautilus/odometry"


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
    def __init__(self):
        super().__init__("sawtooth_sim_test_driver")
        self.gauge_pressure_pa: list[tuple[float, float]] = []
        self.odom_samples: list[tuple[float, Odometry]] = []
        self.target_samples: list[tuple[float, Pose]] = []
        self.imu_msg_count: int = 0
        # First (t, pitch_rad) sample whose magnitude exceeded
        # PITCH_OVERSHOOT_MAX_RAD; remains None if the body stayed
        # within the cap. Used by the test to fail-fast.
        self.pitch_violation: tuple[float, float] | None = None

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)
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
        t = time.monotonic()
        self.odom_samples.append((t, msg))
        if self.pitch_violation is None:
            pitch = _gt_pitch_rad(msg)
            if abs(pitch) > PITCH_OVERSHOOT_MAX_RAD:
                self.pitch_violation = (t, pitch)

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


def _window(
    samples: list[tuple[float, object]],
    t_start: float,
    t_end: float | None = None,
) -> list[tuple[float, object]]:
    if t_end is None:
        return [(t, s) for (t, s) in samples if t >= t_start]
    return [(t, s) for (t, s) in samples if t_start <= t <= t_end]


def _setpoint_pitch_rad(pose: Pose) -> float:
    q = pose.orientation
    _roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
    return float(pitch)


def _gt_pitch_rad(odom: Odometry) -> float:
    """Inclination of the body X axis above world horizontal.

    See ``test_sawtooth_sim_gt._gt_pitch_rad`` for why this avoids ZYX
    Euler decomposition: model spawns at (roll=pi, yaw=pi/2), which would
    couple roll/yaw into the Euler pitch reading.
    """
    q = odom.pose.pose.orientation
    bx_x = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    bx_y = 2.0 * (q.x * q.y + q.w * q.z)
    bx_z = 2.0 * (q.x * q.z - q.w * q.y)
    horizontal = math.sqrt(bx_x * bx_x + bx_y * bx_y)
    return math.atan2(bx_z, horizontal)


def _find_descend_to_ascend_flip(
    target_samples: list[tuple[float, Pose]],
) -> tuple[float, int] | None:
    seen_negative = False
    for i, (t, pose) in enumerate(target_samples):
        pitch = _setpoint_pitch_rad(pose)
        if pitch < -1e-3:
            seen_negative = True
        elif pitch > 1e-3 and seen_negative:
            return (t, i)
    return None


def _nearest_pressure_sample(
    pressure_samples: list[tuple[float, float]],
    t_target: float,
) -> tuple[float, float] | None:
    if not pressure_samples:
        return None
    return min(pressure_samples, key=lambda ts: abs(ts[0] - t_target))


@pytest.mark.sim
class SawtoothSimTest(unittest.TestCase):
    """Behavior: SAWTOOTH cycle through the full EKF + control stack."""

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

    def test_sawtooth_one_full_cycle(self):
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # Identical to the GT variant -- see test_sawtooth_sim_gt.py for
        # rationale on each value.
        mission_duration_s = 300.0
        drain_s = 2.0
        pitch_peak_min_rad = math.radians(25.0)
        # Overshoot cap (PITCH_OVERSHOOT_MAX_RAD) is module-level; the
        # driver's _on_odom flags the first violation and the test
        # fails-fast on it.
        leg_response_window_s = 30.0
        leg_response_min_rad = math.radians(15.0)
        surface_pressure_max_pa = SURFACE_THRESHOLD_PA + 2000.0

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
        # Bail as soon as the driver records an overshoot; otherwise
        # spin the full mission window.
        self._spin_until(
            lambda: self.driver.pitch_violation is not None,
            timeout_s=mission_duration_s,
        )
        self._spin_for(drain_s)
        mission_end_t = time.monotonic() - drain_s

        # Fail-fast on overshoot.
        if self.driver.pitch_violation is not None:
            t_v, pitch_v = self.driver.pitch_violation
            self.fail(
                f"GT pitch reached |{math.degrees(pitch_v):.1f}| deg at "
                f"t+{t_v - mission_start_t:.1f}s -- exceeds the "
                f"{math.degrees(PITCH_OVERSHOOT_MAX_RAD):.1f} deg cap. "
                "Mission only commands +/- 30 deg."
            )

        targets_during = _window(
            self.driver.target_samples, mission_start_t, mission_end_t
        )
        self.assertGreaterEqual(
            len(targets_during),
            200,
            f"pathfinding broadcast only {len(targets_during)} setpoints in "
            f"{mission_duration_s}s -- expected ~{int(mission_duration_s * 10)}",
        )

        z_values = {round(p.position.z, 3) for (_t, p) in targets_during}
        self.assertIn(
            float(TARGET_PRESSURE_PA),
            z_values,
            f"never saw a descend setpoint with position.z = "
            f"{TARGET_PRESSURE_PA} Pa",
        )
        self.assertIn(
            0.0,
            z_values,
            "never saw an ascend setpoint with position.z = 0.0",
        )

        flip = _find_descend_to_ascend_flip(targets_during)
        self.assertIsNotNone(
            flip, "no descend->ascend pitch sign flip seen in POSITION_TARGET"
        )
        t_flip, _flip_idx = flip

        descend_odoms = [
            o for (t, o) in self.driver.odom_samples if mission_start_t <= t <= t_flip
        ]
        self.assertGreaterEqual(
            len(descend_odoms),
            5,
            f"only {len(descend_odoms)} GT odom samples in the descend leg",
        )
        descend_pitch_peak = min(_gt_pitch_rad(o) for o in descend_odoms)
        self.assertLessEqual(
            descend_pitch_peak,
            -pitch_peak_min_rad,
            f"descent leg peak pitch was only {math.degrees(descend_pitch_peak):.1f} deg "
            f"(want <= {-math.degrees(pitch_peak_min_rad):.1f} deg)",
        )

        ascend_odoms = [
            o for (t, o) in self.driver.odom_samples if t_flip <= t <= mission_end_t
        ]
        self.assertGreaterEqual(
            len(ascend_odoms),
            5,
            f"only {len(ascend_odoms)} GT odom samples in the ascend leg",
        )
        ascend_pitch_peak = max(_gt_pitch_rad(o) for o in ascend_odoms)
        self.assertGreaterEqual(
            ascend_pitch_peak,
            pitch_peak_min_rad,
            f"ascent leg peak pitch was only {math.degrees(ascend_pitch_peak):.1f} deg "
            f"(want >= {math.degrees(pitch_peak_min_rad):.1f} deg)",
        )

        descend_pressures = [
            p for (t, p) in self.driver.gauge_pressure_pa if mission_start_t <= t <= t_flip
        ]
        self.assertGreater(len(descend_pressures), 0)
        descend_pressure_peak = max(descend_pressures)
        self.assertGreaterEqual(
            descend_pressure_peak,
            TARGET_PRESSURE_PA - DESCEND_TOLERANCE_PA,
            f"descent peak gauge pressure {descend_pressure_peak:.0f} Pa is "
            f"below the trigger threshold "
            f"{TARGET_PRESSURE_PA - DESCEND_TOLERANCE_PA:.0f} Pa",
        )

        for i, (_t, odom) in enumerate(_window(self.driver.odom_samples, mission_start_t)):
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

        flip_pressure = _nearest_pressure_sample(
            self.driver.gauge_pressure_pa, t_flip
        )
        self.assertIsNotNone(flip_pressure)
        flip_t, flip_pa = flip_pressure
        self.assertLess(abs(t_flip - flip_t), 0.5)
        self.assertAlmostEqual(
            flip_pa,
            TARGET_PRESSURE_PA,
            delta=DESCEND_TOLERANCE_PA,
            msg=(
                f"leg flip happened at gauge pressure {flip_pa:.0f} Pa, "
                f"but the SAWTOOTH trigger requires it within "
                f"+/-{DESCEND_TOLERANCE_PA:.0f} Pa of "
                f"{TARGET_PRESSURE_PA:.0f} Pa"
            ),
        )

        odom_at_flip = min(
            self.driver.odom_samples,
            key=lambda ts: abs(ts[0] - t_flip),
        )
        odom_after_window = min(
            self.driver.odom_samples,
            key=lambda ts: abs(ts[0] - (t_flip + leg_response_window_s)),
        )
        pitch_at_flip = _gt_pitch_rad(odom_at_flip[1])
        pitch_after_window = _gt_pitch_rad(odom_after_window[1])
        leg_response_delta = pitch_after_window - pitch_at_flip
        self.assertGreater(
            leg_response_delta,
            leg_response_min_rad,
            f"GT pitch only rotated {math.degrees(leg_response_delta):.1f} deg "
            f"toward +pitch in {leg_response_window_s}s after the leg flip "
            f"(want > {math.degrees(leg_response_min_rad):.1f} deg); "
            f"started at {math.degrees(pitch_at_flip):.1f} deg, ended at "
            f"{math.degrees(pitch_after_window):.1f} deg",
        )

        ascend_pressures = [
            p
            for (t, p) in self.driver.gauge_pressure_pa
            if t_flip <= t <= mission_end_t
        ]
        self.assertGreater(len(ascend_pressures), 0)
        ascend_pressure_min = min(ascend_pressures)
        self.assertLessEqual(
            ascend_pressure_min,
            surface_pressure_max_pa,
            f"ascent min gauge pressure {ascend_pressure_min:.0f} Pa never "
            f"got below {surface_pressure_max_pa:.0f} Pa",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class SawtoothSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        reap_lingering_gz()
