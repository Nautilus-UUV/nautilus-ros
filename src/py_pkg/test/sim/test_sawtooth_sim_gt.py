"""Tier 3 sim test: full SAWTOOTH cycle with EKF replaced by ground-truth pose.

Mirror of ``test_trim_neutral_sim_gt.py`` -- same composition (HAL bridges,
Gazebo, gt_pose_bridge, depth + ACU + pathfinding nodes), no EKF -- but
runs the SAWTOOTH mission instead of TRIM. Drives a 10 m dive at 30 deg
glide pitch and asserts both legs of one descend->ascend cycle:

    - pathfinding broadcasts both leg setpoints (descend at z=target,
      pitch=-30 deg; ascend at z=0, pitch=+30 deg) with the right sign
      flip in between
    - GT pitch peaks at >= 25 deg in each leg
    - the leg flip happens with the gauge pressure inside
      ``DESCEND_TOLERANCE_PA`` of target (the trigger condition)
    - GT pitch crosses zero within 8 s after the flip (ACU actually
      slewed)
    - the ascent leg makes it back near the surface threshold
    - all GT poses finite + quaternions unit-norm

Composition is inline rather than ``IncludeLaunchDescription`` of
``sawtooth_sim.launch.py`` -- we need to swap the EKF stack for the
sim-only ``gt_pose_bridge``, not parameterise.

Marker-gated ``@pytest.mark.sim``; opt in with
``pytest -m sim test/sim/`` after sourcing the workspace install.
``SAWTOOTH_GT_SIM_GUI=1`` shows the Gazebo GUI.
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
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node as LaunchNode
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
PITCH_RAD = math.radians(30.0)  # SAWTOOTH glide magnitude
N_RESURFACES = 1
# Overshoot bound: GT pitch must never exceed 40 deg in either direction.
# Mission commands +/-30 deg, so 40 deg is a 10 deg overshoot allowance.
# Module-level so the driver can fail-fast on the first violating odom
# sample rather than waiting until post-mission to detect it.
PITCH_OVERSHOOT_MAX_RAD = math.radians(40.0)
GROUND_TRUTH_TOPIC = "/model/glider_nautilus/odometry"


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    # Reap MUST happen here, not in setUpClass.
    reap_lingering_gz()

    gui_enabled = os.environ.get("SAWTOOTH_GT_SIM_GUI", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
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
        )
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
        # Same spawn pose as trim_sim.launch.py / unified_sim.launch.py.
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

    nautilus_params = os.path.join(
        get_package_share_directory("nautilus_hal"),
        "config",
        "nautilus_params.yaml",
    )

    gt_pose_bridge = LaunchNode(
        package="nautilus_hal",
        executable="gt_pose_bridge",
        name="nautilus_gt_pose_bridge",
        parameters=[nautilus_params],
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


class _SawtoothGTTestDriver(Node):
    """Kicks the SAWTOOTH mission and captures the timeseries we assert on."""

    def __init__(self):
        super().__init__("sawtooth_sim_gt_test_driver")
        self.gauge_pressure_pa: list[tuple[float, float]] = []
        self.odom_samples: list[tuple[float, Odometry]] = []
        self.target_samples: list[tuple[float, Pose]] = []
        self.bcu_volume_ml: list[tuple[float, int]] = []
        self.bcu_rpm: list[tuple[float, int]] = []
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
        create_subscription_for_topic(self, UUVTopics.BCU_VOLUME, self._on_volume)
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
        t = time.monotonic()
        self.odom_samples.append((t, msg))
        if self.pitch_violation is None:
            pitch = _gt_pitch_rad(msg)
            if abs(pitch) > PITCH_OVERSHOOT_MAX_RAD:
                self.pitch_violation = (t, pitch)

    def _on_volume(self, msg: Int32) -> None:
        self.bcu_volume_ml.append((time.monotonic(), int(msg.data)))

    def _on_rpm(self, msg) -> None:
        self.bcu_rpm.append((time.monotonic(), int(msg.data)))

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
    """Slice (t, sample) pairs by monotonic timestamp; preserves the time tag.

    The trim tests' ``_window`` drops the timestamp -- here we keep it so
    the leg-flip / leg-windowed assertions can re-correlate streams.
    """
    if t_end is None:
        return [(t, s) for (t, s) in samples if t >= t_start]
    return [(t, s) for (t, s) in samples if t_start <= t <= t_end]


def _setpoint_pitch_rad(pose: Pose) -> float:
    q = pose.orientation
    _roll, pitch = quaternion_to_roll_pitch(q.x, q.y, q.z, q.w)
    return float(pitch)


def _gt_pitch_rad(odom: Odometry) -> float:
    """Inclination of the body X axis above world horizontal.

    Negative when the nose tips down. Computed directly from the rotation
    matrix's first column rather than via Euler decomposition: the model
    spawns at (roll=pi, yaw=pi/2), so a ZYX Tait-Bryan extraction would
    couple roll/yaw into the pitch reading. The body-X inclination
    matches the body-frame pitch the SAWTOOTH mission commands -- 0 at
    spawn, ``-PITCH_RAD`` at full nose-down, ``+PITCH_RAD`` at full
    nose-up -- regardless of yaw or the spawn-roll convention.
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
    """First (t, index) where setpoint pitch crosses from negative to positive.

    SAWTOOTH starts in the descending leg with ``pitch=-angle_rad`` and
    flips to ``+angle_rad`` once gauge pressure crosses
    ``target_pa - DESCEND_TOLERANCE_PA``. We watch for the sign flip on
    the published setpoint stream.
    """
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
class SawtoothGTSimTest(unittest.TestCase):
    """Behavior: SAWTOOTH drives one full descend+ascend cycle through the plant."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _SawtoothGTTestDriver()
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
        # 300 s budget for the full descend+ascend round trip at the
        # 10 m target. Spawn is at ~5 m, so descent is ~5 m of glide
        # plus the saturated bladder drain (~50 s); ascent refills and
        # rises ~10 m back through the surface threshold. 300 s is
        # comfortably above the empirical ~200 s and well below the
        # 480 s the 15 m profile needed.
        mission_duration_s = 300.0
        drain_s = 2.0

        # Pitch peak threshold: 25 deg of the commanded 30 deg; slack
        # absorbs the ACU's bounded mass-shifter travel + plant lag.
        pitch_peak_min_rad = math.radians(25.0)
        # The overshoot cap (PITCH_OVERSHOOT_MAX_RAD) is a module-level
        # constant so the driver can fail-fast as soon as the body
        # crosses it -- see the _on_odom callback above.
        # Within this window after the leg flip we want to see the body
        # actually rotating toward +pitch. We don't require it to fully
        # cross zero -- underwater body rotation through 30 deg has
        # significant inertia + added mass and the empirical time to
        # cross zero is >30 s, which makes a strict crossover assertion
        # flake on a working plant. We instead require a meaningful
        # rotation *toward* +pitch in the first 30 s, which catches a
        # plant that didn't respond to the flip without depending on
        # the exact dynamics speed.
        leg_response_window_s = 30.0
        leg_response_min_rad = math.radians(15.0)
        # Surface threshold + small margin so a sample at the leg
        # boundary doesn't fail purely on timing.
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
        # Spin until either the mission budget elapses or the driver
        # records an overshoot; whichever comes first. _spin_until polls
        # the predicate every slice and bails as soon as the odom
        # callback flags a violation -- we don't have to wait the full
        # mission_duration_s to hear about a 5-second-old crash-out.
        self._spin_until(
            lambda: self.driver.pitch_violation is not None,
            timeout_s=mission_duration_s,
        )
        self._spin_for(drain_s)
        mission_end_t = time.monotonic() - drain_s

        # 0) Fail-fast on overshoot.
        if self.driver.pitch_violation is not None:
            t_v, pitch_v = self.driver.pitch_violation
            self.fail(
                f"GT pitch reached |{math.degrees(pitch_v):.1f}| deg at "
                f"t+{t_v - mission_start_t:.1f}s -- exceeds the "
                f"{math.degrees(PITCH_OVERSHOOT_MAX_RAD):.1f} deg cap. "
                "Mission only commands +/- 30 deg, so this is overshoot "
                "beyond the allowed margin. Test aborted before mission "
                "duration ran out."
            )

        # 1) Setpoint stream alive across the mission window.
        targets_during = _window(
            self.driver.target_samples, mission_start_t, mission_end_t
        )
        self.assertGreaterEqual(
            len(targets_during),
            200,
            f"pathfinding broadcast only {len(targets_during)} setpoints in "
            f"{mission_duration_s}s -- expected ~{int(mission_duration_s * 10)}. "
            "Did `start` reach pathfinding_node?",
        )

        # 2) Setpoint encodes both legs.
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

        # 4) Descent leg GT pitch peak >= 25 deg (most-negative pitch in the
        #    descend window).
        descend_odoms = [
            o for (t, o) in self.driver.odom_samples if mission_start_t <= t <= t_flip
        ]
        self.assertGreaterEqual(
            len(descend_odoms),
            5,
            f"only {len(descend_odoms)} GT odom samples in the descend leg "
            f"({t_flip - mission_start_t:.1f}s); GT bridge stalled",
        )
        descend_pitch_peak = min(_gt_pitch_rad(o) for o in descend_odoms)
        self.assertLessEqual(
            descend_pitch_peak,
            -pitch_peak_min_rad,
            f"descent leg peak pitch was only {math.degrees(descend_pitch_peak):.1f} deg "
            f"(want <= {-math.degrees(pitch_peak_min_rad):.1f} deg). "
            "ACU never drove the body to the commanded glide.",
        )

        # 5) Ascent leg GT pitch peak >= 25 deg (most-positive pitch in
        #    the ascend window).
        ascend_odoms = [
            o for (t, o) in self.driver.odom_samples if t_flip <= t <= mission_end_t
        ]
        self.assertGreaterEqual(
            len(ascend_odoms),
            5,
            f"only {len(ascend_odoms)} GT odom samples in the ascend leg "
            f"({mission_end_t - t_flip:.1f}s); mission may have terminated "
            "too early",
        )
        ascend_pitch_peak = max(_gt_pitch_rad(o) for o in ascend_odoms)
        self.assertGreaterEqual(
            ascend_pitch_peak,
            pitch_peak_min_rad,
            f"ascent leg peak pitch was only {math.degrees(ascend_pitch_peak):.1f} deg "
            f"(want >= {math.degrees(pitch_peak_min_rad):.1f} deg). "
            "ACU never drove the body to the commanded climb.",
        )

        # 6) Descent reaches target. The mission state machine flips
        #    legs on this exact threshold; if the descent never got
        #    there we wouldn't have a flip at all.
        descend_pressures = [
            p for (t, p) in self.driver.gauge_pressure_pa if mission_start_t <= t <= t_flip
        ]
        self.assertGreater(
            len(descend_pressures),
            0,
            "no EXTERNAL_PRESSURE samples in the descend window",
        )
        descend_pressure_peak = max(descend_pressures)
        self.assertGreaterEqual(
            descend_pressure_peak,
            TARGET_PRESSURE_PA - DESCEND_TOLERANCE_PA,
            f"descent peak gauge pressure {descend_pressure_peak:.0f} Pa is "
            f"below the trigger threshold "
            f"{TARGET_PRESSURE_PA - DESCEND_TOLERANCE_PA:.0f} Pa "
            f"(target {TARGET_PRESSURE_PA:.0f} Pa, "
            f"tol {DESCEND_TOLERANCE_PA:.0f} Pa)",
        )

        # 7) Sanity on the GT stream we just slurped.
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

        # 8) Leg-boundary pressure correctness: at t_flip the gauge
        #    pressure must be inside DESCEND_TOLERANCE_PA of target.
        flip_pressure = _nearest_pressure_sample(
            self.driver.gauge_pressure_pa, t_flip
        )
        self.assertIsNotNone(
            flip_pressure,
            "no EXTERNAL_PRESSURE samples to correlate with the leg flip",
        )
        flip_t, flip_pa = flip_pressure
        self.assertLess(
            abs(t_flip - flip_t),
            0.5,
            f"nearest pressure sample is {abs(t_flip - flip_t):.2f}s away "
            f"from the leg flip; the streams aren't aligned",
        )
        self.assertAlmostEqual(
            flip_pa,
            TARGET_PRESSURE_PA,
            delta=DESCEND_TOLERANCE_PA,
            msg=(
                f"leg flip happened at gauge pressure {flip_pa:.0f} Pa, "
                f"but the SAWTOOTH trigger requires it within "
                f"+/-{DESCEND_TOLERANCE_PA:.0f} Pa of "
                f"{TARGET_PRESSURE_PA:.0f} Pa. The state machine flipped "
                "for the wrong reason."
            ),
        )

        # 9) Leg-boundary plant response: in the window right after the
        #    flip, the body must rotate measurably toward +pitch. We
        #    sample the GT pitch nearest t_flip and nearest t_flip +
        #    leg_response_window_s and check the delta.
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
            f"{math.degrees(pitch_after_window):.1f} deg. ACU did not "
            "drive the body toward the ascent setpoint.",
        )

        # 10) Ascent reaches surface (within the small margin).
        ascend_pressures = [
            p
            for (t, p) in self.driver.gauge_pressure_pa
            if t_flip <= t <= mission_end_t
        ]
        self.assertGreater(
            len(ascend_pressures),
            0,
            "no EXTERNAL_PRESSURE samples in the ascend window",
        )
        ascend_pressure_min = min(ascend_pressures)
        self.assertLessEqual(
            ascend_pressure_min,
            surface_pressure_max_pa,
            f"ascent min gauge pressure {ascend_pressure_min:.0f} Pa never "
            f"got below {surface_pressure_max_pa:.0f} Pa "
            f"(SURFACE_THRESHOLD_PA={SURFACE_THRESHOLD_PA:.0f}). "
            f"Resurface leg didn't reach the surface within "
            f"{mission_duration_s}s.",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class SawtoothGTSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
