"""Tier 3 sim test: two-pressure SAWTOOTH, surfaces after N oscillations.

Exercises the new two-pressure glide: the vehicle dives to a deep pressure,
climbs to a *sub-surface* shallow pressure, repeats for ``n_oscillations``
dives, then makes a final ascent to the surface and the mission terminates.

Pipeline under test (same stack as ``test_sawtooth_sim.py``):

    /path + /command -> pathfinding_node -> /position/target
                                          ^
                       /position/estimation (gauge depth) -|
    /position/target +-> bcu_node  -> /bcu/rpm + /bcu/valves -> bridge -> Gazebo
    /position/target +-> acu_node  -> /acu/pitch + /acu/roll  -> bridge -> Gazebo

The SAWTOOTH state machine only advances on *real* pressure crossings fed back
through ``update`` -- so the published ``POSITION_TARGET.position.z`` stream is a
faithful proxy for the achieved motion. We assert on that stream (deep / shallow
/ surface setpoints) and corroborate with the measured ``EXTERNAL_PRESSURE``:

    1. the setpoint profile is exactly [deep, shallow, deep, surface] -- two
       dives between the two pressures, then a final surfacing leg;
    2. the measured EXTERNAL_PRESSURE confirms the oscillations happen *at the
       targeted depths* -- the gauge reaches the deep target band twice, climbs
       back to the shallow target band between dives, and crosses the surface
       threshold (it resurfaces);
    3. the mission terminated -- POSITION_TARGET goes quiet after the surfacing
       leg (pathfinding resets on is_done), with the last setpoint at z = 0.

Two distinct sub-surface pressures (~3 m / ~6 m) bracketing the ~5 m spawn keep
the dives short so the test terminates fast. Marker-gated ``@pytest.mark.sim``;
opt in with ``pytest -m sim test/sim/`` after sourcing the workspace install.
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
from py_pkg.path.missions.factory import MissionId
from py_pkg.path.missions.profile import (
    DESCEND_TOLERANCE_PA,
    SHALLOW_TOLERANCE_PA,
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
from std_msgs.msg import Bool, Int32

from ._sim_helpers import (
    mission_command,
    reap_lingering_gz,
    sim_gui_enabled,
    spin_for,
    spin_until,
)

# Two distinct sub-surface pressures bracketing the ~5 m spawn (~49 050 Pa
# gauge). Deep is below the spawn so the first dive is genuine; shallow is well
# above it so each climb is a real ascent that stops short of the surface.
DEEP_PRESSURE_PA = 60_000.0  # ~6.1 m
SHALLOW_PRESSURE_PA = 30_000.0  # ~3.1 m
PITCH_RAD = math.radians(30.0)
N_OSCILLATIONS = 2


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    reap_lingering_gz()

    gui_enabled = sim_gui_enabled()

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
        # timing; scenario sets autostart=false. The mission constants in this
        # file remain the source of truth for the driver's publish.
        launch_arguments={
            "scenario": os.path.join(
                os.path.dirname(__file__), "scenarios", "test_sawtooth.yaml"
            ),
            "headless": "false" if gui_enabled else "true",
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


class _TwoPressureTestDriver(Node):
    """Kicks the two-pressure SAWTOOTH and captures the streams we assert on."""

    def __init__(self):
        super().__init__("sawtooth_two_pressure_test_driver")
        self.target_samples: list[tuple[float, Pose]] = []
        self.gauge_pressure_pa: list[tuple[float, float]] = []
        self.imu_msg_count: int = 0
        self.mission_complete: bool = False

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )
        # pathfinding's latched completion event -- the same signal bcu_node and
        # acu_node stop off. Waiting on it turns the mission budget into an
        # upper bound instead of a fixed cost.
        create_subscription_for_topic(
            self, UUVTopics.MISSION_COMPLETE, self._on_mission_complete
        )

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_mission_complete(self, msg: Bool) -> None:
        self.mission_complete = self.mission_complete or bool(msg.data)

    def _on_target(self, msg: Pose) -> None:
        self.target_samples.append((time.monotonic(), msg))

    def _on_pressure(self, msg: Int32) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the stack works in gauge.
        self.gauge_pressure_pa.append(
            (time.monotonic(), gauge_pressure_pa(float(msg.data)))
        )

    def publish_mission(self) -> None:
        self.path_pub.publish(
            mission_command(
                MissionId.SAWTOOTH,
                target_pressure_pa=DEEP_PRESSURE_PA,
                shallow_pressure_pa=SHALLOW_PRESSURE_PA,
                angle_rad=PITCH_RAD,
                n_resurfaces=N_OSCILLATIONS,
            )
        )

    def publish_start(self) -> None:
        msg = Bool()
        msg.data = True
        self.command_pub.publish(msg)


def _collapse(values: list[float]) -> list[float]:
    """Run-length collapse: drop consecutive duplicates.

    Turns the raw setpoint-z stream into the sequence of distinct legs the
    state machine stepped through, e.g. [deep, deep, shallow, shallow, deep,
    0, 0] -> [deep, shallow, deep, 0].
    """
    out: list[float] = []
    for v in values:
        if not out or out[-1] != v:
            out.append(v)
    return out


# The measured gauge pressure must actually enter these bands for a turn to
# count -- the same hysteresis edges the mission's `update` flips legs on, so
# "reached the targeted depth" means the same thing for the plant and the
# controller. Both bands sit well clear of the surface (35 kPa vs 5 kPa), so a
# shallow turn is never confused with surfacing.
DEEP_BAND_PA = DEEP_PRESSURE_PA - DESCEND_TOLERANCE_PA  # "at the deep target"
SHALLOW_BAND_PA = SHALLOW_PRESSURE_PA + SHALLOW_TOLERANCE_PA  # "at the shallow target"


def _measured_turns(pressures: list[tuple[float, float]]):
    """Walk measured gauge pressure through the mission's hysteresis.

    Returns (dive_count, shallow_turn_count, surfaced) where a dive is the
    measured pressure reaching the deep target band while descending, a shallow
    turn is reaching the shallow target band while ascending, and surfaced is
    whether the measured pressure ever crossed the surface threshold. This reads
    the achieved motion straight off EXTERNAL_PRESSURE, independent of what
    pathfinding commanded.
    """
    descending = True
    dives = 0
    shallow_turns = 0
    surfaced = False
    for _t, pa in pressures:
        if pa <= SURFACE_THRESHOLD_PA:
            surfaced = True
        if descending:
            if pa >= DEEP_BAND_PA:
                dives += 1
                descending = False
        else:
            if pa <= SHALLOW_BAND_PA:
                shallow_turns += 1
                descending = True
    return dives, shallow_turns, surfaced


@pytest.mark.sim
class SawtoothTwoPressureSimTest(unittest.TestCase):
    """Behavior: SAWTOOTH glides between two sub-surface pressures for N
    oscillations, then surfaces and self-terminates."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _TwoPressureTestDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    def test_two_oscillations_then_surface(self):
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # Two ~3 m-amplitude oscillations (~3.6 m <-> ~5.6 m turn points) plus a
        # final ~5 m ascent to the surface: ~10 m of glide, comparable to the
        # single-cycle baseline's 300 s budget. This is only the UPPER BOUND --
        # the run ends on /mission/complete, so a healthy mission costs its real
        # duration (~300 s) and only a broken one pays the full 600 s.
        mission_budget_s = 600.0
        drain_s = 2.0
        # The setpoint stream must fall silent for at least this long before the
        # window ends -- that silence is how we detect pathfinding reset on
        # is_done (mission complete).
        quiet_tail_s = 8.0

        sim_ready = spin_until(
            self.executor,
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU never arrived within {startup_timeout_s}s -- "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )

        spin_for(self.executor, post_ready_settle_s)

        self.driver.publish_mission()
        spin_for(self.executor, 0.5)
        self.driver.publish_start()

        mission_start_t = time.monotonic()
        completed = spin_until(
            self.executor,
            lambda: self.driver.mission_complete,
            timeout_s=mission_budget_s,
        )
        self.assertTrue(
            completed,
            f"mission never published /mission/complete within "
            f"{mission_budget_s}s -- it did not self-terminate after surfacing.",
        )
        # Hold the window open past completion so assertion 4 has a genuine
        # silent stretch of POSITION_TARGET to measure. drain_s on top collects
        # in-flight samples that the mission_end_t cutoff then excludes.
        spin_for(self.executor, quiet_tail_s + drain_s)
        mission_end_t = time.monotonic() - drain_s

        targets_during = [
            (t, p)
            for (t, p) in self.driver.target_samples
            if mission_start_t <= t <= mission_end_t
        ]
        pressures_during = [
            (t, pa)
            for (t, pa) in self.driver.gauge_pressure_pa
            if mission_start_t <= t <= mission_end_t
        ]

        # 1) Setpoint stream alive. Crash/wiring floor, not a rate check -- the
        #    mission completes and then goes quiet, so the count is bounded by
        #    the time-to-complete, not the full window.
        self.assertGreaterEqual(
            len(targets_during),
            100,
            f"pathfinding broadcast only {len(targets_during)} setpoints; "
            "did `start` reach pathfinding_node?",
        )
        self.assertGreaterEqual(
            len(pressures_during),
            50,
            f"only {len(pressures_during)} EXTERNAL_PRESSURE samples collected.",
        )

        # 2) Commanded profile is exactly [deep, shallow, deep, surface]: the
        #    state machine only steps a leg on a real pressure crossing, so this
        #    is "two dives between the two pressures, then a final surfacing".
        legs = _collapse([round(p.position.z, 1) for (_t, p) in targets_during])
        self.assertEqual(
            legs[0],
            DEEP_PRESSURE_PA,
            f"SAWTOOTH must start descending to the deep extremum; legs={legs}",
        )
        self.assertEqual(
            legs.count(DEEP_PRESSURE_PA),
            N_OSCILLATIONS,
            f"expected exactly {N_OSCILLATIONS} dives to the deep extremum "
            f"({DEEP_PRESSURE_PA} Pa); legs={legs}",
        )
        first_deep = legs.index(DEEP_PRESSURE_PA)
        last_deep = len(legs) - 1 - legs[::-1].index(DEEP_PRESSURE_PA)
        self.assertIn(
            SHALLOW_PRESSURE_PA,
            legs[first_deep + 1 : last_deep],
            f"never climbed to the shallow extremum ({SHALLOW_PRESSURE_PA} Pa) "
            f"between the two dives; legs={legs}",
        )
        self.assertEqual(
            legs[-1],
            0.0,
            f"mission must end on the surfacing leg (z = 0); legs={legs}",
        )

        # 3) The oscillations actually happen *at the targeted depths*, read
        #    straight off the measured EXTERNAL_PRESSURE (not just commanded):
        #    the vehicle reaches the deep target band twice, climbs to the
        #    shallow target band between the dives, and crosses the surface
        #    threshold (it resurfaces). gauge_values also pins the extremes.
        measured_dives, measured_shallow_turns, surfaced = _measured_turns(
            pressures_during
        )
        gauge_values = [pa for (_t, pa) in pressures_during]
        self.assertGreaterEqual(
            measured_dives,
            N_OSCILLATIONS,
            f"measured gauge reached the deep target band (>= {DEEP_BAND_PA:.0f} "
            f"Pa) only {measured_dives} time(s); expected {N_OSCILLATIONS} dives. "
            f"max gauge seen = {max(gauge_values):.0f} Pa.",
        )
        self.assertGreaterEqual(
            measured_shallow_turns,
            1,
            f"measured gauge never climbed back to the shallow target band "
            f"(<= {SHALLOW_BAND_PA:.0f} Pa) between dives -- the oscillation "
            "didn't return to the shallow pressure.",
        )
        self.assertTrue(
            surfaced,
            f"measured gauge never crossed the surface threshold "
            f"({SURFACE_THRESHOLD_PA:.0f} Pa); min gauge = {min(gauge_values):.0f} "
            "Pa -- the vehicle did not resurface.",
        )

        # 4) Mission terminated by surfacing: the last commanded setpoint is the
        #    surface, and POSITION_TARGET fell silent well before the window end
        #    (pathfinding resets and stops publishing on is_done).
        last_target_t, last_target = targets_during[-1]
        self.assertEqual(
            round(last_target.position.z, 1),
            0.0,
            "last published setpoint was not the surface -- mission didn't reach "
            "the final surfacing leg within the budget.",
        )
        quiet_tail = mission_end_t - last_target_t
        self.assertGreaterEqual(
            quiet_tail,
            quiet_tail_s,
            f"POSITION_TARGET only went quiet for {quiet_tail:.1f}s before the "
            f"window end (need >= {quiet_tail_s}s) -- the mission never reached "
            "is_done, so it didn't self-terminate after surfacing.",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class SawtoothTwoPressureSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort."""
        reap_lingering_gz()
