"""Tier 3 sim test: short SAWTOOTH cycle across three sampled drag regimes.

Proves the Jinja-rendered SDF pipeline round-trips through Gazebo and
the closed-loop control stack at three perturbed plant points
(``test_hydro_sample_low.yaml`` / ``mid.yaml`` / ``high.yaml``). Each
sample runs the same short sawtooth and is asserted independently.

The terminology is sampling-strategy-agnostic on purpose: today these
three YAMLs are hand-picked, but a future LHS / Sobol / Halton sweep
would generate analogous YAMLs and feed them through the same pipeline.

The Tier 3 parity test in `nautilus_hal/test/` already locks in
"rendering at defaults equals the canonical SDF byte-for-byte" — this
test is the integration counterpart: the rendered SDF is actually
spawnable, the simulation produces sensible telemetry, and the mission
state machine reaches the ascent leg regardless of which drag regime
the sampled YAML asked for.

Determinism: every scenario uses ``seed: 0`` and fault injection is off.
Sim physics is reproducible run-to-run on the same machine; the
assertions intentionally stay qualitative ("descend setpoint observed",
"ascent setpoint observed", "leg-flip threshold crossed") rather than
numeric so a slow drag regime doesn't flake on a tight threshold.

Composed via ``nautilus_hal/launch/sawtooth_sim.launch.py`` with
``scenario:=test/sim/scenarios/test_hydro_sample_{low,mid,high}.yaml``.
Marker-gated ``@pytest.mark.sim``.
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
from py_pkg.path.missions.factory import MissionId
from py_pkg.path.missions.sawtooth import DESCEND_TOLERANCE_PA
from py_pkg.physics import ATMOSPHERIC_PRESSURE_PA
from py_pkg.scenarios.loader import load_scenario

# `nautilus_hal` lives in the dave sister repo. py_pkg's CI (nautilus-ros
# alone, no dave checkout) imports this file during collection even for
# `-m "not sim"` runs; a module-level `from nautilus_hal...` would
# explode there. Deferred to inside `generate_test_description`.
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from ._sim_helpers import reap_lingering_gz, sim_gui_enabled

# Short sawtooth: spawn at z=-5 m (~50 kPa gauge) → dive to ~6 m
# (~60 kPa) → return to surface. About 1 m of glide each leg, a tiny
# excursion compared to the regular sawtooth test, so all three sampled
# regimes (slipperier / nominal-via-jinja / draggier) complete inside
# the wall budget even with the high-drag plant pushing settling time up.
TARGET_PRESSURE_PA = 60000.0
PITCH_RAD = math.radians(20.0)
N_OSCILLATIONS = 1

_SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
_SCENARIO_YAMLS = (
    "test_hydro_sample_low.yaml",
    "test_hydro_sample_mid.yaml",
    "test_hydro_sample_high.yaml",
)


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
@launch_testing.parametrize("scenario_yaml", list(_SCENARIO_YAMLS))
def generate_test_description(scenario_yaml):
    # Deferred import: see module-top comment — keeps nautilus-ros-only
    # CI happy when the dave sister repo isn't checked out.
    from nautilus_hal.render_sdf import description_file_for_scenario

    reap_lingering_gz()
    scenario_path = os.path.join(_SCENARIO_DIR, scenario_yaml)

    # Precondition: the YAML must actually carry a hydrodynamics block,
    # otherwise this run would silently fall back to the canonical SDF
    # and we'd be testing the wrong code path. Catch that here, not at
    # the end of a 90 s sim.
    scenario = load_scenario(scenario_path)
    assert scenario.rig.hydrodynamics is not None, (
        f"{scenario_yaml}: rig.hydrodynamics missing -- this test exists "
        "to exercise the Jinja render path; remove the YAML or add the "
        "block."
    )
    rendered_path = description_file_for_scenario(scenario_path)
    canonical_basename = "model.sdf"
    assert os.path.basename(rendered_path) != canonical_basename, (
        f"description_file_for_scenario returned the canonical SDF "
        f"({rendered_path}) for {scenario_yaml}; expected a temp render."
    )

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
        # Driver publishes /path + /command manually for deterministic timing.
        launch_arguments={
            "scenario": scenario_path,
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
        {"scenario_yaml": scenario_yaml},
    )


class _SawtoothSamplingDriver(Node):
    """Mission kick + telemetry capture for the sampled-sawtooth test."""

    def __init__(self):
        super().__init__("hydro_sampling_sim_test_driver")
        self.target_samples: list[tuple[float, Pose]] = []
        self.pressure_samples: list[tuple[float, float]] = []
        self.imu_msg_count: int = 0

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)
        create_subscription_for_topic(self, UUVTopics.POSITION_TARGET, self._on_target)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_pressure
        )

    def _on_imu(self, _msg: Imu) -> None:
        self.imu_msg_count += 1

    def _on_target(self, msg: Pose) -> None:
        self.target_samples.append((time.monotonic(), msg))

    def _on_pressure(self, msg) -> None:
        self.pressure_samples.append((time.monotonic(), float(msg.data)))

    def publish_mission(self) -> None:
        cmd = MissionCommand()
        cmd.mission_id = int(MissionId.SAWTOOTH)
        cmd.target_pressure_pa = float(TARGET_PRESSURE_PA)
        cmd.shallow_pressure_pa = 0.0  # legacy: climb to the surface each dive
        cmd.angle_rad = float(PITCH_RAD)
        cmd.n_oscillations = int(N_OSCILLATIONS)
        self.path_pub.publish(cmd)

    def publish_start(self) -> None:
        msg = Bool()
        msg.data = True
        self.command_pub.publish(msg)


@pytest.mark.sim
class HydroSamplingSimTest(unittest.TestCase):
    """Behavior: each sampled drag regime renders a fresh SDF, the sim
    spawns against it, and the SAWTOOTH state machine completes one
    descend + ascend cycle within the wall budget."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _SawtoothSamplingDriver()
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

    def test_sampled_sawtooth_round_trip(self, scenario_yaml):
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        # 120 s wall budget for the short sawtooth (~1 m descent + ascent
        # past surface). High-drag regime is the slowest; this gives it
        # headroom without flirting with the test runner's idle timeout.
        mission_duration_s = 120.0
        drain_s = 2.0

        sim_ready = self._spin_until(
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"[{scenario_yaml}] IMU never arrived within "
            f"{startup_timeout_s}s -- did the rendered SDF spawn? "
            "Check the launch log for the 'Spawning SDF: ...' line.",
        )

        self._spin_for(post_ready_settle_s)

        self.driver.publish_mission()
        self._spin_for(0.5)
        self.driver.publish_start()

        mission_start_t = time.monotonic()
        self._spin_for(mission_duration_s)
        self._spin_for(drain_s)
        mission_end_t = time.monotonic() - drain_s

        targets_during = [
            (t, p)
            for (t, p) in self.driver.target_samples
            if mission_start_t <= t <= mission_end_t
        ]
        self.assertGreaterEqual(
            len(targets_during),
            100,
            f"[{scenario_yaml}] pathfinding broadcast only "
            f"{len(targets_during)} setpoints in {mission_duration_s}s -- "
            "did `start` reach pathfinding_node?",
        )

        # The SAWTOOTH state machine stamps the descent leg setpoint with
        # position.z = target_pa and the ascent leg setpoint with
        # position.z = 0.0. Seeing both proves we got through descent and
        # past the leg flip into ascent.
        z_values = {round(p.position.z, 3) for (_t, p) in targets_during}
        self.assertIn(
            float(TARGET_PRESSURE_PA),
            z_values,
            f"[{scenario_yaml}] never saw a descend setpoint with "
            f"position.z = {TARGET_PRESSURE_PA} Pa; SAWTOOTH never started",
        )
        self.assertIn(
            0.0,
            z_values,
            f"[{scenario_yaml}] never saw an ascend setpoint (z=0); "
            f"SAWTOOTH never flipped to ascent within {mission_duration_s}s. "
            "Drag regime may need more wall budget, or the BCU loop did "
            "not close through the rendered hydrodynamics.",
        )

        # Pressure stream proves the BCU control loop is actually closing
        # through the rendered SDF's hydrodynamics: pressure should cross
        # the target setpoint at least once during the mission window.
        pressure_during = [
            v
            for (t, v) in self.driver.pressure_samples
            if mission_start_t <= t <= mission_end_t
        ]
        self.assertGreaterEqual(
            len(pressure_during),
            100,
            f"[{scenario_yaml}] only {len(pressure_during)} EXTERNAL_PRESSURE "
            "samples; the HAL bridge or sensor plugin may be silent.",
        )
        # EXTERNAL_PRESSURE is absolute Pa; the SAWTOOTH state machine
        # flips to ascent at `target_pa - DESCEND_TOLERANCE_PA` (gauge),
        # not at the full target_pa. So the BCU's job is to drive the
        # glider to that flip threshold, not all the way to target_pa.
        # That's the contract this assertion checks.
        flip_threshold_absolute_pa = ATMOSPHERIC_PRESSURE_PA + (
            float(TARGET_PRESSURE_PA) - DESCEND_TOLERANCE_PA
        )
        max_pa = max(pressure_during)
        self.assertGreaterEqual(
            max_pa,
            flip_threshold_absolute_pa,
            f"[{scenario_yaml}] EXTERNAL_PRESSURE never reached the descend "
            f"leg-flip threshold ({flip_threshold_absolute_pa:.0f} Pa abs, "
            f"= target {TARGET_PRESSURE_PA:.0f} gauge minus tolerance "
            f"{DESCEND_TOLERANCE_PA:.0f} Pa). Peak was {max_pa:.0f} Pa abs. "
            "BCU/depth loop did not drive the glider deep enough through "
            "this drag regime within the wall budget.",
        )


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class HydroSamplingSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        reap_lingering_gz()
