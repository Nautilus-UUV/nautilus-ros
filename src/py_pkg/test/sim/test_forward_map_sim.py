"""Tier 3 sim test: forward map sampled SDF.

Proves that the forward map generated SDF is valid and works in simulation.
"""

import math
import os
import tempfile
import time
import unittest
import yaml

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
from py_pkg.scenarios.compile import forward_map

from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from ._sim_helpers import reap_lingering_gz

TARGET_PRESSURE_PA = 60000.0
PITCH_RAD = math.radians(20.0)
N_RESURFACES = 1

def generate_forward_map_scenario(drag_multiplier: float) -> str:
    base_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "py_pkg",
        "scenarios",
        "library",
        "nominal_with_hydrodynamics.yaml"
    )
    with open(base_path, "r") as f:
        scenario = yaml.safe_load(f)
    
    knobs_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "py_pkg",
        "scenarios",
        "library",
        "nominal_knobs.yaml"
    )
    with open(knobs_path, "r") as f:
        knobs = yaml.safe_load(f)["physics_knobs"]
        
    knobs["C_d_c"] *= drag_multiplier
    knobs["C_p_base"] *= drag_multiplier
    
    hydro_spec = forward_map(knobs)
    
    if "rig" not in scenario:
        scenario["rig"] = {}
    scenario["rig"]["hydrodynamics"] = hydro_spec.model_dump()
    
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(scenario, f)
        
    return path


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
@launch_testing.parametrize("drag_multiplier", [1.2]) # High drag
def generate_test_description(drag_multiplier):
    from nautilus_hal.render_sdf import description_file_for_scenario

    reap_lingering_gz()
    
    scenario_path = generate_forward_map_scenario(drag_multiplier)

    gui_enabled = os.environ.get("HYDRO_SAMPLING_SIM_GUI", "").lower() in (
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
        {"scenario_path": scenario_path},
    )


class _SawtoothSamplingDriver(Node):
    def __init__(self):
        super().__init__("hydro_sampling_sim_test_driver")
        self.target_samples: list[tuple[float, Pose]] = []
        self.pressure_samples: list[tuple[float, float]] = []
        self.imu_msg_count: int = 0

        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)

        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)
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
        cmd.angle_rad = float(PITCH_RAD)
        cmd.n_resurfaces = int(N_RESURFACES)
        self.path_pub.publish(cmd)

    def publish_start(self) -> None:
        msg = String()
        msg.data = "start"
        self.command_pub.publish(msg)


@pytest.mark.sim
class HydroSamplingSimTest(unittest.TestCase):
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

    def test_sampled_sawtooth_round_trip(self, scenario_path):
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        mission_duration_s = 120.0
        drain_s = 2.0

        sim_ready = self._spin_until(
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(sim_ready, "IMU_LEFT never arrived")

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
        self.assertGreaterEqual(len(targets_during), 100)

        z_values = {round(p.position.z, 3) for (_t, p) in targets_during}
        self.assertIn(float(TARGET_PRESSURE_PA), z_values)
        self.assertIn(0.0, z_values)

        pressure_during = [
            v
            for (t, v) in self.driver.pressure_samples
            if mission_start_t <= t <= mission_end_t
        ]
        self.assertGreaterEqual(len(pressure_during), 100)
        
        flip_threshold_absolute_pa = ATMOSPHERIC_PRESSURE_PA + (
            float(TARGET_PRESSURE_PA) - DESCEND_TOLERANCE_PA
        )
        max_pa = max(pressure_during)
        self.assertGreaterEqual(max_pa, flip_threshold_absolute_pa)


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class HydroSamplingSimPostShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, 1, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        reap_lingering_gz()
