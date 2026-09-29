"""Tier 3 sim test for the BCU control path.

Drives BCU_RPM and asserts the simulated bladder fills: BCU_FLOW_RATE
goes positive, BCU_VOLUME (mL) increases monotonically. Verifies the
RPM -> flow -> volume integration end-to-end through the HAL bridge and
Gazebo buoyancy plugin; depth-tracking is intentionally out of scope.

Composed lean (bridges + Gazebo + robot, no controller) so the test
owns BCU_RPM exclusively. Marker-gated ``@pytest.mark.sim``; opt in with
``pytest -m sim test/sim/`` after sourcing the workspace install.
``SIM_GUI=1`` shows the Gazebo GUI. Don't run alongside any other
sim/rclpy process on the host — the production topic names overlap.
"""

import os
import time
import unittest

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK
from py_pkg.scenarios.spec.rig import NoiseSpec
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Int16, Int32, UInt8

from ._sim_helpers import reap_lingering_gz, sim_gui_enabled, spin_for, spin_until

# ---------------------------------------------------------------------------
# Launch description — composed, NOT a wholesale include of an end-to-end
# launch. Keeps the test stack lean (bridges + Gazebo + robot, no oscillator).
# ---------------------------------------------------------------------------


# launch_test: tells the launch_testing pytest plugin to run this file as
# a real launch test (without it, generate_test_description never fires
# and proc_info isn't injected). sim: keeps it out of default runs.
@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    """Lean BCU sim stack: bridges + Gazebo + robot, no oscillator."""
    # Reap MUST happen here, not in setUpClass — by then LaunchService has
    # already spawned this test's own gz sim, and the pkill regex would
    # kill it (causing world-name lookup to time out, model never spawns).
    reap_lingering_gz()

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

    # DAVE convention: `gui` is always "true"; `headless` controls the display.
    gui_enabled = sim_gui_enabled()

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
        # Same spawn pose as the trim/sawtooth sim launches.
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

    return (
        LaunchDescription(
            [
                bridge_launch,
                robot_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


# ---------------------------------------------------------------------------
# Test driver node
# ---------------------------------------------------------------------------


class _BCUTestDriver(Node):
    """Publishes BCU_RPM, captures BCU_FLOW_RATE / BCU_VOLUME / IMU."""

    def __init__(self):
        super().__init__("bcu_sim_test_driver")
        self.received_flow: list[float] = []
        self.received_volume_ml: list[int] = []
        self.received_tank_pa: list[int] = []
        self.received_external_pa: list[int] = []
        self.imu_msg_count: int = 0

        self.rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        create_subscription_for_topic(self, UUVTopics.BCU_FLOW_RATE, self._on_flow)
        create_subscription_for_topic(self, UUVTopics.BCU_VOLUME, self._on_volume)
        create_subscription_for_topic(self, UUVTopics.BCU_PRESSURE, self._on_tank)
        create_subscription_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE, self._on_external
        )
        # IMU is the sim-readiness signal: imu_sim_bridge has no timer,
        # so any message proves Gazebo physics + plugins are alive.
        create_subscription_for_topic(self, UUVTopics.IMU, self._on_imu)

    def _on_flow(self, msg: Float32) -> None:
        self.received_flow.append(float(msg.data))

    def _on_volume(self, msg: Int32) -> None:
        self.received_volume_ml.append(int(msg.data))

    def _on_tank(self, msg: Int32) -> None:
        self.received_tank_pa.append(int(msg.data))

    def _on_external(self, msg: Int32) -> None:
        self.received_external_pa.append(int(msg.data))

    def _on_imu(self, msg: Imu) -> None:
        self.imu_msg_count += 1

    def publish_rpm(self, rpm: int) -> None:
        # Mirror bcu_node's wire shape: a nonzero RPM rides with valve 2
        # (motor way) open, a stop closes the valves. The bridge gates the
        # hydraulic transfer on valve 2, so RPM alone must not move oil.
        msg = Int16()
        msg.data = int(rpm)
        self.rpm_pub.publish(msg)
        valves = UInt8()
        valves.data = BCU_MOTOR_VALVE_MASK if rpm != 0 else 0
        self.valves_pub.publish(valves)


# ---------------------------------------------------------------------------
# Test class — runs after ReadyToTest fires.
# ---------------------------------------------------------------------------


@pytest.mark.sim
class BCUSimTest(unittest.TestCase):
    """Behavior: a sustained positive RPM fills the bladder in sim."""

    @classmethod
    def setUpClass(cls):
        # Pre-launch gz reap lives in generate_test_description; can't move
        # it here without killing this test's own gz process.
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _BCUTestDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)

    def tearDown(self):
        self.executor.remove_node(self.driver)
        self.driver.destroy_node()
        self.executor.shutdown()

    # ---- the test ---------------------------------------------------------

    def test_positive_rpm_fills_bladder(self):
        """Sustained positive RPM -> positive flow + monotonic volume rise.

        Sequence: wait for IMU (sim ready), settle, send RPM=0 a few
        times to fire the bridge's startup clamp to rig.plant.bladder_min_m3
        (otherwise the +RPM accumulation gets subtracted from the SDF's
        ~1250 mL initial volume and looks like a decrease), snapshot the
        starting volume, drive +RPM at 10 Hz, then assert direction and
        sign — not exact volumetric rate (that's Tier 1).
        """
        target_rpm = 2000  # within [BCU_MOTOR_MIN_RPM, MAX]
        # CPU-only software rendering can need 20-40s for the world.
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        sync_settle_s = 2.0
        drive_duration_s = 5.0
        drive_period_s = 0.1  # 10 Hz, mirrors the production control loop
        settle_s = 2.0

        # 1) Wait for sim. IMU is the readiness signal — BCU_VOLUME
        # isn't (its timer publishes 0 immediately).
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

        # 2) Let the buoyancy plugin finish loading.
        spin_for(self.executor, post_ready_settle_s)

        # 3) Fire the bridge's startup clamp to rig.plant.bladder_min_m3 by
        #    publishing RPM=0, then let the volume roundtrip settle.
        for _ in range(3):
            self.driver.publish_rpm(0)
            self.executor.spin_once(timeout_sec=0.05)
        spin_for(self.executor, sync_settle_s)

        self.assertGreater(
            len(self.driver.received_volume_ml),
            0,
            "BCU_VOLUME never arrived even after IMU confirmed sim "
            "readiness — bcu_sim_bridge may be down.",
        )
        starting_volume_ml = self.driver.received_volume_ml[-1]

        # Drop the zero-flow samples from the RPM=0 prelude.
        self.driver.received_flow.clear()

        # 5) Drive +RPM at 10 Hz for drive_duration_s.
        deadline = time.monotonic() + drive_duration_s
        next_publish = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self.driver.publish_rpm(target_rpm)
                next_publish = now + drive_period_s
            self.executor.spin_once(timeout_sec=0.02)

        # 6) Stop the pump and let queues drain.
        self.driver.publish_rpm(0)
        spin_for(self.executor, settle_s)

        # 7a) Need >=1 positive flow sample (not all — the first can land
        # before the subscription handshake completes).
        positive_flow_samples = [f for f in self.driver.received_flow if f > 0.0]
        self.assertGreaterEqual(
            len(positive_flow_samples),
            1,
            f"expected >=1 positive BCU_FLOW_RATE sample while driving "
            f"+{target_rpm} RPM; saw {self.driver.received_flow!r}",
        )

        # A negative sample means the bridge sign convention flipped.
        negative_flow_samples = [f for f in self.driver.received_flow if f < 0.0]
        self.assertEqual(
            negative_flow_samples,
            [],
            f"unexpected negative flow samples while driving +RPM: "
            f"{negative_flow_samples!r}",
        )

        # 7b) Bladder volume must rise (we sync'd to the floor in step 3).
        ending_volume_ml = self.driver.received_volume_ml[-1]
        self.assertGreater(
            ending_volume_ml,
            starting_volume_ml,
            f"bladder volume did not increase under sustained +RPM: "
            f"start={starting_volume_ml} mL, end={ending_volume_ml} mL",
        )

    def test_sensor_noise_combs(self):
        """Injected sensor noise lands on the real sensors' quantization combs.

        The default scenario (nominal.yaml) runs with lake-fitted noise ON:
        tank pressure is sigma=353 Pa rounded to the 600 Pa grid, external
        pressure is quantization-only on a 100 Pa grid. Collect a window of
        telemetry and assert (a) every sample sits on its comb, and (b) the
        tank stream actually dithers (>=2 distinct values in a steady
        window) — i.e. the Gaussian term is alive, not just rounding.
        """
        startup_timeout_s = 60.0
        n_samples = 50  # 10 Hz publish rate -> ~5 s of telemetry

        sim_ready = spin_until(
            self.executor,
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(sim_ready, "IMU never arrived — sim not up?")

        collected = spin_until(
            self.executor,
            lambda: (
                len(self.driver.received_tank_pa) >= n_samples
                and len(self.driver.received_external_pa) >= n_samples
            ),
            timeout_s=30.0,
        )
        self.assertTrue(
            collected,
            f"expected >= {n_samples} tank + external samples; got "
            f"{len(self.driver.received_tank_pa)} tank / "
            f"{len(self.driver.received_external_pa)} external",
        )

        tank = self.driver.received_tank_pa[-n_samples:]
        external = self.driver.received_external_pa[-n_samples:]

        # Comb steps come from the spec defaults; a Tier 1 test locks the
        # launched nominal.yaml to those same values, so a noise re-fit
        # updates this test automatically.
        noise = NoiseSpec()
        tank_step = round(noise.tank_pressure.quantization_pa)
        external_step = round(noise.external_pressure.quantization_pa)

        off_comb_tank = [v for v in tank if v % tank_step != 0]
        self.assertEqual(
            off_comb_tank,
            [],
            f"tank pressure samples off the {tank_step} Pa comb: {off_comb_tank[:5]!r}",
        )
        self.assertGreaterEqual(
            len(set(tank)),
            2,
            f"tank pressure never dithered (sigma="
            f"{noise.tank_pressure.sigma_pa:.0f} Pa should move it "
            f"across the {tank_step} Pa grid): {sorted(set(tank))!r}",
        )

        off_comb_external = [v for v in external if v % external_step != 0]
        self.assertEqual(
            off_comb_external,
            [],
            f"external pressure samples off the {external_step} Pa comb: "
            f"{off_comb_external[:5]!r}",
        )


# ---------------------------------------------------------------------------
# Post-shutdown tests — run after the launched processes are torn down.
# ---------------------------------------------------------------------------


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class BCUSimPostShutdown(unittest.TestCase):
    """Sanity-check + cleanup after launched processes are torn down."""

    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort, no assert."""
        reap_lingering_gz()
