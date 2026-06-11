"""Tier 3 sim test for the lifeguard failsafe, end to end over real MQTT.

The full deploy-day chain: a mosquitto broker (standing in for the mission
laptop), the control stack with a shortened lifeguard window, and the Gazebo
glider. A test-side paho client plays the frontend -- it arms the lifeguard
and heartbeats; then the heartbeats stop (the tether "drops" -- from the
bridge watchdog's view, heartbeat silence and a severed cable are the same
signal). The glider must stop the mission and continuously command
EMERGENCY_SURFACE_RPM with valve 2 (the motor way) open, and the simulated
bladder must actually inflate.

Marker-gated ``@pytest.mark.sim``; opt in with ``pytest -m sim test/sim/``
after sourcing the workspace install. ``BCU_SIM_GUI=1`` shows the Gazebo GUI.
Skipped entirely when no mosquitto binary is installed.
"""

import json
import os
import shutil
import time
import unittest

import pytest

# Debian installs mosquitto into /usr/sbin, which is not on a normal user
# PATH -- check both before giving up.
MOSQUITTO_BIN = shutil.which("mosquitto") or (
    "/usr/sbin/mosquitto" if os.path.exists("/usr/sbin/mosquitto") else None
)
if MOSQUITTO_BIN is None:
    pytest.skip("mosquitto broker not installed", allow_module_level=True)

import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import paho.mqtt.client as mqtt
import rclpy
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from py_pkg.debug.bcu_debug_node import EMERGENCY_SURFACE_RPM
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK
from py_pkg.uuv_ros_core import UUVTopics, create_subscription_for_topic
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Int32, UInt8

from ._sim_helpers import reap_lingering_gz, sim_gui_enabled, spin_for, spin_until

BROKER_PORT = 1884  # off the default 1883 so a dev broker is never touched
LIFEGUARD_TIMEOUT_S = 5.0  # shortened from the production 120 s


@pytest.mark.launch_test
@pytest.mark.sim
@launch_testing.markers.keep_alive
def generate_test_description():
    """Broker + bridges + Gazebo robot + full control stack."""
    # Reap MUST happen here, not in setUpClass -- by then LaunchService has
    # already spawned this test's own gz sim, and the pkill regex would
    # kill it (causing world-name lookup to time out, model never spawns).
    reap_lingering_gz()

    mosquitto = ExecuteProcess(
        cmd=[
            MOSQUITTO_BIN,
            "-c",
            os.path.join(os.path.dirname(__file__), "lifeguard_mosquitto.conf"),
        ],
        output="screen",
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

    control_stack_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    FindPackageShare("py_pkg").find("py_pkg"),
                    "launch",
                    "control_stack.launch.py",
                )
            ]
        ),
        launch_arguments={
            "mqtt_broker_host": "127.0.0.1",
            "mqtt_broker_port": str(BROKER_PORT),
            "lifeguard_timeout_s": str(LIFEGUARD_TIMEOUT_S),
        }.items(),
    )

    return (
        LaunchDescription(
            [
                mosquitto,
                bridge_launch,
                robot_launch,
                control_stack_launch,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


# ---------------------------------------------------------------------------
# Test driver node -- observes the BCU wire; all commanding goes over MQTT.
# ---------------------------------------------------------------------------


class _LifeguardTestDriver(Node):
    """Captures BCU_RPM / BCU_VALVES / BCU_VOLUME / BCU_PRESSURE / IMU_LEFT."""

    def __init__(self):
        super().__init__("lifeguard_sim_test_driver")
        self.received_rpm: list[int] = []
        self.received_valves: list[int] = []
        self.received_volume_ml: list[int] = []
        self.received_tank_pa: list[int] = []
        self.imu_msg_count: int = 0

        create_subscription_for_topic(self, UUVTopics.BCU_RPM, self._on_rpm)
        create_subscription_for_topic(self, UUVTopics.BCU_VALVES, self._on_valves)
        create_subscription_for_topic(self, UUVTopics.BCU_VOLUME, self._on_volume)
        # Live tank pressure, for the stand-down test to anchor its
        # registered empty endpoint against.
        create_subscription_for_topic(self, UUVTopics.BCU_PRESSURE, self._on_tank)
        # IMU_LEFT is the sim-readiness signal: imu_sim_bridge has no timer,
        # so any message proves Gazebo physics + plugins are alive.
        create_subscription_for_topic(self, UUVTopics.IMU_LEFT, self._on_imu)

    def _on_rpm(self, msg: Int16) -> None:
        self.received_rpm.append(int(msg.data))

    def _on_valves(self, msg: UInt8) -> None:
        self.received_valves.append(int(msg.data))

    def _on_volume(self, msg: Int32) -> None:
        self.received_volume_ml.append(int(msg.data))

    def _on_tank(self, msg: Int32) -> None:
        self.received_tank_pa.append(int(msg.data))

    def _on_imu(self, msg: Imu) -> None:
        self.imu_msg_count += 1


# ---------------------------------------------------------------------------
# Test class -- runs after ReadyToTest fires.
# ---------------------------------------------------------------------------


@pytest.mark.sim
class LifeguardSimTest(unittest.TestCase):
    """Armed lifeguard + heartbeat silence -> continuous emergency surface."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.driver = _LifeguardTestDriver()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.driver)
        # The mission laptop's stand-in. mosquitto is spawned by the same
        # launch description, so its listener may not be bound yet when
        # ReadyToTest fires -- retry instead of racing it.
        self.laptop = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        deadline = time.monotonic() + 10.0
        while True:
            try:
                self.laptop.connect("127.0.0.1", BROKER_PORT)
                break
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        self.laptop.loop_start()

    def tearDown(self):
        try:
            # Disarm (retained) so nothing stays latched during teardown. The
            # broker's retained store is in-memory (no persistence in the
            # conf), so it evaporates with the broker anyway.
            self.laptop.publish(
                "nautilus/cmd/lifeguard", '{"data": false}', qos=1, retain=True
            )
            self.laptop.loop_stop()
            self.laptop.disconnect()
        finally:
            self.executor.remove_node(self.driver)
            self.driver.destroy_node()
            self.executor.shutdown()

    # ---- the test ---------------------------------------------------------

    def test_heartbeat_loss_blows_ballast(self):
        startup_timeout_s = 60.0
        post_ready_settle_s = 2.0
        beat_hold_s = 4.0  # < LIFEGUARD_TIMEOUT_S while beating
        # Window + 1 s lifeguard tick + bridge->bcu_debug hop, with margin.
        engage_timeout_s = LIFEGUARD_TIMEOUT_S + 10.0

        # 1) Wait for sim readiness.
        sim_ready = spin_until(
            self.executor,
            lambda: self.driver.imu_msg_count >= 1,
            timeout_s=startup_timeout_s,
        )
        self.assertTrue(
            sim_ready,
            f"IMU_LEFT never arrived within {startup_timeout_s}s -- "
            "is Gazebo up and is the model spawned with its IMU plugin?",
        )
        spin_for(self.executor, post_ready_settle_s)

        # 2) Arm (retained, as the UI publishes it) and heartbeat at 1 Hz.
        # While beats flow, the emergency rpm must never appear.
        self.laptop.publish(
            "nautilus/cmd/lifeguard", '{"data": true}', qos=1, retain=True
        )
        deadline = time.monotonic() + beat_hold_s
        while time.monotonic() < deadline:
            self.laptop.publish("nautilus/cmd/heartbeat", "{}", qos=0)
            spin_for(self.executor, 1.0)
        self.assertNotIn(
            EMERGENCY_SURFACE_RPM,
            self.driver.received_rpm,
            "lifeguard fired while heartbeats were still flowing",
        )

        # 3) The tether drops: heartbeats stop. (The bridge's watchdog keys
        # on heartbeat freshness, so silence and a severed cable are the
        # same trigger.) The emergency surface must engage within the window.
        engaged = spin_until(
            self.executor,
            lambda: EMERGENCY_SURFACE_RPM in self.driver.received_rpm,
            timeout_s=engage_timeout_s,
        )
        self.assertTrue(
            engaged,
            f"no {EMERGENCY_SURFACE_RPM} RPM command within {engage_timeout_s}s "
            f"of heartbeat loss; saw {self.driver.received_rpm[-20:]!r}",
        )

        # 4) Continuous hold: bcu_debug heartbeats the emergency command at
        # 10 Hz, so a 2 s window must show it repeatedly -- and nothing else
        # on the wire (the engage stopped the mission; depth_node is silent).
        # Settle first: the engage's one-shot /command=false makes depth_node
        # emit a single safe-stop 0 that must not leak into the clean window.
        spin_for(self.executor, 1.0)
        self.driver.received_rpm.clear()
        self.driver.received_valves.clear()
        spin_for(self.executor, 2.0)
        self.assertGreaterEqual(
            len(self.driver.received_rpm),
            5,
            f"emergency rpm not held continuously: {self.driver.received_rpm!r}",
        )
        self.assertTrue(
            all(r == EMERGENCY_SURFACE_RPM for r in self.driver.received_rpm),
            f"foreign rpm on the wire during emergency: {self.driver.received_rpm!r}",
        )
        self.assertTrue(
            self.driver.received_valves
            and all(v == BCU_MOTOR_VALVE_MASK for v in self.driver.received_valves),
            f"valve 2 (motor way) not held open: {self.driver.received_valves!r}",
        )

        # 5) The sim plant responds: the bladder inflates. Baseline taken
        # well after engage -- the HAL bridge's startup clamp pulls the
        # volume down to bladder_min on the first rpm message, so an
        # earlier baseline would see that drop, not the inflate.
        self.assertTrue(
            spin_until(
                self.executor,
                lambda: len(self.driver.received_volume_ml) >= 1,
                timeout_s=5.0,
            ),
            "BCU_VOLUME never arrived -- bcu_sim_bridge may be down",
        )
        v0 = self.driver.received_volume_ml[-1]
        inflated = spin_until(
            self.executor,
            lambda: self.driver.received_volume_ml[-1] > v0,
            timeout_s=5.0,
        )
        self.assertTrue(
            inflated,
            f"bladder did not inflate under the emergency surface: "
            f"start={v0} mL, end={self.driver.received_volume_ml[-1]} mL",
        )

    def test_tank_empty_stand_down(self):
        """A registered empty endpoint stands the blow down mid-surface.

        Runs after test_heartbeat_loss_blows_ballast (alphabetical order)
        against the same launch session; that test's teardown disarmed the
        lifeguard, so this one starts from a clean, silent BCU wire. The
        registered empty endpoint is anchored to the LIVE tank reading --
        stand-down must trigger once the blow drains the tank 2 kPa below
        where it is now -- so the runtime stays bounded regardless of how
        much the previous test already inflated the bladder.
        """
        # Retained bridge confirmations: lifeguard status, plus the
        # ROS-confirmed echo of the dive registration.
        statuses: list[dict] = []
        init_echoes: list[dict] = []

        def _on_status(_client, _userdata, msg):
            payload = json.loads(msg.payload.decode("utf-8"))
            if msg.topic == "nautilus/status/lifeguard":
                statuses.append(payload)
            else:
                init_echoes.append(payload)

        self.laptop.on_message = _on_status
        self.laptop.subscribe("nautilus/status/lifeguard", qos=1)
        self.laptop.subscribe("nautilus/status/init", qos=1)

        # 1) Live tank reading -> registration anchored just above the band.
        self.assertTrue(
            spin_until(
                self.executor,
                lambda: len(self.driver.received_tank_pa) >= 1,
                timeout_s=30.0,
            ),
            "BCU_PRESSURE never arrived -- bcu_sim_bridge may be down",
        )
        tank_now = self.driver.received_tank_pa[-1]
        drain_to_stand_down_pa = 2_000
        empty_pa = (tank_now - drain_to_stand_down_pa) / 1.10
        self.laptop.publish(
            "nautilus/cmd/init",
            f'{{"surface_pressure_pa": 101325.0, "tank_empty_pa": {empty_pa:.1f}, '
            f'"tank_full_pa": {tank_now + 50_000:.1f}}}',
            qos=1,
            retain=True,
        )
        # The retained status/init echo confirms the registration actually
        # reached the ROS graph -- deterministic, unlike a fixed settle.
        registered = spin_until(
            self.executor,
            lambda: init_echoes
            and abs(init_echoes[-1]["tank_empty_pa"] - empty_pa) < 1.0,
            timeout_s=10.0,
        )
        self.assertTrue(
            registered,
            f"registration never echoed on nautilus/status/init; "
            f"got {init_echoes[-3:]!r}",
        )

        # 2) Arm and stay silent -> engage within the window, blow starts.
        self.laptop.publish(
            "nautilus/cmd/lifeguard", '{"data": true}', qos=1, retain=True
        )
        engaged = spin_until(
            self.executor,
            lambda: EMERGENCY_SURFACE_RPM in self.driver.received_rpm,
            timeout_s=LIFEGUARD_TIMEOUT_S + 10.0,
        )
        self.assertTrue(
            engaged,
            f"blow never started; rpm history {self.driver.received_rpm[-20:]!r}",
        )

        # 3) The blow drains the tank into the band -> bcu_debug is stood
        # down (one False from the bridge -> 0 RPM + valves closed + flush).
        stood_down = spin_until(
            self.executor,
            lambda: self.driver.received_rpm
            and self.driver.received_rpm[-1] == 0
            and self.driver.received_tank_pa[-1] <= empty_pa * 1.10,
            timeout_s=60.0,
        )
        self.assertTrue(
            stood_down,
            f"blow never stood down: tank={self.driver.received_tank_pa[-1]} Pa, "
            f"band edge={empty_pa * 1.10:.0f} Pa, "
            f"rpm tail {self.driver.received_rpm[-10:]!r}",
        )

        # 4) Quiet wire afterwards: no emergency rpm reappears (the latch is
        # engaged but the pump has nothing left to move).
        spin_for(self.executor, 1.0)  # let bcu_debug's trailing-zero flush finish
        self.driver.received_rpm.clear()
        spin_for(self.executor, 2.0)
        self.assertNotIn(
            EMERGENCY_SURFACE_RPM,
            self.driver.received_rpm,
            f"blow restarted after stand-down: {self.driver.received_rpm!r}",
        )

        # 5) The bridge's retained status confirms: still engaged, stood down.
        got_status = spin_until(
            self.executor,
            lambda: statuses
            and statuses[-1].get("engaged")
            and statuses[-1].get("stood_down"),
            timeout_s=10.0,
        )
        self.assertTrue(
            got_status,
            f"expected engaged+stood_down on nautilus/status/lifeguard, "
            f"got {statuses[-3:]!r}",
        )


# ---------------------------------------------------------------------------
# Post-shutdown tests -- run after the launched processes are torn down.
# ---------------------------------------------------------------------------


@launch_testing.post_shutdown_test()
@pytest.mark.sim
class LifeguardSimPostShutdown(unittest.TestCase):
    """Sanity-check + cleanup after launched processes are torn down."""

    def test_exit_codes(self, proc_info):
        # SIGTERM/SIGINT are the normal ros2/Gazebo/mosquitto shutdown paths.
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )

    def test_reap_lingering_gz_servers(self):
        """Reap gz-sim children launch_testing missed; best-effort, no assert."""
        reap_lingering_gz()
