"""Tier 2 scaffolding for in-process rclpy tests.

Spins the node-under-test alongside a tester node in a SingleThreadedExecutor.
Tester nodes use uuv_ros_core factories so QoS auto-matches whatever the
node-under-test expects — manual create_publisher / create_subscription calls
with mismatched QoS will silently drop messages.

Layout:
* ``NodeHarness`` — generic: holds a node-under-test + tester node, drives
  them with an executor, and exposes ``spin_for`` / ``spin_until`` helpers.
* ``_DepthTesterNode`` / ``_ACUTesterNode`` — per-node tester surfaces
  declaring the inbound publishers and outbound subscriptions specific to
  each node-under-test.
* ``depth_node_harness`` / ``acu_node_harness`` — function-scoped pytest
  fixtures wiring node-under-test class + tester class into ``NodeHarness``.
"""

import math
import os
import time

import pytest
import rclpy
from geometry_msgs.msg import Pose
from nautilus_msgs.msg import BcuPumpCommand, MissionCommand
from py_pkg.debug.bcu_debug_node import BcuDebugNode
from py_pkg.path.pathfinding import PathfindingNode
from py_pkg.pid.acu_node import ACUControlNode
from py_pkg.pid.depth_node import DepthControlNode
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Int16, Int32, String, UInt8


def _isolated_ros_domain_id() -> int:
    """Per-PID ROS_DOMAIN_ID. Tests use production topic names, so any
    sibling rclpy participant on the default domain (stray `ros2` CLI,
    leftover Gazebo, parallel pytest) would inject traffic into the
    harness and corrupt assertions."""
    return (os.getpid() % 101) + 1


@pytest.fixture(scope="session", autouse=True)
def rclpy_session():
    """Init/shutdown rclpy once per session, on an isolated ROS_DOMAIN_ID."""
    os.environ["ROS_DOMAIN_ID"] = str(_isolated_ros_domain_id())
    rclpy.init()
    yield
    rclpy.shutdown()


class NodeHarness:
    """Generic Tier 2 harness: node-under-test + tester node + executor.

    Owns lifecycle (executor add/remove, destroy_node, executor shutdown).
    Subclasses or callers supply the node-under-test and a tester node;
    the tester node owns the test-side publishers/subscribers.
    """

    def __init__(self, node_under_test_cls, tester_cls):
        self.node = node_under_test_cls()
        self.tester = tester_cls()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.tester)

    def spin_for(self, duration_s: float, slice_s: float = 0.02) -> None:
        """Spin the executor for at least ``duration_s`` of wall time."""
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def spin_until(self, predicate, timeout: float = 2.0, slice_s: float = 0.02):
        """Spin until ``predicate()`` is truthy or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.executor.spin_once(timeout_sec=slice_s)
        if not predicate():
            raise TimeoutError(f"predicate did not become true within {timeout}s")

    def shutdown(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.executor.remove_node(self.tester)
        finally:
            self.node.destroy_node()
            self.tester.destroy_node()
            self.executor.shutdown()


# ---------------------------------------------------------------------------
# Depth node harness
# ---------------------------------------------------------------------------


class _DepthTesterNode(Node):
    """Drives DepthControlNode and captures BCU_RPM + BCU_VALVES emissions."""

    def __init__(self):
        super().__init__("depth_node_tester")
        self.received_rpm: list[int] = []
        self.received_valves: list[int] = []

        self.target_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.manual_override_pub = create_publisher_for_topic(
            self, UUVTopics.CONTROL_MANUAL_OVERRIDE
        )
        self.bcu_rpm_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_RPM, self._on_rpm
        )
        self.bcu_valves_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_VALVES, self._on_valves
        )

    def _on_rpm(self, msg: Int16) -> None:
        self.received_rpm.append(int(msg.data))

    def _on_valves(self, msg: UInt8) -> None:
        self.received_valves.append(int(msg.data))

    def publish_target_pressure(self, value_pa: float) -> None:
        # depth_node treats position.z as the gauge-pressure setpoint
        # (Pa). Other Pose fields are zeroed.
        msg = Pose()
        msg.position.z = float(value_pa)
        self.target_pose_pub.publish(msg)

    def publish_external_pressure(self, value_pa: int) -> None:
        msg = Int32()
        msg.data = int(value_pa)
        self.external_pressure_pub.publish(msg)

    def publish_manual_override(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self.manual_override_pub.publish(msg)


class DepthNodeHarness(NodeHarness):
    """NodeHarness specialised for DepthControlNode + _DepthTesterNode."""

    def __init__(self):
        super().__init__(DepthControlNode, _DepthTesterNode)

    @property
    def received_rpm(self) -> list[int]:
        return self.tester.received_rpm

    @property
    def received_valves(self) -> list[int]:
        return self.tester.received_valves

    def publish_target_pressure(self, value_pa: float) -> None:
        self.tester.publish_target_pressure(value_pa)

    def publish_external_pressure(self, value_pa: int) -> None:
        self.tester.publish_external_pressure(value_pa)

    def publish_manual_override(self, active: bool) -> None:
        self.tester.publish_manual_override(active)


@pytest.fixture
def depth_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = DepthNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# ACU node harness
# ---------------------------------------------------------------------------


def _quat_from_roll_deg(roll_deg: float):
    """Roll-only quaternion (pitch=yaw=0). Inverse of
    math_utils.quaternion_to_roll_pitch — pitch isn't part of the ACU
    target/estimation surface anymore now that pitch is driven by
    pressure error rather than a target attitude."""
    r = math.radians(roll_deg) / 2.0
    qw = math.cos(r)
    qx = math.sin(r)
    return qx, 0.0, 0.0, qw


class _ACUTesterNode(Node):
    """Drives ACUControlNode and captures ACU_PITCH (Int16 mm) /
    ACU_ROLL (Int16 cdeg).

    POSITION_TARGET carries the roll setpoint in its orientation and the
    *target gauge pressure* on position.z (pathfinding's TRIM
    convention). EXTERNAL_PRESSURE carries the *absolute* pressure
    sensor reading the bang-bang pitch loop compares against."""

    def __init__(self):
        super().__init__("acu_node_tester")
        self.received_pitch_mm: list[int] = []
        self.received_roll_cdeg: list[int] = []

        self.target_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )
        self.estimation_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.pitch_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_PITCH, self._on_pitch
        )
        self.roll_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_ROLL, self._on_roll
        )

    def _on_pitch(self, msg: Int16) -> None:
        self.received_pitch_mm.append(int(msg.data))

    def _on_roll(self, msg: Int16) -> None:
        self.received_roll_cdeg.append(int(msg.data))

    @staticmethod
    def _pose(roll_deg: float, target_pressure_pa: float = 0.0) -> Pose:
        qx, qy, qz, qw = _quat_from_roll_deg(roll_deg)
        msg = Pose()
        msg.position.z = float(target_pressure_pa)
        msg.orientation.x = float(qx)
        msg.orientation.y = float(qy)
        msg.orientation.z = float(qz)
        msg.orientation.w = float(qw)
        return msg

    def publish_target(
        self, roll_deg: float = 0.0, target_pressure_pa: float = 0.0
    ) -> None:
        self.target_pose_pub.publish(self._pose(roll_deg, target_pressure_pa))

    def publish_current_attitude(self, roll_deg: float = 0.0) -> None:
        self.estimation_pose_pub.publish(self._pose(roll_deg))

    def publish_external_pressure(self, value_pa: int) -> None:
        msg = Int32()
        msg.data = int(value_pa)
        self.external_pressure_pub.publish(msg)


class ACUNodeHarness(NodeHarness):
    """NodeHarness specialised for ACUControlNode + _ACUTesterNode."""

    def __init__(self):
        super().__init__(ACUControlNode, _ACUTesterNode)

    @property
    def received_pitch_mm(self) -> list[int]:
        return self.tester.received_pitch_mm

    @property
    def received_roll_cdeg(self) -> list[int]:
        return self.tester.received_roll_cdeg

    def publish_target(
        self, roll_deg: float = 0.0, target_pressure_pa: float = 0.0
    ) -> None:
        self.tester.publish_target(
            roll_deg=roll_deg, target_pressure_pa=target_pressure_pa
        )

    def publish_current_attitude(self, roll_deg: float = 0.0) -> None:
        self.tester.publish_current_attitude(roll_deg=roll_deg)

    def publish_external_pressure(self, value_pa: int) -> None:
        self.tester.publish_external_pressure(value_pa)


@pytest.fixture
def acu_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = ACUNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# Pathfinding node harness
# ---------------------------------------------------------------------------


class _PathfindingTesterNode(Node):
    """Drives PathfindingNode and captures POSITION_TARGET emissions."""

    def __init__(self):
        super().__init__("pathfinding_node_tester")
        self.received_targets: list = []

        self.estimation_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.command_pub = create_publisher_for_topic(self, UUVTopics.COMMAND)
        self.path_pub = create_publisher_for_topic(self, UUVTopics.PATH)
        self.target_sub = create_subscription_for_topic(
            self, UUVTopics.POSITION_TARGET, self._on_target
        )

    def _on_target(self, msg: Pose) -> None:
        self.received_targets.append(msg)

    def publish_pose_estimation(self, x: float, y: float, z: float) -> None:
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.orientation.w = 1.0
        self.estimation_pub.publish(msg)

    def publish_external_pressure(self, value_pa: int) -> None:
        # EXTERNAL_PRESSURE is absolute Pa; the node converts to gauge.
        msg = Int32()
        msg.data = int(value_pa)
        self.external_pressure_pub.publish(msg)

    def publish_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)

    def publish_mission_command(
        self,
        mission_id: int,
        target_pressure_pa: float = 0.0,
        angle_rad: float = 0.0,
        n_resurfaces: int = 0,
    ) -> None:
        msg = MissionCommand()
        msg.mission_id = int(mission_id)
        msg.target_pressure_pa = float(target_pressure_pa)
        msg.angle_rad = float(angle_rad)
        msg.n_resurfaces = int(n_resurfaces)
        self.path_pub.publish(msg)


class PathfindingNodeHarness(NodeHarness):
    """NodeHarness specialised for PathfindingNode + _PathfindingTesterNode."""

    def __init__(self):
        super().__init__(PathfindingNode, _PathfindingTesterNode)

    @property
    def received_targets(self) -> list:
        return self.tester.received_targets

    def publish_pose_estimation(self, x: float, y: float, z: float) -> None:
        self.tester.publish_pose_estimation(x, y, z)

    def publish_external_pressure(self, value_pa: int) -> None:
        self.tester.publish_external_pressure(value_pa)

    def publish_command(self, command: str) -> None:
        self.tester.publish_command(command)

    def publish_mission_command(
        self,
        mission_id: int,
        target_pressure_pa: float = 0.0,
        angle_rad: float = 0.0,
        n_resurfaces: int = 0,
    ) -> None:
        self.tester.publish_mission_command(
            mission_id,
            target_pressure_pa=target_pressure_pa,
            angle_rad=angle_rad,
            n_resurfaces=n_resurfaces,
        )


@pytest.fixture
def pathfinding_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = PathfindingNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# BCU debug node harness
# ---------------------------------------------------------------------------


class _BcuDebugTesterNode(Node):
    """Drives BcuDebugNode and captures BCU_RPM + CONTROL_MANUAL_OVERRIDE
    emissions with timestamps.

    The debug node owns a duration timer, so tests need to reason about
    *when* each rpm value arrives -- a passthrough subscriber that only
    keeps the values would lose the "did the stop fire on time" signal."""

    def __init__(self):
        super().__init__("bcu_debug_tester")
        self.received: list[tuple[float, int]] = []
        self.override_events: list[tuple[float, bool]] = []

        self.cmd_pub = create_publisher_for_topic(self, UUVTopics.DEBUG_BCU_RPM)
        self.rpm_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_RPM, self._on_rpm
        )
        self.override_sub = create_subscription_for_topic(
            self, UUVTopics.CONTROL_MANUAL_OVERRIDE, self._on_override
        )

    def _on_rpm(self, msg: Int16) -> None:
        self.received.append((time.monotonic(), int(msg.data)))

    def _on_override(self, msg: Bool) -> None:
        self.override_events.append((time.monotonic(), bool(msg.data)))

    def publish_pump(self, rpm: int, duration_s: float) -> None:
        msg = BcuPumpCommand()
        msg.rpm = int(rpm)
        msg.duration_s = float(duration_s)
        self.cmd_pub.publish(msg)


class BcuDebugNodeHarness(NodeHarness):
    """NodeHarness specialised for BcuDebugNode + _BcuDebugTesterNode."""

    def __init__(self):
        super().__init__(BcuDebugNode, _BcuDebugTesterNode)

    @property
    def received(self) -> list[tuple[float, int]]:
        return self.tester.received

    @property
    def received_rpm(self) -> list[int]:
        return [rpm for _, rpm in self.tester.received]

    @property
    def override_events(self) -> list[tuple[float, bool]]:
        return self.tester.override_events

    @property
    def override_states(self) -> list[bool]:
        return [state for _, state in self.tester.override_events]

    def publish_pump(self, rpm: int, duration_s: float) -> None:
        self.tester.publish_pump(rpm, duration_s)


@pytest.fixture
def bcu_debug_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = BcuDebugNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()
