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
import time

import pytest
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32, Int32

from py_pkg.pid.acu_node import ACUControlNode
from py_pkg.pid.depth_node import DepthControlNode
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


@pytest.fixture(scope="session", autouse=True)
def rclpy_session():
    """Init/shutdown rclpy exactly once per test session."""
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
            raise TimeoutError(
                f"predicate did not become true within {timeout}s"
            )

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
    """Drives DepthControlNode and captures BCU_RPM emissions."""

    def __init__(self):
        super().__init__("depth_node_tester")
        self.received_rpm: list[int] = []

        self.target_depth_pub = create_publisher_for_topic(
            self, UUVTopics.TARGET_DEPTH
        )
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.bcu_rpm_sub = create_subscription_for_topic(
            self, UUVTopics.BCU_RPM, self._on_rpm
        )

    def _on_rpm(self, msg: Int32) -> None:
        self.received_rpm.append(int(msg.data))

    def publish_target_depth(self, value: float) -> None:
        msg = Float32()
        msg.data = float(value)
        self.target_depth_pub.publish(msg)

    def publish_external_pressure(self, value_pa: int) -> None:
        msg = Int32()
        msg.data = int(value_pa)
        self.external_pressure_pub.publish(msg)


class DepthNodeHarness(NodeHarness):
    """NodeHarness specialised for DepthControlNode + _DepthTesterNode."""

    def __init__(self):
        super().__init__(DepthControlNode, _DepthTesterNode)

    @property
    def received_rpm(self) -> list[int]:
        return self.tester.received_rpm

    def publish_target_depth(self, value: float) -> None:
        self.tester.publish_target_depth(value)

    def publish_external_pressure(self, value_pa: int) -> None:
        self.tester.publish_external_pressure(value_pa)


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


def _quat_from_roll_pitch_deg(roll_deg: float, pitch_deg: float):
    """Build a unit quaternion (x, y, z, w) for the given roll/pitch (yaw=0).

    Inverse of math_utils.quaternion_to_roll_pitch under ZYX Tait-Bryan with
    yaw=0; verified to round-trip exactly for the angles used in tests.
    """
    r = math.radians(roll_deg) / 2.0
    p = math.radians(pitch_deg) / 2.0
    qw = math.cos(r) * math.cos(p)
    qx = math.sin(r) * math.cos(p)
    qy = math.cos(r) * math.sin(p)
    qz = -math.sin(r) * math.sin(p)
    return qx, qy, qz, qw


class _ACUTesterNode(Node):
    """Drives ACUControlNode and captures ACU_PITCH_STEPS / ACU_ROLL_STEPS."""

    def __init__(self):
        super().__init__("acu_node_tester")
        self.received_pitch_steps: list[int] = []
        self.received_roll_steps: list[int] = []

        self.target_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )
        self.estimation_pose_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.pitch_steps_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_PITCH_STEPS, self._on_pitch
        )
        self.roll_steps_sub = create_subscription_for_topic(
            self, UUVTopics.ACU_ROLL_STEPS, self._on_roll
        )

    def _on_pitch(self, msg: Int32) -> None:
        self.received_pitch_steps.append(int(msg.data))

    def _on_roll(self, msg: Int32) -> None:
        self.received_roll_steps.append(int(msg.data))

    @staticmethod
    def _pose_from_roll_pitch(roll_deg: float, pitch_deg: float) -> Pose:
        qx, qy, qz, qw = _quat_from_roll_pitch_deg(roll_deg, pitch_deg)
        msg = Pose()
        msg.orientation.x = float(qx)
        msg.orientation.y = float(qy)
        msg.orientation.z = float(qz)
        msg.orientation.w = float(qw)
        return msg

    def publish_target_attitude(self, roll_deg: float, pitch_deg: float) -> None:
        self.target_pose_pub.publish(
            self._pose_from_roll_pitch(roll_deg, pitch_deg)
        )

    def publish_current_attitude(self, roll_deg: float, pitch_deg: float) -> None:
        self.estimation_pose_pub.publish(
            self._pose_from_roll_pitch(roll_deg, pitch_deg)
        )


class ACUNodeHarness(NodeHarness):
    """NodeHarness specialised for ACUControlNode + _ACUTesterNode."""

    def __init__(self):
        super().__init__(ACUControlNode, _ACUTesterNode)

    @property
    def received_pitch_steps(self) -> list[int]:
        return self.tester.received_pitch_steps

    @property
    def received_roll_steps(self) -> list[int]:
        return self.tester.received_roll_steps

    def publish_target_attitude(self, roll_deg: float, pitch_deg: float) -> None:
        self.tester.publish_target_attitude(roll_deg, pitch_deg)

    def publish_current_attitude(self, roll_deg: float, pitch_deg: float) -> None:
        self.tester.publish_current_attitude(roll_deg, pitch_deg)


@pytest.fixture
def acu_node_harness():
    """Function-scoped harness. Tears both nodes down on exit."""
    harness = ACUNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()
