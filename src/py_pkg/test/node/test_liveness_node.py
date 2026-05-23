"""Tier 2 in-process rclpy test for LivenessNode.

Spins the node alongside a tester that feeds a couple of source topics and
captures the published DiagnosticArray. Assertions key off
``DiagnosticStatus.message`` ("online"/"offline") -- the same field the operator
UI reads across the tether, since the byte ``level`` doesn't survive the JSON
egress as a number.
"""

import time

import pytest
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from std_msgs.msg import UInt8

from py_pkg.liveness.liveness_node import LivenessNode
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)

STALENESS_S = 0.5
ALL_ROWS = {
    "acu_pitch",
    "acu_roll",
    "bcu_pump",
    "bcu_valve_1",
    "bcu_valve_2",
    "imu_left",
    "imu_right",
    "external_pressure",
    "tank_pressure",
}


class _LivenessTesterNode(Node):
    """Feeds a couple of liveness sources and captures /status/liveness."""

    def __init__(self):
        super().__init__("liveness_tester")
        self.latest: DiagnosticArray | None = None
        self.imu_left_pub = create_publisher_for_topic(self, UUVTopics.IMU_LEFT)
        self.valves_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_FEEDBACK_VALVES
        )
        self.liveness_sub = create_subscription_for_topic(
            self, UUVTopics.STATUS_LIVENESS, self._on_liveness
        )

    def _on_liveness(self, msg: DiagnosticArray) -> None:
        self.latest = msg

    def feed_imu_left(self) -> None:
        self.imu_left_pub.publish(Imu())

    def feed_valves(self) -> None:
        m = UInt8()
        m.data = 0
        self.valves_pub.publish(m)


class LivenessNodeHarness:
    def __init__(self):
        self.node = LivenessNode(
            parameter_overrides=[
                Parameter(
                    "staleness_timeout_s", Parameter.Type.DOUBLE, STALENESS_S
                ),
                Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 20.0),
            ]
        )
        self.tester = _LivenessTesterNode()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.tester)

    def spin_for(self, duration_s: float, slice_s: float = 0.02) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def spin_until(self, predicate, timeout: float = 2.0, slice_s: float = 0.02):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.executor.spin_once(timeout_sec=slice_s)
        if not predicate():
            raise TimeoutError(f"predicate did not become true within {timeout}s")

    def states(self) -> dict[str, str]:
        if self.tester.latest is None:
            return {}
        return {s.name: s.message for s in self.tester.latest.status}

    def shutdown(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.executor.remove_node(self.tester)
        finally:
            self.node.destroy_node()
            self.tester.destroy_node()
            self.executor.shutdown()


@pytest.fixture
def liveness_harness():
    harness = LivenessNodeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


class TestLivenessTransitions:
    def test_idle_reports_full_row_set_all_offline(self, liveness_harness):
        h = liveness_harness
        h.spin_until(lambda: h.tester.latest is not None, timeout=2.0)
        states = h.states()
        # Every subsystem row is present, and all read offline before any
        # source has fed the watchdog.
        assert set(states) == ALL_ROWS
        assert all(v == "offline" for v in states.values())

    def test_fed_source_goes_online(self, liveness_harness):
        h = liveness_harness
        for _ in range(6):
            h.tester.feed_imu_left()
            h.tester.feed_valves()
            h.spin_for(0.05)
        states = h.states()
        assert states["imu_left"] == "online"
        # One bitmask proves both valves alive.
        assert states["bcu_valve_1"] == "online"
        assert states["bcu_valve_2"] == "online"
        # An unfed source stays offline.
        assert states["bcu_pump"] == "offline"

    def test_source_goes_offline_after_staleness(self, liveness_harness):
        h = liveness_harness
        for _ in range(4):
            h.tester.feed_imu_left()
            h.spin_for(0.05)
        h.spin_until(lambda: h.states().get("imu_left") == "online", timeout=1.0)
        # Stop feeding; after the staleness window it must flip back to offline.
        h.spin_until(
            lambda: h.states().get("imu_left") == "offline",
            timeout=STALENESS_S + 1.0,
        )
        assert h.states()["imu_left"] == "offline"
