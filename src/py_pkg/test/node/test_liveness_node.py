"""Tier 2 in-process rclpy test for LivenessNode.

Spins the node alongside a tester that feeds a couple of source topics and
captures the published DiagnosticArray. Assertions key off
``DiagnosticStatus.message`` ("online"/"offline") -- the same field the operator
UI reads across the tether, since the byte ``level`` doesn't survive the JSON
egress as a number.
"""

import pytest
from diagnostic_msgs.msg import DiagnosticArray
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

from conftest import NodeHarness

STALENESS_S = 0.5
ALL_ROWS = {
    "acu_pitch",
    "acu_roll",
    "bcu_pump",
    "bcu_valve_1",
    "bcu_valve_2",
    "imu",
    "external_pressure",
    "tank_pressure",
}


class _LivenessTesterNode(Node):
    """Feeds a couple of liveness sources and captures /status/liveness."""

    def __init__(self):
        super().__init__("liveness_tester")
        self.latest: DiagnosticArray | None = None
        # One physical IMU now -> a single "imu" subsystem row fed by IMU.
        self.imu_pub = create_publisher_for_topic(self, UUVTopics.IMU)
        self.valves_pub = create_publisher_for_topic(
            self, UUVTopics.BCU_FEEDBACK_VALVES
        )
        self.liveness_sub = create_subscription_for_topic(
            self, UUVTopics.STATUS_LIVENESS, self._on_liveness
        )

    def _on_liveness(self, msg: DiagnosticArray) -> None:
        self.latest = msg

    def feed_imu(self) -> None:
        self.imu_pub.publish(Imu())

    def feed_valves(self) -> None:
        m = UInt8()
        m.data = 0
        self.valves_pub.publish(m)


class LivenessNodeHarness(NodeHarness):
    """NodeHarness specialised for LivenessNode (needs parameter overrides)."""

    def __init__(self):
        super().__init__(
            lambda: LivenessNode(
                parameter_overrides=[
                    Parameter(
                        "staleness_timeout_s", Parameter.Type.DOUBLE, STALENESS_S
                    ),
                    Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 20.0),
                ]
            ),
            _LivenessTesterNode,
        )

    def states(self) -> dict[str, str]:
        if self.tester.latest is None:
            return {}
        return {s.name: s.message for s in self.tester.latest.status}


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
            h.tester.feed_imu()
            h.tester.feed_valves()
            h.spin_for(0.05)
        states = h.states()
        assert states["imu"] == "online"
        # One bitmask proves both valves alive.
        assert states["bcu_valve_1"] == "online"
        assert states["bcu_valve_2"] == "online"
        # An unfed source stays offline.
        assert states["bcu_pump"] == "offline"

    def test_source_goes_offline_after_staleness(self, liveness_harness):
        h = liveness_harness
        for _ in range(4):
            h.tester.feed_imu()
            h.spin_for(0.05)
        h.spin_until(lambda: h.states().get("imu") == "online", timeout=1.0)
        # Stop feeding; after the staleness window it must flip back to offline.
        h.spin_until(
            lambda: h.states().get("imu") == "offline",
            timeout=STALENESS_S + 1.0,
        )
        assert h.states()["imu"] == "offline"
