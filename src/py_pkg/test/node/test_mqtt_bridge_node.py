"""Tier 2 in-process rclpy tests for MqttBridge.

The bridge talks to a real paho MQTT client in production. In tests we
inject a fake via the ``mqtt_client_factory`` constructor hook -- so the
node spins under a SingleThreadedExecutor exactly like the others, but
``_mqtt.publish`` records into an in-memory list instead of going over
TCP. ``on_message`` is the bridge's own method and can be called
directly to drive the ingress path.

Three behaviours under test:

* Egress: every published ROS message on a mapped topic appears as an
  MQTT publish on the mirror topic, with QoS 0 and JSON body matching
  the ROS fields.
* Throttling: the per-topic rate cap drops messages that arrive within
  the period; on-change topics dedupe identical payloads.
* Mission mirror: ingress on ``nautilus/cmd/path`` +
  ``nautilus/cmd/command`` updates retained
  ``nautilus/telemetry/mission/active`` with the right IDLE / LOADED /
  RUNNING transitions.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest
import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Int32, UInt8

from py_pkg.mqtt.mqtt_bridge_node import (
    EGRESS_MAP,
    MISSION_ACTIVE_TOPIC,
    MqttBridge,
)
from py_pkg.uuv_ros_core import UUVTopics, create_publisher_for_topic


@dataclass
class FakePublish:
    topic: str
    payload: Any
    qos: int
    retain: bool


class FakeMqttClient:
    """Records publishes; no-ops everything else.

    Mirrors the paho-mqtt v2 surface the bridge actually uses. We do not
    emulate the broker (no echo, no retained-store), but the bridge
    doesn't need that to test its egress + mirror semantics. Ingress
    tests drive ``bridge._on_mqtt_message`` directly.
    """

    def __init__(self, *_args, **_kwargs):
        self.published: list[FakePublish] = []
        self.subscribed: list[tuple[str, int]] = []
        self.will_topic: str | None = None
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None

    def will_set(self, topic, payload=None, qos=0, retain=False):
        self.will_topic = topic

    def reconnect_delay_set(self, *_args, **_kwargs):
        pass

    def connect_async(self, *_args, **_kwargs):
        pass

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, qos=0):
        self.subscribed.append((topic, qos))

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append(
            FakePublish(topic=topic, payload=payload, qos=qos, retain=retain)
        )

    # --- test helpers ---------------------------------------------------

    def publishes_on(self, topic: str) -> list[FakePublish]:
        return [p for p in self.published if p.topic == topic]

    def last_payload_on(self, topic: str) -> dict | None:
        items = self.publishes_on(topic)
        if not items:
            return None
        body = items[-1].payload
        return json.loads(body) if isinstance(body, str) else body


class _BridgeTesterNode(Node):
    """Publishes onto every ROS topic the bridge mirrors out.

    QoS auto-matches via uuv_ros_core; if QoS drifts on either side this
    node will silently drop messages, which is precisely what we want
    surfaced as a test failure.
    """

    def __init__(self):
        super().__init__("mqtt_bridge_tester")
        self.pose_estimation_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_ESTIMATION
        )
        self.position_target_pub = create_publisher_for_topic(
            self, UUVTopics.POSITION_TARGET
        )
        self.imu_left_pub = create_publisher_for_topic(
            self, UUVTopics.IMU_FILTERED_LEFT
        )
        self.imu_right_pub = create_publisher_for_topic(
            self, UUVTopics.IMU_FILTERED_RIGHT
        )
        self.bcu_pressure_pub = create_publisher_for_topic(self, UUVTopics.BCU_PRESSURE)
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.bcu_rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.bcu_valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        self.acu_pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)
        self.acu_roll_pub = create_publisher_for_topic(self, UUVTopics.ACU_ROLL)

    @staticmethod
    def _int32(v: int) -> Int32:
        m = Int32()
        m.data = int(v)
        return m

    @staticmethod
    def _int16(v: int) -> Int16:
        m = Int16()
        m.data = int(v)
        return m

    @staticmethod
    def _uint8(v: int) -> UInt8:
        m = UInt8()
        m.data = int(v)
        return m


class MqttBridgeHarness:
    """Spins MqttBridge + a tester node behind a SingleThreadedExecutor."""

    def __init__(self):
        self.fake = FakeMqttClient()
        self.node = MqttBridge(mqtt_client_factory=lambda client_id: self.fake)
        self.tester = _BridgeTesterNode()
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

    def receive_mqtt(self, topic: str, payload: dict) -> None:
        """Simulate a broker -> bridge message arrival."""

        class _Msg:
            pass

        msg = _Msg()
        msg.topic = topic
        msg.payload = json.dumps(payload).encode("utf-8")
        self.node._on_mqtt_message(self.fake, None, msg)

    def shutdown(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.executor.remove_node(self.tester)
        finally:
            self.node.destroy_node()
            self.tester.destroy_node()
            self.executor.shutdown()


@pytest.fixture
def bridge_harness():
    harness = MqttBridgeHarness()
    try:
        yield harness
    finally:
        harness.shutdown()


# ---------------------------------------------------------------------------
# Wiring smoke
# ---------------------------------------------------------------------------


class TestEgressWiring:
    def test_every_egress_topic_has_a_subscription(self, bridge_harness):
        # The bridge should subscribe to each ROS source topic in EGRESS_MAP.
        ros_topics = {s.topic_name for s in bridge_harness.node.subscriptions}
        for em in EGRESS_MAP:
            assert em.ros_topic in ros_topics, (
                f"missing ROS subscription for {em.ros_topic}"
            )


# ---------------------------------------------------------------------------
# Egress per topic
# ---------------------------------------------------------------------------


class TestScalarEgress:
    def test_bcu_pressure_publishes_qos0_json(self, bridge_harness):
        h = bridge_harness
        h.tester.bcu_pressure_pub.publish(_BridgeTesterNode._int32(123_456))
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/bcu/pressure"),
            timeout=1.0,
        )
        last = h.fake.publishes_on("nautilus/telemetry/bcu/pressure")[-1]
        assert last.qos == 0
        assert last.retain is False
        assert json.loads(last.payload) == {"data": 123_456}

    def test_external_pressure_publishes(self, bridge_harness):
        h = bridge_harness
        h.tester.external_pressure_pub.publish(_BridgeTesterNode._int32(180_000))
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/external/pressure"),
            timeout=1.0,
        )
        assert h.fake.last_payload_on("nautilus/telemetry/external/pressure") == {
            "data": 180_000
        }

    def test_bcu_rpm_publishes(self, bridge_harness):
        h = bridge_harness
        h.tester.bcu_rpm_pub.publish(_BridgeTesterNode._int16(2500))
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/bcu/rpm"), timeout=1.0
        )
        assert h.fake.last_payload_on("nautilus/telemetry/bcu/rpm") == {"data": 2500}

    def test_acu_pitch_publishes(self, bridge_harness):
        h = bridge_harness
        h.tester.acu_pitch_pub.publish(_BridgeTesterNode._int16(35))
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/acu/pitch"), timeout=1.0
        )
        assert h.fake.last_payload_on("nautilus/telemetry/acu/pitch") == {"data": 35}

    def test_acu_roll_publishes(self, bridge_harness):
        h = bridge_harness
        h.tester.acu_roll_pub.publish(_BridgeTesterNode._int16(-450))
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/acu/roll"), timeout=1.0
        )
        assert h.fake.last_payload_on("nautilus/telemetry/acu/roll") == {"data": -450}


class TestStructuredEgress:
    def test_pose_estimation_round_trips(self, bridge_harness):
        h = bridge_harness
        p = Pose()
        p.position.x = 1.0
        p.position.y = 2.0
        p.position.z = -5.0
        p.orientation.w = 1.0
        h.tester.pose_estimation_pub.publish(p)
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/position/estimation"),
            timeout=1.0,
        )
        payload = h.fake.last_payload_on("nautilus/telemetry/position/estimation")
        assert payload["position"]["x"] == pytest.approx(1.0)
        assert payload["position"]["z"] == pytest.approx(-5.0)
        assert payload["orientation"]["w"] == pytest.approx(1.0)

    def test_imu_left_carries_header_and_axes(self, bridge_harness):
        h = bridge_harness
        m = Imu()
        m.header.frame_id = "imu_left"
        m.orientation.w = 1.0
        m.angular_velocity.x = 0.05
        m.linear_acceleration.z = -9.81
        h.tester.imu_left_pub.publish(m)
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/imu/left"), timeout=1.0
        )
        payload = h.fake.last_payload_on("nautilus/telemetry/imu/left")
        assert payload["header"]["frame_id"] == "imu_left"
        assert payload["angular_velocity"]["x"] == pytest.approx(0.05)
        assert payload["linear_acceleration"]["z"] == pytest.approx(-9.81)


# ---------------------------------------------------------------------------
# Throttle / on-change
# ---------------------------------------------------------------------------


class TestThrottle:
    def test_rapid_publishes_are_rate_capped(self, bridge_harness):
        """bcu/pressure is capped at 10 Hz (100 ms period); 6 rapid
        publishes spaced ~25 ms apart over ~150 ms should yield at most
        2 MQTT publishes -- one at t=0 and at most one more after the
        period elapses."""
        h = bridge_harness
        for v in range(6):
            h.tester.bcu_pressure_pub.publish(_BridgeTesterNode._int32(100_000 + v))
            h.spin_for(0.025)
        # First passes; subsequent within the period are dropped. 1-2 is
        # acceptable depending on timing jitter, but never 6.
        n = len(h.fake.publishes_on("nautilus/telemetry/bcu/pressure"))
        assert 1 <= n <= 2, f"expected throttled to 1-2 publishes, got {n}"


class TestOnChange:
    def test_valves_dedupe_identical_payloads(self, bridge_harness):
        """valves is on_change + retained: 5 publishes of the same bitmap
        should emit exactly one MQTT message."""
        h = bridge_harness
        for _ in range(5):
            h.tester.bcu_valves_pub.publish(_BridgeTesterNode._uint8(0b10))
            h.spin_for(0.02)
        publishes = h.fake.publishes_on("nautilus/telemetry/bcu/valves")
        assert len(publishes) == 1
        assert publishes[0].retain is True
        assert publishes[0].qos == 0
        assert json.loads(publishes[0].payload) == {"data": 2}

    def test_valves_emits_again_when_bitmap_changes(self, bridge_harness):
        h = bridge_harness
        h.tester.bcu_valves_pub.publish(_BridgeTesterNode._uint8(0b00))
        h.spin_for(0.05)
        h.tester.bcu_valves_pub.publish(_BridgeTesterNode._uint8(0b10))
        h.spin_for(0.05)
        h.tester.bcu_valves_pub.publish(_BridgeTesterNode._uint8(0b10))  # dup
        h.spin_for(0.05)
        publishes = h.fake.publishes_on("nautilus/telemetry/bcu/valves")
        # 00 -> 10 emits twice (initial + change). The duplicate 10 is
        # suppressed by the dedupe.
        assert len(publishes) == 2

    def test_position_target_dedupe_then_change(self, bridge_harness):
        """position/target is on_change + retained; identical Pose dropped."""
        h = bridge_harness
        p1 = Pose()
        p1.position.z = 75383.0
        p1.orientation.w = 1.0
        h.tester.position_target_pub.publish(p1)
        h.spin_for(0.05)
        h.tester.position_target_pub.publish(p1)  # identical
        h.spin_for(0.05)
        p2 = Pose()
        p2.position.z = 100_000.0
        p2.orientation.w = 1.0
        h.tester.position_target_pub.publish(p2)
        h.spin_for(0.05)
        publishes = h.fake.publishes_on("nautilus/telemetry/position/target")
        assert len(publishes) == 2
        assert all(p.retain is True for p in publishes)


# ---------------------------------------------------------------------------
# Mission-active mirror
# ---------------------------------------------------------------------------


def _mission_states(harness) -> list[str]:
    """All mission-active state values published so far, in order."""
    return [
        json.loads(p.payload)["state"]
        for p in harness.fake.publishes_on(MISSION_ACTIVE_TOPIC)
    ]


class TestMissionMirror:
    def test_path_alone_loads(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(
            "nautilus/cmd/path",
            {
                "mission_id": 0,
                "target_pressure_pa": 75383.0,
                "angle_rad": 0.0,
                "n_resurfaces": 0,
            },
        )
        states = _mission_states(h)
        assert states[-1] == "LOADED"
        last = json.loads(h.fake.publishes_on(MISSION_ACTIVE_TOPIC)[-1].payload)
        assert last["mission_id"] == 0
        assert last["target_pressure_pa"] == pytest.approx(75383.0)

    def test_start_then_running(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(
            "nautilus/cmd/path",
            {
                "mission_id": 1,
                "target_pressure_pa": 147150.0,
                "angle_rad": 0.6109,
                "n_resurfaces": 2,
            },
        )
        h.receive_mqtt("nautilus/cmd/command", {"data": "start"})
        states = _mission_states(h)
        assert states[-1] == "RUNNING"

    def test_stop_returns_to_idle_but_keeps_cache(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(
            "nautilus/cmd/path",
            {"mission_id": 0, "target_pressure_pa": 75383.0,
             "angle_rad": 0.0, "n_resurfaces": 0},
        )
        h.receive_mqtt("nautilus/cmd/command", {"data": "start"})
        h.receive_mqtt("nautilus/cmd/command", {"data": "stop"})
        last = json.loads(h.fake.publishes_on(MISSION_ACTIVE_TOPIC)[-1].payload)
        assert last["state"] == "IDLE"
        # Stop preserves the cached mission so the UI can show what was
        # loaded; abort would clear it.
        assert last["mission_id"] == 0

    def test_abort_clears_cache(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(
            "nautilus/cmd/path",
            {"mission_id": 0, "target_pressure_pa": 75383.0,
             "angle_rad": 0.0, "n_resurfaces": 0},
        )
        h.receive_mqtt("nautilus/cmd/command", {"data": "start"})
        h.receive_mqtt("nautilus/cmd/command", {"data": "abort"})
        last = json.loads(h.fake.publishes_on(MISSION_ACTIVE_TOPIC)[-1].payload)
        assert last["state"] == "IDLE"
        assert last["mission_id"] is None

    def test_start_without_loaded_mission_is_noop(self, bridge_harness):
        """A stray ``start`` arriving before ``/path`` is buffered by the
        pathfinder (it has start_pending logic) but the mirror should not
        claim RUNNING -- there is no mission to run."""
        h = bridge_harness
        # Note: pre-existing seed publish on connect won't have fired
        # here because the fake's on_connect was never invoked.
        h.receive_mqtt("nautilus/cmd/command", {"data": "start"})
        states = _mission_states(h)
        # Either no mirror publish at all (nothing changed) or, if one
        # exists, it must not be RUNNING.
        assert "RUNNING" not in states
