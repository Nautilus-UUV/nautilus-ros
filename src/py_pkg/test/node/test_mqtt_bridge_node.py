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
* Lifeguard: the deploy-time dead-man failsafe. Armed via
  ``nautilus/cmd/lifeguard`` and fed by ``nautilus/cmd/heartbeat``, both
  consumed by the bridge itself; heartbeat silence past the (shortened)
  window must stop the mission and latch ``/debug/emergency_surface``,
  and only an explicit disarm stands it down.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

import pytest
from geometry_msgs.msg import Pose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from std_msgs.msg import Int16, Int32, UInt8

from py_pkg.mqtt.mqtt_bridge_node import (
    COMMAND_CMD_TOPIC,
    DEBUG_RESET_CMD_TOPIC,
    EGRESS_MAP,
    EMERGENCY_SURFACE_CMD_TOPIC,
    INGRESS_MAP,
    INIT_CMD_TOPIC,
    LIFEGUARD_CMD_TOPIC,
    LIFEGUARD_HEARTBEAT_TOPIC,
    LIFEGUARD_STATUS_TOPIC,
    MISSION_ACTIVE_TOPIC,
    PATH_CMD_TOPIC,
    STATUS_ONLINE,
    STATUS_TOPIC,
    MqttBridge,
)
from py_pkg.scenarios.spec.rig import PlantSpec
from py_pkg.uuv_ros_core import (
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


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
        # paho drops QoS-0 publishes while the socket is down. Default True so
        # every existing test runs "connected"; the late-broker regression
        # test flips this to reproduce a publish that never lands.
        self.connected = True

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

    def is_connected(self):
        # The reconnect watchdog reads this as paho's own link view. Tests flip
        # `connected` to simulate a drop or a stale-but-claimed connection.
        return self.connected

    def subscribe(self, topic, qos=0):
        self.subscribed.append((topic, qos))

    def publish(self, topic, payload=None, qos=0, retain=False):
        # Mirror paho: a QoS-0 publish on a down socket is dropped, not
        # queued. retain=True only seeds the broker if it actually reaches it.
        if not self.connected:
            return
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
        self.imu_pub = create_publisher_for_topic(self, UUVTopics.IMU_FILTERED)
        self.bcu_pressure_pub = create_publisher_for_topic(self, UUVTopics.BCU_PRESSURE)
        self.external_pressure_pub = create_publisher_for_topic(
            self, UUVTopics.EXTERNAL_PRESSURE
        )
        self.bcu_rpm_pub = create_publisher_for_topic(self, UUVTopics.BCU_RPM)
        self.bcu_valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        self.acu_pitch_pub = create_publisher_for_topic(self, UUVTopics.ACU_PITCH)
        self.acu_roll_pub = create_publisher_for_topic(self, UUVTopics.ACU_ROLL)

        # Lifeguard observers: the failsafe fires through the bridge's own
        # ingress publishers, so the tester records what lands ROS-side.
        self.received_emergency: list[bool] = []
        self.received_command: list[bool] = []
        self.emergency_sub = create_subscription_for_topic(
            self, UUVTopics.DEBUG_EMERGENCY_SURFACE, self._on_emergency
        )
        self.command_sub = create_subscription_for_topic(
            self, UUVTopics.COMMAND, self._on_command
        )

        # Dive-init observer: ingress on nautilus/cmd/init must land a
        # DiveInit on the ROS graph.
        self.received_init: list = []
        self.init_sub = create_subscription_for_topic(
            self, UUVTopics.DIVE_INIT, self._on_init
        )

        # Mission observer: ingress on nautilus/cmd/path must land a
        # MissionCommand on /path with all fields decoded.
        self.received_path: list = []
        self.path_sub = create_subscription_for_topic(
            self, UUVTopics.PATH, self._on_path
        )

    def _on_emergency(self, msg) -> None:
        self.received_emergency.append(bool(msg.data))

    def _on_command(self, msg) -> None:
        self.received_command.append(bool(msg.data))

    def _on_init(self, msg) -> None:
        self.received_init.append(msg)

    def _on_path(self, msg) -> None:
        self.received_path.append(msg)

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


# Production ticks the lifeguard at 1 Hz; that would floor every engage
# and every "stays quiet for several ticks" window at whole seconds, so
# all harnesses run a fast tick.
TICK_PERIOD_S = 0.1


class MqttBridgeHarness:
    """Spins MqttBridge + a tester node behind a SingleThreadedExecutor."""

    def __init__(
        self,
        lifeguard_timeout_s: float | None = None,
        reconnect_grace_s: float | None = None,
        rx_silence_s: float | None = None,
        watchdog_period_s: float | None = None,
    ):
        # The factory builds a fresh fake per call and records them, so a
        # reconnect-watchdog rebuild is observable as a new entry. `fake` stays
        # the first (initial) client, which is what the egress/mirror/lifeguard
        # tests drive.
        self.fakes: list[FakeMqttClient] = []

        def _factory(_client_id):
            fake = FakeMqttClient()
            self.fakes.append(fake)
            return fake

        overrides = [
            Parameter("lifeguard_tick_period_s", Parameter.Type.DOUBLE, TICK_PERIOD_S)
        ]
        # Park the watchdog timer far out by default so it never fires mid-test;
        # the reconnect tests drive node._connection_watchdog() directly.
        overrides.append(
            Parameter(
                "connection_watchdog_period_s",
                Parameter.Type.DOUBLE,
                watchdog_period_s if watchdog_period_s is not None else 1000.0,
            )
        )
        if lifeguard_timeout_s is not None:
            overrides.append(
                Parameter(
                    "lifeguard_timeout_s",
                    Parameter.Type.DOUBLE,
                    lifeguard_timeout_s,
                )
            )
        if reconnect_grace_s is not None:
            overrides.append(
                Parameter("reconnect_grace_s", Parameter.Type.DOUBLE, reconnect_grace_s)
            )
        if rx_silence_s is not None:
            overrides.append(
                Parameter("rx_silence_s", Parameter.Type.DOUBLE, rx_silence_s)
            )
        self.node = MqttBridge(
            mqtt_client_factory=_factory,
            parameter_overrides=overrides,
        )
        self.fake = self.fakes[0]
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


@pytest.fixture
def lifeguard_harness():
    """Bridge harness with a short dead-man window so engage tests stay
    fast; with the fast tick, engage lands within ~0.6 s of arming."""
    harness = MqttBridgeHarness(lifeguard_timeout_s=0.5)
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
            assert (
                em.ros_topic in ros_topics
            ), f"missing ROS subscription for {em.ros_topic}"


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

    def test_imu_sends_only_axes(self, bridge_harness):
        # The filtered IMU egresses compactly: only angular_velocity and
        # linear_acceleration cross the tether. orientation, the covariance
        # arrays, and the header are trimmed (nothing past the tether reads
        # them, and dropping them keeps the DB from logging dead channels).
        h = bridge_harness
        m = Imu()
        m.header.frame_id = "imu"
        m.orientation.w = 1.0
        m.angular_velocity.x = 0.05
        m.linear_acceleration.z = -9.81
        h.tester.imu_pub.publish(m)
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/imu"), timeout=1.0
        )
        payload = h.fake.last_payload_on("nautilus/telemetry/imu")
        assert payload["angular_velocity"]["x"] == pytest.approx(0.05)
        assert payload["linear_acceleration"]["z"] == pytest.approx(-9.81)
        # Trimming contract: the fat fields must not be on the wire.
        assert payload.keys() == {"angular_velocity", "linear_acceleration"}
        assert "header" not in payload
        assert "orientation" not in payload
        assert "orientation_covariance" not in payload
        assert "angular_velocity_covariance" not in payload
        assert "linear_acceleration_covariance" not in payload


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
# Late-broker re-seed (regression)
# ---------------------------------------------------------------------------


class TestReconnectReseedsRetainedEgress:
    """The vehicle's control stack (this bridge included) routinely boots
    before the laptop broker; same shape on a tether reconnect after the
    broker restarted and lost its retained store. paho drops QoS-0 publishes
    while the socket is down, but the on-change dedup still caches them -- so
    without a re-seed on connect the broker ends up with NO retained copy of
    a state-like topic, every byte-identical follow-up is suppressed as "no
    change", and a UI tab joining later renders the whole liveness grid
    offline while Tether sits green. _on_connect must re-publish the cached
    on-change payloads so late joiners get real state.
    """

    def test_onchange_encoded_while_down_is_reseeded_on_connect(
        self, bridge_harness
    ):
        h = bridge_harness
        # Broker not up yet (bridge started first).
        h.fake.connected = False

        # A valves transition the bridge encodes now is dropped on the wire
        # but still caches in _last_payload -- valves stands in for liveness
        # here: identical on-change/retained code path, already wired in the
        # tester.
        h.tester.bcu_valves_pub.publish(_BridgeTesterNode._uint8(0b10))
        h.spin_for(0.1)
        assert h.fake.publishes_on("nautilus/telemetry/bcu/valves") == [], (
            "publish while disconnected must not reach the broker"
        )

        # Broker comes up; paho fires on_connect. The re-seed must restore the
        # retained valves payload even though the cached frame never changes
        # again (no further publishes from the tester).
        h.fake.connected = True
        h.node._on_connect(h.fake, None, None, 0)

        seeded = h.fake.publishes_on("nautilus/telemetry/bcu/valves")
        assert len(seeded) == 1, "on_connect must re-seed the cached on-change payload"
        assert seeded[-1].retain is True
        assert json.loads(seeded[-1].payload) == {"data": 2}

    def test_reconnect_reseeds_without_a_fresh_ros_message(self, bridge_harness):
        # Connected the whole time: the first frame lands normally, then the
        # tether drops and the broker restarts (retained store wiped). A bare
        # reconnect -- no new ROS message, the value hasn't changed -- must
        # still restore the retained copy from cache.
        h = bridge_harness
        h.tester.bcu_valves_pub.publish(_BridgeTesterNode._uint8(0b01))
        h.spin_until(
            lambda: h.fake.publishes_on("nautilus/telemetry/bcu/valves"), timeout=1.0
        )
        before = len(h.fake.publishes_on("nautilus/telemetry/bcu/valves"))

        h.node._on_connect(h.fake, None, None, 0)

        after = h.fake.publishes_on("nautilus/telemetry/bcu/valves")
        assert len(after) == before + 1
        assert after[-1].retain is True
        assert json.loads(after[-1].payload) == {"data": 1}


# ---------------------------------------------------------------------------
# Mission-active mirror
# ---------------------------------------------------------------------------


def _mission_states(harness) -> list[str]:
    """All mission-active state values published so far, in order."""
    return [
        json.loads(p.payload)["state"]
        for p in harness.fake.publishes_on(MISSION_ACTIVE_TOPIC)
    ]


# The stock mission dispatch used wherever a test just needs "a mission
# is loaded" -- TRIM at the nominal target. Keys mirror
# nautilus_msgs/MissionCommand exactly; an unknown field would fail
# set_message_fields and drop the command before the mirror sees it.
PATH_PAYLOAD = {
    "mission_id": 0,
    "target_pressure_pa": 75383.0,
    "shallow_pressure_pa": 0.0,
    "angle_rad": 0.0,
    "n_oscillations": 0,
}


class TestMissionMirror:
    def test_path_alone_loads(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(PATH_CMD_TOPIC, PATH_PAYLOAD)
        states = _mission_states(h)
        assert states[-1] == "LOADED"
        last = json.loads(h.fake.publishes_on(MISSION_ACTIVE_TOPIC)[-1].payload)
        assert last["mission_id"] == PATH_PAYLOAD["mission_id"]
        assert last["target_pressure_pa"] == pytest.approx(
            PATH_PAYLOAD["target_pressure_pa"]
        )

    def test_start_then_running(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(
            PATH_CMD_TOPIC,
            {
                "mission_id": 1,
                "target_pressure_pa": 147150.0,
                "shallow_pressure_pa": 49050.0,
                "angle_rad": 0.6109,
                "n_oscillations": 2,
            },
        )
        h.receive_mqtt(COMMAND_CMD_TOPIC, {"data": True})
        states = _mission_states(h)
        assert states[-1] == "RUNNING"

    def test_sawtooth_path_decodes_with_two_pressure_fields(self, bridge_harness):
        # End-to-end ingress: a sawtooth PATH payload shaped exactly as the
        # frontend builds it must decode onto /path with the new
        # shallow_pressure_pa + n_oscillations fields intact (set_message_fields
        # is generic, so this is the bridge-side "propagated all the way" proof).
        h = bridge_harness
        h.receive_mqtt(
            PATH_CMD_TOPIC,
            {
                "mission_id": 1,
                "target_pressure_pa": 60_000.0,
                "shallow_pressure_pa": 30_000.0,
                "angle_rad": 0.6109,
                "n_oscillations": 2,
            },
        )
        h.spin_until(lambda: h.tester.received_path, timeout=1.0)
        msg = h.tester.received_path[-1]
        assert msg.mission_id == 1
        assert msg.target_pressure_pa == pytest.approx(60_000.0)
        assert msg.shallow_pressure_pa == pytest.approx(30_000.0)
        assert msg.n_oscillations == 2

    def test_stop_returns_to_idle_and_clears_cache(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(PATH_CMD_TOPIC, PATH_PAYLOAD)
        h.receive_mqtt(COMMAND_CMD_TOPIC, {"data": True})
        h.receive_mqtt(COMMAND_CMD_TOPIC, {"data": False})
        last = json.loads(h.fake.publishes_on(MISSION_ACTIVE_TOPIC)[-1].payload)
        assert last["state"] == "IDLE"
        # Stop clears the cached mission too, mirroring pathfinding clearing
        # its loaded mission on /command=false.
        assert last["mission_id"] is None

    def test_start_without_loaded_mission_is_noop(self, bridge_harness):
        """A stray ``start`` arriving before ``/path`` is buffered by the
        pathfinder (it has start_pending logic) but the mirror should not
        claim RUNNING -- there is no mission to run."""
        h = bridge_harness
        # Note: pre-existing seed publish on connect won't have fired
        # here because the fake's on_connect was never invoked.
        h.receive_mqtt(COMMAND_CMD_TOPIC, {"data": True})
        states = _mission_states(h)
        # Either no mirror publish at all (nothing changed) or, if one
        # exists, it must not be RUNNING.
        assert "RUNNING" not in states


# ---------------------------------------------------------------------------
# Lifeguard
# ---------------------------------------------------------------------------


def _lifeguard_status(harness) -> dict:
    """The most recent retained lifeguard status payload."""
    return json.loads(harness.fake.publishes_on(LIFEGUARD_STATUS_TOPIC)[-1].payload)


class TestLifeguard:
    def test_off_by_default_stays_silent(self, lifeguard_harness):
        # Bench scenario: never armed, so heartbeat silence means nothing.
        h = lifeguard_harness
        h.spin_for(1.0)  # well past the 0.5 s window + several ticks
        assert h.tester.received_emergency == []
        assert h.tester.received_command == []

    def test_arm_publishes_retained_status(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        status = h.fake.publishes_on(LIFEGUARD_STATUS_TOPIC)[-1]
        assert status.retain is True
        assert json.loads(status.payload) == {
            "armed": True,
            "engaged": False,
            "timeout_s": 0.5,
            "stood_down": False,
        }

    def test_heartbeat_silence_engages(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)
        # The engage stops the mission first (depth PID must go silent before
        # bcu_debug blows ballast) and the retained status flips to engaged.
        # Separate topic, so its delivery can trail the emergency sample.
        h.spin_until(lambda: False in h.tester.received_command, timeout=1.0)
        assert _lifeguard_status(h)["engaged"] is True

    def test_heartbeats_hold_it_off(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            h.receive_mqtt(LIFEGUARD_HEARTBEAT_TOPIC, {})
            h.spin_for(0.2)
        assert True not in h.tester.received_emergency

    def test_latch_survives_heartbeat_return_and_rearm(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)
        # Tether comes back: beats resume and the broker replays the retained
        # arm command. Neither may stand the failsafe down.
        h.receive_mqtt(LIFEGUARD_HEARTBEAT_TOPIC, {})
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.tester.received_emergency.clear()
        h.spin_for(0.5)  # several ticks
        assert (
            True in h.tester.received_emergency
        ), "engage must keep re-publishing while latched"
        assert False not in h.tester.received_emergency
        assert _lifeguard_status(h)["engaged"] is True

    def test_engaged_restops_a_restarted_mission(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)
        h.spin_until(lambda: False in h.tester.received_command, timeout=1.0)
        # Steady engaged state leaves /command alone -- bcu_node safe-stops
        # on every false, so per-tick re-sends would chatter the valves
        # against the emergency hold.
        h.tester.received_command.clear()
        h.spin_for(0.5)  # several ticks
        assert h.tester.received_command == []
        # But an operator starting a mission over a restored link without
        # disarming first is re-stopped within a tick.
        h.receive_mqtt(PATH_CMD_TOPIC, PATH_PAYLOAD)
        h.receive_mqtt(COMMAND_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_command, timeout=2.0)

    def test_disarm_stands_down(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": False})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=1.0)
        assert _lifeguard_status(h) == {
            "armed": False,
            "engaged": False,
            "timeout_s": 0.5,
            "stood_down": False,
        }
        # And it stays down: no further engage publishes after the stand-down.
        h.tester.received_emergency.clear()
        h.spin_for(0.5)  # several ticks
        assert h.tester.received_emergency == []


# ---------------------------------------------------------------------------
# Dive-init ingress + ROS-confirmed echo
# ---------------------------------------------------------------------------


INIT_STATUS = "nautilus/status/init"

# Tank endpoints mirror the sim plant via PlantSpec (rig.py) -- single
# source of truth, so a plant change can't silently strand these tests.
# Ints on purpose: the UI sends ints on the wire, coercion must hold.
_PLANT = PlantSpec()
TANK_EMPTY_PA = int(_PLANT.tank_pressure_empty_pa)
TANK_FULL_PA = int(_PLANT.tank_pressure_full_pa)
TANK_IN_BAND_PA = int(TANK_EMPTY_PA * 1.05)  # inside the 10% stand-down band
TANK_ABOVE_BAND_PA = int(TANK_EMPTY_PA * 1.5)  # well clear of the band

DIVE_INIT_PAYLOAD = {
    "surface_pressure_pa": 101_325,
    "tank_empty_pa": TANK_EMPTY_PA,
    "tank_full_pa": TANK_FULL_PA,
}


class TestDiveInitIngress:
    def test_init_decodes_onto_the_ros_graph(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        h.spin_until(lambda: h.tester.received_init, timeout=1.0)
        msg = h.tester.received_init[-1]
        assert msg.surface_pressure_pa == pytest.approx(101_325.0)
        assert msg.tank_empty_pa == pytest.approx(TANK_EMPTY_PA)
        assert msg.tank_full_pa == pytest.approx(TANK_FULL_PA)

    def test_init_echoes_retained_ros_confirmed_status(self, bridge_harness):
        # Load-bearing for the UI contract: the bridge's own egress
        # subscription must hear its ingress publisher (rclpy delivers
        # local publications), so nautilus/status/init reflects what
        # actually reached the ROS graph.
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        h.spin_until(lambda: h.fake.publishes_on(INIT_STATUS), timeout=2.0)
        echo = h.fake.publishes_on(INIT_STATUS)[-1]
        assert echo.retain is True
        assert json.loads(echo.payload) == {
            "surface_pressure_pa": 101_325.0,
            "tank_empty_pa": float(TANK_EMPTY_PA),
            "tank_full_pa": float(TANK_FULL_PA),
        }

    def test_garbage_init_is_dropped(self, bridge_harness):
        # Unknown fields fail set_message_fields; the bridge logs and drops
        # without publishing ROS-side or echoing.
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, {"bogus_field": 1.0})
        h.spin_for(0.3)
        assert h.tester.received_init == []
        assert h.fake.publishes_on(INIT_STATUS) == []


# ---------------------------------------------------------------------------
# Lifeguard tank-empty stand-down
# ---------------------------------------------------------------------------


def _publish_tank(h, value_pa: int) -> None:
    """Publish a tank reading and wait until the bridge has cached it."""
    h.tester.bcu_pressure_pub.publish(_BridgeTesterNode._int32(value_pa))
    h.spin_until(lambda: h.node._tank_pa == float(value_pa), timeout=1.0)


class TestLifeguardTankStandDown:
    """With a registered empty endpoint, an engaged blow is stood down
    once the tank is within 10% of it -- the latch stays engaged, only a
    disarm clears the state, and with no registration nothing changes."""

    def test_blow_stands_down_when_tank_reaches_band(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_ABOVE_BAND_PA)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)

        # Tank drains into the band -> one False, then quiet.
        _publish_tank(h, TANK_IN_BAND_PA)
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=3.0)
        status = _lifeguard_status(h)
        assert status["engaged"] is True, "latch must survive the stand-down"
        assert status["stood_down"] is True

        h.tester.received_emergency.clear()
        h.spin_for(0.5)  # several ticks
        assert (
            h.tester.received_emergency == []
        ), "stood-down ticks must stay quiet on the emergency topic"

    def test_tank_already_in_band_at_engage_never_blows(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        # The engage tick runs the band check before the first blow
        # publish: a single stand-down False, never a True.
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=4.0)
        assert True not in h.tester.received_emergency
        assert _lifeguard_status(h)["stood_down"] is True

    def test_mission_restop_survives_stand_down(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=4.0)

        # Operator restarts a mission without disarming: still re-stopped,
        # still no blow.
        h.tester.received_command.clear()
        h.tester.received_emergency.clear()
        h.receive_mqtt(PATH_CMD_TOPIC, PATH_PAYLOAD)
        h.receive_mqtt(COMMAND_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_command, timeout=2.0)
        assert True not in h.tester.received_emergency

    def test_arm_replay_does_not_restart_the_blow(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=4.0)

        # Broker replays the retained arm (reconnect): the stood-down state
        # must hold -- re-blowing at an empty tank buys nothing.
        h.tester.received_emergency.clear()
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_for(0.5)  # several ticks
        assert True not in h.tester.received_emergency

    def test_disarm_clears_and_fresh_engage_reblows(self, lifeguard_harness):
        h = lifeguard_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=4.0)

        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": False})
        h.spin_for(0.1)
        assert _lifeguard_status(h)["stood_down"] is False

        # Tank refilled (e.g. bench reset) -> a fresh arm/engage cycle
        # blows again.
        _publish_tank(h, TANK_ABOVE_BAND_PA)
        h.tester.received_emergency.clear()
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)

    def test_no_registration_means_continuous_blow(self, lifeguard_harness):
        # Tank data flowing but no dive-init: the blow is as dumb as ever.
        h = lifeguard_harness
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(LIFEGUARD_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=4.0)
        h.tester.received_emergency.clear()
        h.spin_for(0.5)  # several ticks
        assert True in h.tester.received_emergency
        assert False not in h.tester.received_emergency


# ---------------------------------------------------------------------------
# Manual emergency blow: tank-empty stand-down
# ---------------------------------------------------------------------------


class TestManualBlowStandDown:
    """The operator's manual emergency surface rides the same topic and
    gets the same tank-empty band check from the bridge -- without the
    lifeguard's latch: stand-down just ends the blow, and the operator's
    own cancel (or the red reset) clears the guard."""

    def test_manual_blow_stands_down_in_band(self, bridge_harness):
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_ABOVE_BAND_PA)
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=1.0)

        # Tank drains into the band -> the bridge ends the blow with one
        # False on the next tick.
        _publish_tank(h, TANK_IN_BAND_PA)
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=3.0)

        # And stays quiet -- no re-engage, no repeated False.
        h.tester.received_emergency.clear()
        h.spin_for(0.5)  # several ticks
        assert h.tester.received_emergency == []

    def test_no_registration_leaves_manual_blow_alone(self, bridge_harness):
        h = bridge_harness
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=1.0)
        h.spin_for(0.5)  # several guard ticks
        assert False not in h.tester.received_emergency

    def test_operator_cancel_clears_the_guard(self, bridge_harness):
        # Operator engages then cancels before the tank reaches the band;
        # the band crossing afterwards must not produce a second False.
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_ABOVE_BAND_PA)
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": True})
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": False})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=1.0)

        h.tester.received_emergency.clear()
        _publish_tank(h, TANK_IN_BAND_PA)
        h.spin_for(0.5)  # several guard ticks
        assert h.tester.received_emergency == []

    def test_reset_clears_the_guard(self, bridge_harness):
        # The red reset cancels the blow at bcu_debug; the guard must not
        # chase it with a stray False when the tank later hits the band.
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_ABOVE_BAND_PA)
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: True in h.tester.received_emergency, timeout=1.0)
        h.receive_mqtt(DEBUG_RESET_CMD_TOPIC, {})

        h.tester.received_emergency.clear()
        _publish_tank(h, TANK_IN_BAND_PA)
        h.spin_for(0.5)  # several guard ticks
        assert h.tester.received_emergency == []

    def test_fresh_engage_after_stand_down_is_reevaluated(self, bridge_harness):
        # No latch: after a stand-down the operator can engage again; with
        # the tank still in the band the guard ends it again within a tick.
        h = bridge_harness
        h.receive_mqtt(INIT_CMD_TOPIC, DIVE_INIT_PAYLOAD)
        _publish_tank(h, TANK_IN_BAND_PA)
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=3.0)

        h.tester.received_emergency.clear()
        h.receive_mqtt(EMERGENCY_SURFACE_CMD_TOPIC, {"data": True})
        h.spin_until(lambda: False in h.tester.received_emergency, timeout=3.0)


# ---------------------------------------------------------------------------
# Reconnect watchdog (E-001)
# ---------------------------------------------------------------------------


class TestReconnectWatchdog:
    """The bridge's reconnect backstop. paho's loop_start() normally reconnects
    on its own; when it wedges after a yanked tether -- stuck disconnected, or
    'connected' but mute -- the watchdog rebuilds the client in-process, which
    used to need a Pi reboot. A rebuild surfaces as a fresh fake in
    ``harness.fakes``; the bridge's client is repointed at it, and on the fresh
    client's connect the bridge re-subscribes and re-publishes online so
    telemetry and the UI recover on their own. The watchdog timer is parked far
    out (harness default), so these drive ``_connection_watchdog()`` directly."""

    def test_disconnect_past_grace_rebuilds_and_resubscribes(self):
        # grace 0: the first watchdog tick that sees a dropped link escalates.
        h = MqttBridgeHarness(reconnect_grace_s=0.0)
        try:
            assert len(h.fakes) == 1
            assert h.node._mqtt is h.fakes[0]
            h.fakes[0].connected = False  # tether yanked; paho reports down

            h.node._connection_watchdog()

            # A fresh client was built and is now the bridge's live client.
            assert len(h.fakes) == 2, "a wedged-disconnected link must rebuild"
            assert h.node._mqtt is h.fakes[1]

            # Its on_connect (fired by paho on a real reconnect; driven here)
            # re-subscribes to every ingress topic and republishes online.
            new = h.fakes[1]
            new.connected = True
            h.node._on_connect(new, None, None, 0)
            subscribed = {t for t, _ in new.subscribed}
            for m in INGRESS_MAP:
                assert m.mqtt_topic in subscribed, f"missing resubscribe {m.mqtt_topic}"
            assert LIFEGUARD_CMD_TOPIC in subscribed
            assert LIFEGUARD_HEARTBEAT_TOPIC in subscribed
            online = [
                p for p in new.publishes_on(STATUS_TOPIC) if p.payload == STATUS_ONLINE
            ]
            assert online, "reconnect must republish online"
            assert online[-1].retain is True
        finally:
            h.shutdown()

    def test_stale_connected_link_rebuilds_then_does_not_thrash(self):
        # rx_silence 0: an inbound message followed by silence while paho still
        # claims 'connected' is the half-open / stale-socket wedge.
        h = MqttBridgeHarness(rx_silence_s=0.0)
        try:
            h.receive_mqtt(LIFEGUARD_HEARTBEAT_TOPIC, {})  # sets _last_rx
            assert h.node._last_rx_monotonic is not None
            assert h.fakes[0].connected is True  # paho still claims a link

            h.node._connection_watchdog()
            assert len(h.fakes) == 2, "stale-connected link must rebuild"
            assert h.node._mqtt is h.fakes[1]
            # rx clock cleared on rebuild, so a healthy-but-quiet fresh client
            # (no operator beats) does not get rebuilt again next tick.
            assert h.node._last_rx_monotonic is None

            h.node._connection_watchdog()
            assert len(h.fakes) == 2, "must not thrash after a rebuild"
        finally:
            h.shutdown()

    def test_healthy_link_does_not_rebuild(self):
        # Default windows (grace 12 s, rx_silence 20 s). Connected with a fresh
        # inbound beat: the watchdog must leave the client alone.
        h = MqttBridgeHarness()
        try:
            h.receive_mqtt(LIFEGUARD_HEARTBEAT_TOPIC, {})
            assert h.fakes[0].connected is True
            for _ in range(5):
                h.node._connection_watchdog()
            assert len(h.fakes) == 1, "a healthy link must never rebuild"
            assert h.node._mqtt is h.fakes[0]
        finally:
            h.shutdown()
