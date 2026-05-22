#!/usr/bin/env python3
"""ROS 2 <-> MQTT bridge for the Nautilus glider.

Runs on the mission laptop alongside the topside ROS graph while tethered.

Two directions:

* **Ingress** (MQTT -> ROS): UI commands. ``nautilus/cmd/...`` payloads are
  JSON, get materialised into the matching ROS message via
  ``set_message_fields`` and republished into the ROS graph. QoS 1 -- the
  receiving controllers (depth setpoint, ACU setpoint, pathfinding
  "start"/"stop"/"abort") are idempotent on duplicates, but at-least-once
  is the right floor for command traffic.

* **Egress** (ROS -> MQTT): telemetry. The bridge subscribes to the ROS
  topics the frontend wants to render, JSON-encodes them via
  ``message_to_ordereddict``, and publishes on ``nautilus/telemetry/...``
  at MQTT QoS 0. Fire-and-forget: a slow tether/broker must not
  head-of-line-block ROS callbacks. Per-topic throttles cap the rate so
  high-frequency streams (IMU at 200 Hz from the prefilter) don't
  saturate the link.

Special bridge state that doesn't come from a ROS topic:

* ``nautilus/telemetry/mission/active`` (retained) -- the bridge mirrors
  the last ``nautilus/cmd/path`` and ``nautilus/cmd/command`` it forwarded
  so the UI can read "what mission is loaded and what state is it in"
  without needing pathfinding_node to publish its own status topic. State
  vocabulary: IDLE / LOADED / RUNNING.

JSON wire format: field names match the ROS message exactly. No unit
conversion at the bridge -- Pa stays Pa, centidegrees stay centidegrees.
The UI owns presentation units.
"""

import json
from dataclasses import dataclass
from typing import Any

import paho.mqtt.client as mqtt
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields

from py_pkg.uuv_ros_core import (
    TOPIC_MESSAGE_MAP,
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)


@dataclass(frozen=True)
class IngressMapping:
    ros_topic: str  # always a UUVTopics.* constant
    mqtt_topic: str
    mqtt_qos: int  # 1 = at-least-once (commands)


@dataclass(frozen=True)
class EgressMapping:
    ros_topic: str  # always a UUVTopics.* constant
    mqtt_topic: str
    # Soft rate cap. 0.0 means "publish every message" (use sparingly --
    # the prefilter IMU runs at 200 Hz). On-change topics ignore this.
    max_rate_hz: float
    # On-change topics publish only when the JSON-encoded payload bytes
    # differ from the last cached value. They also set retain=True so a
    # fresh UI tab sees the last-known immediately. Use for state-like
    # signals where the value is stable for long stretches (valves,
    # setpoint Pose).
    on_change: bool = False


# What the UI is allowed to send into ROS. Both ROS topics are registered
# in uuv_ros_core (topics.py + message_types.py + qos_profiles.py).
INGRESS_MAP: tuple[IngressMapping, ...] = (
    IngressMapping(UUVTopics.COMMAND, "nautilus/cmd/command", 1),
    IngressMapping(UUVTopics.PATH, "nautilus/cmd/path", 1),
    IngressMapping(UUVTopics.DEBUG_BCU_RPM, "nautilus/cmd/debug/bcu/rpm", 1),
    IngressMapping(UUVTopics.DEBUG_BCU_VALVES, "nautilus/cmd/debug/bcu/valves", 1),
    IngressMapping(UUVTopics.DEBUG_ACU_PITCH, "nautilus/cmd/debug/acu/pitch", 1),
    IngressMapping(UUVTopics.DEBUG_ACU_ROLL, "nautilus/cmd/debug/acu/roll", 1),
    IngressMapping(
        UUVTopics.DEBUG_EMERGENCY_SURFACE, "nautilus/cmd/debug/emergency_surface", 1
    ),
    # Operator manual-override slider: the UI raises/lowers both flags to enter
    # or leave manual mode. depth_node / acu_node stand down while True; the
    # debug nodes drive the wire only while True.
    IngressMapping(
        UUVTopics.CONTROL_MANUAL_OVERRIDE, "nautilus/cmd/control/manual_override", 1
    ),
    IngressMapping(
        UUVTopics.CONTROL_ACU_OVERRIDE, "nautilus/cmd/control/acu_override", 1
    ),
)


# What the bridge mirrors out to the operator UI. 10 Hz across the
# board for periodic signals -- gives the strip charts enough resolution
# to see oscillations and short transients without burning serious
# tether-side bandwidth at JSON scalars. State-like signals (valves,
# setpoint Pose) stay on-change with retain=True so transitions
# propagate immediately and a fresh UI tab sees the last-known value
# without waiting.
EGRESS_MAP: tuple[EgressMapping, ...] = (
    EgressMapping(
        UUVTopics.POSITION_ESTIMATION, "nautilus/telemetry/position/estimation", 10.0
    ),
    EgressMapping(
        UUVTopics.POSITION_TARGET,
        "nautilus/telemetry/position/target",
        0.0,
        on_change=True,
    ),
    EgressMapping(UUVTopics.IMU_FILTERED_LEFT, "nautilus/telemetry/imu/left", 10.0),
    EgressMapping(UUVTopics.IMU_FILTERED_RIGHT, "nautilus/telemetry/imu/right", 10.0),
    EgressMapping(UUVTopics.BCU_PRESSURE, "nautilus/telemetry/bcu/pressure", 10.0),
    EgressMapping(
        UUVTopics.EXTERNAL_PRESSURE, "nautilus/telemetry/external/pressure", 10.0
    ),
    EgressMapping(UUVTopics.BCU_RPM, "nautilus/telemetry/bcu/rpm", 10.0),
    EgressMapping(
        UUVTopics.BCU_VALVES, "nautilus/telemetry/bcu/valves", 0.0, on_change=True
    ),
    EgressMapping(UUVTopics.ACU_PITCH, "nautilus/telemetry/acu/pitch", 10.0),
    EgressMapping(UUVTopics.ACU_ROLL, "nautilus/telemetry/acu/roll", 10.0),
    # Manual-override flags: state-like, mirror on-change/retained so the UI
    # can show "manual mode active" without polling.
    EgressMapping(
        UUVTopics.CONTROL_MANUAL_OVERRIDE,
        "nautilus/telemetry/control/manual_override",
        0.0,
        on_change=True,
    ),
    EgressMapping(
        UUVTopics.CONTROL_ACU_OVERRIDE,
        "nautilus/telemetry/control/acu_override",
        0.0,
        on_change=True,
    ),
)


STATUS_TOPIC = "nautilus/status/bridge"
MISSION_ACTIVE_TOPIC = "nautilus/telemetry/mission/active"
HEARTBEAT_PERIOD_S = 2.0

# Bridge link-state vocabulary (retained on STATUS_TOPIC):
#   "online"     -- bridge is connected and ROS<->MQTT plumbing is live.
#   "offline"    -- bridge shut down cleanly (Ctrl-C, ROS shutdown, etc.).
#   "link_lost"  -- broker observed an unclean TCP drop (LWT). Means EITHER
#                   the bridge crashed OR the tether went down -- broker
#                   cannot tell. Either way, ROS<->cloud plumbing is broken.
STATUS_ONLINE = "online"
STATUS_OFFLINE = "offline"
STATUS_LINK_LOST = "link_lost"

# Mission state vocabulary mirrored on MISSION_ACTIVE_TOPIC. Roughly
# parallels pathfinding_node's internal modes, collapsed to what the UI
# actually needs to render:
#   IDLE     -- no mission cached or last command was stop/abort.
#   LOADED   -- a /path was received, no start yet.
#   RUNNING  -- /command:start observed after a mission was loaded.
MISSION_STATE_IDLE = "IDLE"
MISSION_STATE_LOADED = "LOADED"
MISSION_STATE_RUNNING = "RUNNING"


def _safe_json(payload_dict: Any) -> str:
    """JSON-encode a ROS message dict, replacing NaN/Inf with null.

    rosidl_runtime_py emits float fields directly; an unfilled covariance
    entry in an Imu message can come through as NaN. Default ``json.dumps``
    would emit ``NaN`` tokens that ``JSON.parse`` rejects, so we sanitise
    on the way out -- the UI can render "missing" instead of throwing.
    """
    return json.dumps(payload_dict, allow_nan=False, default=lambda _: None)


class MqttBridge(Node):
    """ROS 2 node that bridges Nautilus topics to/from an MQTT broker."""

    def __init__(self, mqtt_client_factory=None) -> None:
        super().__init__("mqtt_bridge")

        # Broker is parameterised. Default to localhost (mosquitto -v on the
        # mission laptop); override with the static IP when running against
        # the real tether broker (e.g. broker_host:=192.168.2.1).
        self.declare_parameter("broker_host", "127.0.0.1")
        self.declare_parameter("broker_port", 1883)
        self.declare_parameter("client_id", "nautilus_bridge")
        self.declare_parameter("keepalive_s", 30)

        host = self.get_parameter("broker_host").get_parameter_value().string_value
        port = self.get_parameter("broker_port").get_parameter_value().integer_value
        client_id = self.get_parameter("client_id").get_parameter_value().string_value
        keepalive = (
            self.get_parameter("keepalive_s").get_parameter_value().integer_value
        )

        # paho v2 callback API; ROS callbacks must never block on the broker.
        # Tests inject a fake via ``mqtt_client_factory`` so the bridge can
        # spin without a real broker on the loopback.
        if mqtt_client_factory is None:
            self._mqtt = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id,
            )
        else:
            self._mqtt = mqtt_client_factory(client_id)
        # LWT fires only on UNCLEAN disconnect (crash, tether drop, kernel
        # kills the TCP session). A graceful shutdown sends a DISCONNECT
        # packet first, which suppresses the will -- so 'offline' (clean)
        # and 'link_lost' (unclean) stay distinguishable on the UI side.
        self._mqtt.will_set(STATUS_TOPIC, payload=STATUS_LINK_LOST, qos=1, retain=True)
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_disconnect = self._on_disconnect
        self._mqtt.on_message = self._on_mqtt_message
        # Built-in exponential backoff on reconnect.
        self._mqtt.reconnect_delay_set(min_delay=1, max_delay=30)

        # --- ingress -----------------------------------------------------
        self._ingress_pubs: dict[str, Any] = {}
        for m in INGRESS_MAP:
            pub = create_publisher_for_topic(self, m.ros_topic)
            self._ingress_pubs[m.mqtt_topic] = (pub, TOPIC_MESSAGE_MAP[m.ros_topic])

        # --- egress ------------------------------------------------------
        # _last_emit holds the monotonic timestamp of the previous publish
        # per egress topic (throttling). _last_payload holds the previous
        # JSON-encoded payload (on-change dedup + retained semantics).
        self._last_emit: dict[str, float] = {}
        self._last_payload: dict[str, str] = {}
        self._egress_subs: list = []
        for em in EGRESS_MAP:
            sub = create_subscription_for_topic(
                self, em.ros_topic, self._make_egress_callback(em)
            )
            self._egress_subs.append(sub)

        # --- mission-active mirror --------------------------------------
        # Cached MissionCommand JSON dict from the most recent
        # nautilus/cmd/path, plus the most recently observed command-state.
        # Combined into MISSION_ACTIVE_TOPIC on every change so the UI can
        # render "current mission" without pathfinding_node owning its own
        # status topic.
        self._mission_cache: dict | None = None
        self._mission_state: str = MISSION_STATE_IDLE

        # connect_async + loop_start: broker absence at boot must not block
        # node init. The mqtt thread services reconnect in the background.
        self._mqtt.connect_async(host, port, keepalive=keepalive)
        self._mqtt.loop_start()

        self._heartbeat_timer = self.create_timer(
            HEARTBEAT_PERIOD_S, self._publish_heartbeat
        )

        self.get_logger().info(
            f"mqtt_bridge: broker={host}:{port}, "
            f"ingress={len(INGRESS_MAP)} egress={len(EGRESS_MAP)}"
        )

    # --- egress: ROS -> MQTT --------------------------------------------

    def _make_egress_callback(self, mapping: EgressMapping):
        """Closure that captures the mapping and returns a ROS subscription
        callback. One per egress entry so each topic has its own throttle
        state via ``mapping.mqtt_topic`` as the key into _last_emit /
        _last_payload."""

        def _callback(msg) -> None:
            try:
                payload_dict = message_to_ordereddict(msg)
                payload_str = _safe_json(payload_dict)
            except Exception as exc:
                # Don't kill the subscription on one bad message; log and
                # drop.
                self.get_logger().warning(
                    f"egress encode failed for {mapping.mqtt_topic}: {exc}"
                )
                return

            if mapping.on_change:
                if self._last_payload.get(mapping.mqtt_topic) == payload_str:
                    return
                self._last_payload[mapping.mqtt_topic] = payload_str
                self._mqtt.publish(
                    mapping.mqtt_topic, payload=payload_str, qos=0, retain=True
                )
                return

            if mapping.max_rate_hz > 0.0:
                period_s = 1.0 / mapping.max_rate_hz
                now = self.get_clock().now().nanoseconds * 1e-9
                last = self._last_emit.get(mapping.mqtt_topic, 0.0)
                if now - last < period_s:
                    return
                self._last_emit[mapping.mqtt_topic] = now

            self._mqtt.publish(mapping.mqtt_topic, payload=payload_str, qos=0)

        return _callback

    # --- ingress: MQTT -> ROS -------------------------------------------

    def _on_mqtt_message(self, _client, _userdata, mqtt_msg) -> None:
        entry = self._ingress_pubs.get(mqtt_msg.topic)
        if entry is None:
            # subscribed via wildcard one day? log so it's visible.
            self.get_logger().debug(f"unmapped ingress topic {mqtt_msg.topic}")
            return
        pub, msg_cls = entry
        try:
            payload = json.loads(mqtt_msg.payload.decode("utf-8"))
            ros_msg = msg_cls()
            set_message_fields(ros_msg, payload)
        except Exception as exc:
            # Commands are SAFETY-adjacent: surface decode failures loudly.
            self.get_logger().error(
                f"ingress decode failed for {mqtt_msg.topic}: {exc}"
            )
            return
        pub.publish(ros_msg)

        # Mission-active mirror: piggy-back on the ingress decode so the
        # UI's "what mission is loaded" view stays in sync with the
        # commands actually dispatched. The pathfinder owns the real
        # state machine; this is just a UI-facing shadow.
        self._update_mission_mirror(mqtt_msg.topic, payload)

    def _update_mission_mirror(self, mqtt_topic: str, payload: dict) -> None:
        changed = False

        if mqtt_topic == "nautilus/cmd/path":
            self._mission_cache = dict(payload)
            # First /path after IDLE -> LOADED; if we were already RUNNING
            # (a re-dispatch mid-run) keep the running state, the
            # pathfinder loads the new mission and continues.
            if self._mission_state == MISSION_STATE_IDLE:
                self._mission_state = MISSION_STATE_LOADED
            changed = True

        elif mqtt_topic == "nautilus/cmd/command":
            data = (payload.get("data") or "").lower()
            if data == "start" and self._mission_cache is not None:
                if self._mission_state != MISSION_STATE_RUNNING:
                    self._mission_state = MISSION_STATE_RUNNING
                    changed = True
            elif data in ("stop", "abort"):
                if self._mission_state != MISSION_STATE_IDLE:
                    self._mission_state = MISSION_STATE_IDLE
                    if data == "abort":
                        # Abort drops the cached mission too, matching
                        # pathfinding_node's behaviour.
                        self._mission_cache = None
                    changed = True

        if changed:
            self._publish_mission_active()

    def _publish_mission_active(self) -> None:
        snapshot: dict = {"state": self._mission_state}
        if self._mission_cache is not None:
            snapshot.update(self._mission_cache)
        else:
            snapshot["mission_id"] = None
        self._mqtt.publish(
            MISSION_ACTIVE_TOPIC,
            payload=_safe_json(snapshot),
            qos=0,
            retain=True,
        )

    # --- mqtt lifecycle -------------------------------------------------

    def _on_connect(self, client, _userdata, _flags, reason_code, _props=None):
        if reason_code != 0:
            self.get_logger().error(
                f"mqtt connect failed (rc={reason_code}); paho will retry"
            )
            return

        for m in INGRESS_MAP:
            client.subscribe(m.mqtt_topic, qos=m.mqtt_qos)
        client.publish(STATUS_TOPIC, payload=STATUS_ONLINE, qos=1, retain=True)
        # Seed MISSION_ACTIVE_TOPIC on connect (and reconnect) so a fresh
        # UI tab sees a defined state immediately rather than waiting for
        # the first command to flow.
        self._publish_mission_active()
        self.get_logger().info("mqtt connected")

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _props=None):
        # Surface as warning -- paho's loop thread handles the reconnect.
        self.get_logger().warning(f"mqtt disconnected (rc={reason_code})")

    def _publish_heartbeat(self) -> None:
        self._mqtt.publish(STATUS_TOPIC + "/tick", payload="alive", qos=0)

    def destroy_node(self) -> bool:
        try:
            self._mqtt.publish(STATUS_TOPIC, payload=STATUS_OFFLINE, qos=1, retain=True)
            self._mqtt.loop_stop()
            self._mqtt.disconnect()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    # Catch SIGINT/SIGTERM so the process exits 0 instead of 1 on Ctrl-C —
    # matches the pattern in pathfinding/depth/acu nodes so launch_testing's
    # exit-code check stays happy.
    rclpy.init(args=args)
    node = MqttBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
