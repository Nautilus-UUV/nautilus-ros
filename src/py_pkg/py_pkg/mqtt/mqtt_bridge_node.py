#!/usr/bin/env python3
"""ROS 2 <-> MQTT bridge for the Nautilus glider.

Runs on the mission laptop alongside the topside ROS graph while tethered.
Currently ingress-only: the UI sends a command (String) and a mission
selector (`nautilus_msgs/MissionCommand`) into ROS. Telemetry egress is
deferred until the frontend defines its contract.

MQTT topic tree:

    nautilus/cmd/<topic>      -- MQTT -> ROS (commands)
    nautilus/status/bridge    -- bridge liveness (LWT + heartbeat)

`/path` payload: `MissionCommand` with `mission_id` plus per-mission
parameters (`target_pressure_pa`, `angle_rad`, `n_resurfaces`). The
mission factory in `py_pkg.path.missions` resolves the id; missions
ignore fields they don't consume.

JSON format via `set_message_fields` / `message_to_ordereddict`;
field names match the ROS message exactly. MQTT QoS 1 on ingress:
at-least-once required, duplicates tolerable because controllers are
idempotent on setpoint topics.
"""

import json
from dataclasses import dataclass
from typing import Any

import paho.mqtt.client as mqtt
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rosidl_runtime_py.set_message import set_message_fields

from py_pkg.uuv_ros_core import (
    TOPIC_MESSAGE_MAP,
    UUVTopics,
    create_publisher_for_topic,
)


@dataclass(frozen=True)
class IngressMapping:
    ros_topic: str  # always a UUVTopics.* constant
    mqtt_topic: str
    mqtt_qos: int  # 1 = at-least-once (commands)


# What the UI is allowed to send into ROS. Both ROS topics are registered
# in uuv_ros_core (topics.py + message_types.py + qos_profiles.py).
# Telemetry egress is deferred until the frontend defines its contract.
TOPIC_MAP: tuple[IngressMapping, ...] = (
    IngressMapping(UUVTopics.COMMAND, "nautilus/cmd/command", 1),
    IngressMapping(UUVTopics.PATH, "nautilus/cmd/path", 1),
)


STATUS_TOPIC = "nautilus/status/bridge"
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


class MqttBridge(Node):
    """ROS 2 node that bridges Nautilus topics to/from an MQTT broker."""

    def __init__(self) -> None:
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
        self._mqtt = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
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

        self._ingress_pubs: dict[str, Any] = {}
        for m in TOPIC_MAP:
            pub = create_publisher_for_topic(self, m.ros_topic)
            self._ingress_pubs[m.mqtt_topic] = (pub, TOPIC_MESSAGE_MAP[m.ros_topic])

        # connect_async + loop_start: broker absence at boot must not block
        # node init. The mqtt thread services reconnect in the background.
        self._mqtt.connect_async(host, port, keepalive=keepalive)
        self._mqtt.loop_start()

        self._heartbeat_timer = self.create_timer(
            HEARTBEAT_PERIOD_S, self._publish_heartbeat
        )

        self.get_logger().info(
            f"mqtt_bridge: broker={host}:{port}, ingress={len(TOPIC_MAP)}"
        )

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

    # --- mqtt lifecycle -------------------------------------------------

    def _on_connect(self, client, _userdata, _flags, reason_code, _props=None):
        if reason_code != 0:
            self.get_logger().error(
                f"mqtt connect failed (rc={reason_code}); paho will retry"
            )
            return

        for m in TOPIC_MAP:
            client.subscribe(m.mqtt_topic, qos=m.mqtt_qos)
        client.publish(STATUS_TOPIC, payload=STATUS_ONLINE, qos=1, retain=True)
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
