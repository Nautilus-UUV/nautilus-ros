#!/usr/bin/env python3
"""ROS 2 <-> MQTT bridge for the Nautilus glider.

Runs on the vehicle alongside the control stack; the broker lives on the
mission laptop across the tether. That placement matters: the lifeguard
failsafe below only works because this node keeps running (and keeps its
local ROS publishers) when the tether -- and with it the broker -- goes away.

Two directions:

* **Ingress** (MQTT -> ROS): UI commands. ``nautilus/cmd/...`` payloads are
  JSON, get materialised into the matching ROS message via
  ``set_message_fields`` and republished into the ROS graph. QoS 1 -- the
  receiving controllers (depth setpoint, ACU setpoint, pathfinding
  start/stop) are idempotent on duplicates, but at-least-once is the right
  floor for command traffic.

* **Egress** (ROS -> MQTT): telemetry. The bridge subscribes to the ROS
  topics the frontend wants to render, JSON-encodes them via
  ``message_to_ordereddict``, and publishes on ``nautilus/telemetry/...``
  at MQTT QoS 0. Fire-and-forget: a slow tether/broker must not
  head-of-line-block ROS callbacks. Per-topic throttles cap the rate so
  high-frequency streams (IMU at 200 Hz from the prefilter) don't
  saturate the link.

Special bridge state that doesn't come from a ROS topic:

* **Lifeguard** -- the deploy-time dead-man failsafe. The UI arms it via
  ``nautilus/cmd/lifeguard`` (retained) and heartbeats on
  ``nautilus/cmd/heartbeat``; both are consumed here, never forwarded to
  ROS. When armed and the heartbeat has been silent for
  ``lifeguard_timeout_s``, the bridge stops the mission and latches the
  emergency-surface command through its existing ingress publishers --
  bcu_debug then blows ballast continuously. If the operator registered
  the tank's empty endpoint pre-dive (``nautilus/cmd/init``), the bridge
  stands the blow down once the tank is within 10% of it -- bladder full,
  nothing left to pump -- while the latch stays engaged; without a
  registration the blow is continuous and dumb as ever. The same band
  check guards the operator's MANUAL emergency surface (the UI slider
  rides the same ``/debug/emergency_surface`` topic through ingress),
  minus the latch semantics: stand-down just ends that blow, and a fresh
  engage is re-evaluated from scratch. Status (armed/engaged/stood_down)
  is retained on ``nautilus/status/lifeguard``.

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
import threading
import time
from dataclasses import dataclass
from typing import Any

import paho.mqtt.client as mqtt
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields

from py_pkg.mqtt.lifeguard import Lifeguard, tank_blow_exhausted
from py_pkg.uuv_ros_core import (
    TOPIC_MESSAGE_MAP,
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
    spin_node,
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


# MQTT command topics the bridge also addresses by name OUTSIDE the map
# (mission mirror, lifeguard failsafe, manual-blow guard). Named once so
# a rename can't strand an interior comparison or _ingress_pubs lookup.
COMMAND_CMD_TOPIC = "nautilus/cmd/command"
PATH_CMD_TOPIC = "nautilus/cmd/path"
INIT_CMD_TOPIC = "nautilus/cmd/init"
EMERGENCY_SURFACE_CMD_TOPIC = "nautilus/cmd/debug/emergency_surface"
DEBUG_RESET_CMD_TOPIC = "nautilus/cmd/debug/reset"

# What the UI is allowed to send into ROS. Both ROS topics are registered
# in uuv_ros_core (topics.py + message_types.py + qos_profiles.py).
INGRESS_MAP: tuple[IngressMapping, ...] = (
    IngressMapping(UUVTopics.COMMAND, COMMAND_CMD_TOPIC, 1),
    IngressMapping(UUVTopics.PATH, PATH_CMD_TOPIC, 1),
    IngressMapping(UUVTopics.DEBUG_BCU_RPM, "nautilus/cmd/debug/bcu/rpm", 1),
    IngressMapping(
        UUVTopics.DEBUG_BCU_RPM_UNTIL_PRESSURE,
        "nautilus/cmd/debug/bcu/rpm_until_pressure",
        1,
    ),
    IngressMapping(UUVTopics.DEBUG_BCU_VALVES, "nautilus/cmd/debug/bcu/valves", 1),
    IngressMapping(UUVTopics.DEBUG_ACU_PITCH, "nautilus/cmd/debug/acu/pitch", 1),
    IngressMapping(UUVTopics.DEBUG_ACU_ROLL, "nautilus/cmd/debug/acu/roll", 1),
    IngressMapping(UUVTopics.DEBUG_EMERGENCY_SURFACE, EMERGENCY_SURFACE_CMD_TOPIC, 1),
    IngressMapping(UUVTopics.DEBUG_RESET, DEBUG_RESET_CMD_TOPIC, 1),
    # Pre-dive registration (surface pressure + tank endpoints). The UI
    # publishes it retained, so the broker replays it to a restarted
    # bridge -- that replay, plus the TRANSIENT_LOCAL latch on the ROS
    # side, is what makes the registration persistent.
    IngressMapping(UUVTopics.DIVE_INIT, INIT_CMD_TOPIC, 1),
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
    # Per-subsystem health. State-like, so on-change/retained: a payload only
    # crosses the tether on a real online<->offline transition (the liveness
    # node leaves header.stamp zero to keep the JSON byte-stable otherwise),
    # and a fresh UI tab gets the last-known states from the retained message.
    EgressMapping(
        UUVTopics.STATUS_LIVENESS,
        "nautilus/status/liveness",
        0.0,
        on_change=True,
    ),
    # ROS-confirmed echo of the dive registration: the bridge's own
    # egress subscription hears the ingress publisher's DIVE_INIT (rclpy
    # delivers local publications), so the UI renders registration state
    # from what actually reached the ROS graph, retained for fresh tabs.
    EgressMapping(UUVTopics.DIVE_INIT, "nautilus/status/init", 0.0, on_change=True),
)


STATUS_TOPIC = "nautilus/status/bridge"
MISSION_ACTIVE_TOPIC = "nautilus/telemetry/mission/active"
HEARTBEAT_PERIOD_S = 2.0

# Lifeguard surfaces. The cmd/heartbeat pair is consumed by the bridge itself
# (no ROS forwarding); status is retained so a fresh UI tab -- and the bridge's
# own reconnect -- always carries the current armed/engaged truth.
LIFEGUARD_CMD_TOPIC = "nautilus/cmd/lifeguard"  # {"data": bool}, QoS 1, retained by UI
LIFEGUARD_HEARTBEAT_TOPIC = "nautilus/cmd/heartbeat"  # QoS 0, ~1 Hz, payload ignored
LIFEGUARD_STATUS_TOPIC = "nautilus/status/lifeguard"
LIFEGUARD_TICK_PERIOD_S = 1.0  # default for the lifeguard_tick_period_s param

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
#   IDLE     -- no mission cached, or last command was stop.
#   LOADED   -- a /path was received, no start yet.
#   RUNNING  -- /command=true (start) observed after a mission was loaded.
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

    def __init__(self, mqtt_client_factory=None, **kwargs) -> None:
        super().__init__("mqtt_bridge", **kwargs)

        # Broker is parameterised. Default to localhost (mosquitto -v on the
        # mission laptop); override with the static IP when running against
        # the real tether broker (e.g. broker_host:=192.168.2.1).
        self.declare_parameter("broker_host", "127.0.0.1")
        self.declare_parameter("broker_port", 1883)
        self.declare_parameter("client_id", "nautilus_bridge")
        self.declare_parameter("keepalive_s", 30)
        # Dead-man window for the lifeguard, once armed. The tick period is
        # how often silence (and the tank-empty band) is re-checked; tests
        # shorten it so engage latency isn't floored at 1 s.
        self.declare_parameter("lifeguard_timeout_s", 15.0)
        self.declare_parameter("lifeguard_tick_period_s", LIFEGUARD_TICK_PERIOD_S)

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

        # --- lifeguard -----------------------------------------------------
        # State is mutated from the paho network thread (_on_mqtt_message)
        # and read from the rclpy timer thread. The lock also pairs each
        # transition with its ROS/MQTT publish, so a disarm's stand-down can
        # never be overtaken by a stale engage re-publish from the tick.
        #
        # Time source is time.monotonic(), not the node clock: the node clock
        # is wall time, and an NTP step when the tether returns is exactly
        # this feature's active window -- a forward step would mis-fire, a
        # backward one would delay the failsafe.
        timeout_s = (
            self.get_parameter("lifeguard_timeout_s").get_parameter_value().double_value
        )
        self._lifeguard = Lifeguard(timeout_s)
        self._lifeguard_lock = threading.Lock()

        # Tank-empty stand-down state, all under the same lock: the
        # registered empty endpoint and the live tank pressure arrive on
        # the rclpy executor (ROS subscriptions below), the stood-down
        # flag is written there too, and _tick_lifeguard reads all three.
        # Stood-down means: the latch stays engaged, but the blow has
        # been stopped because the tank is within 10% of empty -- there's
        # nothing left to pump. Cleared only by disarm. Process-local: a
        # bridge restart while engaged may re-blow for ~a tick until the
        # retained init replay and the first tank sample stand it down
        # again.
        self._init_tank_empty_pa: float | None = None
        self._tank_pa: float | None = None
        self._lifeguard_stood_down = False

        # The operator's manual emergency surface (the UI slider) rides the
        # same /debug/emergency_surface topic and deserves the same
        # tank-empty stand-down. The bridge sees the command pass through
        # its own ingress, so it tracks "a manual blow is running" here and
        # the 1 s tick applies the band check. Unlike the lifeguard there
        # is no latch to preserve: stand-down just clears the flag, and the
        # next manual engage is re-evaluated from scratch. A blow commanded
        # directly on the ROS graph (ros2 topic pub) bypasses MQTT and this
        # guard -- the UI path is what's covered.
        self._manual_emergency_active = False

        create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._on_tank_pressure
        )
        # The registration is consumed as the typed ROS message, not by
        # re-parsing the MQTT JSON: rclpy delivers the bridge's own
        # ingress publish locally (same mechanism the status/init echo
        # rides), and the TRANSIENT_LOCAL latch replays it on discovery.
        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)

        # connect_async + loop_start: broker absence at boot must not block
        # node init. The mqtt thread services reconnect in the background.
        self._mqtt.connect_async(host, port, keepalive=keepalive)
        self._mqtt.loop_start()

        self._heartbeat_timer = self.create_timer(
            HEARTBEAT_PERIOD_S, self._publish_heartbeat
        )
        self._lifeguard_timer = self.create_timer(
            self.get_parameter("lifeguard_tick_period_s")
            .get_parameter_value()
            .double_value,
            self._tick_lifeguard,
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

    def _reseed_retained_egress(self, client) -> None:
        """Re-publish the last-known payload of every on-change egress topic,
        retained, on (re)connect.

        paho drops (doesn't queue) QoS-0 publishes while disconnected, but
        the on-change dedup updates _last_payload regardless -- so the first
        liveness/valves/setpoint frame the bridge encodes before the broker
        is reachable never lands, and every byte-identical follow-up is then
        suppressed as "no change". The broker ends up with no retained copy,
        and a fresh UI tab subscribing later sees nothing: the whole liveness
        grid defaults to offline while Tether (re-seeded below on every
        connect) sits green. This is the common boot order -- the vehicle's
        control stack, this bridge included, comes up before the laptop
        broker; same on a tether reconnect after the broker restarted and
        lost its retained store. Re-seeding from the cache here restores the
        retained copy immediately for ALL on-change topics, regardless of how
        rarely they next change. Snapshot the dict: egress callbacks on the
        rclpy executor thread mutate it while this runs on the paho thread."""
        for mqtt_topic, payload_str in list(self._last_payload.items()):
            client.publish(mqtt_topic, payload=payload_str, qos=0, retain=True)

    # --- ingress: MQTT -> ROS -------------------------------------------

    def _on_mqtt_message(self, _client, _userdata, mqtt_msg) -> None:
        # Lifeguard surfaces are consumed by the bridge itself -- they never
        # touch the ROS graph, so they're intercepted before the ingress map.
        if mqtt_msg.topic == LIFEGUARD_HEARTBEAT_TOPIC:
            with self._lifeguard_lock:
                # Payload deliberately ignored: arrival IS the signal.
                self._lifeguard.beat(time.monotonic())
            return
        if mqtt_msg.topic == LIFEGUARD_CMD_TOPIC:
            self._on_lifeguard_cmd(mqtt_msg.payload)
            return

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
        # Same piggy-back for the manual emergency surface: the blow guard
        # tracks engage/cancel (and the red reset) off the ingress decode.
        self._update_manual_emergency(mqtt_msg.topic, payload)

    def _update_mission_mirror(self, mqtt_topic: str, payload: dict) -> None:
        changed = False

        if mqtt_topic == PATH_CMD_TOPIC:
            self._mission_cache = dict(payload)
            # First /path after IDLE -> LOADED; if we were already RUNNING
            # (a re-dispatch mid-run) keep the running state, the
            # pathfinder loads the new mission and continues.
            if self._mission_state == MISSION_STATE_IDLE:
                self._mission_state = MISSION_STATE_LOADED
            changed = True

        elif mqtt_topic == COMMAND_CMD_TOPIC:
            start = bool(payload.get("data"))
            if start and self._mission_cache is not None:
                if self._mission_state != MISSION_STATE_RUNNING:
                    self._mission_state = MISSION_STATE_RUNNING
                    changed = True
            elif not start:
                # Stop -> clean idle. Drop the cached mission too, mirroring
                # pathfinding_node clearing its loaded mission on /command=false.
                if (
                    self._mission_state != MISSION_STATE_IDLE
                    or self._mission_cache is not None
                ):
                    self._mission_state = MISSION_STATE_IDLE
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

    # --- lifeguard --------------------------------------------------------

    def _on_tank_pressure(self, msg) -> None:
        # Tank pressure in the sensor's own frame (tank relative to
        # hull), same stream the registered empty endpoint was sampled
        # from. Cached for the stand-down check in _tick_lifeguard.
        with self._lifeguard_lock:
            self._tank_pa = float(msg.data)

    def _on_dive_init(self, msg) -> None:
        # Registered empty endpoint for the lifeguard's stand-down check.
        with self._lifeguard_lock:
            empty_pa = float(msg.tank_empty_pa)
            # Non-positive (including a missing field decoding as 0.0)
            # means "not registered" -- tank_blow_exhausted double-guards.
            self._init_tank_empty_pa = empty_pa if empty_pa > 0.0 else None

    def _update_manual_emergency(self, mqtt_topic: str, payload: dict) -> None:
        # Track the operator's manual blow so the tick can stand it down at
        # the tank-empty band. The operator's own cancel and the red reset
        # both end the blow at bcu_debug, so they clear the flag too.
        if mqtt_topic == EMERGENCY_SURFACE_CMD_TOPIC:
            with self._lifeguard_lock:
                self._manual_emergency_active = bool(payload.get("data"))
        elif mqtt_topic == DEBUG_RESET_CMD_TOPIC:
            with self._lifeguard_lock:
                self._manual_emergency_active = False

    def _on_lifeguard_cmd(self, raw: bytes) -> None:
        try:
            arm = bool(json.loads(raw.decode("utf-8")).get("data"))
        except Exception as exc:
            self.get_logger().error(f"lifeguard decode failed: {exc}")
            return
        with self._lifeguard_lock:
            if arm:
                self._lifeguard.arm(time.monotonic())
                self.get_logger().info(
                    f"lifeguard armed ({self._lifeguard.timeout_s:.0f} s heartbeat window)"
                )
            else:
                was_engaged = self._lifeguard.engaged
                self._lifeguard.disarm()
                # Disarm is also the only thing that clears a tank-empty
                # stand-down -- the arm branch deliberately doesn't, so a
                # retained-arm replay can't restart a blow at an empty tank.
                # It's a total stand-down, so the manual-blow flag goes too.
                self._lifeguard_stood_down = False
                self._manual_emergency_active = False
                if was_engaged:
                    # Stand the latched emergency surface down, once.
                    self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, False)
                self.get_logger().info("lifeguard disarmed")
            self._publish_lifeguard_status()

    def _tick_lifeguard(self) -> None:
        with self._lifeguard_lock:
            was_engaged = self._lifeguard.engaged
            if not self._lifeguard.tick(time.monotonic()):
                # No lifeguard engagement -> the same tick guards a manual
                # blow against pumping past the tank-empty band.
                self._tick_manual_blow_guard()
                return
            if not was_engaged:
                self.get_logger().error(
                    "LIFEGUARD: laptop heartbeat lost -- engaging emergency surface"
                )
                self._publish_lifeguard_status()
            # Stop the mission so the depth PID goes silent before bcu_debug
            # blows ballast -- same sequencing as the UI's emergency slider.
            # Unconditionally on the engage transition; afterwards only while
            # the mission mirror shows one loaded/running (an operator starting
            # a mission over a restored link without disarming first). depth_node
            # safe-stops on EVERY /command=false, so re-sending it each tick in
            # the steady engaged state would chatter the valves against the
            # emergency hold.
            if not was_engaged or self._mission_state != MISSION_STATE_IDLE:
                self._publish_ingress_bool(COMMAND_CMD_TOPIC, False)
                self._update_mission_mirror(COMMAND_CMD_TOPIC, {"data": False})
            # Tank-empty stand-down: once the tank is within 10% of the
            # registered empty endpoint there is nothing left to pump, so
            # stop the blow (one False -> bcu_debug zeroes RPM and closes
            # the valves) and go quiet. The latch stays engaged -- only a
            # disarm clears it -- and the mission re-stop above keeps
            # running. With no registration this never fires and the blow
            # stays continuous and dumb, exactly as before.
            if self._lifeguard_stood_down:
                return
            if tank_blow_exhausted(self._tank_pa, self._init_tank_empty_pa):
                self._lifeguard_stood_down = True
                self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, False)
                self._publish_lifeguard_status()
                self.get_logger().warning(
                    f"LIFEGUARD: tank within 10% of empty "
                    f"({self._tank_pa:.0f} Pa vs {self._init_tank_empty_pa:.0f} Pa) "
                    "-- blow stood down, latch stays engaged"
                )
                return
            # The engage itself IS re-published every tick (idempotent at
            # bcu_debug) so a DEBUG_RESET or a bcu_debug restart cannot quietly
            # stand the failsafe down.
            self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, True)

    def _tick_manual_blow_guard(self) -> None:
        """Stand a MANUAL emergency blow down at the tank-empty band.

        Called from the lifeguard tick with the lock held. One False stops
        bcu_debug (0 RPM, valves closed, flush); clearing the flag means a
        fresh manual engage starts a fresh evaluation -- no latch, unlike
        the lifeguard. With no registration this never fires.
        """
        if not self._manual_emergency_active:
            return
        if not tank_blow_exhausted(self._tank_pa, self._init_tank_empty_pa):
            return
        self._manual_emergency_active = False
        self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, False)
        self.get_logger().warning(
            f"manual emergency surface: tank within 10% of empty "
            f"({self._tank_pa:.0f} Pa vs {self._init_tank_empty_pa:.0f} Pa) "
            "-- blow stood down"
        )

    def _publish_ingress_bool(self, mqtt_topic: str, value: bool) -> None:
        """Publish on a ROS topic through the existing ingress mapping, as if
        the command had arrived over MQTT."""
        pub, msg_cls = self._ingress_pubs[mqtt_topic]
        msg = msg_cls()
        msg.data = value
        pub.publish(msg)

    def _publish_lifeguard_status(self) -> None:
        # timeout_s rides along so the UI can count a tether drop down
        # without hardcoding the window.
        self._mqtt.publish(
            LIFEGUARD_STATUS_TOPIC,
            payload=json.dumps(
                {
                    "armed": self._lifeguard.armed,
                    "engaged": self._lifeguard.engaged,
                    "timeout_s": self._lifeguard.timeout_s,
                    "stood_down": self._lifeguard_stood_down,
                }
            ),
            qos=1,
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
        client.subscribe(LIFEGUARD_CMD_TOPIC, qos=1)
        client.subscribe(LIFEGUARD_HEARTBEAT_TOPIC, qos=0)
        client.publish(STATUS_TOPIC, payload=STATUS_ONLINE, qos=1, retain=True)
        # Seed MISSION_ACTIVE_TOPIC on connect (and reconnect) so a fresh
        # UI tab sees a defined state immediately rather than waiting for
        # the first command to flow.
        self._publish_mission_active()
        # Re-seed lifeguard status too: paho drops (doesn't queue) publishes
        # while disconnected, so an engage during a tether outage would
        # otherwise leave a stale retained status behind.
        with self._lifeguard_lock:
            self._publish_lifeguard_status()
        # Same hazard for the on-change egress topics (liveness, valves,
        # setpoint, init): re-publish their last-known retained payloads so a
        # late-joining UI gets real state instead of a default-offline grid.
        self._reseed_retained_egress(client)
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
    rclpy.init(args=args)
    node = MqttBridge()
    spin_node(node)


if __name__ == "__main__":
    main()
