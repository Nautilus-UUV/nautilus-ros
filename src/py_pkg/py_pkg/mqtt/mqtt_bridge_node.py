#!/usr/bin/env python3
"""ROS 2 <-> MQTT bridge for the Nautilus glider.

Runs on the vehicle, alongside the control stack. The broker lives on the
mission laptop, across the tether. The bridge keeps running -- and keeps its
local ROS publishers -- when the tether drops, which is what lets the
lifeguard failsafe below fire.

Two directions:

* **Ingress** (MQTT -> ROS): UI commands. ``nautilus/cmd/...`` payloads are
  JSON. The bridge builds the matching ROS message with
  ``set_message_fields`` and publishes it into the ROS graph. QoS 1
  (at-least-once); the receiving controllers are idempotent on duplicates.

* **Egress** (ROS -> MQTT): telemetry. The bridge subscribes to the ROS
  topics the frontend renders, JSON-encodes them with
  ``message_to_ordereddict``, and publishes on ``nautilus/telemetry/...`` at
  QoS 0. The publish returns at once, keeping ROS callbacks free of broker
  backpressure. Per-topic throttles cap the rate to keep high-frequency
  streams (IMU at 200 Hz) off the link.

Special bridge state, not from a ROS topic:

* **Lifeguard** -- the deploy-time dead-man failsafe. The UI arms it on
  ``nautilus/cmd/lifeguard`` (retained) and heartbeats on
  ``nautilus/cmd/heartbeat``. The bridge consumes both. Once armed, if the
  heartbeat stays silent for ``lifeguard_timeout_s`` the bridge stops the
  mission and latches the emergency-surface command, and bcu_debug blows
  ballast. If the operator registered the tank's empty endpoint pre-dive
  (``nautilus/cmd/init``), the bridge stops the blow once the tank reaches
  within 10% of empty, while the latch stays engaged. The same band check
  guards the operator's manual emergency surface (the UI slider, on the same
  ``/debug/emergency_surface`` topic). Status (armed/engaged/stood_down) is
  retained on ``nautilus/status/lifeguard``.

* ``nautilus/telemetry/mission/active`` (retained) -- the bridge mirrors the
  last ``nautilus/cmd/path`` and ``nautilus/cmd/command`` it forwarded, so the
  UI can read which mission is loaded and its state. States: IDLE / LOADED /
  RUNNING.

JSON wire format: field names match the ROS message exactly. Units pass
through unchanged (Pa stays Pa, centidegrees stay centidegrees). The UI owns
presentation units.
"""

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

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
    # Max publish rate. 0.0 publishes every message (the prefilter IMU runs at
    # 200 Hz, so use sparingly). On-change topics ignore this.
    max_rate_hz: float
    # On-change topics publish only when the encoded payload differs from the
    # last one, with retain=True so a fresh UI tab gets the last value. Use for
    # state-like signals that hold steady (valves, setpoint Pose).
    on_change: bool = False
    # Custom message -> dict encoder. None falls back to message_to_ordereddict
    # (the full message). Set it to trim a fat message down to the fields the UI
    # and DB actually use (see _encode_imu_compact).
    encoder: Callable[[Any], dict] | None = None


# MQTT command topics the bridge references by name (mission mirror, lifeguard,
# manual-blow guard). Named once, so the constant is the single source for
# these string comparisons and _ingress_pubs lookups.
COMMAND_CMD_TOPIC = "nautilus/cmd/command"
PATH_CMD_TOPIC = "nautilus/cmd/path"
INIT_CMD_TOPIC = "nautilus/cmd/init"
EMERGENCY_SURFACE_CMD_TOPIC = "nautilus/cmd/debug/emergency_surface"
DEBUG_RESET_CMD_TOPIC = "nautilus/cmd/debug/reset"

# Commands the UI may send into ROS. Each ROS topic is registered in
# uuv_ros_core (topics.py + message_types.py + qos_profiles.py).
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
    # publishes it retained, and the ROS side latches it TRANSIENT_LOCAL. Both
    # replay on reconnect, so the registration survives a bridge restart.
    IngressMapping(UUVTopics.DIVE_INIT, INIT_CMD_TOPIC, 1),
)


def _encode_imu_compact(msg) -> dict:
    """Trim the filtered Imu to the two fields the UI and DB actually use.

    The prefilter passes orientation + all three covariance matrices straight
    through from the raw IMU; nothing past the tether reads them (the 3D
    attitude model runs off position/estimation, the covariances are unfilled).
    Sending only the two vectors shrinks the frame (~450 B -> ~95 B) and keeps
    the DB from logging ~30 dead covariance/orientation channels per sample.
    """
    av, la = msg.angular_velocity, msg.linear_acceleration
    return {
        "angular_velocity": {"x": av.x, "y": av.y, "z": av.z},
        "linear_acceleration": {"x": la.x, "y": la.y, "z": la.z},
    }


# Telemetry the bridge mirrors out to the UI. Periodic signals run at 10 Hz --
# enough resolution for the strip charts to show oscillations and short
# transients, cheap at JSON-scalar sizes. State-like signals (valves, setpoint
# Pose) are on-change with retain=True, so transitions propagate at once and a
# fresh UI tab gets the last value.
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
    EgressMapping(
        UUVTopics.IMU_FILTERED,
        "nautilus/telemetry/imu",
        10.0,
        encoder=_encode_imu_compact,
    ),
    EgressMapping(UUVTopics.BCU_PRESSURE, "nautilus/telemetry/bcu/pressure", 10.0),
    EgressMapping(
        UUVTopics.EXTERNAL_PRESSURE, "nautilus/telemetry/external/pressure", 10.0
    ),
    # Seawater temperature off the STM. Slowly varying (~1 Hz housekeeping on
    # hardware), so the 10 Hz cap is just a ceiling -- the native rate passes
    # through unthrottled, matching the external/pressure sibling above.
    EgressMapping(
        UUVTopics.EXTERNAL_TEMPERATURE, "nautilus/telemetry/external/temperature", 10.0
    ),
    EgressMapping(UUVTopics.BCU_RPM, "nautilus/telemetry/bcu/rpm", 10.0),
    # Measured pump RPM from the STM (~1 Hz on hardware; the sim HAL matches the
    # shape). The 10 Hz cap sits above that rate, so it passes through and the
    # UI can show commanded vs reported.
    EgressMapping(
        UUVTopics.BCU_FEEDBACK_RPM, "nautilus/telemetry/bcu/feedback/rpm", 10.0
    ),
    EgressMapping(
        UUVTopics.BCU_VALVES, "nautilus/telemetry/bcu/valves", 0.0, on_change=True
    ),
    EgressMapping(UUVTopics.ACU_PITCH, "nautilus/telemetry/acu/pitch", 10.0),
    EgressMapping(UUVTopics.ACU_ROLL, "nautilus/telemetry/acu/roll", 10.0),
    # Per-subsystem health. On-change/retained: a payload crosses the tether
    # only on a real online<->offline transition (the liveness node zeroes
    # header.stamp to keep the JSON byte-stable between transitions). A fresh
    # UI tab gets the last states from the retained message.
    EgressMapping(
        UUVTopics.STATUS_LIVENESS,
        "nautilus/status/liveness",
        0.0,
        on_change=True,
    ),
    # ROS-confirmed echo of the dive registration. The bridge's egress
    # subscription hears its own ingress DIVE_INIT publish (rclpy delivers local
    # publications), so the UI renders registration state from what reached the
    # ROS graph. Retained for fresh tabs.
    EgressMapping(UUVTopics.DIVE_INIT, "nautilus/status/init", 0.0, on_change=True),
)


STATUS_TOPIC = "nautilus/status/bridge"
MISSION_ACTIVE_TOPIC = "nautilus/telemetry/mission/active"
HEARTBEAT_PERIOD_S = 2.0

# Lifeguard surfaces. The bridge consumes cmd and heartbeat itself. Status is
# retained, so a fresh UI tab and the bridge's own reconnect both get the
# current armed/engaged state.
LIFEGUARD_CMD_TOPIC = "nautilus/cmd/lifeguard"  # {"data": bool}, QoS 1, retained by UI
LIFEGUARD_HEARTBEAT_TOPIC = "nautilus/cmd/heartbeat"  # QoS 0, ~1 Hz, payload ignored
LIFEGUARD_STATUS_TOPIC = "nautilus/status/lifeguard"
LIFEGUARD_TICK_PERIOD_S = 1.0  # default for the lifeguard_tick_period_s param

# Bridge link-state vocabulary (retained on STATUS_TOPIC):
#   "online"     -- connected; ROS<->MQTT plumbing is live.
#   "offline"    -- clean shutdown (Ctrl-C, ROS shutdown).
#   "link_lost"  -- unclean TCP drop, seen by the broker as the LWT. The bridge
#                   crashed or the tether dropped; either way the plumbing is
#                   broken.
STATUS_ONLINE = "online"
STATUS_OFFLINE = "offline"
STATUS_LINK_LOST = "link_lost"

# Mission state mirrored on MISSION_ACTIVE_TOPIC. A UI-facing collapse of
# pathfinding_node's internal modes:
#   IDLE     -- no mission cached, or last command was stop.
#   LOADED   -- a /path arrived, not yet started.
#   RUNNING  -- /command=true seen after a mission loaded.
MISSION_STATE_IDLE = "IDLE"
MISSION_STATE_LOADED = "LOADED"
MISSION_STATE_RUNNING = "RUNNING"


def _safe_json(payload_dict: Any) -> str:
    """JSON-encode a ROS message dict, mapping NaN/Inf to null.

    Float fields can arrive as NaN -- e.g. an unfilled covariance entry in an
    Imu message. ``allow_nan=False`` turns those into ``null``, which the UI
    parses cleanly and renders as "missing".
    """
    return json.dumps(payload_dict, allow_nan=False, default=lambda _: None)


class MqttBridge(Node):
    """ROS 2 node that bridges Nautilus topics to/from an MQTT broker."""

    def __init__(self, mqtt_client_factory=None, **kwargs) -> None:
        super().__init__("mqtt_bridge", **kwargs)

        # Broker address is a parameter. Defaults to localhost (mosquitto on the
        # mission laptop); set the static IP for the real tether broker (e.g.
        # broker_host:=192.168.2.1).
        self.declare_parameter("broker_host", "127.0.0.1")
        self.declare_parameter("broker_port", 1883)
        self.declare_parameter("client_id", "nautilus_bridge")
        self.declare_parameter("keepalive_s", 30)
        # Dead-man window for the lifeguard once armed. The tick period sets how
        # often the bridge re-checks silence and the tank-empty band; tests
        # shorten it to speed up engage latency.
        self.declare_parameter("lifeguard_timeout_s", 15.0)
        self.declare_parameter("lifeguard_tick_period_s", LIFEGUARD_TICK_PERIOD_S)

        host = self.get_parameter("broker_host").get_parameter_value().string_value
        port = self.get_parameter("broker_port").get_parameter_value().integer_value
        client_id = self.get_parameter("client_id").get_parameter_value().string_value
        keepalive = (
            self.get_parameter("keepalive_s").get_parameter_value().integer_value
        )

        # paho v2 callback API. Tests inject a fake via ``mqtt_client_factory``
        # so the bridge spins without a real broker.
        if mqtt_client_factory is None:
            self._mqtt = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id,
            )
        else:
            self._mqtt = mqtt_client_factory(client_id)
        # Last will: the broker publishes link_lost if the connection drops
        # uncleanly (crash, tether loss). A clean shutdown sends DISCONNECT
        # first, which suppresses the will, so offline and link_lost stay
        # distinct on the UI side.
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
        # Per-topic egress state: _last_emit tracks the previous publish time
        # (throttling), _last_payload the previous encoded payload (on-change
        # dedup + reseed).
        self._last_emit: dict[str, float] = {}
        self._last_payload: dict[str, str] = {}
        self._egress_subs: list = []
        for em in EGRESS_MAP:
            sub = create_subscription_for_topic(
                self, em.ros_topic, self._make_egress_callback(em)
            )
            self._egress_subs.append(sub)

        # --- mission-active mirror --------------------------------------
        # Mission mirror: the last MissionCommand from nautilus/cmd/path plus
        # the current command state. Published on MISSION_ACTIVE_TOPIC on every
        # change, so the UI can show the current mission. Guarded by _state_lock
        # (below): the paho thread (ingress) and the timer thread (the
        # lifeguard's mission re-stop) both write it.
        self._mission_cache: dict | None = None
        self._mission_state: str = MISSION_STATE_IDLE

        # --- lifeguard -----------------------------------------------------
        # _state_lock guards every piece of bridge state shared across threads:
        # the lifeguard latch, the tank/registration fields, the manual-blow
        # flag, and the mission mirror above. The paho thread (_on_mqtt_message)
        # and the rclpy timer (_tick_lifeguard) both touch it, so one lock keeps
        # them in step and pairs each transition with its publish. The helpers
        # that touch this state (_update_mission_mirror, _publish_mission_active,
        # _update_manual_emergency) run with the lock already held; it is not
        # re-entrant.
        #
        # The lifeguard times with time.monotonic(), not the node clock. The
        # node clock is wall time, and the tether's return can bring an NTP step
        # right inside the failsafe's active window. The monotonic clock is
        # immune to that step.
        timeout_s = (
            self.get_parameter("lifeguard_timeout_s").get_parameter_value().double_value
        )
        self._lifeguard = Lifeguard(timeout_s)
        self._state_lock = threading.Lock()

        # Tank-empty stand-down state, all under _state_lock. The registered
        # endpoints and the live tank pressure arrive on the rclpy executor
        # (subscriptions below); _tick_lifeguard reads them. Stood-down means
        # the latch stays engaged but the blow has stopped: the tank is within
        # 10% of empty, so the bladder is full. Disarm clears it. The flag is
        # process-local, so a restart while engaged re-blows for about one tick
        # until the retained init and first tank sample stand it down.
        self._init_tank_empty_pa: float | None = None
        self._init_tank_full_pa: float | None = None
        self._tank_pa: float | None = None
        self._lifeguard_stood_down = False

        # The operator's manual emergency surface rides the same
        # /debug/emergency_surface topic and gets the same tank-empty stand-down.
        # The bridge sees the command pass through its own ingress and tracks
        # "a manual blow is running" here; the tick applies the band check.
        # Stand-down clears this flag, so the next manual engage starts fresh.
        # This guards the UI path.
        self._manual_emergency_active = False

        create_subscription_for_topic(
            self, UUVTopics.BCU_PRESSURE, self._on_tank_pressure
        )
        # The bridge reads the registration as the typed ROS message. rclpy
        # delivers its own ingress publish locally (the same path the
        # status/init echo rides), and the TRANSIENT_LOCAL latch replays it on
        # discovery.
        create_subscription_for_topic(self, UUVTopics.DIVE_INIT, self._on_dive_init)

        # connect_async + loop_start: node init proceeds even if the broker is
        # down at boot. The mqtt thread handles reconnect in the background.
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
        """Build the ROS subscription callback for one egress mapping.

        One callback per entry, each keyed on ``mapping.mqtt_topic`` for its own
        throttle/dedup state in _last_emit and _last_payload.
        """

        encode = mapping.encoder or message_to_ordereddict

        def _callback(msg) -> None:
            try:
                payload_dict = encode(msg)
                payload_str = _safe_json(payload_dict)
            except Exception as exc:
                # Log and drop one bad message; keep the subscription alive.
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
        """Re-publish every on-change topic's last payload, retained, on
        (re)connect.

        paho drops QoS-0 publishes while disconnected, but the on-change dedup
        still records each one in _last_payload. So an on-change frame encoded
        before the broker is reachable can be the only copy the bridge holds.
        The vehicle stack usually boots before the laptop broker, and a broker
        restart drops its retained store, so this is the common case. Re-seeding
        from the cache hands the broker a fresh retained copy of all on-change
        topics, so a late-joining UI tab gets real state. Snapshot the dict:
        egress callbacks on the rclpy executor mutate it while this runs on the
        paho thread."""
        for mqtt_topic, payload_str in list(self._last_payload.items()):
            client.publish(mqtt_topic, payload=payload_str, qos=0, retain=True)

    # --- ingress: MQTT -> ROS -------------------------------------------

    def _on_mqtt_message(self, _client, _userdata, mqtt_msg) -> None:
        # Lifeguard surfaces are handled here, before the ingress map: the
        # bridge consumes them itself.
        if mqtt_msg.topic == LIFEGUARD_HEARTBEAT_TOPIC:
            with self._state_lock:
                # Payload ignored: arrival is the signal.
                self._lifeguard.beat(time.monotonic())
            return
        if mqtt_msg.topic == LIFEGUARD_CMD_TOPIC:
            self._on_lifeguard_cmd(mqtt_msg.payload)
            return

        entry = self._ingress_pubs.get(mqtt_msg.topic)
        if entry is None:
            # Unmapped topic (e.g. a future wildcard subscribe). Log it.
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

        # Mission-active mirror + manual-blow guard: piggy-back on the ingress
        # decode so the UI's "what mission is loaded" view and the blow guard
        # stay in sync with the commands actually dispatched. Both touch state
        # the lifeguard timer also mutates, so take _state_lock; the helpers
        # assume it's held.
        with self._state_lock:
            self._update_mission_mirror(mqtt_msg.topic, payload)
            self._update_manual_emergency(mqtt_msg.topic, payload)

    def _update_mission_mirror(self, mqtt_topic: str, payload: dict) -> None:
        # Caller holds _state_lock.
        changed = False

        if mqtt_topic == PATH_CMD_TOPIC:
            self._mission_cache = dict(payload)
            # First /path moves IDLE -> LOADED. A /path while RUNNING is a
            # mid-run re-dispatch: keep RUNNING, the pathfinder swaps missions.
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
                # Stop -> IDLE, and drop the cached mission, matching
                # pathfinding_node on /command=false.
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
        # Caller holds _state_lock (reads _mission_state / _mission_cache).
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
        # Tank pressure in the sensor's own frame (tank relative to hull), the
        # same stream the registered endpoints came from. Cached for the
        # stand-down check in _tick_lifeguard.
        with self._state_lock:
            self._tank_pa = float(msg.data)

    def _on_dive_init(self, msg) -> None:
        # Registered tank endpoints for the stand-down band. Empty sets the
        # floor; full sets the span the 10% is measured against.
        with self._state_lock:
            empty_pa = float(msg.tank_empty_pa)
            full_pa = float(msg.tank_full_pa)
            # Non-positive (a missing field decodes to 0.0) means "not
            # registered". tank_blow_exhausted re-checks this too.
            self._init_tank_empty_pa = empty_pa if empty_pa > 0.0 else None
            self._init_tank_full_pa = full_pa if full_pa > 0.0 else None

    def _update_manual_emergency(self, mqtt_topic: str, payload: dict) -> None:
        # Track the operator's manual blow so the tick can stand it down at the
        # tank-empty band. Cancel and the red reset both end the blow, so they
        # clear the flag. Caller holds _state_lock.
        if mqtt_topic == EMERGENCY_SURFACE_CMD_TOPIC:
            self._manual_emergency_active = bool(payload.get("data"))
        elif mqtt_topic == DEBUG_RESET_CMD_TOPIC:
            self._manual_emergency_active = False

    def _on_lifeguard_cmd(self, raw: bytes) -> None:
        try:
            arm = bool(json.loads(raw.decode("utf-8")).get("data"))
        except Exception as exc:
            self.get_logger().error(f"lifeguard decode failed: {exc}")
            return
        with self._state_lock:
            if arm:
                self._lifeguard.arm(time.monotonic())
                self.get_logger().info(
                    f"lifeguard armed ({self._lifeguard.timeout_s:.0f} s heartbeat window)"
                )
            else:
                was_engaged = self._lifeguard.engaged
                self._lifeguard.disarm()
                # Disarm is the only thing that clears a tank-empty stand-down.
                # It's a full stand-down, so the manual-blow flag clears too.
                self._lifeguard_stood_down = False
                self._manual_emergency_active = False
                if was_engaged:
                    # Stand the latched emergency surface down, once.
                    self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, False)
                self.get_logger().info("lifeguard disarmed")
            self._publish_lifeguard_status()

    def _tick_lifeguard(self) -> None:
        with self._state_lock:
            was_engaged = self._lifeguard.engaged
            if not self._lifeguard.tick(time.monotonic()):
                # Not engaged: the same tick guards a manual blow at the
                # tank-empty band.
                self._tick_manual_blow_guard()
                return
            if not was_engaged:
                self.get_logger().error(
                    "LIFEGUARD: laptop heartbeat lost -- engaging emergency surface"
                )
                self._publish_lifeguard_status()

            # Stop the mission so the depth PID goes quiet before bcu_debug blows
            # ballast -- same order as the UI's emergency slider. Always on the
            # engage transition; afterwards only while the mirror shows a mission
            # loaded/running (an operator restarted one over a restored link).
            # Skipping it in the steady engaged state keeps the valves from
            # chattering against the emergency hold.
            if not was_engaged or self._mission_state != MISSION_STATE_IDLE:
                self._publish_ingress_bool(COMMAND_CMD_TOPIC, False)
                self._update_mission_mirror(COMMAND_CMD_TOPIC, {"data": False})

            # Tank-empty stand-down: once the tank is within 10% of empty the
            # bladder is full, so stop the blow (one False zeroes RPM and closes
            # the valves) and go quiet. The latch stays engaged until disarm, and
            # the mission re-stop above continues.
            if self._lifeguard_stood_down:
                return
            if tank_blow_exhausted(
                self._tank_pa, self._init_tank_empty_pa, self._init_tank_full_pa
            ):
                self._lifeguard_stood_down = True
                self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, False)
                self._publish_lifeguard_status()
                self.get_logger().warning(
                    f"LIFEGUARD: tank within 10% of empty "
                    f"({self._tank_pa:.0f} Pa vs {self._init_tank_empty_pa:.0f} Pa) "
                    "-- blow stood down, latch stays engaged"
                )
                return
            # Re-publish the engage every tick (idempotent at bcu_debug), so it
            # survives a DEBUG_RESET or a bcu_debug restart.
            self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, True)

    def _tick_manual_blow_guard(self) -> None:
        """Stand a manual emergency blow down at the tank-empty band.

        Called from the lifeguard tick with the lock held. One False stops
        bcu_debug (0 RPM, valves closed). Clearing the flag lets the next manual
        engage start fresh.
        """
        if not self._manual_emergency_active:
            return
        if not tank_blow_exhausted(
            self._tank_pa, self._init_tank_empty_pa, self._init_tank_full_pa
        ):
            return
        self._manual_emergency_active = False
        self._publish_ingress_bool(EMERGENCY_SURFACE_CMD_TOPIC, False)
        self.get_logger().warning(
            f"manual emergency surface: tank within 10% of empty "
            f"({self._tank_pa:.0f} Pa vs {self._init_tank_empty_pa:.0f} Pa) "
            "-- blow stood down"
        )

    def _publish_ingress_bool(self, mqtt_topic: str, value: bool) -> None:
        """Publish a bool on a ROS topic through the ingress mapping, as if the
        command had arrived over MQTT."""
        pub, msg_cls = self._ingress_pubs[mqtt_topic]
        msg = msg_cls()
        msg.data = value
        pub.publish(msg)

    def _publish_lifeguard_status(self) -> None:
        # Include timeout_s so the UI can count the tether-drop window down.
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
        # Seed MISSION_ACTIVE_TOPIC and re-seed lifeguard status on (re)connect,
        # so a fresh UI tab gets a defined state at once and the broker holds the
        # current retained status (paho drops publishes while disconnected). Both
        # read bridge state the executor/timer threads mutate, so under
        # _state_lock.
        with self._state_lock:
            self._publish_mission_active()
            self._publish_lifeguard_status()
        # Same for the on-change egress topics (liveness, valves, setpoint,
        # init): re-publish their retained payloads so a late-joining UI gets
        # real state.
        self._reseed_retained_egress(client)
        self.get_logger().info("mqtt connected")

    def _on_disconnect(self, _client, _userdata, _flags, reason_code, _props=None):
        # Warn only; paho's loop thread handles the reconnect.
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
