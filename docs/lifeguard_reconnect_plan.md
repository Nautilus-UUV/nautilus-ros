# E-001: Lifeguard / tether-reconnect MQTT recovery

> Status: **implemented** (Tier 1 + Tier 2 green). Tracks test E-001.

## Context

**Symptom (test E-001):** After the Lifeguard protocol engages (glider lost the
mission-laptop tether and surfaced), reconnecting the tether did *not* bring the
UI back online — no telemetry, no command packets. The operator could still
`ping`/`ssh` the main board, but only **rebooting the Pi** restored data flow.

**Root cause:** The mosquitto **broker runs on the mission laptop**; the glider's
`mqtt_bridge_node` (paho client) connects up the tether over TCP 1883. The
frontend talks to the broker over localhost WebSocket, so the frontend↔broker
link never breaks on a tether drop — the frontend auto-recovers the instant the
bridge republishes `online` + its 2 s tick. The bug was glider-side: the bridge
relied **solely on paho's `loop_start()` background-thread auto-reconnect**, with
`_on_disconnect` only logging and **no application-level supervision**. When the
tether is physically pulled, the Pi's TCP socket goes half-open (no FIN/RST) and
paho can wedge — either stuck disconnected even after the cable returns, or left
believing it is still connected while nothing flows. The bridge process stays
alive (SSH/ping work) but publishes telemetry into a dead socket; only a process
restart (Pi reboot) gives a fresh `connect_async`. The Lifeguard latch is **not**
the cause of "no data" — egress telemetry is independent of it and it works as
designed.

**Decisions:** keep **manual disarm** of the Lifeguard after reconnect (the
anti-flapping latch is deliberate; the operator clears it from the existing
`LifeguardToggle`); **in-process reconnect supervisor only**, no systemd.

## Implementation

### `ReconnectSupervisor` — pure backstop (`py_pkg/mqtt/reconnect.py`)

A tiny pure state machine (no ROS, no paho, no clock), fed `(connected, rx_age_s,
now)` on a timer, returning whether to force a full client rebuild:

- **Disconnected past a grace window** → rebuild. While `connected is False`, run
  a `_down_since` clock; rebuild once `now - _down_since >= down_grace_s`. The
  grace lets paho's own backoff try first; only persistent failure escalates.
- **Stale-connected wedge** → rebuild. `connected is True` but `rx_age_s >=
  rx_silence_s` (nothing inbound for far longer than the laptop's ~1 Hz
  heartbeat). `rx_age_s is None` (nothing ever received) never trips.
- `note_rebuilt()` resets timers so a fresh, briefly-disconnected client doesn't
  immediately re-trip.

### `MqttBridge` wiring (`py_pkg/mqtt/mqtt_bridge_node.py`)

- `_build_client()` extracted from `__init__`: create (via factory or paho), set
  will, wire callbacks, `reconnect_delay_set`, `connect_async`, `loop_start`. It
  repoints `self._mqtt` at the new client **before** `loop_start()` so the
  retained re-seed in `_on_connect` lands on the right socket. Used by both
  startup and the rebuild path.
- `_on_mqtt_message` stamps `self._last_rx_monotonic` (under `_state_lock`) — the
  only field crossing the paho↔executor boundary; the inbound-liveness signal.
- `_connection_watchdog()` on a ROS timer reads `self._mqtt.is_connected()` +
  `rx_age`, and on `supervisor.should_rebuild(...)` calls `_rebuild_client()`.
- `_rebuild_client()`: `loop_stop()` (joins the old paho thread) + `disconnect()`
  best-effort, build a fresh client, reset the rx clock, `note_rebuilt()`. The
  existing `_on_connect` then re-subscribes, republishes `online`, restarts the
  tick, and re-seeds retained mission/lifeguard/egress state — so telemetry and
  the UI recover, with the Lifeguard shown `engaged` for the operator to disarm.
- New ROS params: `reconnect_grace_s` (12 s), `rx_silence_s` (20 s),
  `connection_watchdog_period_s` (5 s).

**Concurrency:** `spin_node` uses `rclpy.spin()` → single-threaded executor, so
the watchdog, egress publishes, heartbeat, and lifeguard tick are serialized on
one thread; the client swap can't race a publish. `loop_stop()` joins the old
paho thread before the swap. No new lock required.

No changes to `lifeguard.py`, the frontend, or `mosquitto.conf`.

## Tests

- **Tier 1** `test/unit/test_reconnect_supervisor.py` — grace, stale-connected,
  fresh-boot, reconnect-clears-clock, `note_rebuilt` reset (7 tests).
- **Tier 2** `test/node/test_mqtt_bridge_node.py` — `TestReconnectWatchdog`:
  disconnect-past-grace rebuilds + resubscribes + republishes online;
  stale-connected rebuilds then doesn't thrash; healthy link never rebuilds.
  `FakeMqttClient` gained `is_connected()`; the harness factory builds a fresh
  fake per call so a rebuild is observable as a new `harness.fakes` entry.

Run:

```bash
cd src/nautilus-ros/src/py_pkg
/usr/bin/python3 -m pytest test/unit/test_reconnect_supervisor.py test/node/test_mqtt_bridge_node.py -q
```

## End-to-end verification (bench)

Start mosquitto + frontend on the laptop and the bridge on the Pi
(`mqtt_broker_host:=<laptop-ip>`). Arm the lifeguard, confirm telemetry. **Unplug
the tether** → UI flips to `link_lost` in ~3.5 s; after the window the lifeguard
engages and the glider surfaces. **Reconnect the tether** → within
~`reconnect_grace_s` the bridge log shows a rebuild ("mqtt connected" from a fresh
client), telemetry resumes, and the Lifeguard toggle shows `engaged` — **without
rebooting the Pi**. Operator hits disarm → emergency-surface stands down → fully
operational. E-001 satisfied.
