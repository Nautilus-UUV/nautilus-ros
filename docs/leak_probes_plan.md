# Plan: Wire leak probes from STM → MQTT → frontend (gentle, under DEPTH) + DB

## Context

The STM firmware reports a hull-leak bitmask on var id `0x2420` (`uint8`, each bit a
distinct leak probe). `stm_com_node.py` already decodes it and republishes it on ROS
(`_publish_leaks`, stm_com_node.py:335 → `UInt8MultiArray(data=[raw])` on
`UUVTopics.INTERNAL_LEAK`), but the signal stops at the ROS graph — it never crosses the
tether to the mission laptop. We want it to reach the operator UI and the time-series DB.

Goal: surface the leak bits **very gently** under the existing DEPTH panel (a passive row
of 8 dots that softly tint amber when a bit is set), and store them in the DuckDB logger.
Per the request, **no detection/alerting logic** — false positives are expected, so the UI
is a pure mirror of the raw bits (no debounce, no thresholds, no red/alarm, no actions).

## What already works (no change needed)

- **STM → ROS**: `stm_com_node._publish_leaks` publishes the bitmask on `INTERNAL_LEAK`
  (`/internal/leak`) as `UInt8MultiArray(data=[raw])`.
- **Topic registry**: `INTERNAL_LEAK` is registered in `uuv_ros_core` —
  `topics.py:23` (`"/internal/leak"`), `message_types.py:38` (`UInt8MultiArray`),
  `qos_profiles.py:45` (`SAFETY_CRITICAL`).
- **DB logger**: `nautilus-command-bridge-frontend/db/` subscribes the wildcard
  `nautilus/#` and auto-flattens **anything** under `nautilus/telemetry/` into the generic
  `sensor` table (`flatten.py`, `writer.py:127`). A `{"data": N}` payload on
  `nautilus/telemetry/internal/leak` lands as one row `(channel="telemetry/internal/leak/data",
  value=N)`. **Zero DB code changes.**

## The one design decision

`UInt8MultiArray`'s default JSON (`message_to_ordereddict`) is
`{"layout": {...}, "data": [5]}` — `data` is an **array**. That breaks the frontend's
`bindScalar` (telemetry.ts:75 requires `typeof msg.data === 'number'`) and would log an
awkward `telemetry/internal/leak/data/0` channel in the DB. Fix with a tiny egress
**encoder** that emits a clean scalar `{"data": <uint8>}` — exactly the established pattern
of `_encode_imu_compact` (mqtt_bridge_node.py:126). Then `bindScalar` works unchanged and
the DB channel is clean.

## Changes

### 1. MQTT bridge — `src/nautilus-ros/src/py_pkg/py_pkg/mqtt/mqtt_bridge_node.py`

- Add a module-level encoder next to `_encode_imu_compact` (~line 140):
  ```python
  def _encode_leaks(msg) -> dict:
      """Collapse the leak UInt8MultiArray to the single bitmask byte the UI/DB use.

      The STM sends one byte (data=[raw]); send it as a scalar so bindScalar reads
      it and the DB logs one clean channel instead of an indexed array element.
      """
      return {"data": int(msg.data[0]) if len(msg.data) else 0}
  ```
- Add one entry to `EGRESS_MAP` (~line 172, with the other STM sensor signals):
  ```python
  EgressMapping(
      UUVTopics.INTERNAL_LEAK,
      "nautilus/telemetry/internal/leak",
      0.0,
      on_change=True,
      encoder=_encode_leaks,
  ),
  ```
  `on_change=True` (retained) is right for a state-like bitmask: every transition — including
  false-positive flickers — propagates faithfully, a fresh UI tab gets the last value, and the
  DB stores transition rows rather than 10 Hz redundancy. Mirrors `BCU_VALVES` (line 180).
  No `uuv_ros_core` edits — the topic/type/QoS are already registered.

### 2. Frontend store — `nautilus-command-bridge-frontend/src/store/telemetry.ts`

- Add a ring buffer ref (near line 58): `const leaks = ref<Sample<number>[]>([])`
- Add a subscription with the existing helper (near line 138):
  `bindScalar('nautilus/telemetry/internal/leak', leaks, CAP_STATE)`
- Export `leaks` in the store's return object (lines 162–176).

No new TelemetryType — reuses `ScalarMsg`.

### 3. Frontend UI — `nautilus-command-bridge-frontend/src/components/DepthValveReadout.vue`

This component **is** the DEPTH panel (left instrument strip: depth block + valve block).
Add a third block below the valves.

- Pull `leaks` from the store (extend the `storeToRefs` destructure at line 17).
- Derive the 8 bits (pure reflection, no logic):
  ```ts
  const latestLeaks = computed(() => leaks.value[0]?.value ?? null)
  const leakBits = computed(() =>
    Array.from({ length: 8 }, (_, i) =>
      latestLeaks.value !== null && (latestLeaks.value & (1 << i)) !== 0),
  )
  ```
- Template — a `.dv-leaks` block after `.dv-valves` (template lines 62–71): a small `LEAK`
  caption (reuse `.dv-cap` styling) over a `.dv-leak-row` of 8 dots:
  ```html
  <div class="dv-leaks">
    <div class="dv-cap">Leak</div>
    <div class="dv-leak-row">
      <span v-for="(wet, i) in leakBits" :key="i" class="dv-leak-dot" :class="{ wet }" />
    </div>
  </div>
  ```
- Styles — mirror the valve `.dv-dot` (lines 139–152) but **gentle**: 8px dots, muted
  `var(--gauge-track)` when dry, soft amber `var(--status-q-text)` when `wet`, **no**
  `box-shadow`/glow, **no** red. e.g.:
  ```css
  .dv-leaks { display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
  .dv-leak-row { display: flex; gap: 6px; }
  .dv-leak-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--gauge-track); border: 1px solid var(--border-btn);
    transition: background 0.15s ease, border-color 0.15s ease;
  }
  .dv-leak-dot.wet { background: var(--status-q-text); border-color: var(--status-q-text); }
  ```

### 4. Tests — `src/nautilus-ros/src/py_pkg/test/node/test_mqtt_bridge_node.py`

Mirror the valve on-change tests (lines 480–504) and the IMU-encoder test (line 431):
- Add a leak publisher to `_BridgeTesterNode` (like `bcu_valves_pub`, an `UInt8MultiArray`
  on `INTERNAL_LEAK`).
- `test_leaks_encode_scalar`: publish `data=[0b101]` → exactly one MQTT message on
  `nautilus/telemetry/internal/leak` with `payload == {"data": 5}`, `retain True`, `qos 0`
  (asserts the array→scalar encoder contract).
- `test_leaks_dedupe_then_change`: repeat same byte → one publish; change byte → a second.
- `test_every_egress_topic_has_a_subscription` (line 350) covers the new mapping automatically.

## Verification

**Tier 2 (node tests)** — from `src/nautilus-ros/src/py_pkg`, ros + install sourced:
```bash
colcon build --packages-select py_pkg && source install/setup.bash   # from dave_ws root
cd src/nautilus-ros/src/py_pkg
/usr/bin/python3 -m pytest test/node/test_mqtt_bridge_node.py -v
```

**End-to-end smoke** (mosquitto + bridge + UI + DB on the laptop):
```bash
# 1. bridge running against the local broker, then inject a leak on ROS:
ros2 topic pub --once /internal/leak std_msgs/UInt8MultiArray "{data: [5]}"
# 2. confirm the clean scalar crossed the tether:
mosquitto_sub -t nautilus/telemetry/internal/leak -v      # => {"data": 5}
# 3. UI: npm run dev (:3000) — bits 0 and 2 tint soft amber under the DEPTH strip.
# 4. DB: query the logger's DuckDB for the stored channel:
#    SELECT ts, value FROM sensor WHERE channel = 'telemetry/internal/leak/data' ORDER BY ts DESC;
```

## Notes / scope

- Frontend lives in its own repo (`nautilus-command-bridge-frontend/`); bridge + tests in
  `src/nautilus-ros/` (commit/PR into its `dev`). No edits to `uuv_ros_core` or the DB code.
- `leaks` is left in the telemetry store for any future Telemetry-tab chart, but this change
  only renders it under the DEPTH panel as requested.
