# BCU debug: "Pump RPM until tank pressure Y" command

## Context

`bcu_debug_node` today exposes a *timed* pump command: "run at X RPM for N seconds, then stop." Useful for short bench pokes, but for bring-up tasks like "fill the bladder until the tank reaches Y kPa" the operator currently has to eyeball the live tank-pressure telemetry and stop the pump by hand (often after a few duration-bounded trial pumps). We want a closed-loop variant that watches `/bcu/pressure` and stops itself when the tank crosses the target. Same UI pattern as the existing "Pump In Bladder / Pump Out Bladder" buttons, sitting right below them in the Debug Commands panel.

Stop logic follows from the sign of the commanded RPM, matching the existing inflate/deflate convention: pumping IN (positive RPM) moves fluid out of the tank → tank pressure drops → stop when `tank_pa ≤ target_pa`; pumping OUT (negative RPM) moves fluid back to the tank → tank pressure rises → stop when `tank_pa ≥ target_pa`. The existing `MAX_PUMP_S = 30 s` cap is reused as a hard safety floor; a stuck sensor or unrealistic target can't leave the motor on indefinitely.

## Approach

### 1. New ROS message and topic registration

- **New message**: `src/nautilus-ros/src/nautilus_msgs/msg/BcuPumpUntilPressureCommand.msg`

  ```
  # Debug/manual BCU pump command with a tank-pressure stop condition.
  # Consumed by bcu_debug_node, which drives /bcu/rpm at `rpm` until the
  # tank pressure (/bcu/pressure) crosses `target_pressure_pa` in the
  # direction implied by the sign of `rpm`. A hard MAX_PUMP_S safety cap
  # always applies. Bypasses pathfinding/depth PID -- bench testing only.

  int16 rpm
  int32 target_pressure_pa
  ```

- Register the new message in `src/nautilus-ros/src/nautilus_msgs/CMakeLists.txt` next to `BcuPumpCommand.msg`.

- **Registry edits** (one line each, alongside the existing `DEBUG_BCU_RPM` entries):
  - `uuv_ros_core/topics.py`: `DEBUG_BCU_RPM_UNTIL_PRESSURE = "/debug/bcu/rpm_until_pressure"`
  - `uuv_ros_core/message_types.py`: map it to `BcuPumpUntilPressureCommand`
  - `uuv_ros_core/qos_profiles.py`: `UUVQoS.COMMAND` (same as `DEBUG_BCU_RPM`)

### 2. `bcu_debug_node.py` — add closed-loop pump mode

File: `src/nautilus-ros/src/py_pkg/py_pkg/debug/bcu_debug_node.py`.

The node already subscribes to `EXTERNAL_PRESSURE`; add a sibling subscription to `UUVTopics.BCU_PRESSURE` via `create_subscription_for_topic`. Cache the latest reading as `self._tank_pa: float | None`.

Add a new command subscription `DEBUG_BCU_RPM_UNTIL_PRESSURE → self._on_rpm_until_pressure_cmd`. The callback:

- Runs through the same gates as `_on_rpm_cmd` (emergency-active → ignore; manual-override-off → ignore + warn).
- Validates: `rpm != 0`; if `self._tank_pa is None` (no sample yet) → warn and ignore. If the target is already on the "stop side" of the current pressure for the requested RPM sign → warn ("already past target") and zero the motor.
- Stores `self._held_rpm`, `self._target_pressure_pa`, and re-uses `self._pump_deadline_s = now + MAX_PUMP_S` as the hard safety cap (same field as the timed mode — the two modes are mutually exclusive on this node).
- Warns about valve 1 not being open, same as the timed mode.
- Publishes the first RPM sample immediately, same as `_on_rpm_cmd`.

In `_on_tick`, extend the held-pump branch with a pressure-target check that runs *before* the deadline check:

- If `self._target_pressure_pa` is set: compute `crossed = (held_rpm > 0 and tank_pa <= target) or (held_rpm < 0 and tank_pa >= target)` (skip if `tank_pa is None`). If crossed → log "tank pressure target reached", zero motor, clear `_target_pressure_pa` and `_pump_deadline_s`.
- The existing deadline branch still fires as the safety floor — same `_publish_rpm(0)` path; just additionally clear `_target_pressure_pa`.

Reuse `_publish_rpm`, `_held_rpm`, and the 10 Hz `_on_tick` heartbeat exactly as the timed mode does. A new command of either kind (or a duration=0 stop, or override drop, or emergency engage) clears `_target_pressure_pa` along with `_pump_deadline_s` and `_held_rpm` — consolidate this in `_on_override`, `_on_emergency_cmd`, and any "explicit stop" path.

Docstring at the top of the file gets a new bullet under "Command surfaces" describing the new mode, in the same conversational tone as the existing ones.

### 3. MQTT bridge — register the new command

File: `src/nautilus-ros/src/py_pkg/py_pkg/mqtt/mqtt_bridge_node.py`.

Add one line to `INGRESS_MAP` next to the existing `DEBUG_BCU_RPM` entry:

```python
IngressMapping(UUVTopics.DEBUG_BCU_RPM_UNTIL_PRESSURE,
               "nautilus/cmd/debug/bcu/rpm_until_pressure", 1),
```

`set_message_fields` auto-handles the new message type — no other bridge changes needed.

### 4. Frontend — new control in DebugCommandsPanel

File: `nautilus-command-bridge-frontend/src/components/CommandPanels/DebugCommandsPanel.vue`.

Insert a new `<div class="qc-section">` block right after the existing "BCU Pump (RPM for X seconds)" section (after line 151, before "BCU valves"). Mirror the existing structure exactly:

- Section label: `"BCU Pump (RPM until tank pressure Y)"`
- Two `.qc-input-group` inputs in a `.qc-pump-inputs` grid: `RPM` and `Tank Pressure Target (Pa)` (numbers; same step convention as the existing RPM input). Pa is the wire unit per `mqtt_bridge_node.py`'s "no unit conversion at the bridge" rule, so the operator types Pa, matching what they read off the existing tank-pressure telemetry.
- Same two-button `.qc-row`: "Pump In Bladder" and "Pump Out Bladder", with the same `mdi-arrow-up-bold` / `mdi-arrow-down-bold` icons, calling a new `sendPumpUntilPressure(action)` that signs the RPM the same way `sendPump` does and publishes to a new module-scope constant `const PUMP_UNTIL_TOPIC = 'nautilus/cmd/debug/bcu/rpm_until_pressure'`.
- Reuse the existing `valve1Open` computed for the "valve 1 closed" warning — same `<p v-if="!valve1Open" class="qc-warn">` block under the buttons.
- Inline live-readout under the inputs: read `telemetry.bcuPressure[0]?.value` (already wired in `store/telemetry.ts` via `bindScalar('nautilus/telemetry/bcu/pressure', bcuPressure, …)`) and render it as "Current: X Pa" in the section label area for operator context. Use the existing `.qc-section-label` / hint styling rather than adding new CSS.

Publish payload shape:

```ts
{ rpm: <signed number>, target_pressure_pa: <number> }
```

Field names match the ROS message exactly — same wire convention as the existing `{ rpm, duration_s }` payload.

No new components, no new store, no new CSS classes — everything reuses existing `.qc-section` / `.qc-pump-inputs` / `.qc-input-group` / `.qc-btn` / `.qc-row` / `.qc-warn` and the `mqttBridge.publish` + `useTelemetryStore` paths.

## Critical files

- `src/nautilus-ros/src/nautilus_msgs/msg/BcuPumpUntilPressureCommand.msg` (new)
- `src/nautilus-ros/src/nautilus_msgs/CMakeLists.txt`
- `src/nautilus-ros/src/py_pkg/py_pkg/uuv_ros_core/topics.py`
- `src/nautilus-ros/src/py_pkg/py_pkg/uuv_ros_core/message_types.py`
- `src/nautilus-ros/src/py_pkg/py_pkg/uuv_ros_core/qos_profiles.py`
- `src/nautilus-ros/src/py_pkg/py_pkg/debug/bcu_debug_node.py`
- `src/nautilus-ros/src/py_pkg/py_pkg/mqtt/mqtt_bridge_node.py`
- `nautilus-command-bridge-frontend/src/components/CommandPanels/DebugCommandsPanel.vue`

## Verification

1. **Rebuild + source**: from `/home/girji/dave_ws` →
   `colcon build --packages-select nautilus_msgs py_pkg && source install/setup.bash`.
2. **Tier 2 node test** (optional but recommended): add a focused node test under `src/nautilus-ros/src/py_pkg/test/node/test_bcu_debug_node.py` (or extend an existing one) that spins `BcuDebugNode` in a `SingleThreadedExecutor`, fakes a sequence of `BCU_PRESSURE` messages on either side of a target, publishes a `DEBUG_BCU_RPM_UNTIL_PRESSURE` command after raising the manual override, and asserts `/bcu/rpm` settles to 0 within the expected number of ticks. Run with `/usr/bin/python3 -m pytest test/node/ -v` per the testing convention.
3. **Live sim**:
   ```
   ros2 launch nautilus_hal trim_sim.launch.py headless:=false mission_autostart:=false
   ```
   In another shell, run `mosquitto_pub -t nautilus/cmd/control/manual_override -m '{"data": true}'` to raise the override, then `mosquitto_pub -t nautilus/cmd/debug/bcu/rpm_until_pressure -m '{"rpm": 500, "target_pressure_pa": 110000}'`. Watch `ros2 topic echo /bcu/rpm` and `/bcu/pressure` and confirm RPM drops to 0 once tank pressure crosses 110 kPa.
4. **Frontend**: in `nautilus-command-bridge-frontend/` run `npm run dev`, open the Debug Commands panel, enable Manual Override, type RPM + target Pa, click "Pump In Bladder" — confirm tank-pressure live readout updates and the motor stops when the target is reached. Cross-check the warning fires when valve 1 is closed.
