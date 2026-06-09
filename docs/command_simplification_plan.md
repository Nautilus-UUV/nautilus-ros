# Plan: Simplify the command/override interface (boolean `/command`, no manual-override mode, red Reset)

## Context

The Nautilus control interface has accreted complexity that this change removes:

- **`/command` is a 3-value string** (`"start"`/`"stop"`/`"abort"`) where `stop` only *freezes* the last
  setpoint and `abort` does a hidden resurface. Operators want one unambiguous thing: **run the loaded
  mission, or stop and go safe.**
- **A "Do Nothing" mission** (`MissionId.DO_NOTHING=3`) exists only to drive the controllers back to a
  quiet, no-setpoint hold. Once `stop` resets the system to that exact state, Do-Nothing is redundant.
- **A manual-override mode** (two latched Bool topics `/control/manual_override`, `/control/acu_override`
  plus a UI slider) gates whether the PID controllers or the debug nodes own the actuators. It's a second
  mode the operator has to reason about and keep in sync with the mission state.

Target outcome (matches the supplied architecture diagram, which shows **no override topics** and the debug
nodes driving the actuator topics directly):

1. `/command` becomes a **`std_msgs/Bool`**: `true` = start the loaded Pathfinding mission, `false` = **stop**
   = reset to a clean initial state (no RPM published, no mission, no path loaded, valves closed). No resurface.
2. **Remove manual-override mode.** Engaging any manual/debug command first **stops the active mission**
   (which silences the controllers), then drives the actuator via the debug node. Different debug nodes
   (BCU vs ACU) drive disjoint topics so they never interfere; the same debug node still preempts itself.
3. **A red Reset button** (single click) that stops every debug node *and* stops the mission — full all-stop
   to the clean initial state, including cancelling an in-progress Emergency Surface.

### The mechanism that makes override removal work (read this first)

Today, contention between a controller and a debug node is avoided by the override flag forcing the
controller to stand down. We replace that with a **silence model**:

- A controller (`depth_node`, `acu_node`) drives its actuator topic **only while it has an active target**
  (i.e. a mission is running and publishing `POSITION_TARGET`). With no target it publishes the safe state
  **once** and then **goes silent** (stops publishing entirely).
- **The controllers subscribe to `/command` directly.** On `/command`=false they drop their target, publish
  one safe-stop (0 RPM + valves closed / neutral ACU), and go silent. The actuator wire is now free for a
  debug node with no contention. (This replaces the old `/control/reset` relay — see below.)
- The debug nodes no longer gate on an override flag — receiving a command means "drive the wire."

**`/control/reset` is removed.** It was only an internal relay: pathfinding's way to tell the controllers
"drop target, go safe" on a stop (its sole publisher was pathfinding; its sole subscribers were the two
controllers). Since the stop is already broadcast on `/command`, the controllers just subscribe to that
instead — one fewer topic, no relay. Trade-off: the controllers now couple to the command plane (the diagram
routes `/command` only to pathfinding), and mission **self-completion** (`is_done`) still does not reset the
controllers (they hold the last setpoint) — same as today; only an explicit `/command`=false goes safe.

**Load-bearing safety fact (verified in `depth_node.py:166-172`):** the STM/CAN comms have *no staleness
watchdog* — they re-ship the last cached setpoint forever. So "go silent" is only safe because every stop
path publishes **one** zero/closed/neutral *before* the silence. This is mandatory, not optional.

**The one ordering subtlety:** when a manual command stops the mission, the controllers' reset must NOT also
cancel the just-issued debug command. That's why there are **two disjoint stop signals**:
`/command`=false (pathfinding + the two controllers) and `/debug/reset` (debug nodes only, published only by
the red Reset button). Debug nodes do **not** subscribe to `/command`; controllers do **not** subscribe to
`/debug/reset`.

---

## New topic / payload contract

| Topic (ROS / MQTT) | Type | Meaning |
|---|---|---|
| `/command` / `nautilus/cmd/command` | `std_msgs/Bool` | `true`=start loaded mission, `false`=stop+reset-to-idle. **Subscribed by pathfinding AND both controllers** (controllers reset+silence on `false`). |
| `/debug/reset` / `nautilus/cmd/debug/reset` | `std_msgs/Empty` | **NEW** — stop both debug nodes (zero/close/neutral, cancel emergency, go silent) |
| `/control/reset` | `std_msgs/Empty` | **REMOVED** — folded into `/command`=false (controllers subscribe to `/command` directly) |
| `/control/manual_override`, `/control/acu_override` | — | **REMOVED** (both ingress + telemetry mirror) |

Removed payloads: `/command` string values `"start"`/`"stop"`/`"abort"`; the Do-Nothing mission id `3`.

**Net topology:** this refactor *deletes three* coordination topics (`/control/manual_override`,
`/control/acu_override`, `/control/reset`) and *adds one* (`/debug/reset`) — net **−2 topics**, and every
actuator-wire writer sheds its override-gate branch (node logic net decreases). A design exploration
(5 framings, adversarially verified) confirmed a second stop-signal beyond `/command` is **provably
unavoidable**: the node set "stop-everything" must reach, minus the set `/command`=false already reaches
(pathfinding + both controllers), is *exactly the two debug nodes*, and no existing topic has that subscriber
set; folding it into `/command` breaks the Bool contract (C1), the manual-engage ordering (C3, debug command
self-cancels), or debug-tool faithfulness (C5, sentinel overload). `/debug/reset` is the single irreducible
coordination topic that survives.

> Step 0 (on implementation, post-approval): per project convention, save this plan to
> `src/nautilus-ros/docs/command_simplification_plan.md`.

---

## Part A — ROS stack (`src/nautilus-ros/src/py_pkg/`)

Implement in this dependency order. Run `pytest test/unit test/node` after A1–A6.

### A1. Registry triplet (`py_pkg/uuv_ros_core/`)
- `topics.py`: ADD `DEBUG_RESET = "/debug/reset"`; REMOVE `CONTROL_MANUAL_OVERRIDE`, `CONTROL_ACU_OVERRIDE`, **and `CONTROL_RESET`**.
- `message_types.py`: change `UUVTopics.COMMAND: String` → `Bool`; ADD `DEBUG_RESET: Empty`; REMOVE the two override entries and `CONTROL_RESET`; drop now-unused `String` import (`Empty` is still used by `DEBUG_RESET`).
- `qos_profiles.py`: ADD `DEBUG_RESET: UUVQoS.CONTROL` (RELIABLE + volatile — a momentary event, not state to replay). Volatile is correct here **because the debug nodes construct in the silent/idle state**: a node that is down at publish-time isn't driving the wire and comes up safe, so a missed reset can never strand the actuator. (Conservative alternative if desired: `UUVQoS.COMMAND` so a late joiner latches the last reset — harmless but semantically odd for a one-shot.) REMOVE the two override entries and `CONTROL_RESET`. `COMMAND` keeps `UUVQoS.COMMAND` (reliable + transient-local, so a late-joining controller still sees the latest start/stop).
- `node_factory.py` / `__init__.py`: no change (generic over the maps).

### A2. Remove the "Do Nothing" mission (`py_pkg/path/missions/`)
- DELETE `do_nothing.py`.
- `factory.py`: remove the `DoNothingMission` import, `DO_NOTHING = 3` enum value, and the `_REGISTRY` entry.
- `__init__.py`: remove the `DoNothingMission` import/`__all__` entry.
- `profile.py`: reword the `reference()` docstring to drop the Do-Nothing reference (the "return None = no setpoint this tick → safe hold" contract stays; SURFACE/SAWTOOTH still use it between phases).

### A3. Pathfinding rewrite (`py_pkg/path/pathfinding.py`)
- Module docstring + imports: `from std_msgs.msg import Bool` (drop both `String` and `Empty` — pathfinding no longer emits a reset).
- REMOVE `self._reset_pub` and its `create_publisher_for_topic(self, UUVTopics.CONTROL_RESET)` (controllers now hear stop on `/command` directly).
- State machine becomes `IDLE | LOADED | RUNNING` (drop `STOPPED`).
- Rewrite `_on_command(self, msg: Bool)`: `if bool(msg.data): self._handle_start()` else `self._handle_stop()`.
- `_handle_start` (lines 108-138): REMOVE the `resets_control_on_start` block (131-137). Keep precondition gating + `_start_pending` queueing.
- DELETE `_handle_abort` (140-152).
- ADD `_handle_stop`: clear `_mission`/`_mission_cmd`/`_mission_t0_s`/`_start_pending`, set `_mode="IDLE"`. Publish **no** `POSITION_TARGET` (no resurface). (No reset to emit — the controllers reset themselves off the same `/command`=false.)
- `_tick`: the `_mode != "RUNNING"` guard already covers the dropped `STOPPED`; reword the Do-Nothing comment (172-174) to "a mission may decline to command this tick (reference→None)". Mission self-completion (`is_done`, 164-170) is unchanged — goes IDLE, holds last setpoint (not in scope).

### A4. `depth_node.py` — silence model + override removal + `/command` reset
- Imports: keep `Bool` (now used for the `/command` subscription), keep `Empty, Int16, UInt8`.
- Remove `self._manual_override`, the `manual_override_subscriber` (120-125), and `_on_manual_override` (158-174). Keep `_publish_bcu_stop` (the safe-stop primitive).
- Replace the `CONTROL_RESET` subscription (127-132) with a `UUVTopics.COMMAND` (Bool) subscription → `_on_command`.
- Rename `_on_reset` → `_on_command(self, msg: Bool)`: `if bool(msg.data): return` (start is a no-op; just wait for `POSITION_TARGET`); else do the existing reset body (drop target, `control_system.reset()`, zero state) and ADD `self._publish_bcu_stop()` at the end (one 0-RPM + valves-closed before going silent).
- `control_loop` (195-206): remove the `_manual_override` early-return; **change the no-target branch from `_publish_bcu_stop(); return` to just `return`** (silent). The one-shot safe-stop now lives in `_on_command(false)`.
- `__init__`: after the publishers are created, call `self._publish_bcu_stop()` once so the boot wire state is unambiguously 0/closed (defensive; the comm layers also default to 0).

### A5. `acu_node.py` — silence model + override removal + roll gate + `/command` reset
- Imports: keep `Bool` (now used for `/command`), keep `Empty, Int16, Int32`.
- Remove `self._manual_override`, the `CONTROL_ACU_OVERRIDE` subscription (117-122), and `_on_manual_override` (163-176). Keep `_publish_acu_neutral`.
- Replace the `CONTROL_RESET` subscription (124-129) with a `UUVTopics.COMMAND` (Bool) subscription → `_on_command`.
- ADD `self._target_active = False` in `__init__`; set `True` in `target_pose_callback`; set `False` in the reset body.
- `control_loop` (196-202): replace the `_manual_override` early-return with `if not self._target_active: return`. **This is required** because `_update_roll()` currently runs unconditionally (line 202) and would otherwise keep publishing `/acu/roll` against pose noise after a stop, contending with `acu_debug`.
- Rename `_on_reset` → `_on_command(self, msg: Bool)`: `if bool(msg.data): return`; else the existing state wipe + `self._target_active = False` + `self._publish_acu_neutral()` once (0 pitch + 0 roll) before silence.

### A6. Debug nodes (`py_pkg/debug/`)
**`bcu_debug_node.py`:**
- Remove the `CONTROL_MANUAL_OVERRIDE` subscription, `self._manual_override`, and `_on_override`. ADD a `DEBUG_RESET` subscription → `_on_reset`.
- Remove the `if not self._manual_override:` guards in `_on_rpm_cmd`, `_on_rpm_until_pressure_cmd`, `_on_valve_cmd` (keep the `if self._emergency:` guards). Self-preemption via `_clear_pump_state()` is unchanged.
- `_on_tick` (302-362): remove the `if not self._manual_override: return`. Replace the unconditional heartbeat with an **active-session** notion: heartbeat only while a pump session is live (`_pump_deadline_s is not None`), a valve mask is held, emergency is active, **or** a short trailing-zero flush counter is running. When fully idle → publish nothing (silent). On session-end/reset, set the flush counter (e.g. 5 ticks ≈ 0.5 s of trailing zeros) so the terminal 0 beats the throttled MQTT egress, then go silent.
- ADD `_on_reset(self, _msg: Empty)`: `_clear_pump_state()`, clear valve mask, **cancel emergency** (`_emergency=False`, clear its deadline — per the user's "full all-stop" choice), publish one `0` RPM + `0` valves, arm the trailing-zero flush, then silent.

**`acu_debug_node.py`:**
- Remove the `CONTROL_ACU_OVERRIDE` subscription, `self._manual_override`, and `_on_override`. ADD a `DEBUG_RESET` subscription → `_on_reset`. Drop the now-unused `Bool` import.
- Remove the `if not self._manual_override:` guards in `_on_pitch_cmd`, `_on_roll_cmd`. `_on_tick` heartbeats held setpoints; both `None` → silent (already the body's behavior).
- ADD `_on_reset`: set `_pitch_mm=None`, `_roll_cdeg=None`, publish neutral (0/0) once (with a short flush like BCU to beat the egress throttle), then silent.

**`auto_mission.py`:** `from std_msgs.msg import Bool`; `_emit_start` publishes `Bool(data=True)`. The publisher is created via `create_publisher_for_topic(self, UUVTopics.COMMAND)`, which now resolves to `Bool` automatically.


### A7. MQTT bridge (`py_pkg/mqtt/mqtt_bridge_node.py`)
- `INGRESS_MAP`: keep `nautilus/cmd/command`→`COMMAND` (now ingests `{"data": true/false}` into a `Bool`); REMOVE the two override ingress mappings; ADD `IngressMapping(UUVTopics.DEBUG_RESET, "nautilus/cmd/debug/reset", 1)` (the frontend reaches ROS only through the bridge, so this is required).
- `EGRESS_MAP`: REMOVE the two override telemetry mirrors.
- `_update_mission_mirror` (350-378): the `nautilus/cmd/command` branch now reads a **bool** — `data=bool(payload.get("data"))`; `data and cache → RUNNING`, `not data → IDLE`. DELETE the `"abort"` branch. Clear the mission cache on stop (mirrors pathfinding's new clean-idle).
- Update docstring/vocabulary comments (drop "abort"; `/command` is Bool).

### A8. setup.py / launch / docs
- `setup.py`: no entry-point changes (all four debug/auto nodes stay; Do-Nothing was a library class).
- DELETE `src/dave/hal/nautilus_hal/launch/do_nothing_sim.launch.py` (in the `dave` subrepo — the natural completion of the Do-Nothing removal; nothing else includes it).
- `py_pkg/launch/control_stack.launch.py`: update the composition docstring (drop the manual-override paragraph; `nautilus/cmd/*` now maps to `/command + /path + /debug/* + /debug/reset`).
- `docs/running_sim.md`: change `/command ... String "{data: start}"` examples → `Bool "{data: true}"`; DELETE the "Do Nothing Mission Profile" section.
- `docs/control_topology.wsd`: `/command` edge → `Bool`; optionally add `/debug/reset`.
- `pid/depth_control_system.py` + `pid/acu_axis_controller.py`: `reset()` docstrings — "(Do-Nothing mission)" → "(mission stop)".

### A9. Tests (`test/unit`, `test/node`, `test/sim`)
Update `test/node/conftest.py` **first** (shared harness), then the dependents:
- `conftest.py`: remove `publish_manual_override` / `publish_acu_override` from the depth/acu/bcu-debug/acu-debug tester nodes + harness delegates. For the depth/acu testers, replace the `CONTROL_RESET` publisher (`publish_reset`) with publishing `/command`=false (Bool) — i.e. the controllers are now reset via `publish_command(False)`. Add `publish_reset` via `DEBUG_RESET` to the two **debug**-tester nodes. Remove the pathfinding tester's `CONTROL_RESET` subscriber/`received_resets` (pathfinding no longer emits it). Change `_PathfindingTesterNode.publish_command` to publish `Bool` (signature `publish_command(start: bool)`).
- DELETE `test/unit/test_do_nothing.py`.
- `test/node/test_pathfinding_node.py`: drop the `DO_NOTHING` constant; `publish_command("start"/"stop")`→`(True/False)`; rewrite `test_stop_*` for IDLE + cleared mission + no further `POSITION_TARGET` (no `CONTROL_RESET` assertion — pathfinding no longer emits it); DELETE `TestAbortCommand` and `TestDoNothingMission`.
- `test/node/test_depth_node.py`: DELETE `TestManualOverride`; rewrite the reset test to drive `/command`=false (Bool) instead of `CONTROL_RESET`, asserting the **silence model** (after stop: exactly one 0-RPM+closed, then *no* further emissions — inverse of today's `>= 2`).
- `test/node/test_acu_node.py`: DELETE `TestManualOverride`; reset test drives `/command`=false and asserts one neutral (0/0) then silence (and silence persists without a new target, via `_target_active`).
- `test/node/test_bcu_debug_node.py` / `test_acu_debug_node.py`: drop the `_enter_manual` helper + calls; replace "ignored without override" with "drives wire immediately"; replace "dropping override" with "`/debug/reset` zeros + silences"; rework the idle-heartbeat tests to "idle is silent" + a "session-end flushes trailing zeros" test.
- `test/node/test_mqtt_bridge_node.py`: `{"data":"start"/"stop"}`→`{"data":true/false}`; DELETE `test_abort_clears_cache`; egress-wiring test auto-adjusts.
- `test/sim/` (6 drivers: `test_trim_neutral_sim.py`, `test_trim_neutral_sim_gt.py`, `test_surface_sim.py`, `test_sawtooth_sim.py`, `test_forward_map_sim.py`, `test_hydrodynamics_sampling_sim.py`): `msg = String(); msg.data="start"` → `msg = Bool(); msg.data=True`. Publisher type resolves via the registry automatically.

---

## Part B — Frontend (`nautilus-command-bridge-frontend/`)

The `frontend_claude.md` store map is stale: there is **no `commands` store**. The only store every command
component already imports is `mqttBridge`, so the shared command logic goes there.

### B1. Delete orphaned files
- `src/store/overrides.ts` (whole `useOverridesStore`; this also removes the only consumer of the
  `nautilus/telemetry/control/manual_override` subscription).
- `src/components/CommandPanels/ManualOverrideSlider.vue`.

### B2. `src/store/mqttBridge.ts` — centralize command actions (DRY)
Add module-top constants (`CMD_COMMAND='nautilus/cmd/command'`, `CMD_PATH='nautilus/cmd/path'`,
`CMD_DEBUG_RESET='nautilus/cmd/debug/reset'`) and four actions in the store, exported alongside `publish`:
- `stopMission()` → `publish(CMD_COMMAND, { data: false })` (idempotent; single source of truth for "stop").
- `startMission(cmd)` → `publish(CMD_DEBUG_RESET, {})` then `publish(CMD_PATH, cmd)` then `publish(CMD_COMMAND, { data: true })`. The leading `/debug/reset` makes starting autonomy a **clean slate** — it clears any lingering *persistent* debug hold (open valve / held ACU position) that would otherwise fight the controllers on the same wire once the mission drives. (Tradeoff: starting a mission *during* an emergency surface cancels it — an extreme edge judged acceptable; the emergency control is separate and prominent.)
- `engageManual(run)` → `stopMission()` then `run()`.
- `resetAll()` → `publish(CMD_COMMAND, { data: false })` then `publish(CMD_DEBUG_RESET, {})`.

### B3. `CommandProfilePanel.vue` — Bool command, Stop button, drop Do-Nothing/override
- Remove `useOverridesStore` import + `manualOverride`.
- `MissionKey`/`MISSION_ID`/`profileOptions`/`buildMissionCommand`: drop `doNothing`/`doNothing:3` everywhere; delete its template `v-else` block (212-215) so `surface` is the terminal branch.
- `sendDisabled`: drop the `|| manualOverride` term.
- `onSend`: replace the inline `/path`+`/command:'start'` pair with `mqttStore.startMission(cmd)`.
- ADD a **Stop** button beside Send → `onStop() { mqttStore.stopMission() }`; style red with `--status-err-*` (mirror `.qc-btn-danger`).
- Template: remove the `override-banner`, the `:class="{ locked: manualOverride }"`, and every `:disabled="manualOverride"`. Remove the now-dead `.override-banner` CSS.

### B4. `DebugCommandsPanel.vue` — always-enabled, stop-then-debug
- Remove `useOverridesStore` import + `manualOverride`.
- `commandsEnabled` → just `bridgeOnline.value`.
- Wrap each of the five publishes in `engageManual(() => mqttBridge.publish(...))`: `sendPump`,
  `sendPumpUntilPressure`, `applyValves` (covers the both-valves confirm path too), `movePitch`, `moveRoll`.
- Remove the locked-hint `<p>` and the `:class="{ locked: !manualOverride }"`; drop the unused
  `.qc-locked-hint` CSS. Optional: a small caption "Sending any command here stops the active mission."

### B5. `EmergencySurfaceButton.vue` — stop mission instead of override
- Remove `useOverridesStore`. In the arm branch, replace `overrides.setManualOverride(true)` with
  `mqtt.stopMission()` (or `mqtt.engageManual(...)`) before `publish(EMERGENCY_TOPIC, {data:true})`. Disarm
  branch unchanged. Update the stale comment.

### B6. New `src/components/CommandPanels/ResetButton.vue` — plain red click button
- A single red button (`mdi-restart`/`mdi-backup-restore`), gated on `bridgeStatus==='online'`, that calls
  `mqtt.resetAll()` on click (fires immediately — chosen interaction). Style with `--status-err-*` (reuse
  the `.qc-btn-danger` look from `DebugCommandsPanel.vue`).

### B7. `Commands.vue` — layout
- Remove the `ManualOverrideSlider` import + its `<ManualOverrideSlider/>` (leaving `DebugCommandsPanel`
  in `.command-debug`). Import + add `<ResetButton/>` into the `.command-emergency` cell under
  `<EmergencySurfaceButton/>`; make that cell `display:flex; flex-direction:column; gap:10px;`.

### B8. No-change confirmations
- `views/Charts.vue` mission-state badge (`MISSION_NAMES`, IDLE/LOADED/RUNNING) keeps working — the glider's
  `nautilus/telemetry/mission/active` mirror is unchanged and already has no id-3 entry.
- `types/TelemetryTypes.ts`: no change.

---

## Removal checklists

**Do Nothing** — ROS: `do_nothing.py`, `MissionId.DO_NOTHING`, `_REGISTRY` entry, `__init__` export,
`resets_control_on_start` block in pathfinding, `test_do_nothing.py`, `TestDoNothingMission`,
`running_sim.md` section, `do_nothing_sim.launch.py`, the two `reset()` docstring mentions. Frontend:
`CommandProfilePanel.vue` union/`MISSION_ID`/option/`buildMissionCommand`/template.

**Manual override** — ROS: the two topic constants + map entries (×3 registry files), `depth_node`/`acu_node`
subscriptions+handlers+flags, `bcu_debug`/`acu_debug` gates+handlers+flags, MQTT bridge ingress+egress,
conftest helpers, `TestManualOverride` ×2. Frontend: `overrides.ts`, `ManualOverrideSlider.vue`, all
`useOverridesStore`/`manualOverride` references in the three panels.

**`/control/reset`** — ROS: the topic constant + ×3 registry entries; pathfinding's `_reset_pub`; the
`CONTROL_RESET` subscriptions in `depth_node`/`acu_node` (replaced by `/command` subscriptions); conftest
`reset_pub`/`received_resets` for the depth/acu/pathfinding testers.

## Key behaviors / risks to keep correct

- **Safe-stop-before-silence is mandatory** (no STM watchdog). Every reset path publishes one 0/closed/neutral first (A4, A5, A6).
- **Two disjoint stop signals** (`/command`=false → pathfinding + controllers; `/debug/reset` → debug nodes) so a manual-command's mission-stop doesn't cancel the just-issued debug command. Controllers don't hear `/debug/reset`; debug nodes don't hear `/command`.
- **Trailing-zero flush** on debug session-end/reset so the terminal 0 beats the throttled MQTT egress (don't freeze the UI chart at the last nonzero).
- **Stop clears the loaded mission** — to re-run, the operator re-sends the profile (UI Send does `/debug/reset` → `/path` → `/command:true`). This is the intended "reset to initial state."
- **Reset is a full all-stop**, including cancelling an active Emergency Surface (operator can re-arm).
- **Contention-at-mission-start edge** (surfaced by the design exploration): removing the override gate means a *persistent* debug hold left active when a mission starts would fight the controllers on that wire. Handled at the UI layer — `startMission` leads with `/debug/reset` (B2). There's no ROS-node-level fix that doesn't reintroduce the override flag or make debug nodes subscribe to `/command`, so the UI orchestration is the right seam.

## Verification

1. **Tier 1/2** (from `src/nautilus-ros/src/py_pkg`, ROS + install sourced):
   `/usr/bin/python3 -m pytest test/unit/ test/node/ -v` — green after Part A.
2. **Build**: `colcon build --packages-select py_pkg nautilus_msgs && source install/setup.bash`.
3. **Tier 3 sim** (Gazebo): `ros2 launch nautilus_hal trim_sim.launch.py headless:=false mission_autostart:=true target_pressure_pa:=75383.0`; confirm dive, then `ros2 topic pub --once /command std_msgs/msg/Bool "{data: false}"` → RPM goes to 0, valves close, controllers go silent (`ros2 topic echo /bcu/rpm` quiet). Then a manual pump on `/debug/bcu/rpm` drives the wire with no controller fight.
4. **Reset path**: publish `/debug/reset` (`std_msgs/Empty`) → any active pump/valve/ACU debug + emergency stop and go to 0/closed/neutral.
5. **Frontend**: `npm run dev` (:3000) against the broker; Start a mission, Stop (mission idles, charts flatten), send a debug pump (mission auto-stops, pump runs), click red Reset (everything zeroes). Also: open a debug valve, then Start a mission → the valve clears first (clean-slate start, no `/bcu/valves` contention). Run `npm run build` for the TS type check.
6. **Tier 3 marker-gated**: `/usr/bin/python3 -m pytest -m sim test/sim/ -v` (the 6 drivers now publish Bool).
