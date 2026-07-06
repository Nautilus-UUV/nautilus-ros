# BCU: let manual commands run without the safe-stop reassert fighting them

Status: implemented. Tier-1 + Tier-2 green (552 passed).

## Context

Driving the BCU by hand from the UI was broken in two ways: a manual command
**jittered the valves**, and sending a second manual command (e.g. start the pump
while a valve was held open) **interrupted the first**. Root cause was the UI's
manual-command wrapper plus a level-triggered stop handler in `bcu_node`:

- The frontend wraps *every* manual button in `engageManual`
  (`nautilus-command-bridge-frontend/src/store/mqttBridge.ts`): `stopMission()`
  (publishes `/command`=false) **then** the actual valve/pump command. So a fresh
  `/command`=false is sent **before every single manual command**.
- `bcu_node._on_command` was **level-triggered**: it ran `_begin_safe_stop()` on
  *every* `/command`=false, which re-emitted a zero and **re-armed a 1 s
  "safe-stop reassert burst"** (0-RPM / closed-valves every control tick for
  `STOP_REASSERT_S = 1.0 s`).
- `bcu_node` and `bcu_debug_node` (the manual driver) **both publish to the same
  topics** `/bcu/rpm` + `/bcu/valves` with no arbiter — last writer wins at the
  STM/HAL. So while `bcu_node` bursts zeros and `bcu_debug` heartbeats the manual
  value (10 Hz), the wire flickers; and each new manual command re-sent
  `/command`=false, re-arming the burst → it never drained → the manual commands
  "stopped each other."

The burst is deliberate and must stay for genuine mission stops: the STM has **no
staleness watchdog** (it re-ships the last value forever), so a single dropped
safe-stop would latch a stale command. We keep the burst for real stops and stop
it from fighting manual.

Scope is BCU-only. The ACU emits a *single* neutral on stop (no burst), so it
has no equivalent flicker — `acu_node` is untouched.

## Approach — two complementary changes

**1. Edge-trigger the stop (kills "manual commands stop each other").**
`_on_command(False)` now acts only on the running→stopped *transition*. The clean
signal already exists: `target_pressure_pa is None` means "already stopped" (the
controller only drives the wire while it holds a target), so this is a one-line
guard with no new state. After the first manual command stops the mission, every
subsequent manual command's `/command`=false is a no-op → `bcu_node` stays silent
→ stacked manual commands coexist.

**2. Yield to manual (kills the first-engage 1 s burst).**
`bcu_node` subscribes to the BCU manual command **input** topics (the same ones
`bcu_debug_node` consumes — no new topics, all `UUVQoS.COMMAND`) and cancels its
burst the instant a manual command appears: `DEBUG_BCU_RPM`,
`DEBUG_BCU_RPM_UNTIL_PRESSURE`, `DEBUG_BCU_VALVES`, `DEBUG_EMERGENCY_SURFACE`. The
callback ignores the payload. `DEBUG_RESET` is excluded — after a reset
`bcu_debug` goes silent, so a reset should let `bcu_node` burst normally (and it
is `Empty`/`CONTROL` QoS, not in the COMMAND set).

The burst + yield decision lives in a tiny pure clock-injected class
`BcuSafeStopBurst` (mirrors `BcuCommandGate` / `TankLimitGuard`) so the
stop/manual ordering race is Tier-1 testable; `bcu_node` keeps the rclpy publish.
It survives both message orderings:
- **stop then manual** (the usual one): `begin_stop` arms and the caller emits one
  zero, then `note_manual` cancels the rest → worst case one safe-stop sample.
- **manual then stop** (reverse cross-topic delivery): `note_manual` is seen
  first, so `begin_stop` yields outright → zero zeros.

Free safety win: the lifeguard auto-emergency publishes the same `/command`=false
+ `DEBUG_EMERGENCY_SURFACE` pair (`mqtt_bridge_node.py`), so `bcu_node` also stops
injecting zeros against `bcu_debug`'s 3000-RPM emergency blow.

## Files

- `src/py_pkg/py_pkg/pid/bcu_safe_stop_burst.py` — new `BcuSafeStopBurst` pure class.
- `src/py_pkg/py_pkg/pid/bcu_node.py` — `MANUAL_HOLD_S = 1.5`; the burst counter
  state replaced by `self._safe_stop`; a `_now_s()` helper; four debug-topic
  subscriptions → `_on_manual_activity`; the edge guard in `_on_command`;
  `_begin_safe_stop` / `control_loop` route through `begin_stop` / `tick`.
- `src/py_pkg/test/unit/test_bcu_safe_stop_burst.py` — Tier-1, both orderings +
  drain + hold-expiry + reset.
- `src/py_pkg/test/node/test_bcu_node.py` (+ `conftest.py` harness helper) —
  Tier-2: existing genuine-stop burst preserved, plus `repeated stop does not
  re-arm` and `manual command during stop yields instead of bursting`.

## Verification

```bash
cd src/nautilus-ros/src/py_pkg
source /opt/ros/jazzy/setup.bash
source /home/girji/dave_ws/install/setup.bash
/usr/bin/python3 -m pytest test/unit/test_bcu_safe_stop_burst.py -v
/usr/bin/python3 -m pytest test/node/test_bcu_node.py -v
```

End-to-end (reproduces the original symptom — rebuild first so the sim picks up
the new module: `colcon build --packages-select py_pkg && source install/setup.bash`):
bring up the closed-loop sim, then open a valve and start the pump as two manual
actions — the valve stays open while the pump runs (they don't stop each other)
and `/bcu/valves` shows no jitter.

```bash
ros2 launch nautilus_hal trim_sim.launch.py headless:=false mission_autostart:=true target_pressure_pa:=75383.0
ros2 topic echo /bcu/valves    # watch while issuing manual valve + pump from the UI
```

Regression: a plain mission stop / mission-complete with no manual command still
shows the zero burst on `/bcu/rpm`+`/bcu/valves`, and boot still emits the
safe-stop.
