# BCU safe-stop convergence + near-setpoint chatter fix + trim settling-termination

Status: **implemented** (Tier 1 + Tier 2 green: 348 unit, 169 node). Tier 3 sim
re-baseline still pending (see Risks). Firmware-side watchdog is a separate
recommendation to the STM owner.

## Context

A bench test of the trim/neutral controller via `stm_com_node` on the Pi exposed
a coupled, safety-critical failure:

1. **Trigger — near-setpoint chatter.** A depth target ≈ current depth gives a
   tiny error. With no error deadband, no valve hysteresis, and no dwell in the
   BCU loop, `select_pump_and_valves` picked valves purely from `sign(error)` and
   republished every 10 Hz tick. Sensor quantization (100 Pa/LSB ≈ 1 cm) flipped
   the sign each tick → full ±1000 RPM pump reversals and valve 2 toggling
   open/closed at up to 10 Hz — mechanically abusive ("broke the system").
2. **Danger — stop did not stop the pump.** After Stop, `/bcu/rpm` showed 0 but
   the motor kept pumping. `bcu_node` emitted a *single* safe-stop then went
   silent, and `stm_com` only transmitted when an inbound STM frame clocked it —
   no TX heartbeat, no `destroy_node` safe-stop, and the STM firmware latches the
   last setpoint forever with no staleness watchdog. A link hiccup during the
   chaos stranded the cached 0 undelivered; the STM held the last nonzero RPM.
3. **Latent gap — completion never stopped the actuator.** On normal completion
   `pathfinding._tick` only `_reset()`s and stops publishing setpoints; it never
   sent `/command=false`, so `bcu_node` held the last target and ran the PID
   forever.

## A — Kill the oscillation at source: `BcuCommandGate`

New stateful pure class `py_pkg/pid/bcu_command_gate.py`, applied in
`bcu_node.control_loop` after `solve_bcu_command` (the existing pure solve
functions are unchanged — the gate wraps their output):

- **Error deadband + arm/disarm hysteresis** — pump held idle within
  `error_disarm_pa` of target, re-arms only past the wider `error_arm_pa`;
  between the two it holds state, so noise dither can't toggle the pump.
- **Minimum valve dwell** — the valve bitmask changes at most once per
  `min_valve_dwell_s`, protecting the solenoids in every regime.
- **Safety invariants** — pump never dead-heads against a closed valve; a disarm
  zeroes the pump immediately even while the dwell still holds a valve open;
  with all three knobs 0 the gate is a pass-through (clean disable for sims).

Config lives on `DepthSpec` (`scenarios/spec/control.py`) as `error_arm_pa` /
`error_disarm_pa` / `min_valve_dwell_s` (defaults 4000 / 2000 Pa / 0.5 s — gate
ON by default), with a `model_validator` enforcing `0 ≤ disarm ≤ arm` and
`dwell ≥ 0`. Forward/inverse param mapping mirrors `command_tolerance` in
`scenarios/compile.py` (`params_for_bcu_node` + `bcu_spec_from_node`). Existing
YAMLs inherit the defaults unchanged.

## B — Stop converges regardless of link state

- **B1 `bcu_node`** — `_begin_safe_stop` re-asserts 0 RPM + valves-closed for a
  bounded burst (`STOP_REASSERT_S = 1 s` ≈ 10 ticks) on stop and at boot, then
  goes silent so a debug node can own the wire; plus a `destroy_node` safe-stop
  on teardown (previously missing).
- **B2 `stm_com`** — independent TX heartbeat timer (`tx_period_s`, default 1 Hz)
  re-ships the cached setpoints regardless of the inbound stream; the parasitic
  inbound-cued send is kept for normal-op responsiveness.
- **B3 `stm_com`** — command-staleness watchdog: no fresh `/bcu/rpm` within
  `command_timeout_s` (default 3 s) → fail safe to 0 RPM + valves closed. This is
  the Pi-side dead-man filling the "STM has no staleness watchdog" hole. During
  active control `/bcu/rpm` arrives at 10 Hz so it never fires. (`can_com` has the
  same latch gap and should get the same watchdog later — out of scope here.)

## C — Trim settling-termination + completion safe-stop

- **C1 trim `is_done`** — `path/missions/settling.py::DepthSettlingMonitor`
  (rolling `(t, pressure)` window) + `TrimAndNeutralBuoyancyMission.is_done`
  returns True once **within 1 m of the goal AND** depth peak-to-peak **≤ 0.5 m
  over the last 10 s** (both required). Thresholds are fixed constants in the
  trim module, converted via `physics.WATER_PRESSURE_GRADIENT_PA_PER_M`.
- **C2 `pathfinding`** — on `is_done` completion it publishes `/command=false`
  (the operator-Stop signal) so the controllers run their safe-stop; closes the
  "completion doesn't stop the BCU" gap for all missions.

## Files

`py_pkg/pid/bcu_command_gate.py` (new), `py_pkg/pid/bcu_node.py`,
`py_pkg/stm_com/stm_com_node.py`, `py_pkg/scenarios/spec/control.py`,
`py_pkg/scenarios/compile.py`, `py_pkg/path/missions/settling.py` (new),
`py_pkg/path/missions/trim_and_neutral.py`, `py_pkg/path/pathfinding.py`.

Tests: new `test/unit/test_bcu_command_gate.py`, `test/unit/test_depth_settling.py`;
updated `test/unit/test_trim_and_neutral.py`, `test/unit/test_scenario_loader.py`,
`test/node/test_bcu_node.py`, `test/node/test_stm_com_node.py`,
`test/node/test_pathfinding_node.py`, `test/node/conftest.py`.

## Verification

```bash
cd src/nautilus-ros/src/py_pkg && source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m pytest test/unit/ -q          # Tier 1
source /home/girji/dave_ws/install/setup.bash
/usr/bin/python3 -m pytest test/node/ -q          # Tier 2
```

End-to-end (host, after `colcon build --symlink-install`):

```bash
ros2 launch nautilus_hal trim_sim.launch.py headless:=false \
    mission_autostart:=true target_pressure_pa:=75383.0
```

Watch for: `/bcu/valves` stable at the setpoint (≤1 change per dwell); mission
auto-terminates once settled; on completion and on Stop `/bcu/rpm` goes to 0 and
stays 0; killing `bcu_node` mid-run drives `/bcu/feedback/rpm` to 0 within ~3 s.

## Risks / behavior changes

- Non-zero gate defaults turn the gate ON for `nominal`/`baseline`: intended (the
  fix), but shifts steady-state hold and sim trajectories — Tier 3 sim
  expectations need re-baselining to the new intended behavior (not loosened to
  mask regressions).
- Up to ~1 s actuation latency from `min_valve_dwell_s` — acceptable for the slow
  buoyancy plant; protects solenoids.
- B2/B3 cover controller-side silence, not a physically dead UART (can't write a
  safe value to a dead port); a severed link still needs the firmware-side
  failsafe — a standing recommendation to the STM firmware owner.
- `robot_specs.py` constants untouched; all new tunables live on the scenario
  spec or as mission constants.
```
