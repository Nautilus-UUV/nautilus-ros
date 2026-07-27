# BCU valve flicker at the tank-pressure limit — diagnosis & fix

> **SUPERSEDED (2026-07-27).** The BCU depth loop is now bang-bang, and most of
> the machinery this document designs no longer exists: `BcuCommandGate`,
> `TrimPulser`, the RPM deadband, the `pid_pressure` gains and the
> `tank_release_band` hysteresis were all deleted along with the PID. The
> chatter they suppressed came from a continuously modulated command; with one
> speed, one direction and a turn only at a mission leg boundary, there is
> nothing left to dither. What survives, and is still live: `BcuSafeStopBurst`
> (`control/bcu_safe_stop_burst.py`) with its manual-yield, and
> `TankLimitGuard` (`control/tank_limit_guard.py`), now a single-band latch at
> 5% released only by a command reversal. Kept for the diagnosis and the
> hardware failure modes it records, not as guidance.

## Symptom

Bench "Trim & Neutral" test, depth target 7 m. On the bench the vehicle cannot
dive, so it can never reach the target. The intent was to confirm the BCU stops
actuating once the oil tank gets within 10 % of an endpoint. It did stop the pump
RPM — but the **valve started flickering** (toggling open/closed).

## Root cause

`clamp_to_tank_limits` was a **stateless, hysteresis-free, single-threshold cutoff**
(its own unit test even noted: *"the clamp is per-tick and direction-gated, not a
latch"*).

1. On the bench the depth never changes, so the depth error (`target − current`)
   stays large and constant. The PID demands "fill the tank" at full magnitude
   forever.
2. The loop drives the tank pressure straight to the 10 % guard and **parks exactly
   on it** — the switching surface of the cutoff.
3. Sitting on that threshold, ordinary tank-pressure sensor quantization/jitter
   crosses the guard back and forth every tick. The clamp flips between *force-stop*
   `(0,0,0)` and *pass the raw fill command* `(pump<0, motor_open=1)` → the motor
   valve toggles = the flicker.
4. `BcuCommandGate` could not damp it: its arm/disarm hysteresis keys off **depth
   error**, which on the bench is permanently large, so the gate stays *armed* and
   forwards the toggling valve state. `min_valve_dwell_s` only **rate-limits** the
   toggle (~2 Hz), it doesn't stop it.
5. RPM read ~0 because `out_pump = desired_pump if held_motor else 0` zeros the pump
   whenever the valve is held closed — so RPM looked "stopped" while the **valve
   bitmask** is what oscillated. Exactly the reported symptom.

This is not only a bench artifact: the same chatter occurs **in water** any time the
controller pushes the tank to an endpoint and holds there (a depth unreachable with
available ballast, terminal trim, etc.).

## Fix — `TankLimitGuard` (latch + release hysteresis)

A small pure class modeled on `BcuCommandGate`, applied in `bcu_node.control_loop`
**after** `solve_bcu_command` and **before** `self._gate.apply(...)`. It wraps the
per-tick `clamp_to_tank_limits` decision with a latch:

- **Engage:** when a *filling* command (`pump<0` or `free_open`) reaches the
  `stop_band` (10 %) guard, latch `"full"`; symmetric `"empty"` latch for a
  *draining* command (`pump>0`) at the low guard. While latched and still pushing
  inward, the command is zeroed → `(0,0,0)`.
- **Release** only on one of:
  - the tank retreats past the *wider* `release_band` (12 %) guard — the hysteresis
    gap (~1.6 kPa on the 80 kPa span), wider than sensor noise, so parking at the
    stop guard never releases;
  - the demanded fill direction reverses (flow away from a touched limit passes
    straight through, preserving the old `TestAwayFromLimit` contract);
  - `reset()` — fresh dive-init or mission stop.

Because "latched + still pushing in" and a transient idle both yield `(0,0,0)` at the
endpoint, the **output never toggles** — the flicker is gone at the source. The
still-armed gate downstream just sees a steady `(0,0,0)` and holds the valves closed.

## Files changed (all under `src/nautilus-ros/src/py_pkg/`)

- **New `py_pkg/control/tank_limit_guard.py`** — `class TankLimitGuard`, over the single
  threshold check distilled from `bcu_node.py`'s old `clamp_to_tank_limits` and used
  for both the stop and release thresholds. That check lives in
  `math_utils.at_span_endpoint`, shared with the lifeguard's blow stand-down
  (`mqtt/lifeguard.tank_blow_exhausted`) so the two safety gates can't drift apart.
- **`py_pkg/control/bcu_node.py`** — `solve_bcu_command` drops the clamp step and its tank
  params (returns the raw direction the guard needs); `BCUNode` instantiates
  `self._tank_guard`, applies it in `control_loop`, and resets it on mission-stop and
  dive-init.
- **`py_pkg/scenarios/spec/control.py`** — `DepthSpec` gains `tank_stop_band` (0.10)
  and `tank_release_band` (0.12), with a validator (`0 ≤ band < 0.5`, release ≥ stop).
- **`py_pkg/scenarios/compile.py`** — both fields wired forward (`params_for_bcu_node`)
  and inverse (`bcu_spec_from_node`).
- **Tests** — new `test/unit/test_tank_limit_guard.py` (Tier 1) owns both layers: the
  hold-through-noise regression, and the bare threshold tables (via a `_bare` helper
  driving a fresh no-hysteresis guard per tick) that the since-deleted
  `test/unit/test_tank_limit_clamp.py` used to hold behind a shim named after the
  removed production function; `test/node/test_bcu_node.py::TestTankLimitClamp` passes
  unchanged.

## Verification

```bash
cd src/nautilus-ros/src/py_pkg
source /opt/ros/jazzy/setup.bash && source $WS/install/setup.bash
/usr/bin/python3 -m pytest test/unit/ -q          # 363 passed
/usr/bin/python3 -m pytest test/node/test_bcu_node.py -q   # 30 passed
```

End-to-end: hold the vehicle at the surface with a target unreachable on the bench so
the tank parks at an endpoint; `ros2 topic echo /bcu/valves` should show a flat trace
(RPM 0 **and** valves steady), the latch releasing only on a genuine reversal or a
fresh dive-init.
