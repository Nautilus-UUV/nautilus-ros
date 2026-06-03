# BCU fault model — monotonic degradation ladder

Status: implemented (replaces the transient 3-state injector).

## Why

The HAL's BCU fault injector used to be a **transient, recovering, 3-state**
model (`fault_injection.py`): a 1 Hz Bernoulli check (`probability_per_sec`)
tripped the pump into `DEGRADED` (RPM × `degraded_factor`) for `duration_sec`,
then `clear_fault()` returned it to `NORMAL`. `SEVERE` was never reached by the
trigger path. That models flaky, self-healing faults — not a real pump failure.

It is now a **monotonic, latching degradation ladder**: pump effectiveness walks
100 → 80 → 60 → 40 → 20 → 0 % in equal 20 % steps (6 states, 5 transitions), each
step an independent failure event with a tunable **mean time between failures
(MTTF)**, and there is **no recovery** — once a level is reached it latches and
can only get worse. This suits MC reliability sweeps where the controller must
cope with a permanently weakening actuator.

## Model

- State: integer `level ∈ [0, num_levels]`, starts at 0 (healthy).
- Effectiveness multiplier on commanded RPM: `(num_levels - level) / num_levels`.
  With `num_levels=5`: level 0→1.0, 1→0.8, 2→0.6, 3→0.4, 4→0.2, 5→0.0.
- Stepping: the 1 Hz trigger timer fires; with per-tick probability
  `p = 1 - exp(-tick_period / mttf_sec)` (Poisson inter-arrival, memoryless → the
  same MTTF between every step), `level += 1` if `level < num_levels`. At most one
  step per tick. `mttf_sec <= 0` ⇒ `p = 0` (never steps, fault-free).
- No recovery: no duration timer, no `clear_fault`.
- Telemetry: `/bcu/rpm/fault` (`std_msgs/msg/Int32`) reports the true latched
  level (0..N) at all times — no idle masking. The topic/type is unchanged; only
  the value *semantics* changed (was 0/1/2 fault-state, now 0..N level).

## Knobs (`FaultInjectorSpec`, `rig.faults.bcu_rpm`)

| field        | meaning                                              | default |
|--------------|------------------------------------------------------|---------|
| `mttf_sec`   | mean time between successive 20 % steps; ≤ 0 = off   | `0.0`   |
| `num_levels` | ladder resolution (5 ⇒ 100/80/60/40/20/0)            | `5`     |

The four old knobs (`probability_per_sec`, `duration_sec`, `degraded_factor`,
`severe_factor`) were removed.

## Files changed

- `src/dave/hal/nautilus_hal/nautilus_hal/injectors/fault_injection.py` — core
  rewrite (level/MTTF/effectiveness; dropped `FaultState`, duration timer,
  `clear_fault`, idle masking).
- `src/nautilus-ros/.../scenarios/spec/rig.py` — `FaultInjectorSpec` fields.
- `src/nautilus-ros/.../scenarios/compile.py` — `params_for_bcu_bridge`
  (`fault_mttf_sec`, `fault_num_levels`).
- `src/dave/hal/.../bridges/bcu_sim_bridge.py` — `declare_parameter` +
  `BCUFaultInjector(...)` construction.
- Scenario YAMLs: `library/{nominal,baseline,nominal_with_hydrodynamics,
  suggested_lhs_0071}.yaml` + `test/sim/scenarios/*.yaml` (off → `mttf_sec: 0.0`;
  baseline on → `mttf_sec: 60.0`).
- Sweep configs: `scripts/sweeps/error_static_physics_sweep.yaml` (active dim →
  `rig.faults.bcu_rpm.mttf_sec`, 150–220 s), `scripts/sweeps/pid_calibration.yaml`
  (commented example updated). `run_sweep.py` applies `path:` strings generically
  — unaffected.
- New test: `src/nautilus-ros/src/py_pkg/test/sim/test_bcu_fault_ladder.py`
  (`@pytest.mark.sim`).

## Verification

```bash
cd src/nautilus-ros/src/py_pkg
source /opt/ros/jazzy/setup.bash && source /home/girji/dave_ws/install/setup.bash
/usr/bin/python3 -m pytest -m sim test/sim/test_bcu_fault_ladder.py -v   # ladder behavior
/usr/bin/python3 -m pytest -m sim test/sim/test_bcu_tank_pressure.py -q  # bridge still constructs

# end-to-end: baseline (mttf 60 s) -> /bcu/rpm/fault climbs 0..5 and latches
ros2 launch nautilus_hal trim_sim.launch.py headless:=true mission_autostart:=true \
    scenario:=$(ros2 pkg prefix py_pkg)/share/py_pkg/scenarios/library/baseline.yaml
ros2 topic echo /bcu/rpm/fault   # monotonically increasing, never decreasing
```
