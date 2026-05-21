# Running the Nautilus simulation

Two ways to bring the glider up against the DAVE simulator: as a **Tier 3
pytest test** or as an **unbounded interactive launch** (Gazebo GUI on, runs until you Ctrl-C).

For the containerized flow (Apptainer SIF on a multi-CPU server, no GPU,
no network at run time — the path for MC sweeps), see
[`running_in_apptainer.md`](running_in_apptainer.md).

Both paths assume the workspace has been built and sourced:

```bash
cd /home/$USER/dave_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Unbounded interactive run with the GUI

All three mission launches below accept the same `scenario:=<path>`
argument (defaults to the installed `library/nominal.yaml` —
perturbation-free, fault-injection off). Override with
`library/baseline.yaml` to turn BCU fault injection back on, or with
any custom YAML for an MC sweep. The trim section spells the
override out; the same flag works for the sawtooth and surface
variants and for `bridge.launch.py` / `control_stack.launch.py`.

### 1. Targeted Trim and Neutral Mission Profile
---

The full closed-loop trim demo: HAL bridges + Gazebo + glider model +
the entire control stack (`ekf_prefilter`, `ekf_node`, `depth_node`,
`acu_node`, `pathfinding_node`) + an auto-fired `MissionCommand` that
holds the glider at the requested target depth. Runs forever until
Ctrl-C.

```bash
ros2 launch nautilus_hal trim_sim.launch.py \
    headless:=false \
    mission_autostart:=true \
    target_pressure_pa:=65332.0
```

- `headless:=false` opens the Gazebo GUI; leave at its default `true` for
  faster CPU-only runs without a window.
- `mission_autostart:=true` publishes `MissionCommand{mission_id=0,
  target_pressure_pa=…}` and `/command:start` after an 8/10 s
- `target_pressure_pa` is the depth setpoint in **gauge Pa**:
  - 65 332 ≈ 6.5 m
- `scenario:=<path>` selects the scenario YAML that drives gains, plant,
  bridge publish rates, and fault injection. Defaults to the installed
  `library/nominal.yaml` (perturbation-free, fault-injection off). To
  run with BCU fault injection on (MTTF ~60 s), point it at the
  baseline scenario:

  ```bash
  ros2 launch nautilus_hal trim_sim.launch.py \
      headless:=false \
      mission_autostart:=true \
      target_pressure_pa:=65332.0 \
      scenario:=$(ros2 pkg prefix py_pkg)/share/py_pkg/scenarios/library/baseline.yaml
  ```


#### Re-firing mid-run


```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 0, target_pressure_pa: 60295.0, angle_rad: 0.0, n_resurfaces: 0}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/String "{data: start}"
```


### 2. Sawtooth Mission Profile
---

The same closed-loop stack as TRIM, but driving the SAWTOOTH mission:
the glider dives at `-angle_rad` to `target_pressure_pa`, then ascends
at `+angle_rad` back to the surface, and repeats for `n_resurfaces`
cycles before self-terminating. Gazebo and the controllers stay up
after termination, so you can fire another mission.

```bash
ros2 launch nautilus_hal sawtooth_sim.launch.py \
    headless:=false \
    mission_autostart:=true \
    target_pressure_pa:=147150.0 \
    angle_rad:=0.6109 \
    n_resurfaces:=1
```

- `target_pressure_pa` is the deep extremum in **gauge Pa**:
  - 147 150 ≈ 15 m
- `angle_rad` is the glide pitch magnitude in radians (alternates sign
  each leg). 0.6109 ≈ 35 °.
- `n_resurfaces` controls how many full descend → ascend cycles run
  before the mission terminates.


#### Re-firing mid-run

```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 1, target_pressure_pa: 147150.0, angle_rad: 0.6109, n_resurfaces: 2}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/String "{data: start}"
```


### 3. Surface Mission Profile
---

The same closed-loop stack as above, but with the SURFACE mission
auto-fired instead of TRIM: the glider ascends to gauge pressure 0 and
the mission self-terminates after ~10 s of continuous dwell at the
surface (gauge ≤ 5 kPa, ~0.5 m). Gazebo and the controllers stay up
after self-termination, so you can observe the resting state or fire
another mission.

```bash
ros2 launch nautilus_hal surface_sim.launch.py \
    headless:=false \
    mission_autostart:=true
```

- Spawn is at z=-10 (~10 m), so there's a meaningful ascent to watch.
- `mission_autostart:=true` publishes `MissionCommand{mission_id=2,
  target_pressure_pa=0.0, …}` and `/command:start` after an 8/10 s
  delay.
- SURFACE has no operator-tunable parameters — `target_pressure_pa`,
  `angle_rad`, and `n_resurfaces` are all ignored.


#### Re-firing mid-run

```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 2, target_pressure_pa: 0.0, angle_rad: 0.0, n_resurfaces: 0}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/String "{data: start}"
```

### Inspecting state

```bash
source install/setup.bash

# Ground-truth pose (privileged sim-only — bridged out of Gazebo by
# dave_robot_models/config/glider_nautilus/robot_config.py):
ros2 topic echo /model/glider_nautilus/odometry --once

# What the controller is tracking + commanding:
ros2 topic echo /position/target
ros2 topic echo /external/pressure
ros2 topic echo /bcu/rpm
ros2 topic echo /bcu/volume
ros2 topic echo /acu/pitch
ros2 topic echo /acu/roll

# What the (currently-fragile) EKF reports:
ros2 topic echo /position/estimation
```


## Tier 3 pytest catalog

All sim tests are marker-gated `@pytest.mark.sim` and excluded from the
default `pytest test/` / `colcon test` run. Opt in with `-m sim`. Most
take 30 s – 3 min on CPU-only software rendering; `test_sawtooth_sim`
runs ~5 min because it exercises a full descend → ascend cycle to
~7.5 m and back to the surface.

```bash
# Just one test:
/usr/bin/python3 -m pytest -m sim test/sim/test_trim_neutral_sim_gt.py -v

# All Tier 3 (≈ 6 min serially; flake-prone when chained):
/usr/bin/python3 -m pytest -m sim test/sim/ -v
```

Use `/usr/bin/python3` explicitly: the apt-installed `python3-pytest` is
the same interpreter `colcon test` uses. The conda Python on `$PATH`
typically lacks pytest.

| Test | What it covers | GUI env-var |
|---|---|---|
| `test_bcu_sim` | RPM → flow → bladder volume integration | `BCU_SIM_GUI=1` |
| `test_acu_roll_sim` | ACU_ROLL → roll joint motion | `ACU_SIM_GUI=1` |
| `test_ekf_pipeline_sim` | IMU → prefilter → EKF pose well-formedness | `EKF_SIM_GUI=1` |
| `test_trim_neutral_sim` | Full closed-loop TRIM, EKF in the loop | `TRIM_SIM_GUI=1` |
| `test_trim_neutral_sim_gt` | Full closed-loop TRIM, EKF replaced by ground truth | `TRIM_GT_SIM_GUI=1` |
| `test_sawtooth_sim` | Full closed-loop SAWTOOTH cycle; bang-bang ACU pitch reaches both stroke endpoints in the right legs | `SAWTOOTH_SIM_GUI=1` |
| `test_surface_sim` | Mission-driven ascent to surface + self-termination, EKF in the loop | `SURFACE_SIM_GUI=1` |

Set the env-var to enable the Gazebo GUI for that test:

```bash
TRIM_GT_SIM_GUI=1 /usr/bin/python3 -m pytest -m sim test/sim/test_trim_neutral_sim_gt.py -v -s
```

`-s` keeps stdout flowing so you can see Gazebo logs as they happen.

### Why two trim tests

`test_trim_neutral_sim` runs the production stack including the EKF.
The EKF orientation update is currently non-multiplicative
(see `docs/ekf_node_issues.md`) and drifts over long runs, which feeds
attitude noise into the ACU. So it has looser tolerances and doubles
as a downstream EKF-stability smoke.

`test_trim_neutral_sim_gt` swaps the EKF nodes for a sim-only
`gt_pose_bridge` (in `nautilus_hal`) that re-publishes Gazebo's
ground-truth `/model/glider_nautilus/odometry` onto
`POSITION_ESTIMATION`. With the EKF out of the loop, the only loops
under evaluation are the depth (BCU PID) and ACU controllers — failures
here are real controller regressions. Tighter tolerances reflect that.
