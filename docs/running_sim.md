# Running the Nautilus simulation

Two ways to bring the glider up against the DAVE simulator: as a **Tier 3
pytest test** or as an **unbounded interactive launch** (Gazebo GUI on, runs until you Ctrl-C).

Both paths assume the workspace has been built and sourced:

```bash
cd /home/$USER/dave_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Unbounded interactive run with the GUI (the day-to-day path)

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
  - 60 295 ≈ 6 m
  - **65 332 ≈ 6.5 m  ← what `test_trim_neutral_sim*` aims at**
  - 75 383 ≈ 7.5 m



### Inspecting state in another terminal

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

# What the (currently-fragile) EKF reports:
ros2 topic echo /position/estimation
```

### Retargeting mid-run


```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 0, target_pressure_pa: 60295.0, angle_rad: 0.0, n_resurfaces: 0}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/String "{data: start}"
```


## Tier 3 pytest catalog

All sim tests are marker-gated `@pytest.mark.sim` and excluded from the
default `pytest test/` / `colcon test` run. Opt in with `-m sim`. Each
takes 30 s – 3 min on CPU-only software rendering.

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
| `test_acu_pitch_sim` | ACU_PITCH → pitch joint motion | `ACU_SIM_GUI=1` |
| `test_acu_roll_sim` | ACU_ROLL → roll joint motion | `ACU_SIM_GUI=1` |
| `test_ekf_pipeline_sim` | IMU → prefilter → EKF pose well-formedness | `EKF_SIM_GUI=1` |
| `test_trim_neutral_sim` | Full closed-loop TRIM, EKF in the loop | `TRIM_SIM_GUI=1` |
| `test_trim_neutral_sim_gt` | Full closed-loop TRIM, EKF replaced by ground truth | `TRIM_GT_SIM_GUI=1` |

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
