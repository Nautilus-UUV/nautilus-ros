# Running the Nautilus simulation in Apptainer

The Apptainer flow packages the workspace (ROS 2 Jazzy + Gazebo Harmonic +
all Nautilus / DAVE / HAL code at build time) into a single `.sif` file
that runs anywhere Apptainer is installed, with no GPU and no network
access at run time. It is the supported path for Monte-Carlo-style
scenario sweeps on a multi-CPU server, where many containers run side by
side on the same host.

For interactive controller development on your workstation, prefer the
native colcon flow described in [`running_sim.md`](running_sim.md); for a
reproducible dev environment without the SIF detour, prefer
`docker compose up` described in the workspace-root `infra_claude.md`.

## Build the SIF

The image bakes in the entire `src/` tree at build time, including the
`nautilus-ros` and `dave` sister repos plus the vendored `dockwater` /
`rocker`. Whatever is on disk under `src/` when you run `docker compose
build` is what ends up in the SIF — so commit (or at least save) your
work first if you want it tracked elsewhere, then:

```bash
cd /home/$USER/dave_ws

docker compose build                                                # ~10–20 min
docker save -o dave_nautilus_image.tar dave_nautilus_image:latest
apptainer build nautilus_sim.sif docker-archive://dave_nautilus_image.tar
```

The Docker image pre-downloads the Fuel models the DAVE worlds depend on
(`North East Down frame`, `Coast Water`, `Sand Heightmap`) and flattens
them under `/offline_models/`, and the build step rewrites every
installed `*.world` to replace `https://fuel.gazebosim.org/...` URIs
with `model://`. Net effect: the container can spawn the world without
talking to Fuel, which matters because the cluster has no outbound
network.

If you add a new Fuel model to a world, you must update both the
download block and the flatten step in `Dockerfile` — there is no
generic resolver, just a hand-maintained whitelist.

Sanity-check the new SIF before relying on it:

```bash
apptainer exec nautilus_sim.sif /entrypoint.sh \
    ros2 launch nautilus_hal sawtooth_sim.launch.py --show-args \
    | grep -E 'sampler_id|n_resurfaces|scenario'
```

All three arguments should be advertised.

## Running a sim

The canonical invocation is a plain `apptainer exec`. Run it from the
workspace root so the relative `./sim_data` bind resolves correctly:

```bash
apptainer exec --cleanenv \
    --env GZ_IP=127.0.0.1 \
    --bind ./sim_data:/ros2_ws/sim_data \
    --bind ./scenarios:/ros2_ws/scenarios:ro \
    nautilus_sim.sif \
    /entrypoint.sh ros2 launch nautilus_hal sawtooth_sim.launch.py \
        headless:=true mission_autostart:=true \
        target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=5 \
        record:=true sampler_id:=my_run run_id:=baseline \
        scenario:=/ros2_ws/scenarios/baseline.yaml
```

What each piece does:

- `--cleanenv` strips the host's env (DISPLAY, ROS_DOMAIN_ID, etc.) so
  the container starts from a known baseline.
- `--env GZ_IP=127.0.0.1` keeps the Gazebo transport off any VPN-routable
  interface. Required when the host is on a VPN, harmless otherwise.
- `--bind ./sim_data:/ros2_ws/sim_data` is where recorded bags land —
  the launch builds `[{sampler_id}/]{run_id}_{ts}/raw/` *inside* this
  dir, so bind the parent (not the inner `raw` like older flows did).
- `--bind ./scenarios:/ros2_ws/scenarios:ro` exposes a host directory of
  YAMLs you can edit without rebuilding the SIF. Drop this bind to use
  only the in-image library YAMLs at
  `/ros2_ws/install/py_pkg/share/py_pkg/scenarios/library/`.
- `headless:=true` is required under `--cleanenv` — there's no DISPLAY
  to render a Gazebo GUI to. If you ever need the GUI you'll have to
  forward X11 through (skip `--cleanenv`, bind `/tmp/.X11-unix`, etc.).
- `xvfb-run` is **not** used. The headless server has nothing to render
  to a virtual display, so wrapping with xvfb is dead weight here.

A scenario YAML with a `rig.hydrodynamics:` block is detected by
`nautilus_hal.render_sdf.description_file_for_scenario` and triggers a
Jinja-rendered SDF, so the same `scenario:=` argument that picks the
controller gains can also perturb the simulated plant.

### The `apptainer_exec.sh` shim

`scripts/apptainer_exec.sh` is a thin wrapper that produces the exact
invocation above plus two opt-in extras you'll want as the workflow
matures:

- `--scenarios-dir DIR` — bind `DIR` at `/ros2_ws/scenarios:ro` (the
  same path the launch references).
- `--cpus CPULIST` — pin the apptainer process tree to a subset of
  host CPUs via `taskset`. Useful when running several sims on the
  same multi-CPU server and you want each one isolated to its own
  cores.

```bash
./src/nautilus-ros/scripts/apptainer_exec.sh \
    --scenarios-dir ./scenarios --cpus 0-3 nautilus_sim.sif \
    ros2 launch nautilus_hal sawtooth_sim.launch.py \
        headless:=true mission_autostart:=true \
        target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=5 \
        record:=true sampler_id:=my_run run_id:=baseline \
        scenario:=/ros2_ws/scenarios/baseline_ext.yaml
```

Use the wrapper when one of those two flags is genuinely useful; reach
for the direct command otherwise.

## Single-run examples

### Sawtooth (mission_id=1)

The dive-and-resurface mission used to record sawtooth-shaped pressure
profiles for the anomaly-detection pipeline. `n_resurfaces` is the
single mission knob; the cycle self-terminates after that many
descend → ascend cycles but the stack keeps running so you can fire
another mission from the CLI.

```bash
apptainer exec --cleanenv \
    --env GZ_IP=127.0.0.1 \
    --bind ./sim_data:/ros2_ws/sim_data \
    nautilus_sim.sif \
    /entrypoint.sh ros2 launch nautilus_hal sawtooth_sim.launch.py \
        headless:=true mission_autostart:=true \
        target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=5 \
        record:=true \
        scenario:=/ros2_ws/install/py_pkg/share/py_pkg/scenarios/library/baseline.yaml
```

`baseline.yaml` turns BCU fault injection on at MTTF ~60 s — useful for
generating bags that include fault events. `nominal.yaml` is the
fault-free baseline; `nominal_with_hydrodynamics.yaml` exercises the
SDF render path while keeping the canonical coefficients.

### Trim (mission_id=0)

Hold a target depth indefinitely; useful for collecting steady-state
data or validating buoyancy trim. Runs until you Ctrl-C.

```bash
apptainer exec --cleanenv \
    --env GZ_IP=127.0.0.1 \
    --bind ./sim_data:/ros2_ws/sim_data \
    nautilus_sim.sif \
    /entrypoint.sh ros2 launch nautilus_hal trim_sim.launch.py \
        headless:=true mission_autostart:=true target_pressure_pa:=75383.0
```

### Surface (mission_id=2)

Drive the glider back to gauge 0 and self-terminate once held neutral
on the surface for a configurable hold window.

```bash
apptainer exec --cleanenv \
    --env GZ_IP=127.0.0.1 \
    --bind ./sim_data:/ros2_ws/sim_data \
    nautilus_sim.sif \
    /entrypoint.sh ros2 launch nautilus_hal surface_sim.launch.py \
        headless:=true mission_autostart:=true
```

## Output layout

When `record:=true`, the bridges launch a `ros2 bag record` subprocess
that writes to:

```
sim_data/
  [sampler_id/]            # optional, present iff sampler_id is non-empty
    {run_id}_{YYYY_MM_DD-HH_MM_SS}/
      raw/
        metadata.yaml
        raw_0.mcap.zstd
```

The recorded topics are the six HAL-published surfaces:
`/imu/left`, `/external/pressure`, `/bcu/{rpm,flow_rate,pressure}` and
the sim-only `/bcu/rpm/fault` (fault-injector diagnostic). This is the
exact set `UG-anomaly_detection/src/data/extract.py` knows how to read.

`sampler_id` is the grouping axis a future sampler will own — see
"Future" below. For a one-off run leave it empty and the middle
segment collapses to the historical `sim_data/{run_id}_{ts}/raw/`
layout.

You can override the whole path with `bag_path:=<absolute path>`; that
short-circuits the `sampler_id`/`run_id` synthesis entirely.

## Debugging notes

Things to know before they bite you:

- **`GZ_IP=127.0.0.1`** is required whenever the host is on a VPN, and
  doesn't hurt otherwise. Without it the Gazebo transport binds to a
  routable interface that loops back through the VPN and the simulator
  fails to start.
- **`--cleanenv` + `headless:=true` go together.** `--cleanenv` strips
  `DISPLAY`, so a `headless:=false` launch has nowhere to render and
  Gazebo dies. If you need the GUI you have to skip `--cleanenv`,
  forward `DISPLAY` + `XAUTHORITY`, bind `/tmp/.X11-unix`, and
  `xhost +local:` on the host.
- **Software rendering only**: the SIF was built without `--nv`, so the
  only OpenGL stack inside is Mesa's llvmpipe. `headless:=true` Gazebo
  doesn't use it at all; if you ever pipe a `headless:=false` run
  through and Ogre balks on glX, set `LIBGL_ALWAYS_SOFTWARE=1` +
  `GALLIUM_DRIVER=llvmpipe`.
- **Parallel containers on one host** need to not collide on Gazebo's
  transport. Set `--env GZ_PARTITION=<unique>` per invocation (any
  unique string is fine) and a distinct `ROS_DOMAIN_ID` if your nodes
  publish on shared topics. Single-run flows don't need either.

To poke around inside the container interactively:

```bash
apptainer exec --cleanenv nautilus_sim.sif /entrypoint.sh bash
```

`/entrypoint.sh` has already sourced ROS and the workspace setup by
the time you land in the shell.

## Running an LHS sweep

For Monte Carlo work, two scripts in `src/nautilus-ros/scripts/` cover
the generate-then-dispatch loop:

- `lhs_sample.py` — reads a sweep spec (which scenario dot-paths to
  perturb and how), draws a Latin Hypercube over those dimensions, and
  overlays each sample onto a base scenario from
  `py_pkg/scenarios/library/`. Writes one YAML per sample plus a
  `manifest.json` recording the spec hash and the full sample matrix,
  so post-hoc analysis can join bag results back to scenario
  coordinates without re-parsing YAMLs.
- `run_sweep.py` — consumes the directory of YAMLs, runs them through
  `apptainer_exec.sh` with bounded concurrency, gives every concurrent
  run its own `GZ_PARTITION` / `ROS_DOMAIN_ID` (so Gazebo transports
  and DDS chatter don't collide), and replenishes the pool as slots
  free. Optional `--cpu-budget` slices a host CPU range into disjoint
  per-slot sets so the runs don't fight for cores.

Both scripts run *outside* the SIF; the host needs `numpy`, `pyyaml`,
and `scipy>=1.10` (see `scripts/requirements-sampler.txt`).

### The sweep spec

A short YAML lists what to perturb, what distribution to draw from,
and how many samples to take. See
`scripts/sweeps/example_hydro_faults_pump.yaml` for a worked example
covering hydrodynamics ±20%, BCU fault rate/duration, and the
controller's pump-efficiency belief. The shape:

```yaml
description: "..."
n_samples: 64
seed: 42
base_scenario: nominal_with_hydrodynamics.yaml   # name in py_pkg's library/
dimensions:
  - { path: rig.hydrodynamics.added_mass_xx, distribution: uniform, low: 4.24, high: 6.36 }
  - { path: rig.faults.bcu_rpm.probability_per_sec, distribution: loguniform, low: 0.005, high: 0.05 }
  # ...
```

`path` is a dot-path into the Scenario tree; the sampler writes the
sampled value at that leaf and leaves everything else alone. Use
`nominal_with_hydrodynamics.yaml` as the base whenever you perturb any
`rig.hydrodynamics.*` field — the schema is all-or-nothing, so a
partial block won't validate.

Mission knobs (`target_pressure_pa`, `angle_rad`, `n_resurfaces`) are
not scenario-YAML fields. Pass them once on the `run_sweep.py`
command line; every run in the sweep uses the same values.

### Smoke run (4 scenarios, 2 in parallel)

Use this to confirm the sampler + dispatcher + apptainer plumbing works
end-to-end on your workstation before launching a full sweep. The
sweep is intentionally small: 4 samples over only the depth-loop PID
gains (`control.controllers.depth.pid_pressure.{kp,ki,kd}`) at +/- 50%
of nominal, base scenario is `nominal.yaml` (no SDF render), and the
sawtooth mission is a single descend/ascend cycle.

```bash
cd /home/$USER/dave_ws

# 1. Generate 4 scenario YAMLs + manifest under ./scenarios/smoke/.
./src/nautilus-ros/scripts/lhs_sample.py \
    --spec src/nautilus-ros/scripts/sweeps/smoke_depth_pid.yaml \
    --out ./scenarios --name smoke

# 2. Run them 2-at-a-time on host CPUs 0-7 (two slots, 4 cores each).
./src/nautilus-ros/scripts/run_sweep.py \
    --scenarios-dir ./scenarios/smoke \
    --sif nautilus_sim.sif \
    --concurrency 2 \
    --cpu-budget 0-7 \
    --per-run-timeout 300 \
    --launch sawtooth_sim.launch.py \
    --launch-args target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=1
```

While it runs, you can verify the two slots are isolated:

```bash
pgrep -af apptainer       # expect 2 apptainer exec processes
tail -F sim_data/smoke/sweep_status.csv
ls sim_data/smoke/        # _logs/, sweep_status.csv, lhs_NNNN_{ts}/raw/
```

Total wall time on a workstation is on the order of a few minutes
(two batches of two runs, each sawtooth cycle ~1–2 min). Sims that
self-terminate before the timeout show `exit_code=0, timed_out=False`
in `sweep_status.csv`; sims that hit the timeout show `exit_code=-15,
timed_out=True` — both are expected outcomes for a smoke run (the
mission completes, then the launch hangs waiting for the next
`/path`, and the timeout cleans it up).

### End-to-end

```bash
cd /home/$USER/dave_ws

# 1. Generate 64 scenario YAMLs + manifest.
./src/nautilus-ros/scripts/lhs_sample.py \
    --spec src/nautilus-ros/scripts/sweeps/example_hydro_faults_pump.yaml \
    --out ./scenarios --name lhs_hydro_v1

# 2. Dispatch them, 2 at a time, on host CPUs 0-7.
./src/nautilus-ros/scripts/run_sweep.py \
    --scenarios-dir ./scenarios/lhs_hydro_v1 \
    --sif nautilus_sim.sif \
    --concurrency 2 \
    --cpu-budget 0-7 \
    --per-run-timeout 600 \
    --launch sawtooth_sim.launch.py \
    --launch-args target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=5
```

`--per-run-timeout` is **required in practice** for sawtooth and trim
sweeps. The mission self-terminates after `n_resurfaces` cycles but the
launch keeps Gazebo + the control stack alive waiting for the next
`/path` message, so the subprocess never exits on its own. Pick a value
slightly above `n_resurfaces × single-cycle wall time`. Timed-out runs
get `timed_out=true` in `sweep_status.csv` so they're distinguishable
from real crashes after the fact.

`run_sweep.py` first runs a single short `apptainer exec` to load the
first generated YAML through `load_scenario` — that way a schema typo
in the sweep spec fails once instead of N times. Then it launches
slots, polls every second, reaps finished runs, and refills from the
queue.

### Output layout

```
sim_data/
  lhs_hydro_v1/
    _logs/
      lhs_0000.log         # apptainer_exec.sh stdout+stderr per run
      lhs_0001.log
      ...
    sweep_status.csv       # run_id, slot, start_ts, end_ts, exit_code, yaml_path, duration_sec, timed_out
    lhs_0000_{ts}/raw/...  # bags from each individual run, same layout as single-run flow
    lhs_0001_{ts}/raw/...
    ...
```

The per-run bag layout is unchanged from the single-run flow, so
`UG-anomaly_detection/src/data/extract.py` ingests sweep output without
modification.

### Knobs worth knowing about

- `--concurrency N` (default 2) — N runs in flight at once. Pick based
  on host cores and Gazebo's appetite per sim (~4 cores each is a safe
  starting point).
- `--per-run-timeout SECONDS` — wall-clock budget per slot. See the
  note above; without this set, sawtooth/trim sweeps will hang after
  the first batch of missions complete.
- `--ros-domain-base 50` — slot i runs with `ROS_DOMAIN_ID=base+i`.
  Bump the base if 50–50+N collides with something else on the host.
- `--no-record` — skip `record:=true`; useful for dry-running the
  orchestrator without filling the disk with bags.
- `--no-preflight` — skip the load-scenario validation. Saves a few
  seconds; not recommended on the first run of a new sweep spec.
