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

## Future: sampler orchestrator

A planned `scripts/run_sampler.py` (deferred) will iterate over a host
directory of scenario YAMLs, kick off one `apptainer exec` per
scenario, share a single `sampler_id`, and assign each run a unique
`run_id` derived from the scenario filename. Each run's bag lands at
`sim_data/{sampler_id}/{run_id}_{ts}/raw/`, ready for the
anomaly-detection pipeline to ingest in bulk. Per-run isolation
(`GZ_PARTITION`, `ROS_DOMAIN_ID`, `--cpus`) is the sampler's job to
set; the current launch plumbing — `sampler_id` arg, host-mounted
scenarios — is the contract that sampler will rely on.
