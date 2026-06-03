# Running the Nautilus simulation in Apptainer

## Sample Coefficients

PID Search:
```bash
python3 src/nautilus-ros/scripts/lhs_sample.py \
  --spec src/nautilus-ros/scripts/sweeps/pid_calibration.yaml \
  --out ./scenarios --name nominal_pid_sweep
```

Physics:
```bash
./src/nautilus-ros/scripts/lhs_sample.py \
  --spec src/nautilus-ros/scripts/sweeps/bcu_hydro_coarse.yaml \
  --out ./scenarios \
  --name bcu_hydro_coarse
```

## Build the SIF

```bash
cd /home/$USER/dave_ws

docker compose build   # ~5 min
docker save -o dave_nautilus_image.tar dave_nautilus_image:latest
apptainer build nautilus_sim.sif docker-archive://dave_nautilus_image.tar
```


If you add a new Fuel model to a world, you must update both the
download block and the flatten step in `Dockerfile`. There is no
generic resolver.

Sanity-check the new SIF before relying on it:

```bash
apptainer exec nautilus_sim.sif /entrypoint.sh \
    ros2 launch nautilus_hal sawtooth_sim.launch.py --show-args \
    | grep -E 'sampler_id|n_resurfaces|scenario'
```

All three arguments should be advertised.

## Running a sim

### PID Sweep

```bash
./src/nautilus-ros/scripts/run_sweep.py \
  --scenarios-dir ./scenarios/nominal_pid_sweep_with_min_rpm \
  --sif nautilus_sim.sif \
  --concurrency 8 \
  --cpu-budget 0-31 \
  --per-run-timeout 1200 \
  --launch sawtooth_sim.launch.py \
  --launch-args target_pressure_pa:=100000.0 angle_rad:=0.6109 n_resurfaces:=30
```

### Physics Parameters with Failures


```bash
./src/nautilus-ros/scripts/run_sweep.py \
  --scenarios-dir ./scenarios/bcu_fault_dataset \
  --sif nautilus_sim.sif \
  --concurrency 8 \
  --cpu-budget 0-31 \
  --per-run-timeout 6400 \
  --launch sawtooth_sim.launch.py \
  --launch-args target_pressure_pa:=100000.0 angle_rad:=0.6109 n_resurfaces:=30
```

### Visualization

```bash
/usr/bin/python3 -m py_pkg.analysis.cli sim_data/nominal_pid_sweep_with_min_rpm \
  --include-failed \
  --all-winners \
  --out tmp/nominal_pid_sweep_with_min_rpm.png
```


### Canonical Invocation

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
    target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=1 \
    record:=true sampler_id:=bcu_hydro_coarse run_id:=lhs_0000 \
    scenario:=/ros2_ws/scenarios/bcu_hydro_coarse/lhs_0000.yaml
```

Arguments:

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

