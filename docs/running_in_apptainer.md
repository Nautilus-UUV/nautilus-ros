# Running the Nautilus Simulation in Apptainer

## Sample Coefficients

PID Search:
```bash
python3 src/nautilus-ros/scripts/lhs_sample.py \
  --spec src/nautilus-ros/scripts/sweeps/pid_calibration.yaml \
  --out ./scenarios --name nominal_pid_sweep
```

Lake Envelope Coeff.:
```bash
python3 ./src/nautilus-ros/scripts/lhs_sample.py \
  --spec src/nautilus-ros/scripts/sweeps/nominal_lake_envelope_jun24.yaml \
  --out ./scenarios \
  --name lake_test_env
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
    | grep -E 'sampler_id|n_oscillations|scenario'
```

All three arguments should be advertised.

## Running a sim

`run_sweep.py` dispatches one `apptainer exec` per scenario through
`scripts/apptainer_exec.sh` (per-slot `GZ_PARTITION` / `ROS_DOMAIN_ID`,
optional CPU pinning) — keep the two scripts side by side. Host-side
Python deps for the sampler and analysis live in
`scripts/requirements-sampler.txt` / `scripts/requirements-analysis.txt`.

Robustness knobs added after the first `train_validation_mix_v2` drop
(82% of runs lost to init races, DDS meltdown, and a truncating cap):

- `apptainer_exec.sh` forces `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` and
  `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` — apptainer shares the host
  `/dev/shm`, so Fast DDS's shared-memory transport collides across
  concurrent slots (dead / minutes-late DataReaders); loopback UDP
  doesn't.
- `--stagger-sec` (default 15) spaces slot launches so N containers
  never run Python import + DDS discovery at the same instant.
- `--max-retries` (default 2) requeues runs that end in an
  infrastructure verdict (`abort_init` / `abort_bad_start` /
  `abort_floater` / `abort_sinker` — the sim launches spawn Gazebo
  paused behind a `sim_ready_gate`, so genuine scenario physics is not
  what these mean anymore), crash without a verdict, or leave no bag.
  Failed attempts' bags are parked as `raw.failedN`; `sweep_status.csv`
  gains an `attempt` column and the last row per run_id is the final
  outcome.
- `--per-run-timeout` now REFUSES to cap below any run's scaled mission
  budget (pass `--allow-cap-below-scaled` to truncate anyway). The v2
  drop ran 1800 s against ~17 ks budgets and killed every long mission
  mid-write.
- `--ros-domain-base` defaults to 10 and run_sweep refuses a window
  whose Fast DDS ports (7400 + 250×domain) reach the Linux ephemeral
  port range (domains ≥ 102 — the v2 drop's 64 slots at base 50 put
  slots 52–63 there, where transient sockets sporadically break DDS
  participant creation; those slots showed 3× the no-usable-data rate).
  Keep base + concurrency − 1 ≤ 100.

Knobs added after the `train_validation_mix_v3` drop (interrupted at
700/2048; 30% of launches froze — buoyancy plugin never applied
commanded volume as force, hull pinned at its 0.44 m float, a
load-dependent init coin flip the graph/spawn/stepping checks all pass):

- **Physics-liveness probe.** run_sweep injects `physics_probe:=true`
  into every run: before latching `/sim/ready`, `sim_ready_gate`
  suppresses the HeaveAugment entry servo, waits out the spawn settle,
  commands a bladder deflate and requires the hull to actually sink,
  then requires the plant to self-restore. A frozen run dies as a
  ~2–3 min `abort_init` retry (with a `physics:no-hull-response(...)`
  detail line in `run_verdict.json` and the slot log) instead of a
  wasted wall budget or a garbage bag. The `*_sim` launches default the
  probe OFF for interactive use (`physics_probe:=false` also works as
  an operator override on the sweep command line).
- Retries requeue at the FRONT of the queue (v3's interruption stranded
  all 189 requeued runs behind ~1350 unstarted ones), and the failed
  attempt's `run_verdict.json` is parked inside its `raw.failedN/` for
  forensics.
- The race is load-dependent: v3 ran 64 slots on `0-383`. Pilot any new
  campaign at reduced load (e.g. `--concurrency 32` = 12 cores/slot,
  `--stagger-sec 25`) and measure the `abort_init` rate before
  committing the full sweep — see the v4 section below.

Changing `nautilus_hal` or `py_pkg` (the gate, the consolidated
`record_throttle`, `/command`-armed fault epochs, `SIM_READY`) requires
a SIF rebuild before the next sweep.

### PID Sweep

```bash
./src/nautilus-ros/scripts/run_sweep.py \
  --scenarios-dir ./scenarios/nominal_pid_sweep \
  --sif nautilus_sim.sif \
  --concurrency 8 \
  --cpu-budget 0-31 \
  --per-run-timeout 1200 \
  --launch sawtooth_sim.launch.py \
  --launch-args target_pressure_pa:=100000.0 angle_rad:=0.6109 n_oscillations:=30
```

### Envelope of the Lake Test


```bash
./src/nautilus-ros/scripts/run_sweep.py \
  --scenarios-dir ./scenarios/lake_test_env \
  --sif nautilus_sim.sif \
  --concurrency 64 \
  --cpu-budget 0-383 \
  --per-run-timeout 4800 \
  --launch sawtooth_sim.launch.py \
  --launch-args angle_rad:=0.0 z:=-1.0
```

```bash
./src/nautilus-ros/scripts/run_sweep.py \
  --scenarios-dir ./scenarios/train_validation_mix_v3 \
  --sif nautilus_sim.sif \
  --concurrency 64 \
  --cpu-budget 0-383 \
  --per-run-timeout 18000 \
  --launch sawtooth_sim.launch.py \
  --launch-args angle_rad:=0.0 z:=-0.115 watchdog:=true
```

### train_validation_mix_v4 — pilot first

The v4 campaign is gated on a ~64-run pilot at reduced load (the frozen
race is load-dependent; requires a SIF rebuilt after the probe landed in
`nautilus_hal` + `dave_gz_model_plugins`):

```bash
python3 src/nautilus-ros/scripts/lhs_sample.py \
  --spec src/nautilus-ros/scripts/sweeps/train_validation_mix_v4.yaml \
  --out ./scenarios --name train_validation_mix_v4
mkdir ./scenarios/tv_v4_pilot
cp ./scenarios/train_validation_mix_v4/manifest.json ./scenarios/tv_v4_pilot/
cp $(ls ./scenarios/train_validation_mix_v4/lhs_*.yaml | head -64) ./scenarios/tv_v4_pilot/

./src/nautilus-ros/scripts/run_sweep.py \
  --scenarios-dir ./scenarios/tv_v4_pilot \
  --sif nautilus_sim.sif \
  --concurrency 32 \
  --cpu-budget 0-383 \
  --stagger-sec 25 \
  --per-run-timeout 18000 \
  --launch sawtooth_sim.launch.py \
  --launch-args angle_rad:=0.0 z:=-0.115 watchdog:=true
```

Pilot acceptance before the full 2048 (same command, scenarios-dir
`./scenarios/train_validation_mix_v4`):

- zero `frozen`-kind bags in the decode;
- clean yield ≥ 0.85 of the 64 run_ids after retries;
- retries converge — `attempt>0` rows exist for every retried run_id and
  no run exhausts `--max-retries` on a `physics:*` detail (three
  consecutive probe failures on one scenario = a deterministic bug, not
  the race);
- zero `abort_bad_start` (regression check on the entry-servo
  suppression — its failure signature is a deep mission start);
- median probe overhead ≲ 60 s (gate OPEN line minus graph-complete line
  in `sim_data/<sweep>/_logs/*.log`);
- spot-check bags: pre-mission dip ≤ ~0.5 m, leg-1 entry hump present.

If the pilot's `abort_init` rate is ≤ ~5%, 64 slots may be worth
re-testing to halve wall time — the probe converts residual frozen
launches into cheap retries either way.

### Visualization

`run_analysis.py` runs host-side (outside the SIF) and reads the recorded MCAP bags off disk, writes a markdown statistics report and renders the plot PNG(s).

```bash
python src/nautilus-ros/scripts/run_analysis.py sim_data/nominal_pid_sweep
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
    target_pressure_pa:=147150.0 angle_rad:=0.6109 n_oscillations:=1 \
    record:=true sampler_id:=bcu_hydro_coarse run_id:=lhs_0000 \
    scenario:=/ros2_ws/scenarios/bcu_hydro_coarse/lhs_0000.yaml
```

Arguments:

- `--cleanenv` strips the host's env (DISPLAY, ROS_DOMAIN_ID, etc.) so
  the container starts from a baseline.
- `--env GZ_IP=127.0.0.1` keeps the Gazebo transport off any VPN-routable
  interface. Required when the host is on a VPN, harmless otherwise.
- `--bind ./sim_data:/ros2_ws/sim_data` is where recorded bags land.
  The launch builds `[{sampler_id}/]{run_id}_{ts}/raw/` *inside* this
  dir, so bind the parent.
- `--bind ./scenarios:/ros2_ws/scenarios:ro` exposes a host directory of
  YAMLs you can edit without rebuilding the SIF.
- `headless:=true` is required under `--cleanenv`: there's no DISPLAY
  to render a Gazebo GUI to.

A scenario YAML with a `rig.hydrodynamics:` block is detected by
`nautilus_hal.render_sdf.description_file_for_scenario` and triggers a
Jinja-rendered SDF, so the same `scenario:=` argument that picks the
controller gains can also perturb the simulated plant.

