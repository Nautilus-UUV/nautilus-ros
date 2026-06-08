# Analysis tooling revamp — `scripts/analysis/` + `plot_sweep.py`

> Status: implemented (2026-06-08). This records the plan and the verification
> findings that shaped it.

## Context

The analysis tooling under `scripts/analysis/` grew a verbose, single-purpose pose
plotter plus a heavily-flagged CLI. It was reorganized into a `plotting/` package
(one script per plot), a second plot showing how the BCU fault labels distribute
over time, a sweep loader that drops runs which failed on startup, and a
3-argument `plot_sweep.py`. Driving dataset: `sim_data/bcu_fault_dataset/`
(128 LHS runs of the BCU fault ladder).

## Verified findings (against all 128 runs)

- **All 128 runs are `exit_code=-15` / `timed_out=True`** (fixed-duration runs,
  SIGTERM by design) → `discover_sweep` must be called with `include_failed=True`.
- **"BCU never actuated" does not discriminate**: `/bcu/rpm` → 4000 within <1 s in
  every run; `/bcu/flow_rate` is uniformly 0 (unpopulated). The ACU-roll==0 anomaly
  (11 runs) is a red herring — those runs dove normally.
- **The startup-failure mode is surface-floating, read off Z**: every run spawns at
  z ≈ −5.0 m. Exactly 3 runs (`lhs_0067`, `lhs_0090`, `lhs_0111`) never descend
  below spawn (dive 0.00 m) and bob at the surface; every other run dives ≥ 5.16 m.
  Clean gap → `min_dive_m` threshold (default 2.0 m) drops exactly those 3.
- **Fault label mapping** (`docs/bcu_fault_model_plan.md`): `/bcu/rpm/fault`
  (`std_msgs/Int32`) carries the latched level 0..5; `% = 100 − 20·level` → the 6
  labels 100/80/60/40/20/0. Monotonic ladder ⇒ onset of level L = first timestamp
  it reaches L.

## Decisions

- **Startup filter**: drop surface-floaters via Z (`z[0] − min(z) < min_dive_m`).
- **Error box plot Y**: onset time per run per level (label 100 sits at t≈0).
- **Pose plot**: keep winner-marking + nominal highlighting; strip only the extra
  CLI flags and verbose docstrings.

## Layout

```
scripts/
  plot_sweep.py                  # 3-arg CLI, dispatches to plotting/
  analysis/
    __init__.py
    bag_reader.py                # read_odometry + read_fault_levels
    sweep_loader.py              # discover_sweep + select_dived_runs (floater filter)
    plotting/
      __init__.py
      pose_multi_plot.py         # plot_pose_multi (winners kept)
      error_box_plot.py          # plot_error_box (onset-time boxes)
```

## CLI

```
plot_sweep.py INPUT_PATH [--output-path DIR] [--plot {pose_multi_plot,error_box_plot,all}]
```
- `INPUT_PATH` — dataset dir (form of `sim_data/bcu_fault_dataset/`).
- `--output-path` — default `scripts/output/`, treated as a directory.
- `--plot` — default `all`. Output files `<sweep_name>_<plot>.png`.

Flow: `discover_sweep(include_failed=True)` → `select_dived_runs` (drops floaters,
returns trajectories for reuse) → `plot_pose_multi` (target depth from
`launch_args.txt`) and/or `plot_error_box` (reads `read_fault_levels` per kept run).

## Verification (run host-side with `/usr/bin/python3.12`)

```bash
/usr/bin/python3.12 src/nautilus-ros/scripts/plot_sweep.py sim_data/bcu_fault_dataset --plot all
```
- Drops exactly `lhs_0067, lhs_0090, lhs_0111`; 125 runs plotted.
- `bcu_fault_dataset_pose_multi_plot.png` — overlay + winner highlights, 125/125
  reached target (floaters gone).
- `bcu_fault_dataset_error_box_plot.png` — 6 boxes 100→0, onset time on Y; "100"
  collapses at t≈0, later labels spread (medians ~0.9k/1.9k/2.8k/4.0k/4.4k s), N
  falls 125→122→114→100→87→65 as fewer runs reach deeper degradation.

No new deps (`rosbags`, `matplotlib`, `numpy`, `scipy` already in
`requirements-analysis.txt`).
