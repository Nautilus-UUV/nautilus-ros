"""Discover the runs in a sweep directory and drop the ones that failed on startup.

A sweep on disk is `sim_data/{name}/` with one `{run_id}_{timestamp}/raw/` subdir
per run, plus a `sweep_status.csv` keyed on `run_id` (its `yaml_path` column
identifies the nominal run without re-reading every YAML).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

_TIMESTAMP_SUFFIX = re.compile(r"_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$")


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    run_dir: Path
    bag_dir: Path
    scenario_yaml_path: str | None
    is_nominal: bool


def _looks_nominal(yaml_path: str | None) -> bool:
    return bool(yaml_path) and Path(yaml_path).name == "nominal.yaml"


def _read_status_csv(sweep_dir: Path) -> dict[str, dict]:
    """`run_id -> row` map. Empty if the file is missing (ad-hoc runs, or a sweep
    that crashed before the orchestrator wrote rows)."""
    status_csv = sweep_dir / "sweep_status.csv"
    if not status_csv.is_file():
        return {}
    with status_csv.open() as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def discover_sweep(
    sweep_dir: Path,
    *,
    nominal_run_id: str | None = None,
    include_failed: bool = False,
) -> list[RunEntry]:
    """Return a `RunEntry` per `{run_id}_{ts}/raw/` directory found on disk.

    The filesystem is the source of truth for "which runs exist"; `sweep_status.csv`
    only supplies `yaml_path` (nominal detection) and `exit_code`. Non-zero-exit runs
    are dropped unless `include_failed=True` — pass that for sweeps that time out by
    design (fixed-duration fault/sawtooth runs, where every row is exit_code=-15 but
    the bags are intact). Runs absent from the CSV are always kept.
    """
    sweep_dir = Path(sweep_dir)
    rows = _read_status_csv(sweep_dir)

    entries: list[RunEntry] = []
    for run_dir in sorted(sweep_dir.glob("*_*")):
        bag_dir = run_dir / "raw"
        if not run_dir.is_dir() or not bag_dir.is_dir():
            continue
        run_id = _TIMESTAMP_SUFFIX.sub("", run_dir.name)
        row = rows.get(run_id, {})
        exit_code = str(row.get("exit_code", "")).strip()
        if row and not include_failed and exit_code != "0":
            continue
        yaml_path = row.get("yaml_path") or None
        is_nominal = (
            run_id == nominal_run_id
            if nominal_run_id is not None
            else _looks_nominal(yaml_path)
        )
        entries.append(
            RunEntry(run_id, run_dir, bag_dir, yaml_path, is_nominal)
        )

    if nominal_run_id is not None and not any(e.is_nominal for e in entries):
        raise ValueError(
            f"--nominal-run-id {nominal_run_id!r} matched no run dir under {sweep_dir}"
        )
    return entries


def select_dived_runs(
    entries: Iterable[RunEntry],
    *,
    model_name: str = "glider_nautilus",
    min_dive_m: float = 2.0,
) -> tuple[list[tuple[RunEntry, dict[str, np.ndarray]]], list[str]]:
    """Drop runs that float at the surface and never dive — the startup-failure mode
    in these sweeps (a run spawns at ~-5 m, fails to initialise, and bobs at the
    surface instead of executing the dive). A run is kept when it descends at least
    `min_dive_m` below its spawn depth; the gap between the two populations is wide
    (floaters dive ~0 m, real runs >=5 m), so the threshold is not delicate.

    Reads odometry once per run and returns `(kept_with_traj, dropped_ids)` so the
    pose plot can reuse the trajectories without re-reading the bags.
    """
    # Local import keeps this module's import light (rosbags/scipy load only here).
    from .bag_reader import read_odometry

    kept: list[tuple[RunEntry, dict[str, np.ndarray]]] = []
    dropped: list[str] = []
    for entry in entries:
        traj = read_odometry(entry.bag_dir, model_name=model_name)
        z = traj["z"]
        if z.size and float(z[0] - z.min()) >= min_dive_m:
            kept.append((entry, traj))
        else:
            dropped.append(entry.run_id)
    return kept, dropped


def nominal_entry(entries: Iterable[RunEntry]) -> RunEntry | None:
    return next((e for e in entries if e.is_nominal), None)


def read_launch_args(sweep_dir: Path) -> dict[str, str]:
    """Parse `launch_args.txt` (the `key:=value` argv forwarded to `ros2 launch`)
    into a dict. Returns `{}` when the file is missing."""
    p = Path(sweep_dir) / "launch_args.txt"
    if not p.is_file():
        return {}
    return dict(
        tok.split(":=", 1) for tok in p.read_text().split() if ":=" in tok
    )


def read_scenario_faults(
    entry: RunEntry, *, sweep_dir: Path | None = None
) -> dict[str, float] | None:
    """The BCU fault-injector knobs a run was *initialised* with.

    Reads `rig.faults.bcu_rpm` from the run's scenario YAML and returns
    `{"mttf_sec": ..., "num_levels": ...}` — the configured mean time between
    successive degradation steps, as opposed to the ladder actually realised in
    the bag (`read_fault_levels`). `mttf_sec <= 0` means faults were disabled.

    `entry.scenario_yaml_path` is recorded relative to the sweep's launch cwd
    (the workspace root), so we try it verbatim and then against each parent of
    `sweep_dir`, letting analysis run from any working directory. Returns None
    when the path is missing, unresolvable, or carries no `bcu_rpm` fault block.
    """
    path = _resolve_scenario_yaml(entry.scenario_yaml_path, sweep_dir)
    if path is None:
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    bcu = ((doc.get("rig") or {}).get("faults") or {}).get("bcu_rpm") or {}
    if "mttf_sec" not in bcu:
        return None
    return {
        "mttf_sec": float(bcu["mttf_sec"]),
        "num_levels": int(bcu.get("num_levels", 5)),
    }


def _resolve_scenario_yaml(
    yaml_path: str | None, sweep_dir: Path | None
) -> Path | None:
    """Locate the scenario YAML named by a (usually relative) `yaml_path`.

    Tries it as-is (absolute, or relative to the cwd) first, then joined onto
    each parent of `sweep_dir` — the workspace root is one of those parents, and
    that is what the relative path is anchored to. None if nothing resolves.
    """
    if not yaml_path:
        return None
    p = Path(yaml_path)
    if p.is_absolute():
        return p if p.is_file() else None
    bases = [Path.cwd()]
    if sweep_dir is not None:
        bases += list(Path(sweep_dir).resolve().parents)
    for base in bases:
        cand = base / p
        if cand.is_file():
            return cand
    return None
