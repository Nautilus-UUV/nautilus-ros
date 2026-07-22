"""Discover the runs in a sweep directory and classify each one's viability
(oscillated / floater / sinker / no-odometry), keeping only the oscillated runs.

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

# Viability thresholds live in the installed package (pure stdlib, no ROS) so
# the sim run watchdog's early-abort rule and this offline classifier are the
# same two numbers rather than two copies that must be edited in lockstep.
from py_pkg.watchdog.plausibility import MIN_DIVE_M, MIN_RETURN_M

_TIMESTAMP_SUFFIX = re.compile(r"_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$")

# The non-viable verdicts `classify_run` can return (everything except
# "oscillated"), in report order. Consumers (dataset_stats' viability
# section) key off this tuple, so a new class added there is one edit here.
NON_VIABLE_CLASSES = ("floater", "sinker", "no_odometry")


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    run_dir: Path
    bag_dir: Path
    scenario_yaml_path: str | None
    is_nominal: bool
    # Run-watchdog / sim-gate verdict from sweep_status.csv
    # ("mission_complete" / "abort_floater" / "abort_sinker" /
    # "abort_bad_start" / "abort_init"); "" for legacy sweeps and
    # wall-clock kills — classify from the bag in that case. Retried
    # runs resolve to their final attempt's verdict (last CSV row wins).
    verdict: str = ""


def _looks_nominal(yaml_path: str | None) -> bool:
    return bool(yaml_path) and Path(yaml_path).name == "nominal.yaml"


def _read_status_csv(sweep_dir: Path) -> dict[str, dict]:
    """`run_id -> row` map. Empty if the file is missing (ad-hoc runs, or a sweep
    that crashed before the orchestrator wrote rows). run_sweep's retry
    path appends one row per attempt for the same run_id; the dict
    comprehension keeps the LAST row — the final attempt, whose bag is
    the one on disk (failed attempts' bags are parked as raw.failedN)."""
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
            RunEntry(
                run_id,
                run_dir,
                bag_dir,
                yaml_path,
                is_nominal,
                verdict=str(row.get("verdict", "") or "").strip(),
            )
        )

    if nominal_run_id is not None and not any(e.is_nominal for e in entries):
        raise ValueError(
            f"--nominal-run-id {nominal_run_id!r} matched no run dir under {sweep_dir}"
        )
    return entries


def classify_run(
    z: np.ndarray, *, min_dive_m: float = MIN_DIVE_M, min_return_m: float = MIN_RETURN_M
) -> str:
    """Classify a run's vertical viability from its odometry z trace.

    Returns one of:

    - `"no_odometry"` — the bag recorded no odometry samples;
    - `"floater"`     — never descended `min_dive_m` below its spawn depth
                        (startup failure: bobs at the surface);
    - `"sinker"`      — dived but never climbed back `min_return_m` from its
                        running-deepest point (mis-buoyant plant, sinks forever);
    - `"oscillated"`  — completed at least one dive + climb-back cycle.

    Odometry z is negative-down, so dive depth is `z[0] - z.min()` and the
    climb-back ("drawup") from the running-deepest is
    `(z - np.minimum.accumulate(z)).max()`.

    The thresholds are numerically identical to the downstream `_oscillated`
    gate (`UG-anomaly_detection/src/data/build_dataset.py`), but the basis
    differs deliberately: this classifier reads ground-truth odometry while the
    downstream gate reads external-pressure-derived depth. Downstream stays the
    final authority on what enters a dataset — this exists for generation-time
    visibility into a sweep's viable-run yield.
    """
    if z.size == 0:
        return "no_odometry"
    if float(z[0] - z.min()) < min_dive_m:
        return "floater"
    drawup = float((z - np.minimum.accumulate(z)).max())
    if drawup < min_return_m:
        return "sinker"
    return "oscillated"


def select_oscillated_runs(
    entries: Iterable[RunEntry],
    *,
    model_name: str = "glider_nautilus",
    min_dive_m: float = MIN_DIVE_M,
    min_return_m: float = MIN_RETURN_M,
) -> tuple[list[tuple[RunEntry, dict[str, np.ndarray]]], dict[str, str]]:
    """Keep only runs that oscillated (dived and climbed back — `classify_run`),
    dropping surface-floaters, continuous sinkers, and odometry-less runs.

    Reads odometry once per run and returns `(kept_with_traj, dropped)` — where
    `dropped` maps `run_id -> reason` — so the pose plot can reuse the
    trajectories without re-reading the bags.
    """
    # Local import keeps this module's import light (rosbags/scipy load only here).
    from .bag_reader import read_odometry

    kept: list[tuple[RunEntry, dict[str, np.ndarray]]] = []
    dropped: dict[str, str] = {}
    for entry in entries:
        try:
            traj = read_odometry(entry.bag_dir, model_name=model_name)
        except FileNotFoundError:
            # Truncated recording (mcap present, metadata.yaml never written —
            # run killed mid-write): no readable odometry.
            dropped[entry.run_id] = "no_odometry"
            continue
        verdict = classify_run(
            traj["z"], min_dive_m=min_dive_m, min_return_m=min_return_m
        )
        if verdict == "oscillated":
            kept.append((entry, traj))
        else:
            dropped[entry.run_id] = verdict
    return kept, dropped


def nominal_entry(entries: Iterable[RunEntry]) -> RunEntry | None:
    return next((e for e in entries if e.is_nominal), None)


def read_launch_args(sweep_dir: Path) -> dict[str, str]:
    """Parse `launch_args.txt` (the `key:=value` argv forwarded to `ros2 launch`)
    into a dict. Returns `{}` when the file is missing."""
    p = Path(sweep_dir) / "launch_args.txt"
    if not p.is_file():
        return {}
    return dict(tok.split(":=", 1) for tok in p.read_text().split() if ":=" in tok)


def read_scenario_anomaly(
    entry: RunEntry, *, sweep_dir: Path | None = None
) -> dict | None:
    """The anomaly label a run carries, from its scenario YAML.

    Reads the top-level `anomaly:` label block (written by the sampler
    and validated against `rig.faults` at load time). Returns
    `{"class", "channel", "archetype"}` — a YAML without a label block
    is an explicit `"nominal"` record (absence of the label never has
    to be inferred downstream). Per-timestamp ground truth lives in the
    bag's `/anomaly/label` stream; this is the run-level view.

    `entry.scenario_yaml_path` is recorded relative to the sweep's launch cwd
    (the workspace root), so we try it verbatim and then against each parent of
    `sweep_dir`, letting analysis run from any working directory. Returns None
    only when the YAML itself is missing or unresolvable.
    """
    path = _resolve_scenario_yaml(entry.scenario_yaml_path, sweep_dir)
    if path is None:
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    label = doc.get("anomaly") or {}
    return {
        "class": label.get("anomaly_class", "nominal"),
        "channel": label.get("channel", ""),
        "archetype": label.get("archetype", ""),
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
