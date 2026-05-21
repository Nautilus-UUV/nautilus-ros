"""Walk a sweep directory and decide which run is the nominal one.

A sweep on disk is `sim_data/{name}/` with one `{run_id}_{timestamp}/raw/` subdir per
run plus a `sweep_status.csv` keyed on `run_id`. The CSV's `yaml_path` column tells us
which scenario YAML each run executed against, which is how we identify the nominal
run without re-reading every YAML.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_TIMESTAMP_SUFFIX = re.compile(r"_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$")


@dataclass(frozen=True)
class RunEntry:
    run_id: str
    run_dir: Path
    bag_dir: Path
    scenario_yaml_path: str | None
    is_nominal: bool


def _find_run_dir(sweep_dir: Path, run_id: str) -> Path | None:
    # Run dirs are `{run_id}_{YYYY_MM_DD-HH_MM_SS}`; the timestamp suffix means we
    # can't reconstruct the path from run_id alone.
    matches = sorted(sweep_dir.glob(f"{run_id}_*"))
    return matches[0] if matches else None


def _looks_nominal(yaml_path: str | None) -> bool:
    if not yaml_path:
        return False
    return Path(yaml_path).name == "nominal.yaml"


def _read_status_csv(sweep_dir: Path) -> dict[str, dict]:
    """`run_id -> row` map. Empty if the file is missing — that's OK for sweeps that
    crashed before the orchestrator wrote a row, or for ad-hoc single runs."""
    status_csv = sweep_dir / "sweep_status.csv"
    if not status_csv.is_file():
        return {}
    rows: dict[str, dict] = {}
    with status_csv.open() as f:
        for row in csv.DictReader(f):
            rows[row["run_id"]] = row
    return rows


def discover_sweep(
    sweep_dir: Path,
    *,
    nominal_run_id: str | None = None,
    include_failed: bool = False,
) -> list[RunEntry]:
    """Return a `RunEntry` per `{run_id}_{ts}/raw/` directory found on disk.

    `sweep_status.csv` is read for `yaml_path` and `exit_code` metadata when present,
    but the source of truth for "which runs exist" is the filesystem. The orchestrator
    can crash before flushing late rows (this happened in `crashed_pid_calibration`,
    where the CSV has 24 of 32 runs), and ad-hoc single runs don't write a CSV at all.

    Rows whose `exit_code` is non-zero are dropped by default — bags from killed runs
    are often present and readable but we don't want a crashed run masquerading as the
    nominal trajectory. Pass `include_failed=True` to keep them; that's the right call
    when the whole sweep timed out by design (e.g. `crashed_pid_calibration`, where
    every row has exit_code=-15 but the bags are intact for the duration they ran).
    Runs not in the CSV at all are always kept — there's no exit_code to inspect.
    """
    sweep_dir = Path(sweep_dir)
    rows = _read_status_csv(sweep_dir)

    entries: list[RunEntry] = []
    for run_dir in sorted(sweep_dir.glob("*_*")):
        if not run_dir.is_dir():
            continue
        bag_dir = run_dir / "raw"
        if not bag_dir.is_dir():
            continue
        # `{run_id}_{YYYY_MM_DD-HH_MM_SS}` — strip the trailing timestamp suffix.
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
                run_id=run_id,
                run_dir=run_dir,
                bag_dir=bag_dir,
                scenario_yaml_path=yaml_path,
                is_nominal=is_nominal,
            )
        )

    if nominal_run_id is not None and not any(e.is_nominal for e in entries):
        raise ValueError(
            f"--nominal-run-id {nominal_run_id!r} matched no run dir under {sweep_dir}"
        )
    return entries


def nominal_entry(entries: Iterable[RunEntry]) -> RunEntry | None:
    for e in entries:
        if e.is_nominal:
            return e
    return None


def read_launch_args(sweep_dir: Path) -> dict[str, str]:
    """Parse `launch_args.txt` (written by run_sweep.py) into a key→value dict.

    The file is the literal argv we forwarded to `ros2 launch`, one whitespace-
    separated entry per arg, each of the form `key:=value`. Returns `{}` when
    the file is missing — older sweeps (anything recorded before this lands)
    will simply have nothing to surface.
    """
    p = Path(sweep_dir) / "launch_args.txt"
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for tok in p.read_text().split():
        if ":=" in tok:
            k, v = tok.split(":=", 1)
            out[k] = v
    return out
