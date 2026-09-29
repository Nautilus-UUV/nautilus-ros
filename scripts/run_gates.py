#!/usr/bin/env python3
"""`run_gates` CLI: v2-campaign acceptance gates over a recorded sweep.

Runs the three duty/plant gates (see `analysis.duty_gates`) over a
ros-collected sweep dir (form of `sim_data/tvm_v2_pilot32/`):

  A — nominal in-dive idle-fraction distribution covers [0.40, 0.80];
  B — every run's tank pressure stays inside its scenario's calibrated
      [empty, full] endpoints (+2 kPa noise margin);
  C — nominal runs contain interior (off-rail, nonzero) feedback rpm.

Writes a markdown report + per-run JSON next to the stats output and
exits 0 (all gates pass) / 1 (any gate fails) — the campaign acceptance
record. Use `--pilot` for small pre-campaign sweeps (>= 1 run per idle
bin instead of the 3% share).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src/py_pkg"))

from analysis.bag_reader import read_scalar_streams
from analysis.duty_gates import (
    INTERIOR_PER_RUN_MIN,
    INTERIOR_POOLED_MIN,
    INTERIOR_RUN_SHARE_MIN,
    WATER_PRESSURE_GRADIENT_PA_PER_M,
    gate_a_idle_coverage,
    gate_c_interior_feedback,
    idle_fraction,
    interior_feedback_fraction,
    tank_within_interval,
)
from analysis.sweep_loader import (
    _resolve_scenario_yaml,
    classify_run,
    discover_sweep,
)
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM
from py_pkg.scenarios.spec.rig import PlantSpec
from py_pkg.uuv_ros_core.topics import UUVTopics

_PLANT_DEFAULTS = PlantSpec()


def _scenario_doc(entry, sweep_dir: Path) -> dict:
    """The run's scenario YAML, parsed once per run.

    Both the anomaly label and the tank endpoints come out of this one
    document — resolving the path walks every parent of `sweep_dir`, so
    reading it twice costs a stat walk and a parse per run.
    """
    path = _resolve_scenario_yaml(entry.scenario_yaml_path, sweep_dir)
    if path is None:
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _anomaly_class(doc: dict) -> str:
    """Run-level anomaly label; a YAML without the block is `"nominal"`."""
    return (doc.get("anomaly") or {}).get("anomaly_class", "nominal")


def _plant_endpoints(doc: dict) -> tuple[float, float]:
    """The run's tank endpoints from its scenario YAML (spec defaults absent)."""
    plant = (doc.get("rig") or {}).get("plant") or {}
    return (
        float(
            plant.get("tank_pressure_empty_pa", _PLANT_DEFAULTS.tank_pressure_empty_pa)
        ),
        float(
            plant.get("tank_pressure_full_pa", _PLANT_DEFAULTS.tank_pressure_full_pa)
        ),
    )


def _completed_from_pressure(ext_pa: np.ndarray) -> bool:
    """Legacy fallback (no watchdog verdict): oscillation from the depth trace.

    `sweep_loader.classify_run` itself, fed depth-from-external-pressure
    instead of odometry — negated because odometry z is negative-down. Only
    consulted for nominal runs, where that channel is uncorrupted.
    """
    if len(ext_pa) == 0:
        return False
    depth = (ext_pa - ext_pa[0]) / WATER_PRESSURE_GRADIENT_PA_PER_M
    return classify_run(-depth) == "oscillated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="sweep dataset dir")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_SCRIPTS_DIR / "output",
        help="report output directory (default: scripts/output/)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="pilot thresholds: >= 1 run per idle bin instead of the 3%% share",
    )
    parser.add_argument(
        "--max-rpm",
        type=float,
        default=BCU_MOTOR_MAX_RPM,
        help=(
            "commanded rail for the interior-feedback gate "
            f"(default {BCU_MOTOR_MAX_RPM})"
        ),
    )
    args = parser.parse_args(argv)

    sweep_dir = args.input_path
    entries = discover_sweep(sweep_dir, include_failed=True)
    if not entries:
        print(f"no runs found under {sweep_dir}", file=sys.stderr)
        return 1

    per_run: dict[str, dict] = {}
    idle_fracs: list[float] = []
    interior_fracs: list[float] = []
    tank_failures: list[str] = []
    n_nominal_complete = 0

    for entry in entries:
        doc = _scenario_doc(entry, sweep_dir)
        anomaly_class = _anomaly_class(doc)
        # One bag open per run: these bags are zstd-compressed whole-file, so
        # each Reader() streams the entire archive through a temp file.
        streams = read_scalar_streams(
            entry.bag_dir,
            (
                UUVTopics.EXTERNAL_PRESSURE,
                UUVTopics.BCU_PRESSURE,
                UUVTopics.BCU_RPM,
                UUVTopics.BCU_FEEDBACK_RPM,
            ),
        )
        ext_t, ext = streams[UUVTopics.EXTERNAL_PRESSURE]
        _, tank = streams[UUVTopics.BCU_PRESSURE]
        rec: dict = {"class": anomaly_class, "verdict": entry.verdict}

        # Gate B: every run with a tank stream, against its own endpoints.
        empty_pa, full_pa = _plant_endpoints(doc)
        check = tank_within_interval(tank, empty_pa, full_pa)
        if check is not None:
            rec["tank"] = {
                "min_pa": check.min_pa,
                "max_pa": check.max_pa,
                "bounds": [check.low_bound_pa, check.high_bound_pa],
                "passed": check.passed,
            }
            if not check.passed:
                tank_failures.append(entry.run_id)

        # Gates A + C: nominal runs that actually completed their mission.
        completed = (
            entry.verdict == "mission_complete"
            if entry.verdict
            else _completed_from_pressure(ext)
        )
        rec["completed"] = completed
        if anomaly_class == "nominal" and completed:
            n_nominal_complete += 1
            rpm_t, rpm = streams[UUVTopics.BCU_RPM]
            fb_t, fb = streams[UUVTopics.BCU_FEEDBACK_RPM]
            frac, n_in_dive = idle_fraction(rpm_t, rpm, ext_t, ext)
            if frac is not None:
                rec["idle_fraction"] = frac
                rec["n_in_dive"] = n_in_dive
                idle_fracs.append(frac)
            interior = interior_feedback_fraction(
                fb_t, fb, ext_t, ext, max_rpm=args.max_rpm
            )
            if interior is not None:
                rec["interior_feedback_fraction"] = interior
                interior_fracs.append(interior)

        per_run[entry.run_id] = rec

    a_passed, hist = gate_a_idle_coverage(idle_fracs, pilot=args.pilot)
    b_passed = not tank_failures
    c_passed, c_per_run, c_pooled = gate_c_interior_feedback(
        interior_fracs, pilot=args.pilot
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    (args.output_path / "gates_per_run.json").write_text(
        json.dumps(per_run, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        f"# v2 acceptance gates — {sweep_dir.name}",
        "",
        f"Runs discovered: {len(entries)}; nominal mission-complete analyzed: "
        f"{n_nominal_complete} ({len(idle_fracs)} with in-dive samples).",
        f"Mode: {'pilot' if args.pilot else 'campaign'}.",
        "",
        f"## Gate A — idle coverage: {'PASS' if a_passed else 'FAIL'}",
        "",
        "Per-run in-dive idle-fraction histogram (share of analyzed nominal "
        "runs; real-glider reference: pooled 0.625, per-dive 0.408–0.790):",
        "",
        "| bin | share |",
        "|---|---|",
        *(f"| {k} | {v:.3f} |" for k, v in hist.items()),
        "",
        f"## Gate B — tank interval: {'PASS' if b_passed else 'FAIL'}",
        "",
        (
            f"{len(tank_failures)} run(s) outside [empty−2 kPa, full+2 kPa]: "
            + ", ".join(tank_failures[:10])
            + ("…" if len(tank_failures) > 10 else "")
            if tank_failures
            else "All runs inside their calibrated tank interval."
        ),
        "",
        f"## Gate C — interior feedback: {'PASS' if c_passed else 'FAIL'}",
        "",
        f"Runs ≥ {INTERIOR_PER_RUN_MIN:.0%} interior: {c_per_run:.1%}; pooled "
        f"interior fraction: {c_pooled:.3f} (campaign thresholds: "
        f"≥{INTERIOR_RUN_SHARE_MIN:.0%} of runs, pooled ≥{INTERIOR_POOLED_MIN}).",
        "",
    ]
    report_path = args.output_path / "gates_report.md"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"report: {report_path}")

    return 0 if (a_passed and b_passed and c_passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
