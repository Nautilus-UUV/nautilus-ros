#!/usr/bin/env python3
"""`plot_sweep` CLI: render analysis plots for a ros-collected sweep dataset.

Discovers the runs in a dataset dir (form of `sim_data/bcu_fault_dataset/`), drops
the ones that failed on startup (floated at the surface, never dived), and writes
the requested plot(s).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `analysis.plotting.pose_multi_plot` reaches `py_pkg.physics`; inject the subrepo's
# `src/py_pkg` so that import resolves host-side without a sourced ROS env. The
# `analysis/` package is already importable via this script's own directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src/py_pkg"))

from analysis.bag_reader import read_fault_levels
from analysis.plotting.error_box_plot import plot_error_box
from analysis.plotting.pose_multi_plot import plot_pose_multi
from analysis.sweep_loader import discover_sweep, read_launch_args, select_dived_runs

_CHOICES = ("pose_multi_plot", "error_box_plot", "all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="ros-collected dataset dir (e.g. sim_data/bcu_fault_dataset/)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_SCRIPTS_DIR / "output",
        help="output directory for the PNG(s) (default: scripts/output/)",
    )
    parser.add_argument(
        "--plot",
        choices=_CHOICES,
        default="all",
        help="which plot to generate (default: all)",
    )
    args = parser.parse_args(argv)

    input_path: Path = args.input_path
    if not input_path.is_dir():
        print(f"not a directory: {input_path}", file=sys.stderr)
        return 2

    # All runs in these fixed-duration sweeps are SIGTERM'd at the time limit
    # (exit_code=-15), so keep them and let the floater filter be the quality gate.
    entries = discover_sweep(input_path, include_failed=True)
    if not entries:
        print(f"no runs found in {input_path}", file=sys.stderr)
        return 1

    kept, dropped = select_dived_runs(entries)
    if dropped:
        print(
            f"dropped {len(dropped)} startup-failed (surface-floater) runs: {', '.join(dropped)}"
        )
    if not kept:
        print(
            "no runs survived the startup-failure filter; nothing to plot",
            file=sys.stderr,
        )
        return 1

    out_dir: Path = args.output_path
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = input_path.resolve().name
    want_pose = args.plot in ("pose_multi_plot", "all")
    want_box = args.plot in ("error_box_plot", "all")

    if want_pose:
        target_pa = _target_pressure_pa(input_path)
        written = plot_pose_multi(
            kept,
            out_dir / f"{sweep}_pose_multi_plot.png",
            title=sweep,
            target_pressure_pa=target_pa,
            mark_best_pitch=False,
            mark_best_depth=False,
            mark_best_combined=False,
        )
        print(f"wrote {written}  ({len(kept)} runs)")

    if want_box:
        faults = [(entry, read_fault_levels(entry.bag_dir)) for entry, _ in kept]
        written = plot_error_box(
            faults, out_dir / f"{sweep}_error_box_plot.png", title=sweep
        )
        print(f"wrote {written}  ({len(kept)} runs)")

    return 0


def _target_pressure_pa(input_path: Path) -> float | None:
    """Depth setpoint from the sweep's `launch_args.txt`, if recorded."""
    recorded = read_launch_args(input_path).get("target_pressure_pa")
    if recorded is None:
        return None
    try:
        return float(recorded)
    except ValueError:
        print(
            f"non-numeric target_pressure_pa={recorded!r} in launch_args.txt; ignoring",
            file=sys.stderr,
        )
        return None


if __name__ == "__main__":
    sys.exit(main())
