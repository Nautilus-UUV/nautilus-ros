"""`plot_sweep` CLI: aggregate all runs in a sweep dir onto one 2x3 trajectory plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bag_reader import read_odometry
from .pose_plot import plot_sweep
from .sweep_loader import RunEntry, discover_sweep, read_launch_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="sim_data/{sweep_name}/ directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output PNG path (default: <sweep_dir>/trajectories.png)",
    )
    parser.add_argument(
        "--nominal-run-id",
        type=str,
        default=None,
        help="run_id of the nominal trajectory; overrides yaml-based auto-detect",
    )
    parser.add_argument(
        "--nominal-bag",
        type=Path,
        default=None,
        help=(
            "path to a bag directory recorded by running nominal.yaml directly; "
            "rendered as the highlighted reference trajectory. Use this when the "
            "sweep itself contains only perturbed samples."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="glider_nautilus",
        help="Gazebo model name used to build the odometry topic",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "skip the normalize-to-spawn-pose step (plot world-frame X/Y/Z + "
            "absolute roll/pitch/yaw). Default off because the FRD body "
            "convention parks roll at ±180°."
        ),
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "include runs with non-zero exit_code (use for sweeps that always time out "
            "by design, e.g. fixed-duration sawtooth/pid_calibration)"
        ),
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="plot roll/pitch/yaw in radians (default: degrees)",
    )
    parser.add_argument(
        "--pitch-limit-deg",
        type=float,
        default=35.0,
        help=(
            "draw dashed ±limit guide lines on the pitch subplot (degrees). "
            "Default 35 — pass 0 (or a negative value) to suppress."
        ),
    )
    parser.add_argument(
        "--target-pressure-pa",
        type=float,
        default=None,
        help=(
            "depth controller setpoint in gauge Pa; rendered as a dashed line on "
            "the Z subplot at the equivalent depth (converted via water "
            "hydrostatic gradient). When omitted, falls back to "
            "<sweep_dir>/launch_args.txt if run_sweep.py wrote one."
        ),
    )
    parser.add_argument(
        "--all-winners",
        action="store_true",
        help=(
            "also highlight the per-metric pitch and depth winners (purple and "
            "green respectively) alongside the suggested winner. Off by default; "
            "default plot shows only the suggested/combined winner in red."
        ),
    )
    parser.add_argument(
        "--no-mark-best-combined",
        dest="mark_best_combined",
        action="store_false",
        help=(
            "don't highlight the rank-sum suggested run that aggregates pitch and "
            "depth metrics. On by default; auto-skipped if either of pitch limit / "
            "target pressure is unset."
        ),
    )
    args = parser.parse_args(argv)

    sweep_dir: Path = args.sweep_dir
    if not sweep_dir.is_dir():
        print(f"not a directory: {sweep_dir}", file=sys.stderr)
        return 2

    out_path = args.out or (sweep_dir / "trajectories.png")

    runs = discover_sweep(
        sweep_dir,
        nominal_run_id=args.nominal_run_id,
        include_failed=args.include_failed,
    )
    if not runs:
        print(f"no runs found in {sweep_dir}", file=sys.stderr)
        return 1

    runs_with_traj: list[tuple[RunEntry, object]] = []
    skipped = 0
    for entry in runs:
        try:
            traj = read_odometry(entry.bag_dir, model_name=args.model_name)
        except Exception as exc:
            print(f"[{entry.run_id}] read failed: {exc}", file=sys.stderr)
            skipped += 1
            continue
        if traj["t"].size == 0:
            print(f"[{entry.run_id}] no odometry on /model/{args.model_name}/odometry/throttled")
            skipped += 1
            continue
        runs_with_traj.append((entry, traj))

    if args.nominal_bag is not None:
        traj = read_odometry(args.nominal_bag, model_name=args.model_name)
        if traj["t"].size == 0:
            print(
                f"--nominal-bag {args.nominal_bag} has no odometry; ignoring",
                file=sys.stderr,
            )
        else:
            ext_nominal = RunEntry(
                run_id="nominal",
                run_dir=args.nominal_bag.parent,
                bag_dir=args.nominal_bag,
                scenario_yaml_path=None,
                is_nominal=True,
            )
            runs_with_traj.append((ext_nominal, traj))

    if not runs_with_traj:
        print("no runs had readable odometry; nothing to plot", file=sys.stderr)
        return 1

    pitch_limit = args.pitch_limit_deg if args.pitch_limit_deg > 0 else None
    target_pa = args.target_pressure_pa
    if target_pa is None:
        recorded = read_launch_args(sweep_dir).get("target_pressure_pa")
        if recorded is not None:
            try:
                target_pa = float(recorded)
                print(f"target_pressure_pa={target_pa} (from launch_args.txt)")
            except ValueError:
                print(
                    f"launch_args.txt has non-numeric target_pressure_pa={recorded!r}; ignoring",
                    file=sys.stderr,
                )
    written = plot_sweep(
        runs_with_traj,
        out_path,
        degrees=not args.radians,
        title=sweep_dir.name,
        normalize=not args.raw,
        pitch_limit_deg=pitch_limit,
        target_pressure_pa=target_pa,
        mark_best_pitch=args.all_winners,
        mark_best_depth=args.all_winners,
        mark_best_combined=args.mark_best_combined,
    )
    print(
        f"wrote {written}  ({len(runs_with_traj)} runs plotted, {skipped} skipped)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
