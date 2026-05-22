"""2x3 grid of run-overlaid pose time-series.

Top row: X(t), Y(t), Z(t). Bottom row: roll(t), pitch(t), yaw(t). All runs share a
single time axis. Non-nominal runs are drawn faint so the cloud-shape conveys
deviation; the nominal run (if any) is drawn bold red on top. When no nominal is
flagged we fall back to an empirical median computed on a 1 Hz common grid — this
is the natural reference for an LHS sweep where every run is a perturbation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M

from .sweep_loader import RunEntry


_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_LABELS_DEG = {
    "x": "along-track [m]",
    "y": "cross-track [m]",
    "z": "Z [m]",
    "roll": "$\\Delta$roll [deg]",
    "pitch": "$\\Delta$pitch [deg]",
    "yaw": "$\\Delta$yaw [deg]",
}
_LABELS_RAD = {
    **_LABELS_DEG,
    "roll": "$\\Delta$roll [rad]",
    "pitch": "$\\Delta$pitch [rad]",
    "yaw": "$\\Delta$yaw [rad]",
}


def normalize_trajectory(traj: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Re-express a trajectory relative to its initial pose.

    Position: subtract spawn point and rotate the horizontal plane by the spawn
    heading so body-forward (the glider's nose direction projected onto the
    world horizontal) becomes +x_plot. Z is left alone.

    Attitude: replace absolute roll/pitch/yaw with the body-frame Euler angles of
    `R(q_t) * R(q_0)^-1` — i.e. the rotation the controller has driven the body
    through since spawn. This kills the ±π wrap that appears when the FRD body
    convention puts identity attitude at world-frame roll = 180°.
    """
    if traj["t"].size < 1:
        return traj

    # --- attitude relative to initial ---
    # We don't carry the raw quaternions out of bag_reader, so reconstruct from
    # the absolute Euler angles we did record. as_euler('xyz', extrinsic) is what
    # we plotted in v1, so feed it back the same way to round-trip cleanly.
    eul = np.stack([traj["roll"], traj["pitch"], traj["yaw"]], axis=1)
    rots = Rotation.from_euler("xyz", eul)
    rel = rots[0].inv() * rots
    rel_eul = rel.as_euler("xyz")
    out = dict(traj)
    out["roll"] = rel_eul[:, 0]
    out["pitch"] = rel_eul[:, 1]
    out["yaw"] = rel_eul[:, 2]

    # --- position relative to spawn, rotated by spawn heading ---
    dx = traj["x"] - traj["x"][0]
    dy = traj["y"] - traj["y"][0]
    # Heading vector at spawn: where body-X points in world after the spawn
    # rotation. For FRD-on-ENU at yaw=π/2 this comes out to (0, 1, 0); we
    # compute it per-run so the same code works for any spawn yaw.
    fwd_world = rots[0].apply([1.0, 0.0, 0.0])
    heading = np.arctan2(fwd_world[1], fwd_world[0])
    c, s = np.cos(heading), np.sin(heading)
    # Rotate (dx, dy) by -heading: along-track = projection onto heading vector,
    # cross-track = perpendicular component (left positive).
    out["x"] = c * dx + s * dy
    out["y"] = -s * dx + c * dy
    out["z"] = traj["z"].copy()
    return out


def _pitch_extrema_score(
    traj: dict[str, np.ndarray], pitch_limit_rad: float
) -> float | None:
    """Sum of squared distances of the trajectory's pitch max/min from ±limit (rad²).
    Lower is better. Returns None for trajectories too short to score.
    """
    if traj["t"].size < 2:
        return None
    pitch = traj["pitch"]
    pmax = float(np.max(pitch))
    pmin = float(np.min(pitch))
    return (pmax - pitch_limit_rad) ** 2 + (pmin + pitch_limit_rad) ** 2


def _depth_proximity_info(
    traj: dict[str, np.ndarray], target_depth_m: float
) -> tuple[float, float, float] | None:
    """Return (t_min, z_min, dist) for the run's deepest sample relative to
    z = -target_depth_m. dist = |z_min - target_z| in metres; lower is better
    (the deepest excursion lands closer to the intended setpoint, either by
    barely-reaching or barely-overshooting).

    Eligibility is gated on actually crossing target on descent — without that,
    a run that never reached the setpoint could win simply by being shallow.
    Returns None when the trajectory has < 2 samples or never crossed target.
    """
    if traj["t"].size < 2:
        return None
    target_z = -float(target_depth_m)
    z = traj["z"]
    crossed = bool(((z[:-1] > target_z) & (z[1:] <= target_z)).any())
    if not crossed:
        return None
    i_min = int(np.argmin(z))
    z_min = float(z[i_min])
    t_min = float(traj["t"][i_min])
    return (t_min, z_min, abs(z_min - target_z))


def pick_pitch_extrema_winner(
    runs_with_traj: list[tuple[RunEntry, dict[str, np.ndarray]]],
    pitch_limit_rad: float,
) -> tuple[RunEntry, dict[str, np.ndarray], float] | None:
    """Pick the run whose pitch peaks land closest to ±pitch_limit at the sawtooth's
    turning points.

    Score = (max(pitch) - +limit)² + (min(pitch) - -limit)², both in radians. Lower
    is better. Global max/min stand in for the first peak/trough — runs that never
    reverse direction (failed dives) naturally lose because one of the extrema is
    far from its target. Returns (entry, traj, score) or None when nothing scores.
    """
    best: tuple[RunEntry, dict[str, np.ndarray], float] | None = None
    for entry, traj in runs_with_traj:
        score = _pitch_extrema_score(traj, pitch_limit_rad)
        if score is None:
            continue
        if best is None or score < best[2]:
            best = (entry, traj, score)
    return best


def pick_depth_proximity_winner(
    runs_with_traj: list[tuple[RunEntry, dict[str, np.ndarray]]],
    target_depth_m: float,
) -> tuple[RunEntry, dict[str, np.ndarray], float, float, float] | None:
    """Pick the run whose deepest excursion lands closest to z = -target_depth_m.

    Score = |min(z) - target_z|. Returns (entry, traj, t_min, z_min, dist) or None
    when no eligible run is available. Eligibility (run must actually cross target
    on descent) is inherited from `_depth_proximity_info`.
    """
    best: tuple[RunEntry, dict[str, np.ndarray], float, float, float] | None = None
    for entry, traj in runs_with_traj:
        info = _depth_proximity_info(traj, target_depth_m)
        if info is None:
            continue
        t_min, z_min, dist = info
        if best is None or dist < best[4]:
            best = (entry, traj, t_min, z_min, dist)
    return best


def pick_combined_winner(
    runs_with_traj: list[tuple[RunEntry, dict[str, np.ndarray]]],
    pitch_limit_rad: float,
    target_depth_m: float,
) -> tuple[RunEntry, dict[str, np.ndarray], int, int, int] | None:
    """Suggest a single run that does well on both pitch-extrema and depth-proximity
    metrics, via rank-sum aggregation.

    For each metric, eligible runs are sorted best-to-worst and assigned rank 1..N.
    The combined winner minimises (pitch_rank + depth_rank). Rank-sum sidesteps the
    unit mismatch between rad² and m/s — we don't need a hand-tuned weighting to
    compare them. The trade-off is that rank-sum is insensitive to *how much*
    better a top run is than its neighbours; if two metrics ever disagree sharply
    we may need a continuous score (z-score or relative-ratio), but for the kind
    of LHS clouds these sweeps produce, ranks are a clean readout.

    Only runs eligible for *both* metrics participate (i.e. ≥2 samples AND a
    descent crossing of target depth). Returns
    (entry, traj, pitch_rank, depth_rank, eligible_n) or None when no run
    qualifies on both.
    """
    pitch_scores: dict[str, float] = {}
    depth_scores: dict[str, float] = {}
    by_id: dict[str, tuple[RunEntry, dict[str, np.ndarray]]] = {}

    for entry, traj in runs_with_traj:
        ps = _pitch_extrema_score(traj, pitch_limit_rad)
        di = _depth_proximity_info(traj, target_depth_m)
        if ps is None or di is None:
            # Need both metrics defined to be a combined candidate. A run that
            # never crosses target depth is fundamentally not "the best overall".
            continue
        by_id[entry.run_id] = (entry, traj)
        pitch_scores[entry.run_id] = ps
        depth_scores[entry.run_id] = di[2]

    if not by_id:
        return None

    ids = list(by_id.keys())
    pitch_order = sorted(ids, key=lambda r: pitch_scores[r])
    depth_order = sorted(ids, key=lambda r: depth_scores[r])
    pitch_rank = {r: i + 1 for i, r in enumerate(pitch_order)}
    depth_rank = {r: i + 1 for i, r in enumerate(depth_order)}

    best_id = min(ids, key=lambda r: pitch_rank[r] + depth_rank[r])
    entry, traj = by_id[best_id]
    return (entry, traj, pitch_rank[best_id], depth_rank[best_id], len(ids))


def _plot_highlight(
    axes_flat,
    traj: dict[str, np.ndarray],
    *,
    color: str,
    label: str,
    angle_scale: float,
    zorder: int = 10,
) -> None:
    """Draw one trajectory as a bold highlight across all six subplots. The label is
    attached to the X subplot only — the legend-dedupe pass picks it up from there.
    """
    for ax, key in zip(axes_flat, _KEYS):
        series = traj[key]
        if key in ("roll", "pitch", "yaw"):
            series = series * angle_scale
        ax.plot(
            traj["t"],
            series,
            color=color,
            alpha=0.85,
            lw=1.2,
            zorder=zorder,
            label=label if key == "x" else None,
        )


def plot_sweep(
    runs_with_traj: list[tuple[RunEntry, dict[str, np.ndarray]]],
    out_path: Path,
    *,
    degrees: bool = True,
    title: str | None = None,
    normalize: bool = True,
    pitch_limit_deg: float | None = 35.0,
    target_pressure_pa: float | None = None,
    mark_best_pitch: bool = True,
    mark_best_depth: bool = True,
    mark_best_combined: bool = True,
) -> Path:
    """Render the 2x3 grid and write a PNG to `out_path`. Returns the path written.

    `normalize=True` re-expresses each run's position relative to its spawn point
    (rotated into along-track / cross-track / depth) and each attitude relative
    to its spawn quaternion. Without normalization the FRD body convention puts
    every run's roll at ±180° and a 90° spawn yaw swaps the X/Y axes — the
    normalized form is what reads correctly for controller analysis.

    `pitch_limit_deg` draws dashed ±limit guides on the pitch subplot — the
    glider's mechanical pitch envelope. Pass `None` to suppress. `target_pressure_pa`
    is the depth controller's gauge-pressure setpoint; when set, we draw a dashed
    line on the Z subplot at the equivalent depth (Z is up-positive, so the line
    sits at -depth).

    `mark_best_pitch` highlights the run whose global pitch max/min land closest to
    ±pitch_limit (only computed when pitch_limit_deg is set). `mark_best_depth`
    highlights the run whose deepest excursion lands closest to -target_depth (only
    computed when target_pressure_pa is set). `mark_best_combined`
    highlights the rank-sum suggested run across both metrics (needs both guide
    inputs set). The nominal run is never re-highlighted as a "best" winner — it
    already has its own line. When the combined winner coincides with the pitch or
    depth winner, only the combined line is drawn; its label calls out the dual
    achievement.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        runs_with_traj = [(e, normalize_trajectory(t)) for e, t in runs_with_traj]

    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 7))
    axes_flat = axes.flatten()  # x, y, z, roll, pitch, yaw

    nominal: tuple[RunEntry, dict[str, np.ndarray]] | None = None
    plotted = 0
    angle_scale = (180.0 / np.pi) if degrees else 1.0

    for entry, traj in runs_with_traj:
        if traj["t"].size == 0:
            continue
        plotted += 1
        if entry.is_nominal and nominal is None:
            nominal = (entry, traj)
            continue
        for ax, key in zip(axes_flat, _KEYS):
            series = traj[key]
            if key in ("roll", "pitch", "yaw"):
                series = series * angle_scale
            ax.plot(traj["t"], series, color="steelblue", alpha=0.25, lw=0.8)

    if nominal is not None:
        _, traj = nominal
        _plot_highlight(
            axes_flat, traj, color="crimson", label="nominal",
            angle_scale=angle_scale, zorder=10,
        )

    # --- reference markers ---
    # Pitch envelope: glider's mechanical up/down pitch limit. Drawn in whatever
    # angle units we're plotting in so the guide lines up with the data.
    if pitch_limit_deg is not None:
        pitch_ax = axes_flat[_KEYS.index("pitch")]
        limit = pitch_limit_deg if degrees else (pitch_limit_deg * np.pi / 180.0)
        pitch_ax.axhline(
            +limit, color="gray", ls="--", lw=1.0, alpha=0.7,
            label=f"±{pitch_limit_deg:g}°",
        )
        pitch_ax.axhline(-limit, color="gray", ls="--", lw=1.0, alpha=0.7)

    # Target depth: gauge-pressure setpoint expressed in metres. World-frame Z is
    # up-positive, so a positive depth lands at -depth on this axis. We also
    # surface the count of plotted runs that actually crossed this line — that's
    # the denominator behind the rank/N tags on the suggested-winner label, and
    # otherwise looks suspiciously small versus the full sweep size.
    target_depth_m: float | None = None
    if target_pressure_pa is not None:
        target_depth_m = target_pressure_pa / WATER_PRESSURE_GRADIENT_PA_PER_M
        # Count strictly within the sweep candidates (excludes the nominal, which
        # is the reference rather than a sweep sample). Total plotted = the same
        # count plus the nominal if one exists.
        sweep_only = [
            t for e, t in runs_with_traj if not e.is_nominal and t["t"].size
        ]
        n_reached = sum(
            1 for t in sweep_only if _depth_proximity_info(t, target_depth_m) is not None
        )
        z_ax = axes_flat[_KEYS.index("z")]
        z_ax.axhline(
            -target_depth_m, color="darkgreen", ls="--", lw=1.0, alpha=0.8,
            label=(
                f"target {target_pressure_pa:g} Pa  (z = {-target_depth_m:.2f} m, "
                f"{n_reached}/{len(sweep_only)} runs reached it)"
            ),
        )

    # --- best-of-sweep highlights ---
    # All three picks run on the same post-normalize trajectories the main loop
    # drew, and exclude the nominal run (it has its own crimson highlight; if it
    # also won an aggregate metric the operator already sees that visually). Each
    # metric picks its own actual winner; when the combined-suggested run
    # coincides with either individual winner, only the combined line is drawn so
    # the plot doesn't stack three highlights on the same trajectory.
    sweep_candidates = [(e, t) for e, t in runs_with_traj if not e.is_nominal]
    limit_rad = (
        (pitch_limit_deg * np.pi / 180.0) if pitch_limit_deg is not None else None
    )

    pitch_winner = (
        pick_pitch_extrema_winner(sweep_candidates, limit_rad)
        if (mark_best_pitch and limit_rad is not None) else None
    )
    depth_winner = (
        pick_depth_proximity_winner(sweep_candidates, target_depth_m)
        if (mark_best_depth and target_depth_m is not None) else None
    )
    combined_winner = (
        pick_combined_winner(sweep_candidates, limit_rad, target_depth_m)
        if (mark_best_combined and limit_rad is not None and target_depth_m is not None)
        else None
    )
    combined_id = combined_winner[0].run_id if combined_winner else None

    # Pitch winner — purple. Red is reserved for the suggested/combined winner
    # (the headline), so a single-metric pitch leader uses purple as a distinct
    # contrast against the steelblue cloud. Suppressed when the combined winner
    # is the same run; the red line below absorbs the achievement into its label.
    if pitch_winner is not None and pitch_winner[0].run_id != combined_id:
        entry, traj, score = pitch_winner
        rms_deg = float(np.sqrt(score) * 180.0 / np.pi)
        _plot_highlight(
            axes_flat, traj, color="purple",
            label=f"best pitch peaks: {entry.run_id}  (Δ={rms_deg:.1f}°)",
            angle_scale=angle_scale, zorder=11,
        )

    # Depth winner — green. Same combined-priority suppression rule.
    if depth_winner is not None and depth_winner[0].run_id != combined_id:
        entry, traj, t_min, z_min, dist = depth_winner
        _plot_highlight(
            axes_flat, traj, color="green",
            label=(
                f"closest to target: {entry.run_id}  "
                f"(|Δz|={dist:.2f} m)"
            ),
            angle_scale=angle_scale, zorder=11,
        )

    # Anchor the metric with a dot at the run's deepest sample — that's the
    # point whose distance from target we ranked on. Drawn even when the
    # depth-winner line itself got suppressed (combined-coincidence case).
    if depth_winner is not None:
        _, _, t_min, z_min, _ = depth_winner
        z_ax = axes_flat[_KEYS.index("z")]
        z_ax.plot(
            [t_min], [z_min],
            marker="o", color="green", ms=6, zorder=12,
        )

    # Combined / suggested winner — red, drawn last so it sits on top of any
    # coincident individual-winner line. Red is the "this is the headline"
    # color; the pitch envelope guide is rendered in neutral gray specifically
    # so it doesn't compete with this line on the pitch subplot. The legend
    # label calls out which individual metrics this same run also leads, so a
    # single line communicates the full set of achievements.
    if combined_winner is not None:
        entry, traj, p_rank, d_rank, n = combined_winner
        extras: list[str] = []
        if pitch_winner is not None and pitch_winner[0].run_id == entry.run_id:
            extras.append("also best pitch")
        if depth_winner is not None and depth_winner[0].run_id == entry.run_id:
            extras.append("also closest to target")
        extra_tag = f"  [{', '.join(extras)}]" if extras else ""
        _plot_highlight(
            axes_flat, traj, color="red",
            label=(
                f"suggested: {entry.run_id}  "
                f"(pitch #{p_rank}, depth #{d_rank} of {n} reached target){extra_tag}"
            ),
            angle_scale=angle_scale, zorder=12,
        )

    labels = _LABELS_DEG if degrees else _LABELS_RAD
    for ax, key in zip(axes_flat, _KEYS):
        ax.set_ylabel(labels[key])
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("time [s]")

    # Gather labelled artists across all subplots (nominal lives on the X axis,
    # pitch-envelope on the pitch axis, depth target on Z), then dedupe by label.
    seen: dict[str, object] = {}
    for ax in axes_flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lbl, h)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="upper left")

    fig.suptitle(
        f"{title or out_path.stem}  ({plotted} runs)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
