"""2x3 grid of run-overlaid pose time-series: X/Y/Z (top), roll/pitch/yaw (bottom).

All runs share one time axis. Non-nominal runs are drawn faint so the cloud-shape
conveys deviation; the nominal run (if any) is drawn bold red on top. Best-of-sweep
runs (closest pitch peaks, closest depth, and a rank-sum combined pick) are
highlighted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M
from scipy.spatial.transform import Rotation

from ..sweep_loader import RunEntry

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
    heading so body-forward becomes +x_plot (Z left alone). Attitude: replace
    absolute roll/pitch/yaw with the Euler angles of `R(q_t) * R(q_0)^-1` — the
    rotation driven since spawn — which kills the ±π wrap from the FRD convention.
    """
    if traj["t"].size < 1:
        return traj

    eul = np.stack([traj["roll"], traj["pitch"], traj["yaw"]], axis=1)
    rots = Rotation.from_euler("xyz", eul)
    rel_eul = (rots[0].inv() * rots).as_euler("xyz")
    out = dict(traj)
    out["roll"] = rel_eul[:, 0]
    out["pitch"] = rel_eul[:, 1]
    out["yaw"] = rel_eul[:, 2]

    # Position relative to spawn, rotated into along-track / cross-track by the
    # spawn heading (where body-X points in world after the spawn rotation).
    dx = traj["x"] - traj["x"][0]
    dy = traj["y"] - traj["y"][0]
    fwd_world = rots[0].apply([1.0, 0.0, 0.0])
    heading = np.arctan2(fwd_world[1], fwd_world[0])
    c, s = np.cos(heading), np.sin(heading)
    out["x"] = c * dx + s * dy
    out["y"] = -s * dx + c * dy
    out["z"] = traj["z"].copy()
    return out


def _pitch_extrema_score(
    traj: dict[str, np.ndarray], pitch_limit_rad: float
) -> float | None:
    """Sum of squared distances of the run's pitch max/min from ±limit (rad²).
    Lower is better. None for trajectories too short to score."""
    if traj["t"].size < 2:
        return None
    pitch = traj["pitch"]
    return (float(np.max(pitch)) - pitch_limit_rad) ** 2 + (
        float(np.min(pitch)) + pitch_limit_rad
    ) ** 2


def _depth_proximity_info(
    traj: dict[str, np.ndarray], target_depth_m: float
) -> tuple[float, float, float] | None:
    """Return (t_min, z_min, dist) for the run's deepest sample relative to
    z = -target_depth_m; dist = |z_min - target_z| (lower is better). Eligibility is
    gated on actually crossing target on descent, so a too-shallow run can't win.
    None when the run has < 2 samples or never crossed target."""
    if traj["t"].size < 2:
        return None
    target_z = -float(target_depth_m)
    z = traj["z"]
    if not bool(((z[:-1] > target_z) & (z[1:] <= target_z)).any()):
        return None
    i_min = int(np.argmin(z))
    z_min = float(z[i_min])
    return (float(traj["t"][i_min]), z_min, abs(z_min - target_z))


def pick_pitch_extrema_winner(
    runs_with_traj: list[tuple[RunEntry, dict[str, np.ndarray]]],
    pitch_limit_rad: float,
) -> tuple[RunEntry, dict[str, np.ndarray], float] | None:
    """Run whose global pitch max/min land closest to ±pitch_limit (the sawtooth
    turning points). Returns (entry, traj, score) or None when nothing scores."""
    best = None
    for entry, traj in runs_with_traj:
        score = _pitch_extrema_score(traj, pitch_limit_rad)
        if score is not None and (best is None or score < best[2]):
            best = (entry, traj, score)
    return best


def pick_depth_proximity_winner(
    runs_with_traj: list[tuple[RunEntry, dict[str, np.ndarray]]],
    target_depth_m: float,
) -> tuple[RunEntry, dict[str, np.ndarray], float, float, float] | None:
    """Run whose deepest excursion lands closest to z = -target_depth_m. Returns
    (entry, traj, t_min, z_min, dist) or None when no eligible run is available."""
    best = None
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
    """Suggest one run that does well on both metrics via rank-sum aggregation: each
    metric ranks its eligible runs 1..N, the winner minimises (pitch_rank +
    depth_rank). Rank-sum sidesteps the rad²-vs-metre unit mismatch. Only runs
    eligible for *both* metrics participate. Returns
    (entry, traj, pitch_rank, depth_rank, eligible_n) or None."""
    by_id: dict[str, tuple[RunEntry, dict[str, np.ndarray]]] = {}
    pitch_scores: dict[str, float] = {}
    depth_scores: dict[str, float] = {}
    for entry, traj in runs_with_traj:
        ps = _pitch_extrema_score(traj, pitch_limit_rad)
        di = _depth_proximity_info(traj, target_depth_m)
        if ps is None or di is None:
            continue
        by_id[entry.run_id] = (entry, traj)
        pitch_scores[entry.run_id] = ps
        depth_scores[entry.run_id] = di[2]

    if not by_id:
        return None

    ids = list(by_id.keys())
    pitch_rank = {r: i + 1 for i, r in enumerate(sorted(ids, key=pitch_scores.get))}
    depth_rank = {r: i + 1 for i, r in enumerate(sorted(ids, key=depth_scores.get))}
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
    attached to the X subplot only — the legend-dedupe pass picks it up from there."""
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


def plot_pose_multi(
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

    `normalize` re-expresses each run relative to its spawn pose (along-track /
    cross-track / depth + attitude relative to spawn) — the form that reads correctly
    for controller analysis. `pitch_limit_deg` draws the glider's mechanical pitch
    envelope; `target_pressure_pa` draws the depth setpoint on the Z subplot (at
    -depth). The `mark_best_*` flags highlight the per-metric and rank-sum winners;
    when the combined winner coincides with a per-metric winner only the red combined
    line is drawn, with its label calling out the dual achievement.
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
        _plot_highlight(
            axes_flat, nominal[1], color="crimson", label="nominal", angle_scale=angle_scale
        )

    # Pitch envelope: mechanical up/down limit, drawn in the plotted angle units.
    if pitch_limit_deg is not None:
        pitch_ax = axes_flat[_KEYS.index("pitch")]
        limit = pitch_limit_deg if degrees else (pitch_limit_deg * np.pi / 180.0)
        pitch_ax.axhline(
            +limit, color="gray", ls="--", lw=1.0, alpha=0.7, label=f"±{pitch_limit_deg:g}°"
        )
        pitch_ax.axhline(-limit, color="gray", ls="--", lw=1.0, alpha=0.7)

    # Target depth: gauge-pressure setpoint in metres (world Z up-positive, so at
    # -depth). The label surfaces how many runs actually crossed it — the denominator
    # behind the rank/N tags on the suggested-winner label.
    target_depth_m: float | None = None
    if target_pressure_pa is not None:
        target_depth_m = target_pressure_pa / WATER_PRESSURE_GRADIENT_PA_PER_M
        sweep_only = [t for e, t in runs_with_traj if not e.is_nominal and t["t"].size]
        n_reached = sum(
            1 for t in sweep_only if _depth_proximity_info(t, target_depth_m) is not None
        )
        z_ax = axes_flat[_KEYS.index("z")]
        z_ax.axhline(
            -target_depth_m,
            color="darkgreen",
            ls="--",
            lw=1.0,
            alpha=0.8,
            label=(
                f"target {target_pressure_pa:g} Pa  (z = {-target_depth_m:.2f} m, "
                f"{n_reached}/{len(sweep_only)} runs reached it)"
            ),
        )

    # Best-of-sweep highlights, computed on the post-normalize trajectories and
    # excluding the nominal run (it has its own crimson line).
    sweep_candidates = [(e, t) for e, t in runs_with_traj if not e.is_nominal]
    limit_rad = (pitch_limit_deg * np.pi / 180.0) if pitch_limit_deg is not None else None

    pitch_winner = (
        pick_pitch_extrema_winner(sweep_candidates, limit_rad)
        if (mark_best_pitch and limit_rad is not None)
        else None
    )
    depth_winner = (
        pick_depth_proximity_winner(sweep_candidates, target_depth_m)
        if (mark_best_depth and target_depth_m is not None)
        else None
    )
    combined_winner = (
        pick_combined_winner(sweep_candidates, limit_rad, target_depth_m)
        if (mark_best_combined and limit_rad is not None and target_depth_m is not None)
        else None
    )
    combined_id = combined_winner[0].run_id if combined_winner else None

    # Pitch winner — purple (red is reserved for the combined headline). Suppressed
    # when it is the combined winner; the red line absorbs the achievement.
    if pitch_winner is not None and pitch_winner[0].run_id != combined_id:
        entry, traj, score = pitch_winner
        rms_deg = float(np.sqrt(score) * 180.0 / np.pi)
        _plot_highlight(
            axes_flat,
            traj,
            color="purple",
            label=f"best pitch peaks: {entry.run_id}  (Δ={rms_deg:.1f}°)",
            angle_scale=angle_scale,
            zorder=11,
        )

    # Depth winner — green. Same combined-priority suppression rule.
    if depth_winner is not None and depth_winner[0].run_id != combined_id:
        entry, traj, t_min, z_min, dist = depth_winner
        _plot_highlight(
            axes_flat,
            traj,
            color="green",
            label=f"closest to target: {entry.run_id}  (|Δz|={dist:.2f} m)",
            angle_scale=angle_scale,
            zorder=11,
        )

    # Anchor the depth metric with a dot at the deepest sample we ranked on (drawn
    # even when the depth line itself was suppressed).
    if depth_winner is not None:
        _, _, t_min, z_min, _ = depth_winner
        axes_flat[_KEYS.index("z")].plot(
            [t_min], [z_min], marker="o", color="green", ms=6, zorder=12
        )

    # Combined / suggested winner — red, drawn last so it sits on top. Its label
    # calls out which individual metrics this same run also leads.
    if combined_winner is not None:
        entry, traj, p_rank, d_rank, n = combined_winner
        extras = []
        if pitch_winner is not None and pitch_winner[0].run_id == entry.run_id:
            extras.append("also best pitch")
        if depth_winner is not None and depth_winner[0].run_id == entry.run_id:
            extras.append("also closest to target")
        extra_tag = f"  [{', '.join(extras)}]" if extras else ""
        _plot_highlight(
            axes_flat,
            traj,
            color="red",
            label=(
                f"suggested: {entry.run_id}  "
                f"(pitch #{p_rank}, depth #{d_rank} of {n} reached target){extra_tag}"
            ),
            angle_scale=angle_scale,
            zorder=12,
        )

    labels = _LABELS_DEG if degrees else _LABELS_RAD
    for ax, key in zip(axes_flat, _KEYS):
        ax.set_ylabel(labels[key])
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("time [s]")

    # Dedupe labelled artists across subplots (labels live on different axes).
    seen: dict[str, object] = {}
    for ax in axes_flat:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lbl, h)
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="upper left")

    fig.suptitle(f"{title or out_path.stem}  ({plotted} runs)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
