"""Box plot of when each BCU fault label first appears, across a sweep.

Y axis: the error labels (effectiveness %) 80, 60, 40, 20, 0 — i.e. ladder levels
1..5 via `% = 100 - 20*level`. The 100% level (level 0) is the fault-free start, so
its box collapses at t≈0 and carries no information; it's left off. X axis: onset
time (min:sec), the first timestamp each run reaches that level (one point per run
per level reached). The ladder is monotonic, so later labels spread out to the right
as faults accumulate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..sweep_loader import RunEntry

# Label (effectiveness %) for each ladder level, indexed by level. Level 0 (100%) is
# the fault-free start and is dropped from the plot, so we display levels 1..5.
_PERCENTS = [100, 80, 60, 40, 20, 0]
_DISPLAY_LEVELS = [1, 2, 3, 4, 5]


def _fmt_hhmmss(seconds: float, _pos=None) -> str:
    """Format an onset time (seconds) as `hour:min:sec` for axis ticks."""
    total = int(round(max(seconds, 0.0)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _onset_times(fault: dict[str, np.ndarray]) -> dict[int, float]:
    """`level -> first timestamp the run reaches it`. Empty if the run has no fault
    stream. The ladder never decreases, so first occurrence is the onset."""
    t, level = fault["t"], fault["level"]
    if t.size == 0:
        return {}
    onsets: dict[int, float] = {}
    for lv in np.unique(level):
        onsets[int(lv)] = float(t[int(np.argmax(level == lv))])
    return onsets


def plot_error_box(
    runs_with_fault: list[tuple[RunEntry, dict[str, np.ndarray]]],
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Render the onset-time box plot and write a PNG to `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One list of onset times per displayed label, in _DISPLAY_LEVELS order.
    per_label: list[list[float]] = [[] for _ in _DISPLAY_LEVELS]
    n_runs = 0
    for _, fault in runs_with_fault:
        onsets = _onset_times(fault)
        if not onsets:
            continue
        n_runs += 1
        for level, t in onsets.items():
            if level in _DISPLAY_LEVELS:
                per_label[_DISPLAY_LEVELS.index(level)].append(t)

    fig, ax = plt.subplots(figsize=(9, 6))
    positions = range(1, len(_DISPLAY_LEVELS) + 1)
    bp = ax.boxplot(
        [vals or [np.nan] for vals in per_label],
        positions=list(positions),
        widths=0.6,
        showmeans=True,
        vert=False,
    )
    # Reuse the real boxplot artists as legend handles so the swatches match exactly.
    ax.legend([bp["means"][0], bp["medians"][0]], ["mean", "median"], loc="best")
    # Count of runs that reached each label, at the left edge of each box's row.
    for pos, vals in zip(positions, per_label):
        ax.text(ax.get_xlim()[0], pos, f"n={len(vals)}", ha="left", va="bottom",
                fontsize=8, color="gray")

    ax.set_yticks(list(positions))
    ax.set_yticklabels([f"{_PERCENTS[lv]}%" for lv in _DISPLAY_LEVELS])
    ax.set_ylabel("BCU effectiveness label")
    ax.set_xlabel("failure time [hr:min:sec]")
    ax.xaxis.set_major_formatter(_fmt_hhmmss)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(f"{title or out_path.stem}  (fault onset, {n_runs} runs)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
