"""Dataset statistics summary for a ros-collected sweep.

Counts the oscillated runs, totals their dive time, and breaks that time down
across the BCU fault ladder (`/bcu/rpm/fault`, levels 0..5 / 100..0 %
effectiveness) — the same ladder the error-box plot keys off. Emits a markdown
report — a few summary lines (run count, dive hours, dive time per run, the
initialised MTBF interval) above the fault-ladder dwell table, followed by a
"Run viability" section counting the whole sweep's oscillated / floater /
sinker / no-odometry runs — which `run_analysis.py` both writes to disk and
echoes to the terminal.

The fault level is a latched 10 Hz (1 Hz post-throttle) signal that spans the
whole run, so "hours at level k" is the dwell-weighted integral of that step
signal. We take each run's duration from odometry (authoritative) and split it
across levels by the fault stream's own time fractions, so the per-class hours
always sum back to the total dive hours.
"""

from __future__ import annotations

import numpy as np

from .sweep_loader import NON_VIABLE_CLASSES, RunEntry

# Effectiveness label per ladder level, indexed by level — the canonical 5-step
# 100..0 % ladder, matching plotting/error_box_plot.py.
_PERCENTS = [100, 80, 60, 40, 20, 0]
_LEVELS = list(range(len(_PERCENTS)))  # 0..5


def _level_fractions(fault: dict[str, np.ndarray]) -> dict[int, float]:
    """`level -> fraction of the run's fault-stream time spent at that level`.

    Each sample holds its level until the next one arrives (the signal is
    latched), so a sample's dwell is the gap to its successor; we sum those gaps
    per level and normalise by the total span. Empty when the run has no usable
    fault stream — the caller then attributes the whole run to the healthy
    level 0.
    """
    t, level = fault["t"], fault["level"]
    if t.size < 2:
        return {}
    span = float(t[-1] - t[0])
    if span <= 0:
        return {}
    dt = np.diff(t)
    return {
        int(lv): float(dt[level[:-1] == lv].sum()) / span
        for lv in np.unique(level[:-1])
    }


def summarize_dataset(
    kept: list[tuple[RunEntry, dict[str, np.ndarray]]],
    faults: list[tuple[RunEntry, dict[str, np.ndarray]]],
    *,
    title: str | None = None,
    dropped: dict[str, str] | None = None,
    fault_cfgs: list[dict[str, float] | None],
) -> str:
    """Build the markdown statistics report for the oscillated-run set.

    `kept` carries the odometry trajectories (for run durations); `faults` is the
    matching per-run fault stream from `read_fault_levels`, in the same order.
    `fault_cfgs` is the matching per-run *initialised* fault config from
    `read_scenario_faults` (its `mttf_sec`), summarised into the MTBF interval.
    `dropped` is the `run_id -> reason` map of non-viable runs the caller
    filtered out (`select_oscillated_runs`), rendered as the "Run viability"
    section.
    """
    class_hours: dict[int, float] = {lv: 0.0 for lv in _LEVELS}
    reached: dict[int, int] = {lv: 0 for lv in _LEVELS}
    total_hours = 0.0

    for (_, traj), (_, fault) in zip(kept, faults):
        t = traj["t"]
        dur_h = float(t[-1] - t[0]) / 3600.0 if t.size else 0.0
        total_hours += dur_h

        fracs = _level_fractions(fault)
        if fracs:
            for lv, frac in fracs.items():
                class_hours[lv] = class_hours.get(lv, 0.0) + frac * dur_h
        else:
            class_hours[0] += dur_h  # no fault stream -> count as fully healthy

        levels = fault["level"]
        present = {int(x) for x in np.unique(levels)} if levels.size else {0}
        for lv in present:
            reached[lv] = reached.get(lv, 0) + 1

    mttfs = [c["mttf_sec"] for c in fault_cfgs if c is not None]
    return _render_markdown(
        title or "dataset",
        len(kept),
        dropped or {},
        total_hours,
        class_hours,
        reached,
        mttfs,
    )


def _render_markdown(
    title: str,
    n_runs: int,
    dropped: dict[str, str],
    total_hours: float,
    class_hours: dict[int, float],
    reached: dict[int, int],
    mttfs: list[float],
) -> str:
    note = (
        f"  ({len(dropped)} non-viable dropped — see run viability)" if dropped else ""
    )
    per_run_h = total_hours / n_runs if n_runs else 0.0
    interval = f"{min(mttfs):.0f}–{max(mttfs):.0f} s" if mttfs else "—"
    lines = [
        f"# {title} — dataset statistics",
        "",
        f"total runs: {n_runs}{note}",
        f"total dive hours: {total_hours:.2f}",
        f"dive time per run: {per_run_h:.2f} h",
        f"MTBF interval: {interval}",
        "",
        "| level | effectiveness | dive hours | % of dive hrs | runs reached |",
        "|------:|:-------------:|-----------:|--------------:|-------------:|",
    ]
    for lv in _LEVELS:
        hrs = class_hours.get(lv, 0.0)
        pct = (hrs / total_hours * 100.0) if total_hours > 0 else 0.0
        lines.append(
            f"| {lv} | {_PERCENTS[lv]}% | {hrs:.2f} | {pct:.1f} | {reached.get(lv, 0)} |"
        )
    lines += _viability_lines(n_runs, dropped)
    return "\n".join(lines) + "\n"


def _viability_lines(n_kept: int, dropped: dict[str, str]) -> list[str]:
    """The "Run viability" section: per-class run counts over the whole sweep
    (kept oscillated runs + every dropped class), then the dropped run ids per
    reason so a bad sweep's failures are identifiable without re-reading bags.
    """
    total = n_kept + len(dropped)
    by_reason = {
        reason: sorted(rid for rid, r in dropped.items() if r == reason)
        for reason in NON_VIABLE_CLASSES
    }
    pct = lambda n: (n / total * 100.0) if total else 0.0  # noqa: E731
    lines = [
        "",
        "## Run viability",
        "",
        f"total runs: {total}",
        "",
        "| class | runs | % of runs |",
        "|:------|-----:|----------:|",
        f"| oscillated (kept) | {n_kept} | {pct(n_kept):.1f} |",
    ]
    for reason in NON_VIABLE_CLASSES:
        n = len(by_reason[reason])
        lines.append(f"| {reason} | {n} | {pct(n):.1f} |")
    for reason in NON_VIABLE_CLASSES:
        if by_reason[reason]:
            lines.append(f"\ndropped {reason}: {', '.join(by_reason[reason])}")
    return lines
