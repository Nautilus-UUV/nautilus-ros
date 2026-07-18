"""Dataset statistics summary for a ros-collected sweep.

Counts the oscillated runs, totals their dive time, and breaks both down
per anomaly class (`nominal` / `bcu_pump` / `sensor` / `comms` /
`biofouling` — the run-level labels from `read_scenario_anomaly`).
Anomalies are persistent whole-run at constant severity, so a run's
entire dive time belongs to its class — there is no onset/dwell split
(the old Poisson-ladder dwell table died with the ladder; per-timestamp
ground truth lives in each bag's `/anomaly/label` stream instead).

Emits a markdown report — summary lines above the per-class table,
followed by a "Run viability" section counting the whole sweep's
oscillated / floater / sinker / no-odometry runs — which
`run_analysis.py` both writes to disk and echoes to the terminal.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from py_pkg.scenarios.anomaly import ANOMALY_CLASSES

from .sweep_loader import NON_VIABLE_CLASSES, RunEntry


def summarize_dataset(
    kept: list[tuple[RunEntry, dict[str, np.ndarray]]],
    *,
    title: str | None = None,
    dropped: dict[str, str] | None = None,
    anomaly_records: list[dict | None],
) -> str:
    """Build the markdown statistics report for the oscillated-run set.

    `kept` carries the odometry trajectories (for run durations);
    `anomaly_records` is the matching per-run label record from
    `read_scenario_anomaly`, in the same order — `None` records count
    as "unlabeled", kept distinct from "nominal" so a broken join is
    visible, never silently counted as healthy data. `dropped` is the
    `run_id -> reason` map of non-viable runs the caller filtered out
    (`select_oscillated_runs`), rendered as the "Run viability" section.
    """
    class_hours: Counter[str] = Counter()
    class_runs: Counter[str] = Counter()
    archetype_runs: Counter[str] = Counter()
    total_hours = 0.0

    for (_, traj), record in zip(kept, anomaly_records):
        t = traj["t"]
        dur_h = float(t[-1] - t[0]) / 3600.0 if t.size else 0.0
        total_hours += dur_h

        cls = record["class"] if record is not None else "unlabeled"
        class_hours[cls] += dur_h
        class_runs[cls] += 1
        if record is not None and cls == "sensor":
            archetype_runs[f"{record['channel']}/{record['archetype']}"] += 1

    return _render_markdown(
        title or "dataset",
        len(kept),
        dropped or {},
        total_hours,
        class_hours,
        class_runs,
        archetype_runs,
    )


def _render_markdown(
    title: str,
    n_runs: int,
    dropped: dict[str, str],
    total_hours: float,
    class_hours: dict[str, float],
    class_runs: dict[str, int],
    archetype_runs: dict[str, int],
) -> str:
    note = (
        f"  ({len(dropped)} non-viable dropped — see run viability)" if dropped else ""
    )
    per_run_h = total_hours / n_runs if n_runs else 0.0
    lines = [
        f"# {title} — dataset statistics",
        "",
        f"total runs: {n_runs}{note}",
        f"total dive hours: {total_hours:.2f}",
        f"dive time per run: {per_run_h:.2f} h",
        "",
        "| anomaly class | runs | dive hours | % of dive hrs |",
        "|:--------------|-----:|-----------:|--------------:|",
    ]
    # Canonical classes always render (zeros included); anything else
    # the records carried ("unlabeled", or a class this module has
    # never heard of) appears only when non-empty — never dropped
    # silently. The vocabulary is the sampler's own (py_pkg), so a new
    # class lands in the table without touching this file.
    extras = sorted(set(class_runs) - set(ANOMALY_CLASSES))
    for cls in (*ANOMALY_CLASSES, *extras):
        hrs = class_hours[cls]
        pct = (hrs / total_hours * 100.0) if total_hours > 0 else 0.0
        lines.append(f"| {cls} | {class_runs[cls]} | {hrs:.2f} | {pct:.1f} |")
    if archetype_runs:
        lines += [
            "",
            "sensor runs by channel/archetype: "
            + ", ".join(f"{k} x{n}" for k, n in sorted(archetype_runs.items())),
        ]
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
