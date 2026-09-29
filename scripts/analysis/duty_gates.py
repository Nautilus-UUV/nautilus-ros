"""Pure v2-campaign acceptance-gate computations (no bag/file IO here).

Three gates, computed from a sweep's 1 Hz recorded scalar streams (see
`run_gates.py` for the CLI that feeds them):

  Gate A — pump duty-cycle coverage. v1's nominal runs railed the pump
      ~92% of in-dive time (idle ~2%) while the real glider idles ~62%
      in-dive; every detector learned the gap as a domain feature. The
      per-run in-dive idle-fraction distribution over nominal
      mission-complete runs must populate every bin of [0.40, 0.80].
  Gate B — tank pressure inside the physical interval. The synthesized
      tank sensor must stay within the run's calibrated
      [empty, full] endpoints (+/- a noise margin); v1's free gas-law
      cushion reached 469 kPa against a real max of 195.5 kPa.
  Gate C — feedback-rail overlap. Nominal runs must contain interior
      (0 < |rpm| < rail) feedback samples, so "off the rails" cannot be
      a pump-fault giveaway.

Uniform 1 Hz samples make sample fractions time fractions. The in-dive
mask comes from the external-pressure stream (depth > 0.8 m, the lake
convention) — gates A and C only run on nominal runs, so sensor-fault
corruption of that channel never feeds a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The Pa/m gradient and the commanded rail are properties of the water and
# the actuator, not of these gates — so they come from the installed
# package and a density or EPOS4 change reaches the gates too.
from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M
from py_pkg.robot_specs import BCU_MOTOR_MAX_RPM

# The 2026-06-24 lake analysis dive boundary.
DIVE_THRESHOLD_M = 0.8

# Gate A bins + thresholds. Real-glider reference (2026-06-24 lake
# test): pooled in-dive idle 62.5%, per-dive 40.8-79.0%.
IDLE_BINS = ((0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80))
IDLE_BIN_MIN_SHARE = 0.03  # of analyzed nominal runs, per bin (>=1 run in pilot)

# Gate B margin over the calibrated endpoints: the tank noise chain is
# ~1.18 kPa at 3 sigma (sigma 353 Pa + 600 Pa comb), 2 kPa clears it.
TANK_MARGIN_PA = 2000.0

# Gate C thresholds (provisional until the pilot).
INTERIOR_PER_RUN_MIN = 0.05  # a run counts as having interior feedback above this
INTERIOR_RUN_SHARE_MIN = 0.5  # ... and this share of runs must clear it
INTERIOR_POOLED_MIN = 0.15


def in_dive_mask(
    t_ns: np.ndarray,
    ext_t_ns: np.ndarray,
    ext_pa: np.ndarray,
) -> np.ndarray:
    """Boolean in-dive mask for samples at `t_ns`, from external pressure.

    Depth is gauged against the run's first external sample (surfaced
    start), interpolated onto the target timestamps. All-False when the
    pressure stream is empty.
    """
    if len(ext_pa) == 0 or len(t_ns) == 0:
        return np.zeros(len(t_ns), dtype=bool)
    depth_m = (ext_pa - ext_pa[0]) / WATER_PRESSURE_GRADIENT_PA_PER_M
    depth_at = np.interp(t_ns.astype(np.float64), ext_t_ns.astype(np.float64), depth_m)
    return depth_at > DIVE_THRESHOLD_M


def idle_fraction(
    rpm_t_ns: np.ndarray,
    rpm: np.ndarray,
    ext_t_ns: np.ndarray,
    ext_pa: np.ndarray,
) -> tuple[float | None, int]:
    """(in-dive commanded-idle fraction, n in-dive samples) for one run.

    Idle = commanded rpm exactly 0 (the deadband snaps sub-500 commands
    to 0, so the wire value is discrete). None when the run has no
    in-dive samples.
    """
    mask = in_dive_mask(rpm_t_ns, ext_t_ns, ext_pa)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    return float((np.abs(rpm[mask]) == 0).mean()), n


def interior_feedback_fraction(
    fb_t_ns: np.ndarray,
    fb_rpm: np.ndarray,
    ext_t_ns: np.ndarray,
    ext_pa: np.ndarray,
    max_rpm: float = BCU_MOTOR_MAX_RPM,
) -> float | None:
    """In-dive fraction of feedback samples strictly between 0 and the rail."""
    mask = in_dive_mask(fb_t_ns, ext_t_ns, ext_pa)
    if not mask.any():
        return None
    fb = np.abs(fb_rpm[mask])
    return float(((fb > 0) & (fb < max_rpm)).mean())


@dataclass(frozen=True)
class TankCheck:
    min_pa: float
    max_pa: float
    low_bound_pa: float
    high_bound_pa: float

    @property
    def passed(self) -> bool:
        return self.min_pa >= self.low_bound_pa and self.max_pa <= self.high_bound_pa


def tank_within_interval(
    tank_pa: np.ndarray,
    empty_pa: float,
    full_pa: float,
    margin_pa: float = TANK_MARGIN_PA,
) -> TankCheck | None:
    """Whole-run tank-pressure interval check against the run's endpoints."""
    if len(tank_pa) == 0:
        return None
    return TankCheck(
        min_pa=float(tank_pa.min()),
        max_pa=float(tank_pa.max()),
        low_bound_pa=empty_pa - margin_pa,
        high_bound_pa=full_pa + margin_pa,
    )


def gate_a_idle_coverage(
    idle_fractions: list[float], pilot: bool = False
) -> tuple[bool, dict[str, float]]:
    """Per-bin coverage of the [0.40, 0.80] idle band over nominal runs.

    Returns (passed, histogram) where the histogram maps a bin label to
    the share of runs in it (plus "<0.40" / ">0.80" context rows).
    Pilot mode demands >= 1 run per bin instead of the 3% share.
    """
    n = len(idle_fractions)
    fracs = np.asarray(idle_fractions, dtype=np.float64)
    hist: dict[str, float] = {}
    hist["<0.40"] = float((fracs < IDLE_BINS[0][0]).mean()) if n else 0.0
    passed = n > 0
    for lo, hi in IDLE_BINS:
        # Closed top edge on the last bin so 0.80 itself counts.
        in_bin = (fracs >= lo) & (
            (fracs <= hi) if hi == IDLE_BINS[-1][1] else (fracs < hi)
        )
        share = float(in_bin.mean()) if n else 0.0
        hist[f"[{lo:.2f},{hi:.2f})"] = share
        ok = in_bin.sum() >= 1 if pilot else share >= IDLE_BIN_MIN_SHARE
        passed = passed and bool(ok)
    hist[">0.80"] = float((fracs > IDLE_BINS[-1][1]).mean()) if n else 0.0
    return passed, hist


def gate_c_interior_feedback(
    fractions: list[float], pilot: bool = False
) -> tuple[bool, float, float]:
    """(passed, share of runs above per-run min, pooled mean fraction)."""
    if not fractions:
        return False, 0.0, 0.0
    fracs = np.asarray(fractions, dtype=np.float64)
    per_run_ok = float((fracs >= INTERIOR_PER_RUN_MIN).mean())
    pooled = float(fracs.mean())
    if pilot:
        passed = bool((fracs >= INTERIOR_PER_RUN_MIN).any())
    else:
        passed = per_run_ok >= INTERIOR_RUN_SHARE_MIN and pooled >= INTERIOR_POOLED_MIN
    return passed, per_run_ok, pooled
