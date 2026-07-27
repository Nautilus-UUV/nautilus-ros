"""Pure run-viability verdicts behind the sim run watchdog.

Streaming restatement of ``scripts/analysis/sweep_loader.py::classify_run``,
same axis convention: odometry z is negative-down, so the dive excursion is
``z0 - min(z)`` and the climb-back ("drawup") from the running-deepest point
is ``max(z - running_min(z))``. Where the offline classifier reads a finished
trace, this tracker decides *during* the run -- floaters need a deadline
(the trace may still dive) and sinkers a stall grace (it may still climb).

No ROS and no clock of its own -- ``(t_s, z_m)`` samples are handed in
directly, so every rule is unit-testable against synthetic streams.
"""

from __future__ import annotations

from dataclasses import dataclass

# The one home for the viability thresholds: this module is pure stdlib and
# ships inside the installed package, so ``scripts/analysis/sweep_loader.py``
# imports them from here rather than restating them. An early abort here is
# then exactly a run the offline analysis would drop anyway, by construction
# instead of by comment.
MIN_DIVE_M = 2.0
MIN_RETURN_M = 1.0

# Floater deadline, from arming. 360 s covers the slowest legitimate dive onset
# in the sweep envelope: a 0.55-effectiveness pump behind the lake-fitted
# delay/slew still deflating the full bladder (~150 s) plus sinking to
# MIN_DIVE_M with degraded authority (a 0.6-effectiveness pilot run reached
# only 1.4 m by the old 120 s deadline -- a false floater). Still a fast-fail
# next to the multi-ks mission budgets.
DIVE_DEADLINE_S = 360.0

# The sinker stall grace. The stall clock restarts at every deepening event, so
# this only has to cover one transit leg plus controller response; 240 s is
# generous against the slowest legitimate leg in the sweep envelope, so only a
# genuinely stuck vehicle trips the sinker rule.
STALL_GRACE_S = 240.0


@dataclass(frozen=True)
class PlausibilityConfig:
    """Thresholds for the in-run floater/sinker/bad-start verdicts."""

    min_dive_m: float = MIN_DIVE_M
    min_return_m: float = MIN_RETURN_M
    dive_deadline_s: float = DIVE_DEADLINE_S  # floater check, from arming
    stall_grace_s: float = STALL_GRACE_S  # sinker check
    deepen_epsilon_m: float = 0.05
    # Bad-start guard: a mission must begin from (near) the surface. If
    # the FIRST fed sample is already deeper than this, the run's init
    # went wrong (e.g. the vehicle fell during bringup) and every later
    # rule would misread it — dive_m is measured from that first sample.
    # <= 0 disables the guard (the pre-v3 behavior).
    max_start_depth_m: float = 0.0


class RunPlausibility:
    """Streaming floater/sinker detector. Arms at the first ``feed()`` sample.

    - ``"bad_start"`` -- the first sample is already deeper than
      ``max_start_depth_m`` (guard enabled when > 0): the run did not
      begin at the surface, so no later verdict can be trusted;
    - ``"floater"`` -- at/after ``dive_deadline_s`` since the first sample,
      the dive excursion never reached ``min_dive_m``;
    - ``"sinker"`` -- dived at least ``min_dive_m``, never drew up
      ``min_return_m`` from the running-deepest point, and has not deepened
      by ``deepen_epsilon_m`` for ``stall_grace_s`` (a creep slower than
      epsilon-per-grace still counts as stalled);
    - once drawup reaches ``min_return_m`` the sinker rule disarms for good
      (the vehicle proved it can climb) -- automatic, since max drawup never
      decreases.

    A verdict latches: every later ``feed()`` returns it unchanged.
    """

    def __init__(self, config: PlausibilityConfig) -> None:
        self._config = config
        self._verdict: str | None = None
        self._t0: float | None = None  # None until armed by the first sample
        self._z0 = 0.0
        self._z_min = 0.0
        self._drawup_max = 0.0
        # Deepening hysteresis anchor: the depth at the last deepening
        # event, not the running min -- a slow creep only resets the stall
        # clock once it accumulates a full epsilon below the anchor.
        self._z_at_last_deepen = 0.0
        self._t_last_deepen = 0.0

    @property
    def dive_m(self) -> float:
        """Max dive excursion below the first sample (classify_run's
        ``z[0] - z.min()``). 0.0 before arming."""
        return self._z0 - self._z_min if self._t0 is not None else 0.0

    @property
    def drawup_m(self) -> float:
        """Max climb-back from the running-deepest point (classify_run's
        ``(z - np.minimum.accumulate(z)).max()``). 0.0 before arming."""
        return self._drawup_max

    def feed(self, t_s: float, z_m: float) -> str | None:
        """Ingest one odometry sample; return the latched verdict, if any."""
        if self._verdict is not None:
            return self._verdict
        if self._t0 is None:
            self._t0 = t_s
            self._z0 = z_m
            self._z_min = z_m
            self._z_at_last_deepen = z_m
            self._t_last_deepen = t_s
            # z is negative-down: depth of the first sample is -z_m.
            if (
                self._config.max_start_depth_m > 0.0
                and -z_m > self._config.max_start_depth_m
            ):
                self._verdict = "bad_start"
                return self._verdict

        self._z_min = min(self._z_min, z_m)
        self._drawup_max = max(self._drawup_max, z_m - self._z_min)
        if z_m <= self._z_at_last_deepen - self._config.deepen_epsilon_m:
            self._z_at_last_deepen = z_m
            self._t_last_deepen = t_s

        if (
            t_s - self._t0 >= self._config.dive_deadline_s
            and self.dive_m < self._config.min_dive_m
        ):
            self._verdict = "floater"
        elif (
            self.dive_m >= self._config.min_dive_m
            # Monotone drawup makes the sinker disarm permanent: once >=
            # min_return_m this stays false forever.
            and self._drawup_max < self._config.min_return_m
            and t_s - self._t_last_deepen >= self._config.stall_grace_s
        ):
            self._verdict = "sinker"
        return self._verdict
