"""Tier 1: pure run-viability verdicts for the sim run watchdog.

No ROS, no clock -- ``(t_s, z_m)`` samples are handed in directly. z follows
the odometry convention mirrored from ``scripts/analysis/sweep_loader.py``'s
``classify_run``: negative-down, so diving *decreases* z. Streams start from
a nonzero spawn depth (the sim spawns at z=-5) to lock the excursions to
relative, not absolute, z.
"""

from py_pkg.watchdog.plausibility import (
    MIN_DIVE_M,
    MIN_RETURN_M,
    PlausibilityConfig,
    RunPlausibility,
)

Z_SPAWN = -5.0
DEADLINE = 120.0
GRACE = 300.0


def _rp(**overrides) -> RunPlausibility:
    return RunPlausibility(
        PlausibilityConfig(dive_deadline_s=DEADLINE, stall_grace_s=GRACE, **overrides)
    )


def _dive_to(rp: RunPlausibility, z_from: float, z_to: float, t0: float, dt=1.0):
    """Feed a descent in 0.5 m steps (each > epsilon, so each resets the
    stall clock). Returns the time of the last sample."""
    t = t0
    z = z_from
    while z > z_to:
        z = max(z_to, z - 0.5)
        t += dt
        assert rp.feed(t, z) is None
    return t


class TestConstantsParity:
    def test_config_defaults_are_the_exported_thresholds(self):
        # scripts/analysis/sweep_loader.py imports MIN_DIVE_M / MIN_RETURN_M
        # from this module, so watchdog-vs-analysis parity holds by import.
        # What could still drift is a hand-typed dataclass default.
        cfg = PlausibilityConfig()
        assert cfg.min_dive_m == MIN_DIVE_M
        assert cfg.min_return_m == MIN_RETURN_M


class TestFloater:
    def test_fires_at_deadline_when_dive_short(self):
        rp = _rp()
        # Bobs at the surface: excursions well under min_dive_m.
        assert rp.feed(0.0, Z_SPAWN) is None
        assert rp.feed(60.0, Z_SPAWN - 1.0) is None
        assert rp.feed(DEADLINE - 0.001, Z_SPAWN - 0.5) is None
        # Boundary is inclusive: "at/after the deadline".
        assert rp.feed(DEADLINE, Z_SPAWN - 0.5) == "floater"

    def test_no_floater_when_dived_in_time(self):
        rp = _rp()
        assert rp.feed(0.0, Z_SPAWN) is None
        assert rp.feed(30.0, Z_SPAWN - 2.5) is None  # dived >= min_dive_m
        assert rp.feed(DEADLINE + 10.0, Z_SPAWN - 2.5) is None

    def test_dive_is_relative_to_first_sample(self):
        # An absolute z far below zero is still a floater when the
        # *excursion* from the first sample stays small (classify_run
        # computes z[0] - z.min(), never abs(z)).
        rp = _rp()
        assert rp.feed(0.0, -50.0) is None
        assert rp.feed(DEADLINE, -50.5) == "floater"


class TestSinker:
    def test_fires_after_stall_grace(self):
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        t = _dive_to(rp, Z_SPAWN, Z_SPAWN - 2.5, t0=0.0)
        # Frozen at depth: no drawup, no deepening. Just inside the grace
        # window nothing fires; at the boundary the sinker latches.
        assert rp.feed(t + GRACE - 0.001, Z_SPAWN - 2.5) is None
        assert rp.feed(t + GRACE, Z_SPAWN - 2.5) == "sinker"

    def test_requires_min_dive(self):
        # Stalled shallow (dive < min_dive_m) is a floater at the deadline,
        # never a sinker.
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        assert rp.feed(10.0, Z_SPAWN - 1.5) is None
        assert rp.feed(DEADLINE - 1.0, Z_SPAWN - 1.5) is None
        assert rp.feed(DEADLINE, Z_SPAWN - 1.5) == "floater"

    def test_creep_below_epsilon_still_counts_as_stalled(self):
        # A descent slower than epsilon-per-grace never resets the stall
        # clock: 0.001 m per 10 s accumulates only 0.03 m (< 0.05 epsilon)
        # over the whole 300 s window.
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        t = _dive_to(rp, Z_SPAWN, Z_SPAWN - 2.5, t0=0.0)
        z = Z_SPAWN - 2.5
        verdict = None
        while verdict is None and t < 1000.0:
            t += 10.0
            z -= 0.001
            verdict = rp.feed(t, z)
        assert verdict == "sinker"

    def test_real_descent_resets_stall_clock(self):
        # Deepening by >= epsilon keeps resetting the clock: no sinker while
        # the vehicle is genuinely still going down.
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        t = _dive_to(rp, Z_SPAWN, Z_SPAWN - 2.5, t0=0.0)
        z = Z_SPAWN - 2.5
        for _ in range(10):
            t += GRACE - 1.0
            z -= 0.1  # > epsilon: a fresh deepening event each time
            assert rp.feed(t, z) is None

    def test_drawup_disarms_permanently(self):
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        t = _dive_to(rp, Z_SPAWN, Z_SPAWN - 2.5, t0=0.0)
        # Climbs back >= min_return_m: proved it can climb.
        assert rp.feed(t + 10.0, Z_SPAWN - 1.4) is None
        # Sinks again and stalls for far longer than the grace -- the sinker
        # rule stays disarmed forever.
        t = _dive_to(rp, Z_SPAWN - 1.4, Z_SPAWN - 6.0, t0=t + 10.0)
        for k in range(1, 11):
            assert rp.feed(t + k * GRACE, Z_SPAWN - 6.0) is None


class TestVerdictLatches:
    def test_floater_latches_through_a_later_dive(self):
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        assert rp.feed(DEADLINE, Z_SPAWN - 0.1) == "floater"
        # A deep dive after the verdict cannot un-latch it.
        assert rp.feed(DEADLINE + 60.0, Z_SPAWN - 10.0) == "floater"

    def test_sinker_latches_through_a_later_climb(self):
        rp = _rp()
        rp.feed(0.0, Z_SPAWN)
        t = _dive_to(rp, Z_SPAWN, Z_SPAWN - 2.5, t0=0.0)
        assert rp.feed(t + GRACE, Z_SPAWN - 2.5) == "sinker"
        assert rp.feed(t + GRACE + 60.0, Z_SPAWN) == "sinker"
