"""Tier 1 unit tests for TankLimitGuard.

Pure logic -- no rclpy. Pins the latching tank-endpoint cutoff that wraps
`clamp_to_tank_limits`: it engages a stop when a fill/drain command reaches the
(narrow) `stop_band` guard and *holds* it until the tank retreats past the
wider `release_band` guard, the command reverses, or it is reset.

The bug this guard exists to kill: on the bench the vehicle can't dive, so the
loop drives the tank onto the stop guard and parks there. A bare per-tick
threshold (the clamp alone) then flip-flops between stop and go as ordinary
sensor noise crosses the guard, chattering the valves. `test_holds_through_*`
is that regression -- the same noise must now leave the command rock-steady.

Conventions match the clamp tests: positive bus RPM drains the tank toward
`tank_empty_pa`; negative RPM, or the passive free-valve vent, fills it toward
`tank_full_pa`.
"""

from py_pkg.math_utils import span_band_guards
from py_pkg.pid.tank_limit_guard import TankLimitGuard
from py_pkg.scenarios.spec.rig import PlantSpec

_PLANT = PlantSpec()
EMPTY = _PLANT.tank_pressure_empty_pa
FULL = _PLANT.tank_pressure_full_pa
MID = (EMPTY + FULL) / 2.0

STOP_BAND = 0.10
RELEASE_BAND = 0.12
# Stop guards (the latch point) and the wider release guards (the let-go
# point). For the full side high_release < high_stop; for the empty side
# low_release > low_stop -- the release guard always sits deeper in the span.
LOW_STOP, HIGH_STOP = span_band_guards(EMPTY, FULL, STOP_BAND)
LOW_REL, HIGH_REL = span_band_guards(EMPTY, FULL, RELEASE_BAND)

# Representative raw commands as solve_bcu_command would emit them.
FILL = (-1500, 1, 0)  # negative RPM, motor valve open -> fills toward full
DRAIN = (1500, 1, 0)  # positive RPM -> drains toward empty
VENT = (0, 0, 1)  # passive free-valve vent -> also fills toward full
IDLE = (0, 0, 0)


def _guard(stop=STOP_BAND, release=RELEASE_BAND) -> TankLimitGuard:
    return TankLimitGuard(stop_band=stop, release_band=release)


class TestEngage:
    """A command pushing into an endpoint latches at the stop guard."""

    def test_fill_latches_at_high_stop_guard(self):
        assert _guard().apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE

    def test_drain_latches_at_low_stop_guard(self):
        assert _guard().apply(*DRAIN, LOW_STOP, EMPTY, FULL) == IDLE

    def test_vent_latches_at_full(self):
        assert _guard().apply(*VENT, FULL, EMPTY, FULL) == IDLE

    def test_fill_below_stop_guard_passes(self):
        # Approaching, not yet at the guard -> the command flows untouched.
        assert _guard().apply(*FILL, HIGH_STOP - 1, EMPTY, FULL) == FILL

    def test_vent_below_full_passes(self):
        assert _guard().apply(*VENT, MID, EMPTY, FULL) == VENT


class TestHoldsThroughNoise:
    """The regression: once latched, sensor jitter inside the hysteresis band
    must leave the command at zero every tick -- no valve chatter."""

    def test_holds_through_noise_at_full(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        # Jitter straddling the stop guard but staying above the release guard.
        noisy = [HIGH_STOP - 1, HIGH_STOP + 1, HIGH_REL + 1, HIGH_STOP, HIGH_REL + 50]
        out = [g.apply(*FILL, t, EMPTY, FULL) for t in noisy]
        assert all(o == IDLE for o in out), out

    def test_holds_through_noise_at_empty(self):
        g = _guard()
        assert g.apply(*DRAIN, LOW_STOP, EMPTY, FULL) == IDLE  # latch
        noisy = [LOW_STOP + 1, LOW_STOP - 1, LOW_REL - 1, LOW_STOP, LOW_REL - 50]
        out = [g.apply(*DRAIN, t, EMPTY, FULL) for t in noisy]
        assert all(o == IDLE for o in out), out


class TestRelease:
    """The latch lets go on a genuine retreat, a reversal, or a reset."""

    def test_releases_after_retreat_past_release_guard(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        assert g.apply(*FILL, HIGH_REL + 1, EMPTY, FULL) == IDLE  # still held
        assert g.apply(*FILL, HIGH_REL - 1, EMPTY, FULL) == FILL  # released

    def test_releases_on_direction_reversal(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch full
        # Reverse to drain at the full end -> allowed straight through.
        assert g.apply(*DRAIN, HIGH_STOP, EMPTY, FULL) == DRAIN

    def test_reset_clears_latch(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        g.reset()
        # No stale latch: a below-guard command flows again.
        assert g.apply(*FILL, HIGH_REL + 1, EMPTY, FULL) == FILL

    def test_idle_command_does_not_hold(self):
        # An idle tick at the limit just passes through as idle and drops the
        # latch -- nothing to hold when nothing is being commanded.
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        assert g.apply(*IDLE, HIGH_STOP, EMPTY, FULL) == IDLE


class TestNoHysteresis:
    """stop_band == release_band -> no gap, releases the tick it leaves the
    guard. Degrades to a per-tick latch, still well-defined."""

    def test_releases_immediately_below_guard(self):
        g = _guard(stop=STOP_BAND, release=STOP_BAND)
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE
        assert g.apply(*FILL, HIGH_STOP - 1, EMPTY, FULL) == FILL


class TestInert:
    """Unregistered or invalid limits -> pass through and drop any latch,
    exactly like the bare clamp."""

    def test_unregistered_passes_and_clears_latch(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        assert g.apply(*FILL, None, EMPTY, FULL) == FILL  # inert + cleared
        # Latch was dropped: a below-guard reading still flows.
        assert g.apply(*FILL, HIGH_REL + 1, EMPTY, FULL) == FILL

    def test_inverted_limits_pass_through(self):
        assert _guard().apply(*FILL, HIGH_STOP, FULL, EMPTY) == FILL

    def test_zero_full_passes_through(self):
        assert _guard().apply(*FILL, MID, EMPTY, 0.0) == FILL
