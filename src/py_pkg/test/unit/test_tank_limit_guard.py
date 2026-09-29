"""Tier 1 unit tests for TankLimitGuard.

Pure logic -- no rclpy. Pins the latching tank-endpoint cutoff: it engages a
stop when a fill/drain command reaches the `stop_band` guard and *holds* it
until the command reverses or it is reset.

This is the stop condition every bang-bang leg runs into -- pump one way until
the tank rails at 5%, then coast until the mission's next leg reverses the
command -- and it is also the cure for a bench bug: on the bench the vehicle
can't dive, so the loop drives the tank onto the guard and parks there. A bare
per-tick threshold then flip-flops between stop and go as ordinary sensor noise
crosses the guard, chattering the valves. `test_holds_through_*` is that
regression -- the same noise must now leave the command rock-steady.

There is deliberately no retreat-based release: while latched the pump is
stopped and both valves are shut, so the tank cannot move on its own. Only a
reversal (the mission's next leg) or a reset lets go.

Both layers live here: the latch behaviour, and (via `_bare`, a fresh guard per
tick) the bare threshold contract underneath it that the since-removed
`clamp_to_tank_limits` used to own.

Conventions (same as the rest of the stack): positive bus RPM drains the tank
toward `tank_empty_pa`; negative RPM, or the passive free-valve vent, fills it
toward `tank_full_pa`. Limits come from the operator's pre-dive Initialize and
are sampled off the same /bcu/pressure stream `tank_pa` reports, so all
comparisons share one frame.
"""

from py_pkg.control.tank_limit_guard import TankLimitGuard
from py_pkg.math_utils import span_band_guards
from py_pkg.scenarios.spec.control import DepthSpec
from py_pkg.scenarios.spec.rig import PlantSpec

_PLANT = PlantSpec()
EMPTY = _PLANT.tank_pressure_empty_pa
FULL = _PLANT.tank_pressure_full_pa
MID = (EMPTY + FULL) / 2.0

# The shipped band -- 5% of the empty->full span, per DepthSpec.
STOP_BAND = DepthSpec().tank_stop_band
LOW_STOP, HIGH_STOP = span_band_guards(EMPTY, FULL, STOP_BAND)

# Representative raw commands as solve_bcu_command would emit them.
FILL = (-3000, 1, 0)  # negative RPM, motor valve open -> fills toward full
DRAIN = (3000, 1, 0)  # positive RPM -> drains toward empty
VENT = (0, 0, 1)  # passive free-valve vent -> also fills toward full
IDLE = (0, 0, 0)


def _guard(stop=STOP_BAND) -> TankLimitGuard:
    return TankLimitGuard(stop_band=stop)


def _bare(pump_rpm, motor_open, free_open, tank_pa, empty=EMPTY, full=FULL):
    """One tick through a fresh guard: no latch memory at all.

    This is the bare per-tick threshold the latch is layered on top of, which
    the classes below pin independently of the latch.
    """
    return _guard().apply(pump_rpm, motor_open, free_open, tank_pa, empty, full)


class TestBandIsFivePercent:
    """The band the missions are specified against."""

    def test_stop_band_default_is_five_percent(self):
        assert STOP_BAND == 0.05

    def test_guards_sit_five_percent_inside_each_endpoint(self):
        span = FULL - EMPTY
        assert LOW_STOP == EMPTY + 0.05 * span
        assert HIGH_STOP == FULL - 0.05 * span


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
    """The regression: once latched, sensor jitter around the guard must leave
    the command at zero every tick -- no valve chatter.

    The tank channel is the noisy one (sigma 353 Pa, quantised at 600 Pa in the
    lake-fitted noise model), so the jitter below is realistic, not synthetic.
    """

    def test_holds_through_noise_at_full(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        noisy = [HIGH_STOP - 600, HIGH_STOP + 600, HIGH_STOP - 1, HIGH_STOP, MID]
        out = [g.apply(*FILL, t, EMPTY, FULL) for t in noisy]
        assert all(o == IDLE for o in out), out

    def test_holds_through_noise_at_empty(self):
        g = _guard()
        assert g.apply(*DRAIN, LOW_STOP, EMPTY, FULL) == IDLE  # latch
        noisy = [LOW_STOP + 600, LOW_STOP - 600, LOW_STOP + 1, LOW_STOP, MID]
        out = [g.apply(*DRAIN, t, EMPTY, FULL) for t in noisy]
        assert all(o == IDLE for o in out), out


class TestRelease:
    """The latch lets go on a reversal or a reset -- and on nothing else."""

    def test_releases_on_direction_reversal(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch full
        # Reverse to drain at the full end -> allowed straight through. This
        # is the mission-flips-leg case, the only way a run leaves a rail.
        assert g.apply(*DRAIN, HIGH_STOP, EMPTY, FULL) == DRAIN

    def test_reversal_then_reversing_back_can_relatch(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE
        assert g.apply(*DRAIN, HIGH_STOP, EMPTY, FULL) == DRAIN  # release
        # Still at the guard and pushing in again -> latches again.
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE

    def test_reset_clears_latch(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        g.reset()
        # No stale latch: a below-guard command flows again.
        assert g.apply(*FILL, HIGH_STOP - 1, EMPTY, FULL) == FILL

    def test_idle_command_does_not_hold(self):
        # An idle tick at the limit just passes through as idle and drops the
        # latch -- nothing to hold when nothing is being commanded.
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        assert g.apply(*IDLE, HIGH_STOP, EMPTY, FULL) == IDLE

    def test_retreat_alone_does_not_release(self):
        # The deliberate asymmetry vs. the old release_band: with the pump
        # stopped and both valves shut the tank CANNOT retreat on its own,
        # so a reading that says it did is noise (or a stale sample) and
        # must not restart the pump into the rail.
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        assert g.apply(*FILL, MID, EMPTY, FULL) == IDLE
        assert g.apply(*FILL, EMPTY, EMPTY, FULL) == IDLE


class TestInert:
    """Unregistered or invalid limits -> pass through and drop any latch,
    exactly like the bare clamp."""

    def test_unregistered_passes_and_clears_latch(self):
        g = _guard()
        assert g.apply(*FILL, HIGH_STOP, EMPTY, FULL) == IDLE  # latch
        assert g.apply(*FILL, None, EMPTY, FULL) == FILL  # inert + cleared
        # Latch was dropped: a below-guard reading still flows.
        assert g.apply(*FILL, HIGH_STOP - 1, EMPTY, FULL) == FILL

    def test_inverted_limits_pass_through(self):
        assert _guard().apply(*FILL, HIGH_STOP, FULL, EMPTY) == FILL

    def test_zero_full_passes_through(self):
        assert _guard().apply(*FILL, MID, EMPTY, 0.0) == FILL

    def test_missing_empty_limit_passes_through(self):
        assert _bare(*DRAIN, EMPTY, None, FULL) == DRAIN

    def test_missing_full_limit_passes_through(self):
        assert _bare(*FILL, FULL, EMPTY, None) == FILL

    def test_equal_limits_pass_through(self):
        assert _bare(*DRAIN, MID, MID, MID) == DRAIN

    def test_zero_empty_passes_through(self):
        # A defaulted empty=0.0 would put every reading above the low guard and
        # never stop a drain; a defaulted full=0.0 (above) is the mirror hazard.
        assert _bare(*DRAIN, EMPTY, 0.0, FULL) == DRAIN


class TestBareThresholds:
    """The per-tick threshold contract underneath the latch: stop the pump and
    shut both valves when the commanded oil flow is headed at a registered
    endpoint and the tank is already within `stop_band` of it -- the last sliver
    of travel just dead-heads the pump. Driven through `_bare`, so no latch
    memory participates.
    """

    def test_drain_at_bare_empty_stops(self):
        assert _bare(*DRAIN, EMPTY) == IDLE

    def test_drain_past_bare_empty_stops(self):
        assert _bare(*DRAIN, EMPTY - 500) == IDLE

    def test_drain_just_inside_low_guard_stops(self):
        # Still in the band -> stopped, even though well above bare empty.
        assert _bare(*DRAIN, LOW_STOP - 1) == IDLE

    def test_drain_just_outside_low_guard_passes(self):
        assert _bare(*DRAIN, LOW_STOP + 1) == DRAIN

    def test_fill_at_bare_full_stops(self):
        assert _bare(*FILL, FULL) == IDLE

    def test_fill_past_bare_full_stops(self):
        assert _bare(*FILL, FULL + 500) == IDLE

    def test_fill_just_inside_high_guard_stops(self):
        assert _bare(*FILL, HIGH_STOP + 1) == IDLE


class TestDirectionGating:
    """The cutoff is direction-gated: only flow *toward* a touched endpoint is
    stopped, so the vehicle can always pump its way back off a limit."""

    def test_vent_is_not_gated_at_empty(self):
        # The vent only ever fills the tank -- the empty endpoint can't gate it.
        assert _bare(*VENT, EMPTY) == VENT

    def test_fill_at_empty_limit_passes(self):
        assert _bare(*FILL, EMPTY) == FILL

    def test_drain_at_full_limit_passes(self):
        assert _bare(*DRAIN, FULL) == DRAIN

    def test_idle_at_either_limit_unchanged(self):
        assert _bare(*IDLE, EMPTY) == IDLE
        assert _bare(*IDLE, FULL) == IDLE
