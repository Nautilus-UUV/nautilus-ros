"""Tier 1 unit tests for the bcu_node tank-limit output clamp.

`clamp_to_tank_limits(pump_rpm, motor_open, free_open, tank_pa,
tank_empty_pa, tank_full_pa, band=0.10)` is the last pure decision before
the BCU wire: zero the pump and shut both valves when the commanded oil
flow is headed at a registered tank endpoint and the tank is already
within `band` (10% of the full--empty span) of that endpoint -- the last
sliver of travel just dead-heads the pump.

Conventions (same as the rest of the stack): positive bus RPM inflates
the bladder, draining the tank toward `tank_empty_pa`; negative RPM --
or the passive free-valve vent (free_open) -- fills it toward
`tank_full_pa`. Limits come from the operator's pre-dive Initialize and
are sampled off the same /bcu/pressure stream `tank_pa` reports, so all
comparisons share one frame.
"""

from py_pkg.pid.tank_limit_guard import clamp_to_tank_limits
from py_pkg.scenarios.spec.rig import PlantSpec

# The sim plant's endpoints (rig.py): empty = drained tank (bladder
# full), full = tank full of oil (bladder empty).
_PLANT = PlantSpec()
EMPTY = _PLANT.tank_pressure_empty_pa
FULL = _PLANT.tank_pressure_full_pa
MID = (EMPTY + FULL) / 2.0

# The 10% band guards (empty=70k, full=150k, span=80k -> 78k / 142k).
# Draining clamps at/below LOW_GUARD; filling clamps at/above HIGH_GUARD.
BAND = 0.10
LOW_GUARD = EMPTY + BAND * (FULL - EMPTY)
HIGH_GUARD = FULL - BAND * (FULL - EMPTY)


class TestUnregistered:
    """No registration (or no tank reading) -> the clamp is inert."""

    def test_no_tank_reading_passes_through(self):
        assert clamp_to_tank_limits(-1500, 1, 0, None, EMPTY, FULL) == (-1500, 1, 0)

    def test_no_empty_limit_passes_through(self):
        assert clamp_to_tank_limits(1500, 1, 0, EMPTY, None, FULL) == (1500, 1, 0)

    def test_no_full_limit_passes_through(self):
        assert clamp_to_tank_limits(-1500, 1, 0, FULL, EMPTY, None) == (-1500, 1, 0)


class TestInvalidRegistration:
    """Nonsense limits (e.g. a half-filled payload decoding as zeros)
    must behave exactly like no registration at all."""

    def test_inverted_limits_pass_through(self):
        assert clamp_to_tank_limits(1500, 1, 0, EMPTY, FULL, EMPTY) == (1500, 1, 0)

    def test_equal_limits_pass_through(self):
        assert clamp_to_tank_limits(1500, 1, 0, MID, MID, MID) == (1500, 1, 0)

    def test_zero_empty_passes_through(self):
        # tank_pa <= 0.0-empty would otherwise never clamp ascend, but a
        # zero FULL (below) is the dangerous default -- both are rejected.
        assert clamp_to_tank_limits(1500, 1, 0, EMPTY, 0.0, FULL) == (1500, 1, 0)

    def test_zero_full_passes_through(self):
        # A defaulted full=0.0 would put every reading "at the full limit"
        # and clamp all descend commands forever.
        assert clamp_to_tank_limits(-1500, 1, 0, MID, EMPTY, 0.0) == (-1500, 1, 0)


class TestTowardEmpty:
    """Positive RPM drains the tank; clamp within 10% of empty -- i.e. at or
    below the low guard (empty + 10% of span), well before the bare endpoint."""

    def test_at_empty_limit_clamps(self):
        assert clamp_to_tank_limits(1500, 1, 0, EMPTY, EMPTY, FULL) == (0, 0, 0)

    def test_below_empty_limit_clamps(self):
        assert clamp_to_tank_limits(1500, 1, 0, EMPTY - 500, EMPTY, FULL) == (0, 0, 0)

    def test_at_low_guard_clamps(self):
        assert clamp_to_tank_limits(1500, 1, 0, LOW_GUARD, EMPTY, FULL) == (0, 0, 0)

    def test_just_inside_low_guard_clamps(self):
        # Still in the 10% band -> clamped, even though well above bare empty.
        assert clamp_to_tank_limits(1500, 1, 0, LOW_GUARD - 1, EMPTY, FULL) == (0, 0, 0)

    def test_just_above_low_guard_passes(self):
        assert clamp_to_tank_limits(1500, 1, 0, LOW_GUARD + 1, EMPTY, FULL) == (
            1500,
            1,
            0,
        )


class TestTowardFull:
    """Negative RPM fills the tank; clamp within 10% of full -- i.e. at or
    above the high guard (full - 10% of span), well before the bare endpoint."""

    def test_at_full_limit_clamps(self):
        assert clamp_to_tank_limits(-1500, 1, 0, FULL, EMPTY, FULL) == (0, 0, 0)

    def test_above_full_limit_clamps(self):
        assert clamp_to_tank_limits(-1500, 1, 0, FULL + 500, EMPTY, FULL) == (0, 0, 0)

    def test_at_high_guard_clamps(self):
        assert clamp_to_tank_limits(-1500, 1, 0, HIGH_GUARD, EMPTY, FULL) == (0, 0, 0)

    def test_just_inside_high_guard_clamps(self):
        # Still in the 10% band -> clamped, even though below bare full.
        assert clamp_to_tank_limits(-1500, 1, 0, HIGH_GUARD + 1, EMPTY, FULL) == (
            0,
            0,
            0,
        )

    def test_just_below_high_guard_passes(self):
        assert clamp_to_tank_limits(-1500, 1, 0, HIGH_GUARD - 1, EMPTY, FULL) == (
            -1500,
            1,
            0,
        )


class TestPassiveVent:
    """The deep-descend free-valve vent fills the tank without the pump,
    so it counts as flow toward full and gets clamped the same way."""

    def test_vent_at_full_limit_closes_free_valve(self):
        assert clamp_to_tank_limits(0, 0, 1, FULL, EMPTY, FULL) == (0, 0, 0)

    def test_vent_below_full_limit_stays_open(self):
        assert clamp_to_tank_limits(0, 0, 1, MID, EMPTY, FULL) == (0, 0, 1)

    def test_vent_is_not_clamped_at_empty(self):
        # The vent only ever fills the tank -- the empty endpoint can't
        # gate it.
        assert clamp_to_tank_limits(0, 0, 1, EMPTY, EMPTY, FULL) == (0, 0, 1)


class TestAwayFromLimit:
    """Flow AWAY from a touched limit passes immediately -- the clamp is
    per-tick and direction-gated, not a latch."""

    def test_descend_at_empty_limit_passes(self):
        assert clamp_to_tank_limits(-1500, 1, 0, EMPTY, EMPTY, FULL) == (-1500, 1, 0)

    def test_ascend_at_full_limit_passes(self):
        assert clamp_to_tank_limits(1500, 1, 0, FULL, EMPTY, FULL) == (1500, 1, 0)

    def test_idle_at_either_limit_unchanged(self):
        assert clamp_to_tank_limits(0, 0, 0, EMPTY, EMPTY, FULL) == (0, 0, 0)
        assert clamp_to_tank_limits(0, 0, 0, FULL, EMPTY, FULL) == (0, 0, 0)
