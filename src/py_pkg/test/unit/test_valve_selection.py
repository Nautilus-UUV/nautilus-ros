"""Tier 1 unit tests for the depth_node valve-selection rule.

`select_pump_and_valves(current_pressure_pa, q, pump_rpm, deep_threshold_pa)`
is the pure decision: pass the pump command through with the motor way (the
pump flow path) open, or — below the deep threshold with descent intent —
force the pump off and open the free/bypass way to vent passively. Sign
convention: Z-positive-down, gauge pressure, so `current_pressure_pa >
threshold` means "below the threshold depth", and `q > 0` is a sink intent.

The return tuple is `(pump_rpm, motor_open, free_open)`: motor_open is bit0 of
the wire bitmask (operator-facing "valve 2"), free_open is bit1 ("valve 1").
The bit values are unchanged here -- only the operator-facing valve numbers
were corrected elsewhere in the stack.
"""

from py_pkg.physics import depth_to_pressure_pa, gauge_pressure_pa
from py_pkg.pid.depth_node import select_pump_and_valves


def _gauge_pa_for_depth(depth_m: float) -> float:
    return gauge_pressure_pa(depth_to_pressure_pa(depth_m))


THRESHOLD = _gauge_pa_for_depth(30.0)
SHALLOW = _gauge_pa_for_depth(0.0)
DEEP = _gauge_pa_for_depth(50.0)


class TestShallow:
    """Above the deep threshold the rule is purely pump-driven."""

    def test_descend_active_pump_uses_motor_valve(self):
        # current=0 (surface) is shallower than the threshold even though
        # the descent intent is set, so the motor way carries the pumped flow.
        rpm, motor, free = select_pump_and_valves(SHALLOW, 0.5, -1500, THRESHOLD)
        assert (rpm, motor, free) == (-1500, 1, 0)

    def test_ascend_active_pump_uses_motor_valve(self):
        rpm, motor, free = select_pump_and_valves(SHALLOW, -0.5, 1500, THRESHOLD)
        assert (rpm, motor, free) == (1500, 1, 0)

    def test_idle_pump_closes_both(self):
        rpm, motor, free = select_pump_and_valves(SHALLOW, 0.0, 0, THRESHOLD)
        assert (rpm, motor, free) == (0, 0, 0)

    def test_descend_intent_with_deadbanded_pump_keeps_both_closed(self):
        # Shallow descent intent but pump zeroed by the deadband: passive
        # vent is reserved for the deep regime, so both valves stay shut.
        rpm, motor, free = select_pump_and_valves(SHALLOW, 0.001, 0, THRESHOLD)
        assert (rpm, motor, free) == (0, 0, 0)


class TestDeep:
    """Below the deep threshold a descent intent vents passively."""

    def test_descend_overrides_pump_to_zero_and_opens_free_valve(self):
        # Even with a non-zero pump command, the deep+descend rule forces
        # pump_rpm to 0 and opens the free/bypass way (bit1).
        rpm, motor, free = select_pump_and_valves(DEEP, 0.5, -3000, THRESHOLD)
        assert (rpm, motor, free) == (0, 0, 1)

    def test_descend_with_idle_pump_still_opens_free_valve(self):
        # q > 0 alone is enough to open the free way; pump cmd is irrelevant
        # below the threshold.
        rpm, motor, free = select_pump_and_valves(DEEP, 0.001, 0, THRESHOLD)
        assert (rpm, motor, free) == (0, 0, 1)

    def test_ascend_active_pump_uses_motor_valve(self):
        # No descent intent → free way stays closed; the motor way carries flow.
        rpm, motor, free = select_pump_and_valves(DEEP, -0.5, 1500, THRESHOLD)
        assert (rpm, motor, free) == (1500, 1, 0)

    def test_quiescent_closes_both(self):
        rpm, motor, free = select_pump_and_valves(DEEP, 0.0, 0, THRESHOLD)
        assert (rpm, motor, free) == (0, 0, 0)


class TestThresholdBoundary:
    """`deep` uses strict `>`: at exactly the threshold we are not deep."""

    def test_at_threshold_descend_does_not_vent(self):
        # current_pressure_pa == threshold → not deep → pump-driven path.
        rpm, motor, free = select_pump_and_valves(THRESHOLD, 0.5, -1500, THRESHOLD)
        assert (rpm, motor, free) == (-1500, 1, 0)

    def test_just_above_threshold_descend_vents(self):
        rpm, motor, free = select_pump_and_valves(
            THRESHOLD + 1e-6, 0.5, -1500, THRESHOLD
        )
        assert (rpm, motor, free) == (0, 0, 1)


class TestZeroIntentSign:
    """`q == 0` is not a descent intent — strict `>`."""

    def test_deep_with_q_zero_does_not_vent(self):
        rpm, motor, free = select_pump_and_valves(DEEP, 0.0, 1500, THRESHOLD)
        assert (rpm, motor, free) == (1500, 1, 0)
