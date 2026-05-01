"""Tier 1 unit tests for the depth_node valve-selection rule.

`select_pump_and_valves(current_depth, q, pump_rpm, deep_threshold)` is the
pure decision: pass the pump command through with valve 1 open, or — below
the deep threshold with descent intent — force the pump off and open
valve 2 to vent passively. Sign convention: Z-positive-down, so
`current_depth > threshold` means "below the threshold depth", and `q > 0`
is a sink intent.
"""

from py_pkg.pid.depth_node import select_pump_and_valves

THRESHOLD = 30.0


class TestShallow:
    """Above the deep threshold the rule is purely pump-driven."""

    def test_descend_active_pump_uses_valve1(self):
        # current=0 (surface) is shallower than the threshold even though
        # the descent intent is set, so valve 1 carries the pumped flow.
        rpm, v1, v2 = select_pump_and_valves(0.0, 0.5, -1500, THRESHOLD)
        assert (rpm, v1, v2) == (-1500, 1, 0)

    def test_ascend_active_pump_uses_valve1(self):
        rpm, v1, v2 = select_pump_and_valves(0.0, -0.5, 1500, THRESHOLD)
        assert (rpm, v1, v2) == (1500, 1, 0)

    def test_idle_pump_closes_both(self):
        rpm, v1, v2 = select_pump_and_valves(0.0, 0.0, 0, THRESHOLD)
        assert (rpm, v1, v2) == (0, 0, 0)

    def test_descend_intent_with_deadbanded_pump_keeps_both_closed(self):
        # Shallow descent intent but pump zeroed by the deadband: passive
        # vent is reserved for the deep regime, so both valves stay shut.
        rpm, v1, v2 = select_pump_and_valves(0.0, 0.001, 0, THRESHOLD)
        assert (rpm, v1, v2) == (0, 0, 0)


class TestDeep:
    """Below the deep threshold a descent intent vents passively."""

    def test_descend_overrides_pump_to_zero_and_opens_valve2(self):
        # Even with a non-zero pump command, the deep+descend rule forces
        # pump_rpm to 0 and opens valve 2.
        rpm, v1, v2 = select_pump_and_valves(50.0, 0.5, -3000, THRESHOLD)
        assert (rpm, v1, v2) == (0, 0, 1)

    def test_descend_with_idle_pump_still_opens_valve2(self):
        # q > 0 alone is enough to open valve 2; pump cmd is irrelevant
        # below the threshold.
        rpm, v1, v2 = select_pump_and_valves(50.0, 0.001, 0, THRESHOLD)
        assert (rpm, v1, v2) == (0, 0, 1)

    def test_ascend_active_pump_uses_valve1(self):
        # No descent intent → valve 2 stays closed; valve 1 carries flow.
        rpm, v1, v2 = select_pump_and_valves(50.0, -0.5, 1500, THRESHOLD)
        assert (rpm, v1, v2) == (1500, 1, 0)

    def test_quiescent_closes_both(self):
        rpm, v1, v2 = select_pump_and_valves(50.0, 0.0, 0, THRESHOLD)
        assert (rpm, v1, v2) == (0, 0, 0)


class TestThresholdBoundary:
    """`deep` uses strict `>`: at exactly the threshold we are not deep."""

    def test_at_threshold_descend_does_not_vent(self):
        # current_depth == threshold → not deep → pump-driven path.
        rpm, v1, v2 = select_pump_and_valves(THRESHOLD, 0.5, -1500, THRESHOLD)
        assert (rpm, v1, v2) == (-1500, 1, 0)

    def test_just_above_threshold_descend_vents(self):
        rpm, v1, v2 = select_pump_and_valves(
            THRESHOLD + 1e-6, 0.5, -1500, THRESHOLD
        )
        assert (rpm, v1, v2) == (0, 0, 1)


class TestZeroIntentSign:
    """`q == 0` is not a descent intent — strict `>`."""

    def test_deep_with_q_zero_does_not_vent(self):
        rpm, v1, v2 = select_pump_and_valves(50.0, 0.0, 1500, THRESHOLD)
        assert (rpm, v1, v2) == (1500, 1, 0)
