"""Tier 1 unit tests for BcuCommandGate.

Pure logic -- no rclpy, clock injected. Pins the anti-chatter contract:
error deadband + arm/disarm hysteresis, minimum valve dwell, and the
pump/valve-consistency safety invariants (no dead-heading, immediate pump
stop on disarm). The near-setpoint dither case is the bug this gate exists
to kill.
"""

from py_pkg.pid.bcu_command_gate import BcuCommandGate

# A representative "raw" motor command (pump running, motor valve open) and
# the free/vent command, as solve_bcu_command would emit them.
_MOTOR_CMD = (-1500, 1, 0)
_FREE_CMD = (0, 0, 1)
_IDLE_CMD = (0, 0, 0)


def _gate(arm=4000.0, disarm=2000.0, dwell=0.5) -> BcuCommandGate:
    return BcuCommandGate(
        error_arm_pa=arm, error_disarm_pa=disarm, min_valve_dwell_s=dwell
    )


class TestPassthroughWhenDisabled:
    """arm == disarm == 0 and dwell == 0 -> identical to the raw command."""

    def test_motor_command_passes_through(self):
        g = _gate(arm=0.0, disarm=0.0, dwell=0.0)
        # Large error so the gate is armed; raw motor command must pass intact.
        out = [g.apply(50_000, *_MOTOR_CMD, now_s=0.1 * i) for i in range(5)]
        assert all(o == _MOTOR_CMD for o in out), out

    def test_free_vent_passes_through(self):
        g = _gate(arm=0.0, disarm=0.0, dwell=0.0)
        out = g.apply(50_000, *_FREE_CMD, now_s=0.1)
        assert out == _FREE_CMD


class TestArmDisarmHysteresis:
    """The pump is held idle near the setpoint and re-arms only past the
    wider arm band; between the two bands it keeps its state. Dwell is
    disabled here so the valve responds at once -- the dwell delay on
    closing is pinned separately in TestMinimumValveDwell."""

    def test_starts_disarmed_small_error_stays_idle(self):
        g = _gate(dwell=0.0)
        # Error inside the disarm band from the first tick -> never arms.
        out = [g.apply(1000, *_MOTOR_CMD, now_s=0.1 * i) for i in range(10)]
        assert all(o == _IDLE_CMD for o in out), out

    def test_error_must_exceed_arm_to_start(self):
        g = _gate(dwell=0.0)
        # Inside [disarm, arm): still disarmed because it started disarmed.
        assert g.apply(3000, *_MOTOR_CMD, now_s=0.0) == _IDLE_CMD
        # Past arm: now it actuates.
        assert g.apply(5000, *_MOTOR_CMD, now_s=0.1) == _MOTOR_CMD

    def test_stays_armed_between_bands(self):
        g = _gate(dwell=0.0)
        g.apply(5000, *_MOTOR_CMD, now_s=0.0)  # arm
        # Drop into the hysteresis band: must stay armed (still actuating).
        assert g.apply(3000, *_MOTOR_CMD, now_s=0.1) == _MOTOR_CMD

    def test_disarms_only_below_disarm(self):
        g = _gate(dwell=0.0)
        g.apply(5000, *_MOTOR_CMD, now_s=0.0)  # arm
        # Below disarm -> idle (pump off, valve closes immediately at dwell=0).
        assert g.apply(1500, *_MOTOR_CMD, now_s=0.1) == _IDLE_CMD


class TestNoChatterNearSetpoint:
    """The exact bench failure: tiny error sign-dither must not toggle the
    pump or valves. Starting disarmed inside the band, nothing actuates."""

    def test_sign_dither_inside_band_holds_idle(self):
        g = _gate()
        out = []
        for i in range(40):
            err = 800 if i % 2 == 0 else -800  # dither well inside disarm band
            raw = _MOTOR_CMD if i % 2 == 0 else (1500, 1, 0)
            out.append(g.apply(err, *raw, now_s=0.1 * i))
        valve_states = {(m, f) for _, m, f in out}
        pumps = {p for p, _, _ in out}
        assert valve_states == {(0, 0)}, valve_states
        assert pumps == {0}, pumps


class TestMinimumValveDwell:
    """The valve bitmask changes at most once per dwell, whatever the
    controller asks for."""

    def test_first_change_is_immediate(self):
        g = _gate(dwell=0.5)
        # Large error -> armed; first open is allowed with no prior change.
        out = g.apply(50_000, *_MOTOR_CMD, now_s=0.0)
        assert out == _MOTOR_CMD

    def test_switch_blocked_until_dwell_elapses(self):
        g = _gate(dwell=0.5)
        g.apply(50_000, *_MOTOR_CMD, now_s=0.0)  # held MOTOR at t=0
        # Want FREE at t=0.1: dwell blocks, valves stay MOTOR.
        _, m, f = g.apply(50_000, *_FREE_CMD, now_s=0.1)
        assert (m, f) == (1, 0)
        # At t=0.6 (>0.5 since last change) the switch is allowed.
        _, m2, f2 = g.apply(50_000, *_FREE_CMD, now_s=0.6)
        assert (m2, f2) == (0, 1)

    def test_at_most_one_change_per_dwell_window(self):
        g = _gate(dwell=0.5)
        changes = 0
        prev = (0, 0)
        # Alternate the desired valve state every tick for 2 s at 10 Hz.
        for i in range(20):
            raw = _MOTOR_CMD if i % 2 == 0 else _FREE_CMD
            _, m, f = g.apply(50_000, *raw, now_s=0.1 * i)
            if (m, f) != prev:
                changes += 1
                prev = (m, f)
        # 2 s / 0.5 s dwell -> at most ~4 changes, not one per 0.1 s tick (20).
        assert changes <= 5, changes


class TestSafetyInvariants:
    def test_no_dead_head_pump_waits_for_valve(self):
        g = _gate(dwell=0.5)
        # Open the free vent first so a later MOTOR switch is dwell-blocked.
        g.apply(50_000, *_FREE_CMD, now_s=0.0)
        # Want to pump now, but the motor valve can't open yet (dwell): the
        # pump must be 0, never run against the still-closed motor valve.
        pump, m, f = g.apply(50_000, *_MOTOR_CMD, now_s=0.1)
        assert (m, f) == (0, 1)  # still free, motor valve closed
        assert pump == 0, "pump must not dead-head against a closed motor valve"

    def test_disarm_stops_pump_immediately_even_if_valve_held(self):
        g = _gate(dwell=0.5)
        out = g.apply(50_000, *_MOTOR_CMD, now_s=0.0)
        assert out == _MOTOR_CMD  # pumping, motor valve open
        # Error collapses into the disarm band: pump must go to 0 on this very
        # tick even though the dwell still holds the motor valve open.
        pump, m, f = g.apply(500, *_MOTOR_CMD, now_s=0.1)
        assert pump == 0
        assert (m, f) == (1, 0)  # valve still open (harmless: no flow)

    def test_reset_returns_to_disarmed_closed(self):
        g = _gate()
        g.apply(50_000, *_MOTOR_CMD, now_s=0.0)  # arm + open
        g.reset()
        # After reset, a small error keeps it idle (disarmed) and closed.
        assert g.apply(1000, *_MOTOR_CMD, now_s=0.1) == _IDLE_CMD
