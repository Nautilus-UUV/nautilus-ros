"""Tier 1: pure dead-man logic for the lifeguard.

No ROS, no clock -- timestamps are handed in directly, so each arming /
silence / latch rule is asserted against synthetic times.
"""

from py_pkg.mqtt.lifeguard import Lifeguard, tank_blow_exhausted

TIMEOUT = 120.0


def _lg():
    return Lifeguard(TIMEOUT)


class TestOffByDefault:
    def test_disarmed_silence_never_engages(self):
        lg = _lg()
        # Bench scenario: hours of silence with the lifeguard never armed.
        assert lg.tick(now=100.0) is False
        assert lg.tick(now=100.0 + 100 * TIMEOUT) is False
        assert lg.engaged is False


class TestArming:
    def test_arm_seeds_the_clock(self):
        lg = _lg()
        # Armed at t=1000: the silence window starts here, so a tick just
        # under the timeout stays quiet and one at the timeout fires.
        lg.arm(now=1000.0)
        assert lg.tick(now=1000.0 + TIMEOUT - 0.001) is False
        assert lg.tick(now=1000.0 + TIMEOUT) is True

    def test_rearm_while_armed_is_a_noop(self):
        lg = _lg()
        lg.arm(now=0.0)
        # A retained-command replay must not push the window forward...
        lg.arm(now=TIMEOUT - 1.0)
        assert lg.tick(now=TIMEOUT) is True

    def test_rearm_while_engaged_keeps_the_latch(self):
        lg = _lg()
        lg.arm(now=0.0)
        lg.tick(now=TIMEOUT)
        # ...and must not clear an engaged latch on bridge reconnect.
        lg.arm(now=TIMEOUT + 1.0)
        assert lg.engaged is True


class TestHeartbeat:
    def test_beats_hold_the_failsafe_off(self):
        lg = _lg()
        lg.arm(now=0.0)
        lg.beat(now=100.0)
        assert lg.tick(now=100.0 + TIMEOUT - 0.001) is False

    def test_silence_after_last_beat_engages(self):
        lg = _lg()
        lg.arm(now=0.0)
        lg.beat(now=100.0)
        assert lg.tick(now=100.0 + TIMEOUT) is True

    def test_beat_while_disarmed_is_ignored(self):
        lg = _lg()
        lg.beat(now=50.0)
        # Arming later starts a fresh window from the arm, not that beat.
        lg.arm(now=200.0)
        assert lg.tick(now=200.0 + TIMEOUT - 0.001) is False
        assert lg.tick(now=200.0 + TIMEOUT) is True


class TestLatch:
    def test_returning_heartbeat_does_not_stand_down(self):
        lg = _lg()
        lg.arm(now=0.0)
        assert lg.tick(now=TIMEOUT) is True
        # Tether comes back, beats resume: still surfacing.
        lg.beat(now=TIMEOUT + 1.0)
        assert lg.tick(now=TIMEOUT + 2.0) is True

    def test_disarm_clears_the_latch(self):
        lg = _lg()
        lg.arm(now=0.0)
        lg.tick(now=TIMEOUT)
        lg.disarm()
        assert lg.armed is False
        assert lg.engaged is False
        assert lg.tick(now=10 * TIMEOUT) is False

    def test_rearm_after_disarm_starts_a_fresh_window(self):
        lg = _lg()
        lg.arm(now=0.0)
        lg.tick(now=TIMEOUT)
        lg.disarm()
        lg.arm(now=500.0)
        assert lg.tick(now=500.0 + TIMEOUT - 0.001) is False
        assert lg.tick(now=500.0 + TIMEOUT) is True


class TestTankBlowExhausted:
    """The blow drains the tank toward the registered empty endpoint;
    within 10% of the full--empty span there's nothing left to pump. Shares
    the span-based band with the depth controller's tank clamp."""

    EMPTY = 70_000.0
    FULL = 150_000.0
    # empty + 10% of the (full - empty) span = 70k + 8k = 78k.
    LOW_GUARD = EMPTY + 0.10 * (FULL - EMPTY)

    def test_unregistered_is_never_exhausted(self):
        # No pre-dive Initialize -> blow stays continuous and dumb.
        assert tank_blow_exhausted(0.0, None, self.FULL) is False
        assert tank_blow_exhausted(None, self.EMPTY, self.FULL) is False
        assert tank_blow_exhausted(None, None, None) is False

    def test_non_positive_empty_is_never_exhausted(self):
        # A half-filled payload decodes missing fields as 0.0 -- treat it
        # as not registered, not as "every reading is at the limit".
        assert tank_blow_exhausted(0.0, 0.0, self.FULL) is False
        assert tank_blow_exhausted(50_000.0, -1.0, self.FULL) is False

    def test_missing_or_inverted_full_is_never_exhausted(self):
        # No full endpoint -> no span to measure the 10% against; an inverted
        # or zero span (full <= empty) is equally unusable. Either way the
        # blow stays continuous rather than standing down on bad data.
        assert tank_blow_exhausted(self.EMPTY, self.EMPTY, None) is False
        assert tank_blow_exhausted(self.EMPTY, self.EMPTY, self.EMPTY) is False
        assert tank_blow_exhausted(self.EMPTY, self.EMPTY, self.EMPTY - 1.0) is False

    def test_exactly_at_the_band_boundary_is_exhausted(self):
        assert tank_blow_exhausted(self.LOW_GUARD, self.EMPTY, self.FULL) is True

    def test_just_outside_the_band_is_not_exhausted(self):
        assert tank_blow_exhausted(self.LOW_GUARD + 1.0, self.EMPTY, self.FULL) is False

    def test_below_the_empty_endpoint_is_exhausted(self):
        assert tank_blow_exhausted(self.EMPTY - 5_000.0, self.EMPTY, self.FULL) is True
