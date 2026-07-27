"""Tier 1 unit tests for BcuSafeStopBurst.

Pure logic -- no rclpy, clock injected. Pins the safe-stop contract: a stop
re-asserts the zero for a bounded burst (so the STM latches it), but a manual
command yields the wire so the burst can't flicker against the manual driver.
The two message orderings bcu_node has to survive -- stop-then-manual and the
reverse manual-then-stop -- are the cases this exists to make deterministic.
"""

from py_pkg.control.bcu_safe_stop_burst import BcuSafeStopBurst

_COUNT = 10
_HOLD = 1.5


def _burst(count=_COUNT, hold=_HOLD) -> BcuSafeStopBurst:
    return BcuSafeStopBurst(reassert_count=count, manual_hold_s=hold)


class TestGenuineStopBurst:
    """A stop with no manual command in flight re-asserts the zero, then quits."""

    def test_begin_stop_arms_and_then_drains(self):
        b = _burst()
        assert b.begin_stop(0.0) is True  # caller emits the first zero
        # control_loop then publishes one zero per tick for the burst window.
        assert [b.tick() for _ in range(_COUNT)] == [True] * _COUNT
        # Spent -> silent.
        assert b.tick() is False
        assert b.tick() is False

    def test_boot_with_unset_clock_still_bursts(self):
        # _last_manual_s defaults to None, so a near-zero boot clock is not
        # mistaken for "a manual command just landed".
        b = _burst()
        assert b.begin_stop(0.0) is True
        assert b.tick() is True

    def test_reassert_count_floored_at_one(self):
        b = _burst(count=0)
        assert b.begin_stop(0.0) is True
        assert b.tick() is True
        assert b.tick() is False


class TestStopThenManual:
    """Usual order: /command=false arms the burst, the manual command cancels it."""

    def test_manual_cancels_in_flight_burst(self):
        b = _burst()
        assert b.begin_stop(0.0) is True  # one zero emitted by the caller
        assert b.tick() is True  # a tick or two land before the manual arrives
        b.note_manual(0.05)
        # From here bcu_debug owns the wire -- no more zeros.
        assert b.tick() is False
        assert b.tick() is False


class TestManualThenStop:
    """Reverse cross-topic order: the manual command is seen before the stop."""

    def test_begin_stop_yields_when_manual_recent(self):
        b = _burst()
        b.note_manual(0.0)
        assert b.begin_stop(0.05) is False  # yield -- caller emits nothing
        assert b.tick() is False  # and no burst is queued

    def test_yield_holds_for_the_whole_window(self):
        b = _burst()
        b.note_manual(0.0)
        # Anywhere inside the hold window still yields.
        assert b.begin_stop(_HOLD - 0.01) is False
        assert b.tick() is False


class TestManualHoldExpiry:
    """Once the hold lapses, a stop is a genuine stop again."""

    def test_stop_after_hold_bursts(self):
        b = _burst()
        b.note_manual(0.0)
        assert b.begin_stop(_HOLD + 0.5) is True
        assert b.tick() is True


class TestReset:
    def test_reset_clears_burst_and_manual(self):
        b = _burst()
        b.begin_stop(0.0)
        b.note_manual(0.0)
        b.reset()
        # No burst pending...
        assert b.tick() is False
        # ...and the manual stamp is gone, so a stop bursts normally.
        assert b.begin_stop(0.1) is True
