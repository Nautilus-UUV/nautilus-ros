"""Safe-stop reassert burst for the BCU wire, with a manual-yield.

Sits behind ``bcu_node``'s stop path. Two jobs:

  - **Reassert burst.** On a mission stop the controller emits one safe-stop
    (0 RPM, valves closed) and keeps re-asserting it for a bounded burst, so
    the STM latches the zero even if a single message is dropped -- the STM has
    no staleness watchdog, it re-ships whatever it last heard forever, so a lone
    safe-stop that drops on the wire would leave a stale command running.

  - **Yield to manual.** ``bcu_node`` and the manual driver ``bcu_debug_node``
    both publish to ``/bcu/rpm`` + ``/bcu/valves`` with no arbiter, so a burst
    firing while the operator drives the BCU by hand makes the wire flicker
    (last-writer-wins between the burst's zeros and the manual heartbeat). When
    a manual command is in flight this cancels the burst and holds off arming a
    new one, so the manual command owns the wire cleanly.

Pure and clock-injected like ``BcuCommandGate`` and ``TankLimitGuard``: the node
keeps the rclpy publish and just acts on the booleans this returns, so the
stop/manual ordering race is trivially Tier-1 testable against a synthetic clock.
The two message orderings it has to survive:

  - **stop then manual** (the usual one -- the UI sends ``/command``=false just
    before each manual command): ``begin_stop`` arms and the caller emits one
    zero, then ``note_manual`` cancels the rest. Worst case one zero on the wire.

  - **manual then stop** (the reverse cross-topic delivery order): ``note_manual``
    is seen first, so ``begin_stop`` yields outright -- no zero, no burst.
"""


class BcuSafeStopBurst:
    """Reassert-burst counter + manual-hold for the BCU safe-stop."""

    def __init__(self, reassert_count: int, manual_hold_s: float) -> None:
        # reassert_count: how many idle ticks still publish the safe-stop before
        # the loop goes silent. manual_hold_s: how long a manual command keeps
        # bcu_node off the wire -- only needs to span the delivery skew between
        # the paired /command=false and the manual command.
        self.reassert_count = max(1, int(reassert_count))
        self.manual_hold_s = float(manual_hold_s)
        self.reset()

    def reset(self) -> None:
        """Back to construction state: no burst pending, no manual seen.

        ``_last_manual_s`` is ``None`` (not 0.0) so a near-zero boot/sim clock
        can't read as "a manual command just landed" and suppress the boot burst.
        """
        self._remaining = 0
        self._last_manual_s: float | None = None

    def note_manual(self, now_s: float) -> None:
        """A manual command is driving the BCU: cancel the burst, remember when."""
        self._last_manual_s = float(now_s)
        self._remaining = 0

    def begin_stop(self, now_s: float) -> bool:
        """Register a mission-stop edge.

        Returns ``True`` if the caller should emit one safe-stop zero now (the
        burst is armed), ``False`` if a fresh manual command holds the wire and
        the caller should emit nothing (yield).
        """
        if (
            self._last_manual_s is not None
            and (now_s - self._last_manual_s) < self.manual_hold_s
        ):
            self._remaining = 0
            return False
        self._remaining = self.reassert_count
        return True

    def tick(self) -> bool:
        """One idle control tick. ``True`` -> publish a safe-stop zero this tick."""
        if self._remaining > 0:
            self._remaining -= 1
            return True
        return False
