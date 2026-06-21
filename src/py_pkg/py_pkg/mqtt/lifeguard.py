"""Pure dead-man state machine behind the MQTT bridge's lifeguard.

No ROS and no clock of its own. The
lifeguard is the deploy-time failsafe, armed by the operator before the glider
goes in the water, it watches the laptop heartbeat and *engages* (latches the
emergency-surface command) once the heartbeat has been silent for the timeout.

Off by default: bench tests and pre-water startup legitimately lose
connectivity, so silence only matters after an explicit arm.

Latched on purpose: once engaged, a returning heartbeat does NOT stand it
down — a flapping tether must not toggle the pump. Only an explicit disarm
clears the latch.
"""

from __future__ import annotations

from py_pkg.math_utils import span_band_guards


class Lifeguard:
    """Tracks the laptop heartbeat and decides when to blow ballast."""

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self.armed = False
        self.engaged = False
        self._last_beat: float | None = None

    def arm(self, now: float) -> None:
        """Arm the failsafe, seeding the clock so the silence window starts here"""
        if self.armed:
            return
        self.armed = True
        self.engaged = False
        self._last_beat = now

    def disarm(self) -> None:
        """Stand down. The ONLY way to clear the engaged latch."""
        self.armed = False
        self.engaged = False
        self._last_beat = None

    def beat(self, now: float) -> None:
        """A laptop heartbeat arrived. Refreshes the silence window but never
        clears the latch."""
        if self.armed:
            self._last_beat = now

    def tick(self, now: float) -> bool:
        """Advance the watchdog; returns True while engaged. Engages when
        armed and the heartbeat has been silent for at least the timeout."""
        if (
            self.armed
            and not self.engaged
            and (now - self._last_beat) >= self.timeout_s
        ):
            self.engaged = True
        return self.engaged


def tank_blow_exhausted(
    tank_pa: float | None,
    tank_empty_pa: float | None,
    tank_full_pa: float | None,
    band: float = 0.10,
) -> bool:
    """Is there anything left for the emergency blow to pump?

    The blow inflates the bladder, draining the tank toward its empty
    endpoint. Within ``band`` (10%) of the full--empty span above the empty
    endpoint the bladder is as full as it's going to get.

    Both endpoints come from the pre-dive initialization (DIVE_INIT).
    A missing or non-positive empty endpoint, an absent full endpoint, or a
    non-positive span all return False.
    """
    if tank_pa is None or tank_empty_pa is None or tank_full_pa is None:
        return False
    if not tank_empty_pa > 0.0 or tank_full_pa <= tank_empty_pa:
        return False
    low_guard, _ = span_band_guards(tank_empty_pa, tank_full_pa, band)
    return tank_pa <= low_guard
