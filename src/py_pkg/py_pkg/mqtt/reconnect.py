"""Pure reconnect backstop for the MQTT bridge.

paho's ``loop_start()`` runs a background thread that is supposed to reconnect
on its own. Most of the time it does. This is the backstop for when it doesn't:
yank a tether mid-dive and the Pi's TCP socket goes half-open (no FIN/RST), and
paho can wedge -- either stuck disconnected even after the cable returns, or
left believing it's still connected while nothing actually flows. The bridge
process stays alive the whole time, so a reboot is the only thing that brings
the link back. That's the bug this guards against.

No ROS, no paho, no clock of its own. Fed (connected, ever_connected, rx_age,
now) on a timer, it decides one thing: should the bridge tear the client down
and build a fresh one. A rebuild is the in-process equivalent of that reboot --
a brand new ``connect_async`` that can't inherit the wedged socket.
"""

from __future__ import annotations


class ReconnectSupervisor:
    """Decides when to force a full MQTT client rebuild."""

    def __init__(self, down_grace_s: float, rx_silence_s: float) -> None:
        # How long paho may report "disconnected" before we stop trusting its
        # own backoff and rebuild. Gives a clean, self-healing drop room to
        # recover without a needless teardown.
        self.down_grace_s = down_grace_s
        # How long the link may sit "connected" with nothing arriving before we
        # call it wedged. Sized well above the laptop's ~1 Hz heartbeat, so a
        # live operator's presence beat keeps this from ever tripping on a
        # healthy link.
        self.rx_silence_s = rx_silence_s
        # Monotonic time we first saw "disconnected"; None while connected.
        self._down_since: float | None = None

    def should_rebuild(
        self,
        *,
        connected: bool,
        ever_connected: bool,
        rx_age_s: float | None,
        now: float,
    ) -> bool:
        """Return True when the client should be rebuilt.

        ``connected`` is paho's own view (``is_connected()``). ``ever_connected``
        says whether this client has ever completed a connect. ``rx_age_s`` is
        the time since the last inbound MQTT message, or None if nothing has
        ever arrived. ``now`` is a monotonic timestamp.
        """
        if not connected:
            # Never connected at all is not a wedge -- it's a broker that isn't
            # up yet (the vehicle-boots-before-the-laptop order the bridge
            # docstring calls common, and every sim/sweep run, which has no
            # mosquitto). paho's own 1->30 s backoff is exactly right for that,
            # and rebuilding on a cadence would permanently truncate it. Only a
            # link that once worked can be wedged.
            if not ever_connected:
                self._down_since = None
                return False
            # Stuck disconnected: let paho's backoff try first, then escalate.
            if self._down_since is None:
                self._down_since = now
            return (now - self._down_since) >= self.down_grace_s

        # Connected (or paho believes so). Clear the disconnect clock.
        self._down_since = None

        # Stale-connected wedge: paho still claims a link but nothing has come
        # in for far longer than the heartbeat cadence. rx_age None means we've
        # never received anything (fresh boot, broker not up yet) -- that's not
        # a wedge, so leave it alone.
        if rx_age_s is not None and rx_age_s >= self.rx_silence_s:
            return True
        return False

    def note_rebuilt(self) -> None:
        """Call right after a rebuild, so the fresh (briefly disconnected)
        client doesn't immediately re-trip the down-grace timer."""
        self._down_since = None
