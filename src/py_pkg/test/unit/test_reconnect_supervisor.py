"""Tier 1: pure reconnect-backstop logic.

No ROS, no paho, no clock -- (connected, ever_connected, rx_age, now) are handed
in directly, so each rebuild rule is asserted against synthetic observations.
This is the decision behind the MQTT bridge's self-heal: when paho's own
reconnect wedges after a yanked tether, the supervisor is what says "rebuild the
client" instead of leaving the bridge alive but mute until a reboot.
"""

from py_pkg.mqtt.reconnect import ReconnectSupervisor

GRACE = 12.0
RX_SILENCE = 20.0


def _sup():
    return ReconnectSupervisor(down_grace_s=GRACE, rx_silence_s=RX_SILENCE)


def _ask(sup, *, connected, rx_age_s, now, ever_connected=True):
    """One observation fed to the supervisor.

    `ever_connected` defaults True because every wedge rule is about a link that
    once worked; TestNeverConnected covers the other case explicitly.
    """
    return sup.should_rebuild(
        connected=connected,
        ever_connected=ever_connected,
        rx_age_s=rx_age_s,
        now=now,
    )


class TestHealthyLink:
    def test_connected_with_fresh_rx_never_rebuilds(self):
        sup = _sup()
        # Beats arriving well inside the silence window: nothing to do.
        for t in range(0, 100, 5):
            assert (
                _ask(sup, connected=True, rx_age_s=1.0, now=float(t)) is False
            )

    def test_fresh_boot_connected_without_rx_does_not_rebuild(self):
        # rx_age None == nothing has ever arrived (broker just came up, no
        # operator yet). That's not a wedge, so leave the link alone even past
        # the silence window.
        sup = _sup()
        assert _ask(sup, connected=True, rx_age_s=None, now=0.0) is False
        assert (
            _ask(sup, connected=True, rx_age_s=None, now=10 * RX_SILENCE)
            is False
        )


class TestDisconnectGrace:
    def test_brief_disconnect_within_grace_holds_off(self):
        sup = _sup()
        # paho reports a drop; under the grace window we trust its own backoff.
        assert _ask(sup, connected=False, rx_age_s=5.0, now=100.0) is False
        assert (
            _ask(sup, connected=False, rx_age_s=5.0, now=100.0 + GRACE - 0.1)
            is False
        )

    def test_disconnect_past_grace_rebuilds(self):
        sup = _sup()
        assert _ask(sup, connected=False, rx_age_s=5.0, now=100.0) is False
        assert (
            _ask(sup, connected=False, rx_age_s=5.0, now=100.0 + GRACE) is True
        )

    def test_reconnect_before_grace_clears_the_down_clock(self):
        sup = _sup()
        # Drop at t=100, recover at t=105 (well inside grace), drop again at
        # t=109. Without the clear, 109-100 >= ... could fire early; with it,
        # the second drop starts its own grace window.
        assert _ask(sup, connected=False, rx_age_s=1.0, now=100.0) is False
        assert _ask(sup, connected=True, rx_age_s=1.0, now=105.0) is False
        assert _ask(sup, connected=False, rx_age_s=1.0, now=109.0) is False
        # Only GRACE after the *second* drop does it escalate.
        assert (
            _ask(sup, connected=False, rx_age_s=1.0, now=109.0 + GRACE) is True
        )


class TestStaleConnectedWedge:
    def test_connected_but_silent_past_window_rebuilds(self):
        sup = _sup()
        # paho still claims a link, but no inbound for longer than the silence
        # window: the half-open / stale-socket wedge this guard exists for.
        assert (
            _ask(sup, connected=True, rx_age_s=RX_SILENCE - 1.0, now=0.0)
            is False
        )
        assert (
            _ask(sup, connected=True, rx_age_s=RX_SILENCE, now=0.0) is True
        )


class TestNoteRebuilt:
    def test_note_rebuilt_resets_the_down_clock(self):
        sup = _sup()
        # Escalate on a long disconnect, rebuild, then the fresh (still
        # disconnected) client must get its own full grace window -- not fire
        # again on the very next tick.
        assert _ask(sup, connected=False, rx_age_s=1.0, now=0.0) is False
        assert _ask(sup, connected=False, rx_age_s=1.0, now=GRACE) is True
        sup.note_rebuilt()
        # The fresh client is disconnected too; the first post-rebuild
        # observation (now=GRACE) seeds a new down-clock rather than firing, so
        # the next escalation is a full GRACE later (now=2*GRACE), not at once.
        assert _ask(sup, connected=False, rx_age_s=None, now=GRACE) is False
        assert (
            _ask(sup, connected=False, rx_age_s=None, now=2 * GRACE - 0.1)
            is False
        )
        assert (
            _ask(sup, connected=False, rx_age_s=None, now=2 * GRACE) is True
        )


class TestNeverConnected:
    """A link that has never come up is not wedged -- it's a broker that isn't
    running yet (vehicle boots before the laptop; every sim/sweep run has no
    mosquitto at all). paho's own 1->30 s backoff owns that case; escalating to
    a rebuild cadence would truncate it and churn a client+thread every grace
    window, forever."""

    def test_never_connected_never_rebuilds(self):
        sup = _sup()
        for t in range(0, 10 * int(GRACE), int(GRACE) // 2):
            assert (
                _ask(
                    sup,
                    connected=False,
                    rx_age_s=None,
                    now=float(t),
                    ever_connected=False,
                )
                is False
            )

    def test_first_connect_arms_the_escalation(self):
        sup = _sup()
        # Silent broker for a long while: no rebuilds.
        assert (
            _ask(sup, connected=False, rx_age_s=None, now=0.0, ever_connected=False)
            is False
        )
        assert (
            _ask(
                sup,
                connected=False,
                rx_age_s=None,
                now=100 * GRACE,
                ever_connected=False,
            )
            is False
        )
        # Once the link has worked, a drop starts its own grace window and
        # escalates normally -- the down-clock must not have been left running
        # from the never-connected span.
        assert _ask(sup, connected=True, rx_age_s=1.0, now=100 * GRACE) is False
        assert _ask(sup, connected=False, rx_age_s=1.0, now=100 * GRACE) is False
        assert _ask(sup, connected=False, rx_age_s=1.0, now=101 * GRACE) is True
