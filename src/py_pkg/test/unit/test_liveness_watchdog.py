"""Tier 1: pure freshness logic for the liveness watchdog.

No ROS, no clock -- timestamps are handed in directly, so each freshness rule
is asserted against synthetic times.
"""

from py_pkg.liveness.watchdog import LivenessWatchdog

SUBSYSTEMS = ("a", "b", "c")
TIMEOUT = 2.0


def _wd():
    return LivenessWatchdog(SUBSYSTEMS, TIMEOUT)


class TestFreshness:
    def test_never_seen_is_offline(self):
        assert _wd().is_online("a", now=100.0) is False

    def test_fresh_mark_is_online(self):
        wd = _wd()
        wd.mark_seen(["a"], now=100.0)
        assert wd.is_online("a", now=100.5) is True

    def test_just_under_timeout_is_online(self):
        wd = _wd()
        wd.mark_seen(["a"], now=100.0)
        assert wd.is_online("a", now=100.0 + TIMEOUT - 0.001) is True

    def test_at_timeout_is_offline(self):
        # Boundary is exclusive: online requires (now - last) < timeout.
        wd = _wd()
        wd.mark_seen(["a"], now=100.0)
        assert wd.is_online("a", now=100.0 + TIMEOUT) is False

    def test_past_timeout_is_offline(self):
        wd = _wd()
        wd.mark_seen(["a"], now=100.0)
        assert wd.is_online("a", now=100.0 + TIMEOUT + 5.0) is False

    def test_re_mark_refreshes(self):
        wd = _wd()
        wd.mark_seen(["a"], now=100.0)
        # Would be stale at 103, but a fresh mark at 102 keeps it online.
        wd.mark_seen(["a"], now=102.0)
        assert wd.is_online("a", now=103.0) is True


class TestFanOut:
    def test_one_mark_refreshes_several_subsystems(self):
        # One source (the valve bitmask) proves two rows alive at once.
        wd = LivenessWatchdog(("bcu_valve_1", "bcu_valve_2"), TIMEOUT)
        wd.mark_seen(("bcu_valve_1", "bcu_valve_2"), now=10.0)
        assert wd.is_online("bcu_valve_1", now=10.1) is True
        assert wd.is_online("bcu_valve_2", now=10.1) is True


class TestSnapshot:
    def test_snapshot_preserves_registration_order(self):
        names = [name for name, _ in _wd().snapshot(now=0.0)]
        assert names == list(SUBSYSTEMS)

    def test_snapshot_mixed_states(self):
        wd = _wd()
        wd.mark_seen(["a", "c"], now=100.0)
        assert dict(wd.snapshot(now=100.5)) == {"a": True, "b": False, "c": True}
