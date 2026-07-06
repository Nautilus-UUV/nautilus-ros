"""Tier 1 unit tests for DepthSettlingMonitor.

Pure logic, time injected. Pins the "depth has gone quiet" contract the
trim/neutral test uses to decide it has converged: stationary only once the
retained samples span a full window AND their peak-to-peak is within range.
"""

from py_pkg.path.missions.settling import DepthSettlingMonitor


def _feed(m, values, start_t=0.0, dt=0.1):
    t = start_t
    for v in values:
        m.update(t, v)
        t += dt


class TestWindowMustFill:
    def test_empty_is_not_stationary(self):
        assert DepthSettlingMonitor(10.0, 5000.0).stationary() is False

    def test_short_quiet_history_is_not_stationary(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        _feed(m, [75_000.0] * 50)  # 5 s at 10 Hz, dead quiet -- but too short
        assert m.stationary() is False

    def test_latches_once_window_is_full(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        _feed(m, [75_000.0] * 120)  # 12 s quiet
        assert m.stationary() is True


class TestRangeGate:
    def test_full_quiet_window_is_stationary(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        _feed(m, [75_000.0 + (i % 2) * 100 for i in range(150)])  # tiny jitter
        assert m.stationary() is True

    def test_moving_window_is_not_stationary(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        _feed(m, [75_000.0 + i * 100 for i in range(150)])  # ramp -> wide range
        assert m.stationary() is False

    def test_range_boundary_is_inclusive(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        # Peak-to-peak exactly 5000 over a full window -> still stationary.
        _feed(m, [75_000.0 if i % 2 == 0 else 80_000.0 for i in range(150)])
        assert m.stationary() is True

    def test_spike_unsettles_then_settles_as_it_ages_out(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        _feed(m, [75_000.0] * 120)  # fill, quiet
        assert m.stationary() is True
        # A big spike blows the range...
        m.update(12.0, 90_000.0)
        assert m.stationary() is False
        # ...and once >window_s of quiet samples follow, it ages out.
        _feed(m, [75_000.0] * 120, start_t=12.1)
        assert m.stationary() is True


class TestReset:
    def test_reset_clears_history(self):
        m = DepthSettlingMonitor(window_s=10.0, range_pa=5000.0)
        _feed(m, [75_000.0] * 120)
        assert m.stationary() is True
        m.reset()
        assert m.stationary() is False
