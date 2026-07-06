"""Settling detector for hold-depth missions.

Tracks a rolling window of ``(time, gauge-pressure)`` samples and reports when
the depth has gone quiet. The trim/neutral test uses it to decide it has
converged and may terminate. Pure and clock-injected (the caller passes the
mission time), so it's Tier-1 testable against a synthetic series.
"""

from collections import deque


class DepthSettlingMonitor:
    """Is the depth stationary across a trailing window?

    ``stationary()`` is True only once the retained samples span at least
    ``window_s`` *and* their peak-to-peak pressure is within ``range_pa``.
    Requiring a full window stops a mission that happens to start quiet from
    declaring victory before it has actually watched for long enough.
    """

    def __init__(self, window_s: float, range_pa: float) -> None:
        self.window_s = float(window_s)
        self.range_pa = float(range_pa)
        self._samples: deque[tuple[float, float]] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, t_s: float, pressure_pa: float) -> None:
        """Record one sample and trim the window.

        Keep exactly one sample at or before the horizon (``t - window_s``) so
        the retained span genuinely covers ``window_s`` -- dropping every old
        sample would leave ``newest - oldest`` just under the window and
        ``stationary`` could never latch under discrete sampling.
        """
        self._samples.append((t_s, float(pressure_pa)))
        horizon = t_s - self.window_s
        while len(self._samples) >= 2 and self._samples[1][0] <= horizon:
            self._samples.popleft()

    def stationary(self) -> bool:
        if len(self._samples) < 2:
            return False
        if self._samples[-1][0] - self._samples[0][0] < self.window_s:
            return False  # not enough history yet
        pressures = [p for _, p in self._samples]
        return (max(pressures) - min(pressures)) <= self.range_pa
