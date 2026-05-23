"""Pure freshness watchdog behind the liveness node.

No ROS and no clock of its own: the caller hands in timestamps, so the decision
logic is directly unit-testable (Tier 1). A subsystem reads *online* while its
most recent mark is younger than the staleness timeout, and *offline* once it
ages past that — or until it has ever been marked at all.
"""

from __future__ import annotations

from typing import Iterable


class LivenessWatchdog:
    """Tracks the last-seen time of every subsystem and decides freshness."""

    def __init__(self, subsystems: Iterable[str], staleness_timeout_s: float) -> None:
        self._subsystems = tuple(subsystems)
        self.staleness_timeout_s = staleness_timeout_s
        # Absent until the first mark, so a never-seen subsystem reads offline.
        self._last_seen: dict[str, float] = {}

    def mark_seen(self, subsystems: Iterable[str], now: float) -> None:
        """Refresh one or more subsystems. A single source can feed several
        rows — one valve bitmask proves both valves alive — so this takes an
        iterable rather than a single name."""
        for name in subsystems:
            self._last_seen[name] = now

    def is_online(self, name: str, now: float) -> bool:
        last = self._last_seen.get(name)
        if last is None:
            return False
        return (now - last) < self.staleness_timeout_s

    def snapshot(self, now: float) -> list[tuple[str, bool]]:
        """``(name, online)`` for every subsystem, in registration order."""
        return [(name, self.is_online(name, now)) for name in self._subsystems]
