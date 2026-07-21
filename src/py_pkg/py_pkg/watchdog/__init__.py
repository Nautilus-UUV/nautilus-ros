"""Sim run watchdog.

Ends a sweep run as soon as its outcome is decided: mission completion (the
latched ``/mission/complete`` from pathfinding) or an implausibility verdict
-- surface-floater / bottom-sinker -- from a pure, ROS-free tracker
(``plausibility.py``) that mirrors the sweep analysis classifier. The node
exits with a per-verdict code; the launch side turns that exit into a full
launch ``Shutdown`` and the sweep runner reaps the slot on process exit.
"""
