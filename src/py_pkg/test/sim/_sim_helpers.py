"""Shared utilities for Tier 3 sim tests.

Sim tests aren't safe to run alongside another manual sim launch — both
bind the same gz transport bus. ``reap_lingering_gz`` enforces "no
concurrent simulators of this world" by killing any leftover ``gz sim``
of our world. Needed because launch_testing's SIGTERM doesn't reliably
reap the gz-sim-server child of the Ruby ``gz sim`` wrapper.
"""

import subprocess
import time

# Catches the Ruby wrapper for our world, and the "gz sim server" child
# it forks — once reparented to PID 1 after a crash, that child has no
# world arg in its cmdline, so we can't scope further. ^ anchors so an
# unrelated process that just contains "gz sim" isn't matched.
LINGERING_GZ_PATTERN = r"^gz sim( server|.*dave_ocean_waves)"

# Sibling helpers that launch_testing spawns alongside `gz sim` — the
# `ros_gz_sim/create` model spawner and the `ros_gz_bridge/parameter_bridge`
# topic relay. If the gz server dies they get re-parented to PID 1 and
# linger, holding the model-namespaced topics and blocking the next test's
# spawn (`create-N exited with code 255`). Scope to `glider_nautilus` so
# unrelated parameter_bridge instances on the host aren't touched.
LINGERING_ORPHAN_PATTERN = (
    r"ros_gz_(sim/create|bridge/parameter_bridge).*glider_nautilus"
)

# Self-starting debug nodes (the `auto_*` convention in py_pkg/debug/).
# A leftover oscillator from an earlier manual session quietly publishes
# onto /bcu/rpm (or sibling actuator topics) and corrupts the test —
# every Tier 3 sim test owns the actuator bus exclusively, so any
# `auto_*` survivor under py_pkg's install layout is fair game to reap.
LINGERING_AUTO_DEBUG_PATTERN = r"lib/py_pkg/auto_\w+"

_REAP_PATTERNS = (
    LINGERING_GZ_PATTERN,
    LINGERING_ORPHAN_PATTERN,
    LINGERING_AUTO_DEBUG_PATTERN,
)


def reap_lingering_gz() -> None:
    """SIGTERM then SIGKILL any matching gz-sim, orphaned helper, or
    stray `auto_*` debug node; pkill's no-match nonzero exit is expected,
    so check=False."""
    for pattern in _REAP_PATTERNS:
        subprocess.run(["pkill", "-TERM", "-f", pattern], check=False)
    time.sleep(0.3)
    for pattern in _REAP_PATTERNS:
        subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
