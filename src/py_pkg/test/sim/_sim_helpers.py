"""Shared utilities for Tier 3 sim tests.

Sim tests aren't safe to run alongside another manual sim launch — both
bind the same gz transport bus. ``reap_lingering_gz`` enforces "no
concurrent simulators of this world" by killing any leftover ``gz sim``
of our world. Needed because launch_testing's SIGTERM doesn't reliably
reap the gz-sim-server child of the Ruby ``gz sim`` wrapper.

``spin_for`` / ``spin_until`` are the standard executor-pumping loops
every sim test needs, and ``sim_gui_enabled`` reads the single shared
``SIM_GUI`` switch — defined once here so the polling slice and the
env vocabulary can't drift between tests.

``window`` / ``speed`` / ``omega`` are the timeseries-analysis helpers
the trim/surface convergence assertions share.
"""

import math
import os
import subprocess
import time

from nav_msgs.msg import Odometry


def spin_for(executor, duration_s: float, slice_s: float = 0.05) -> None:
    """Pump an executor for a fixed wall-clock window."""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=slice_s)


def spin_until(executor, predicate, timeout_s: float, slice_s: float = 0.05) -> bool:
    """Pump an executor until ``predicate()`` holds or the timeout passes;
    returns the final predicate value."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        executor.spin_once(timeout_sec=slice_s)
    return predicate()


def sim_gui_enabled() -> bool:
    """``SIM_GUI=1`` (or true/yes/on) shows the Gazebo GUI."""
    return os.environ.get("SIM_GUI", "").lower() in ("1", "true", "yes", "on")


def window(
    samples: list[tuple[float, object]],
    window_start_t: float,
) -> list[object]:
    """Samples at or after ``window_start_t`` from a (t, sample) series."""
    return [s for (t, s) in samples if t >= window_start_t]


def speed(odom: Odometry) -> float:
    """Ground-truth linear speed |v| of one odometry sample."""
    v = odom.twist.twist.linear
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def omega(odom: Odometry) -> float:
    """Ground-truth angular rate |w| of one odometry sample."""
    w = odom.twist.twist.angular
    return math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)


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
