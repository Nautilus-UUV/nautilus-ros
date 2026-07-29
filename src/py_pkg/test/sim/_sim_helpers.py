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

``mission_command`` builds the ``MissionCommand`` every sim driver publishes.
It lives here because each driver used to fill the message field-by-field, so a
new ``.msg`` field meant an identical edit in six files — and the last one added
missed one of them.

``SimClock`` sources Gazebo sim time from the ground-truth odometry
stream. Velocity/time measurements must clock on it, not the wall
clock: the full stack drags host RTF well below 1, and wall-clock
slopes under-read true sim velocities by exactly that factor.
"""

import math
import os
import subprocess
import time

from nautilus_hal.constants import SimTopics
from nautilus_msgs.msg import MissionCommand
from nav_msgs.msg import Odometry
from py_pkg.physics import gauge_pressure_pa

# Depth conversion for the sim's sea-pressure plugin gradient
# (9.80638 kPa/m — the plugin's own constant, deliberately distinct from
# physics.WATER_PRESSURE_GRADIENT_PA_PER_M). Defined once here so the
# lake-matching tests can't drift apart on it.
SIM_PA_PER_M = 9806.38

# Privileged sim-only ground-truth pose stream (deliberately NOT in
# uuv_ros_core, so production controllers can't depend on it). Its
# header stamp IS gz sim time — the clock run_watchdog's plausibility
# rules and every sim-time measurement in these tests use.
GROUND_TRUTH_ODOM_TOPIC = SimTopics.ODOMETRY.format(model_name="glider_nautilus")


def sim_depth_m(absolute_pa: float) -> float:
    """Depth (m) of an absolute external-pressure sample (Pa) under the
    sim's sea-pressure gradient."""
    return gauge_pressure_pa(absolute_pa) / SIM_PA_PER_M


class SimClock:
    """Gazebo sim time for a test driver, read off ground-truth odometry.

    ``now`` is the latest odometry header stamp in seconds (the stream
    runs ~100 Hz, so cross-topic skew is <= 10 ms) or None until the
    first message arrives — sample callbacks should drop data until
    then rather than stamp it with a guess.
    """

    def __init__(self, node) -> None:
        self.now: float | None = None
        node.create_subscription(Odometry, GROUND_TRUTH_ODOM_TOPIC, self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp
        self.now = stamp.sec + stamp.nanosec * 1e-9


def mission_command(mission_id, **fields) -> MissionCommand:
    """One ``MissionCommand`` for a sim driver to publish.

    Every field not named stays at its ``.msg`` default (0 / 0.0), which is what
    the profiles treat as "operator left this alone" -- so a test only spells out
    the parameters its mission actually reads, and a new message field needs no
    edit here or in any driver.

    ``**fields`` are message field names, so the wire name (``n_resurfaces``)
    applies rather than the operator-facing ``n_oscillations``.
    """
    cmd = MissionCommand()
    cmd.mission_id = int(mission_id)
    for name, value in fields.items():
        # getattr first: a typo'd field would otherwise be silently attached to
        # the message object instead of failing the test.
        setattr(cmd, name, type(getattr(cmd, name))(value))
    return cmd


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
