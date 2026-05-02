"""Shared utilities for Tier 3 sim tests.

Sim tests aren't safe to run alongside a manual unified_sim — both bind
the same gz transport bus. ``reap_lingering_gz`` enforces "no concurrent
simulators of this world" by killing any leftover ``gz sim`` of our
world. Needed because launch_testing's SIGTERM doesn't reliably reap the
gz-sim-server child of the Ruby ``gz sim`` wrapper.
"""

import subprocess
import time

# Catches the Ruby wrapper for our world, and the "gz sim server" child
# it forks — once reparented to PID 1 after a crash, that child has no
# world arg in its cmdline, so we can't scope further. ^ anchors so an
# unrelated process that just contains "gz sim" isn't matched.
LINGERING_GZ_PATTERN = r"^gz sim( server|.*dave_ocean_waves)"


def reap_lingering_gz() -> None:
    """SIGTERM then SIGKILL any matching gz-sim process; pkill's
    no-match nonzero exit is expected, so check=False."""
    subprocess.run(
        ["pkill", "-TERM", "-f", LINGERING_GZ_PATTERN], check=False
    )
    time.sleep(0.3)
    subprocess.run(
        ["pkill", "-KILL", "-f", LINGERING_GZ_PATTERN], check=False
    )

