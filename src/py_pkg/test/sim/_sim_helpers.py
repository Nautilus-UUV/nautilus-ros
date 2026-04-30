"""Shared utilities for Tier 3 sim tests.

Sim tests aren't safe to run alongside a manual unified_sim — both bind
the same gz transport bus. ``reap_lingering_gz`` enforces "no concurrent
simulators of this world" by killing any leftover ``gz sim`` of our
world. Needed because launch_testing's SIGTERM doesn't reliably reap the
gz-sim-server child of the Ruby ``gz sim`` wrapper.
"""

import subprocess
import time

# Matches headless or GUI gz sim of this project's world.
LINGERING_GZ_PATTERN = r"gz sim.*dave_ocean_waves\.world"


def reap_lingering_gz() -> None:
    """SIGTERM then SIGKILL any gz-sim session of our world. No-match
    pkill returns non-zero, which is fine — we don't assert on it."""
    subprocess.run(
        ["pkill", "-TERM", "-f", LINGERING_GZ_PATTERN], check=False
    )
    time.sleep(0.3)
    subprocess.run(
        ["pkill", "-KILL", "-f", LINGERING_GZ_PATTERN], check=False
    )

