"""Tier 3: the sim run watchdog self-terminates the launch on mission complete.

Full-Gazebo test. Boots ``nautilus_hal/sawtooth_sim.launch.py`` with
``watchdog:=true`` + ``mission_autostart:=true`` for a shallow 3 m,
single-cycle dive and asserts the whole LAUNCH PROCESS TREE exits on its
own -- the watchdog concludes ``mission_complete`` on pathfinding's
``/mission/complete``, and ``run_watchdog.launch.py``'s ``OnProcessExit``
handler turns that node exit into a full launch ``Shutdown``. It then
checks that ``run_verdict.json`` landed next to the bag with verdict
``mission_complete`` and that the bag directory finalized a
``metadata.yaml``.

Unlike the other Tier 3 sim tests this is NOT driven through
launch_testing + rclpy: the property under test is *self-termination*, so
the launch is run as a plain subprocess and we simply wait for it to end
within a generous wall-clock budget. ``mission_autostart:=true`` also
publishes a latched DiveInit carrying the scenario's tank endpoints, which
arms bcu_node's tank-limit clamp and slows the final ascent near the guard
band -- hence the deliberately large budget.

Marker-gated ``@pytest.mark.sim``; ``SIM_GUI=1`` shows the Gazebo GUI.
Don't run alongside any other sim launch on the host -- both bind the same
gz transport bus.

NOTE: pytest-timeout is not installed in this workspace, so the hard time
bound is enforced by ``subprocess.wait(timeout=...)`` (which reaps and
fails the test on expiry), not a ``@pytest.mark.timeout`` decorator.
"""

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from ._sim_helpers import reap_lingering_gz, sim_gui_enabled

pytestmark = pytest.mark.sim

# 3.0 m * 9806.38 Pa/m gauge -- a shallow single dive.
TARGET_PRESSURE_PA = 29418.0
# Generous: bringup (~15-20 s) + the 8 s autostart delay + one full
# descend/ascend cycle, with the armed tank clamp slowing the guard-band
# approach. The watchdog's 12 s complete-grace rides on top before exit.
WALL_TIMEOUT_S = 600.0
# After the launch process returns, both artifacts are already written
# (the watchdog writes the verdict before its grace, and ros2 bag record
# finalizes metadata during the launch shutdown that precedes exit); poll
# a short window purely to absorb filesystem-flush jitter.
ARTIFACT_SETTLE_S = 15.0


def _tail(path: Path, n: int = 60) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no launch log captured)"


def _wait_for(predicate, timeout_s: float, poll_s: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def test_watchdog_terminates_run_on_mission_complete():
    reap_lingering_gz()

    with tempfile.TemporaryDirectory(prefix="nautilus_run_term_") as run_dir:
        run_path = Path(run_dir)
        bag_path = run_path / "raw"  # must NOT pre-exist (ros2 bag creates it)
        verdict_path = run_path / "run_verdict.json"  # bag_path.parent / ...
        log_path = run_path / "launch.log"

        cmd = [
            "ros2",
            "launch",
            "nautilus_hal",
            "sawtooth_sim.launch.py",
            "headless:=false" if sim_gui_enabled() else "headless:=true",
            "mission_autostart:=true",
            "watchdog:=true",
            f"target_pressure_pa:={TARGET_PRESSURE_PA}",
            "n_oscillations:=1",
            "angle_rad:=0.0",
            "z:=-0.115",
            "record:=true",
            f"bag_path:={bag_path}",
        ]

        t0 = time.monotonic()
        timed_out = False
        with open(log_path, "w") as log:
            # New session so we can reap the whole gz/ros2 process tree if
            # the launch never self-terminates.
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
            try:
                returncode = proc.wait(timeout=WALL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = None
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
            finally:
                reap_lingering_gz()

        wall_s = time.monotonic() - t0
        print(
            f"\n[run_termination] wall={wall_s:.1f}s returncode={returncode} "
            f"timed_out={timed_out}"
        )

        assert not timed_out, (
            f"launch did not self-terminate within {WALL_TIMEOUT_S:.0f}s "
            f"(reaped forcibly at {wall_s:.1f}s) -- watchdog never concluded?"
            f"\n--- launch.log tail ---\n{_tail(log_path)}"
        )

        # The watchdog wrote the verdict before its grace; give the FS a
        # brief moment in case of flush lag.
        assert _wait_for(verdict_path.exists, ARTIFACT_SETTLE_S), (
            f"run_verdict.json never appeared at {verdict_path} "
            f"(self-terminated at {wall_s:.1f}s, rc={returncode})"
            f"\n--- launch.log tail ---\n{_tail(log_path)}"
        )
        record = json.loads(verdict_path.read_text())
        assert record.get("verdict") == "mission_complete", (
            f"run ended on the wrong verdict: {record}"
            f"\n--- launch.log tail ---\n{_tail(log_path)}"
        )

        # The recorded bag directory finalized its rosbag2 metadata.
        metadata_path = bag_path / "metadata.yaml"
        assert _wait_for(metadata_path.exists, ARTIFACT_SETTLE_S), (
            f"bag metadata.yaml missing at {metadata_path}; bag dir contents: "
            f"{sorted(p.name for p in bag_path.iterdir()) if bag_path.exists() else 'MISSING'}"
            f"\n--- launch.log tail ---\n{_tail(log_path)}"
        )
