#!/usr/bin/env python3
"""Concurrent dispatcher for Nautilus scenario sweeps over Apptainer.

Reads a directory of scenario YAMLs (typically the output of
`lhs_sample.py`), runs each one as an `apptainer exec` job through the
`apptainer_exec.sh` wrapper, holds up to N runs in flight at once, and
replenishes the pool as slots free.

Per-slot isolation is the whole point of doing this in one place rather
than ad hoc: every concurrent run gets a unique `GZ_PARTITION` (so the
Gazebo transports don't collide) and a unique `ROS_DOMAIN_ID` (so any
DDS chatter stays inside that run). Optionally also slices a CPU budget
into disjoint sets via `--cpu-budget`, handed to apptainer_exec.sh's
`--cpus` flag so runs don't fight each other for cores.

Mission knobs (`target_pressure_pa`, `shallow_pressure_pa`, `angle_rad`,
`n_oscillations`) are not scenario-YAML fields — pass them as
`--launch-args foo:=bar`. The
same value is used for every run in the sweep, unless the sampler's
manifest.json (next to the scenario YAMLs) carries `mission.*`
dimensions: those become per-run launch args (`mission.target_pressure_pa`
-> `target_pressure_pa:=<value>`), appended after `--launch-args` so the
per-run value wins on collision.

When a run's manifest mission values include `target_pressure_pa`, the
per-run wall-clock budget is scaled to that run's mission length
(conservative leg speeds + bringup margin, see `_scaled_timeout`), with
`--per-run-timeout` acting as the cap. Runs without mission values fall
back to `--per-run-timeout` unchanged.

Before queueing anything, runs one `apptainer exec` to load the first
scenario through `py_pkg.scenarios.loader.load_scenario` so a Pydantic
schema typo fails once instead of N times.

The sawtooth/trim launches keep Gazebo + controllers alive after the
mission completes (sawtooth_sim.launch.py:20 documents this), so
without `--per-run-timeout` every slot will hang forever once its
mission finishes. Set the timeout to roughly n_oscillations times the
single-cycle wall time plus a comfortable margin.

Example:
    run_sweep.py --scenarios-dir ./scenarios/lhs_hydro_v1 \\
                 --sif nautilus_sim.sif --concurrency 2 \\
                 --cpu-budget 0-7 --per-run-timeout 600 \\
                 --launch sawtooth_sim.launch.py \\
                 --launch-args target_pressure_pa:=147150.0 angle_rad:=0.6109 n_oscillations:=5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
APPTAINER_EXEC = SCRIPTS_DIR / "apptainer_exec.sh"

# Same repo walk lhs_sample.py uses. physics itself is stdlib-only, but
# py_pkg/__init__ eagerly imports uuv_ros_core, so this drags in the message-type
# registry (~180 ms) where rclpy is installed and silently degrades where it
# isn't. Paid once per orchestrator process for an hours-long sweep, so it's not
# worth dodging here -- unlike debug/mission_fields.py, which exists precisely
# because a launch file pays it on every evaluation.
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src/py_pkg"))
from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M  # noqa: E402

DEFAULT_LAUNCH = "sawtooth_sim.launch.py"
POLL_INTERVAL_SEC = 1.0
# Gazebo + ros2 launch shutdown is slow — give it generous breathing room
# before escalating to SIGKILL on the whole process group. Empirically a
# clean ros2 launch teardown can take 10-15 s; 20 s leaves margin.
KILL_GRACE_SEC = 20

# apptainer_exec.sh hardcodes the host-side `./sim_data` bind to
# `/ros2_ws/sim_data` in the container, so the runner has to live with
# the same convention when it predicts where bags will land.
CONTAINER_SIM_DATA = "/ros2_ws/sim_data"
HOST_SIM_DATA = Path.cwd() / "sim_data"

# Sampler dimensions under this prefix are per-run launch args, not
# scenario fields (mirrors lhs_sample.py's MISSION_PREFIX).
MISSION_PREFIX = "mission."

# Per-run timeout scaling for runs whose manifest carries mission values.
# Leg speeds are deliberately below the plant's slowest steady legs, so
# the budget expires well after the mission finishes even at a poor
# real-time factor. Ascent is budgeted at the ARMED-tank-clamp speed:
# with scenario:= arming bcu_node's clamp (v2 sweeps), inflation stops
# at the tank guard band and steady ascent drops to ~0.015 m/s near it.
# With watchdog:=true the budget is only a hang fallback — runs end at
# mission completion or on a floater/sinker abort long before it.
TIMEOUT_DESCENT_MPS = 0.10
TIMEOUT_ASCENT_MPS = 0.012
# Bringup budget covers the sim_ready_gate's paused-spawn wait (its own
# ready timeout is 600 s): under load the world now WAITS for the graph
# instead of running away from it, so wall time spent here is bounded
# and non-destructive.
TIMEOUT_BRINGUP_SEC = 600.0
TIMEOUT_SAFETY = 2.0

# Watchdog / gate verdicts that mean "the run's infrastructure failed
# before or during init — the same YAML rerun should come back clean".
# These are retried (up to --max-retries) instead of being recorded as
# the run's final outcome; a genuine mission_complete never retries.
RETRYABLE_VERDICTS = {
    "abort_init",
    "abort_bad_start",
    "abort_floater",
    "abort_sinker",
}


def should_retry(
    verdict: str,
    exit_code: int,
    timed_out: bool,
    bag_present: bool,
    record: bool,
    watchdog: bool,
) -> bool:
    """Pure retry policy for one reaped run.

    - an infra verdict (RETRYABLE_VERDICTS) always retries;
    - no verdict + a crash (nonzero exit that wasn't our own timeout
      kill) retries;
    - no verdict + timed out retries only when a watchdog was in play —
      without one, the wall clock is the NORMAL end of a run;
    - recording enabled but no .mcap on disk retries (the bag never
      materialized).
    A successful conclusion (verdict ``mission_complete``) never retries.
    """
    if verdict == "mission_complete":
        return False
    if verdict in RETRYABLE_VERDICTS:
        return True
    if verdict:
        return False
    if timed_out:
        return watchdog
    if exit_code != 0:
        return True
    return record and not bag_present


# Fast DDS maps a domain's fixed ports as 7400 + 250*domainId (+ small
# participant offsets). The Linux ephemeral port range starts at 32768
# (/proc/sys/net/ipv4/ip_local_port_range), so domains >= 102 place
# their fixed DDS ports where any transient client socket may already
# be bound — sporadic "No unicast locators ... Problem creating
# associated Reader" participant failures. The v2 campaign ran 64 slots
# at base 50: slots 52-63 (domains 102-113) were exposed.
_DDS_PORT_BASE = 7400
_DDS_PORTS_PER_DOMAIN = 250
_EPHEMERAL_PORT_START = 32768


# Domain d owns ports 7400 + 250*d .. 7400 + 250*(d+1) - 1, so the
# highest domain whose WHOLE block clears the ephemeral range is 100.
MAX_SAFE_DOMAIN_ID = (
    _EPHEMERAL_PORT_START - _DDS_PORT_BASE
) // _DDS_PORTS_PER_DOMAIN - 1


def max_safe_domain_base(n_slots: int) -> int:
    """Highest --ros-domain-base whose whole slot window keeps every
    domain's full port block below the ephemeral range."""
    return MAX_SAFE_DOMAIN_ID - (n_slots - 1)


def domain_window_collides_with_ephemeral(base: int, n_slots: int) -> bool:
    """True when any slot's domain maps part of its fixed DDS port block
    at/above the Linux ephemeral port range.

    Defined in terms of max_safe_domain_base so the guard and the remedy
    it prints can never disagree about where the boundary is.
    """
    return base > max_safe_domain_base(n_slots)


def truncating_cap_runs(
    mission_args: dict[str, dict[str, object]], cap: Optional[float]
) -> list[tuple[str, float]]:
    """Runs whose scaled wall budget the --per-run-timeout cap would cut.

    Returns (run_id, scaled_seconds) pairs, worst first. The v2 campaign
    was run with a 1800 s cap against ~17 ks scaled budgets — every long
    mission was killed mid-write. Refuse that silently happening again.
    """
    if cap is None:
        return []
    offenders = [
        (run_id, scaled)
        for run_id, mission in mission_args.items()
        if (scaled := _scaled_timeout(mission, None)) is not None and scaled > cap
    ]
    return sorted(offenders, key=lambda pair: -pair[1])


def load_mission_args(scenarios_dir: Path) -> dict[str, dict[str, object]]:
    """Per-run launch args from the sampler manifest's mission.* dimensions.

    Returns {run_id: {launch_arg_name: value}}; empty when there is no
    manifest or it carries no mission dimensions.
    """
    manifest_path = scenarios_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    mission_args: dict[str, dict[str, object]] = {}
    for sample in manifest.get("samples", []):
        args = {
            path[len(MISSION_PREFIX) :]: value
            for path, value in sample.get("values", {}).items()
            if path.startswith(MISSION_PREFIX)
        }
        if args:
            mission_args[sample["run_id"]] = args
    return mission_args


def _scaled_timeout(
    mission: dict[str, object], cap: Optional[float]
) -> Optional[float]:
    """Wall-clock budget for one run, sized to its sampled mission.

    Round-trip time at conservative leg speeds, times a safety factor for
    real-time-factor jitter, plus bringup. `cap` (--per-run-timeout)
    bounds the result; without a target pressure there is nothing to
    scale and the cap is returned unchanged.
    """
    target_pa = mission.get("target_pressure_pa")
    if target_pa is None:
        return cap
    depth_m = float(target_pa) / WATER_PRESSURE_GRADIENT_PA_PER_M
    if "n_steps" in mission:
        # A staircase is ONE round trip of vertical travel regardless of
        # how many steps it descends through.
        n_osc = 1
    else:
        n_osc = max(1, int(mission.get("n_oscillations", 1)))
    round_trips_sec = (
        n_osc * depth_m * (1.0 / TIMEOUT_DESCENT_MPS + 1.0 / TIMEOUT_ASCENT_MPS)
    )
    scaled = TIMEOUT_BRINGUP_SEC + TIMEOUT_SAFETY * round_trips_sec
    return min(cap, scaled) if cap is not None else scaled

# Per-bag finalize: glob *.mcap in cwd, zstd each into *.mcap.zstd, drop
# the original. We use libzstd via ctypes rather than the `zstd` CLI
# because the SIF ships libzstd1 (rosbag2's compression plugin needs it)
# but not the CLI binary. ZSTD_compress is one-shot in-memory; bags
# under a sensible `--per-run-timeout` are well under 1 GB so reading
# the whole file is fine.
_FINALIZE_PYTHON = r"""
import ctypes, glob, os, sys
lib = ctypes.CDLL('libzstd.so.1')
lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
lib.ZSTD_compressBound.restype  = ctypes.c_size_t
lib.ZSTD_compress.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                              ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
lib.ZSTD_compress.restype  = ctypes.c_size_t
lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
lib.ZSTD_isError.restype  = ctypes.c_uint
COMPRESSION_LEVEL = 3
for mcap in sorted(glob.glob('*.mcap')):
    with open(mcap, 'rb') as f:
        data = f.read()
    cap = lib.ZSTD_compressBound(len(data))
    out = ctypes.create_string_buffer(cap)
    written = lib.ZSTD_compress(out, cap, data, len(data), COMPRESSION_LEVEL)
    if lib.ZSTD_isError(written):
        sys.exit(f'libzstd failed on {mcap}')
    with open(mcap + '.zstd', 'wb') as f:
        f.write(out.raw[:written])
    os.remove(mcap)
"""


def _mark_metadata_compressed(metadata_path: Path, compressed_names: list[str]) -> None:
    """Edit a rosbag2 metadata.yaml in place to declare per-file zstd compression.

    Rosbag2 writes metadata with compression fields empty whenever
    `--compression-mode none` is in effect; reindex does the same. Once
    we have zstd'd the .mcap files into .mcap.zstd, the metadata has to
    match or `ros2 bag info / play` won't decompress on read. The
    `files:` inner block keeps the bare .mcap name -- that's rosbag2's
    own convention for the original (pre-compression) MCAP filename.
    """
    raw = yaml.safe_load(metadata_path.read_text())
    info = raw["rosbag2_bagfile_information"]
    info["compression_format"] = "zstd"
    info["compression_mode"] = "FILE"
    info["relative_file_paths"] = list(compressed_names)
    metadata_path.write_text(yaml.safe_dump(raw, sort_keys=False))


def parse_cpu_list(spec: str) -> list[int]:
    """Expand a CPU-list spec like '0-3,5,8-11' into a sorted list of ints."""
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"CPU range {chunk!r}: high < low")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(chunk))
    return sorted(cpus)


def format_cpu_list(cpus: list[int]) -> str:
    """Inverse of parse_cpu_list: compress contiguous runs into ranges."""
    if not cpus:
        return ""
    parts: list[str] = []
    start = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = c
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def slice_cpus(budget: list[int], n_slots: int) -> list[str]:
    """Carve `budget` into `n_slots` near-equal contiguous-ish slices.

    If the budget doesn't divide evenly, the first few slots get one
    extra CPU. Slot i's CPUs are budget[start:end] in input order, so
    contiguous budgets stay contiguous.
    """
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if len(budget) < n_slots:
        raise ValueError(
            f"CPU budget has {len(budget)} cores; not enough for {n_slots} slots"
        )
    base, extra = divmod(len(budget), n_slots)
    slices = []
    i = 0
    for s in range(n_slots):
        size = base + (1 if s < extra else 0)
        slices.append(format_cpu_list(budget[i : i + size]))
        i += size
    return slices


@dataclass
class Slot:
    index: int
    cpus: Optional[str] = None
    proc: Optional[subprocess.Popen] = None
    run_id: Optional[str] = None
    yaml_path: Optional[Path] = None
    log_file: Optional[object] = None
    start_ts: Optional[float] = None
    timeout_sec: Optional[float] = None
    host_bag_path: Optional[Path] = None
    container_bag_path: Optional[str] = None
    # Which try this is (0 = first). Rides with the work item through the
    # queue rather than living in a side map keyed by run_id, so the count
    # and the run it describes cannot get out of step.
    attempt: int = 0


@dataclass
class StatusRow:
    run_id: str
    slot: int
    start_ts: str
    end_ts: str
    exit_code: int
    yaml_path: str
    duration_sec: float
    timed_out: bool
    # From the run watchdog's / sim gate's run_verdict.json
    # (watchdog:=true launches): "mission_complete" / "abort_floater" /
    # "abort_sinker" / "abort_bad_start" / "abort_init". Empty for
    # legacy runs and wall-clock kills — analysis treats empty as
    # "classify from the bag".
    verdict: str = ""
    # 0 for the first try; retried runs append one row per attempt.
    attempt: int = 0


@dataclass
class SweepRunner:
    sif: Path
    scenarios_dir: Path
    sweep_name: str
    sim_data_dir: Path
    launch_file: str
    extra_launch_args: list[str]
    ros_domain_base: int
    record: bool
    per_run_timeout: Optional[float]
    slots: list[Slot]
    # (run_id, yaml_path, attempt); attempt is 0 for a fresh run.
    queue: list[tuple[str, Path, int]]
    mission_args: dict[str, dict[str, object]] = field(default_factory=dict)
    max_retries: int = 2
    stagger_sec: float = 15.0
    status: list[StatusRow] = field(default_factory=list)
    status_path: Path = field(init=False)
    logs_dir: Path = field(init=False)
    _shutdown: bool = False
    _last_launch_ts: float = 0.0

    def __post_init__(self) -> None:
        self.sweep_dir = self.sim_data_dir / self.sweep_name
        self.logs_dir = self.sweep_dir / "_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.sweep_dir / "sweep_status.csv"
        if not self.status_path.exists():
            with self.status_path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "run_id",
                        "slot",
                        "start_ts",
                        "end_ts",
                        "exit_code",
                        "yaml_path",
                        "duration_sec",
                        "timed_out",
                        "verdict",
                        "attempt",
                    ]
                )
        # Retry-on-timeout only makes sense when a run watchdog concludes
        # runs early — without one, the wall clock is the normal end.
        # Parsed as key:=value rather than matched as one exact string, so
        # `watchdog:=True` doesn't silently read as "no watchdog".
        launch_arg_map = dict(
            arg.split(":=", 1) for arg in self.extra_launch_args if ":=" in arg
        )
        self._watchdog_in_play = (
            launch_arg_map.get("watchdog", "").strip().lower() == "true"
        )
        # Persist the mission knobs we forward to ros2 launch (e.g.
        # target_pressure_pa:=147150.0). These don't live in the scenario YAML
        # so without this file post-hoc tooling has no way to recover them.
        # Overwritten on every invocation — last writer wins, which matches
        # how a re-run would otherwise leave stale state.
        if self.extra_launch_args:
            (self.sweep_dir / "launch_args.txt").write_text(
                " ".join(self.extra_launch_args) + "\n"
            )
        # The sampler manifest carries per-run mission args AND the
        # per-run anomaly labels — copy it whenever it exists so the
        # output dir is self-describing (the dataset builder joins
        # labels on run_id from this copy).
        manifest_src = self.scenarios_dir / "manifest.json"
        if manifest_src.is_file():
            shutil.copy2(manifest_src, self.sweep_dir / "manifest.json")

    def launch_slot(
        self, slot: Slot, run_id: str, yaml_path: Path, attempt: int = 0
    ) -> None:
        log_path = self.logs_dir / f"{run_id}.log"
        log_file = log_path.open("w")
        scenario_in_container = f"/ros2_ws/scenarios/{yaml_path.name}"
        gz_partition = f"{self.sweep_name}_slot{slot.index}"
        ros_domain = self.ros_domain_base + slot.index

        # Pick a deterministic bag path (no in-launch timestamp) so the
        # reaper can find and post-process the bag without globbing.
        # Keeping `sampler_id` consistent with the sweep_name puts the
        # bag under sim_data/<sweep>/<run_id>/raw on both sides of the bind.
        host_bag_path: Optional[Path] = None
        container_bag_path: Optional[str] = None
        if self.record:
            container_bag_path = f"{CONTAINER_SIM_DATA}/{self.sweep_name}/{run_id}/raw"
            host_bag_path = HOST_SIM_DATA / self.sweep_name / run_id / "raw"

        cmd: list[str] = [
            str(APPTAINER_EXEC),
            "--scenarios-dir",
            str(self.scenarios_dir),
            "--env",
            f"GZ_PARTITION={gz_partition}",
            "--env",
            f"ROS_DOMAIN_ID={ros_domain}",
        ]
        if slot.cpus:
            cmd += ["--cpus", slot.cpus]
        cmd += [
            str(self.sif),
            "ros2",
            "launch",
            "nautilus_hal",
            self.launch_file,
            "headless:=true",
            "mission_autostart:=true",
            f"sampler_id:={self.sweep_name}",
            f"run_id:={run_id}",
            f"scenario:={scenario_in_container}",
        ]
        if self.record:
            # bag_compression:=none keeps the recorder's shutdown path
            # trivial — no compress thread to wait on — so a SIGKILL only
            # ever strips the last few in-flight messages, never the
            # metadata.yaml. We then zstd the closed bag in `reap_slot`.
            cmd += [
                "record:=true",
                f"bag_path:={container_bag_path}",
                "bag_compression:=none",
            ]
        cmd += list(self.extra_launch_args)
        # Per-run mission args go last so they win over any identically
        # named global --launch-args value.
        run_mission = self.mission_args.get(run_id, {})
        cmd += [f"{name}:={value}" for name, value in run_mission.items()]

        # Header line in the per-run log makes post-hoc forensics easy —
        # you can see the exact wrapper invocation that produced the bag.
        log_file.write(f"# {' '.join(cmd)}\n")
        log_file.flush()

        # start_new_session=True calls setsid() in the child, putting the
        # whole apptainer -> /entrypoint.sh -> ros2 launch -> gz sim tree
        # into its own session/process group. Without it, terminate()
        # only signals the apptainer wrapper and gz sim survives the
        # kill, reparented to PID 1 and pegging cores forever.
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )

        slot.proc = proc
        slot.run_id = run_id
        slot.yaml_path = yaml_path
        slot.attempt = attempt
        slot.log_file = log_file
        slot.start_ts = time.time()
        slot.timeout_sec = _scaled_timeout(run_mission, self.per_run_timeout)
        slot.host_bag_path = host_bag_path
        slot.container_bag_path = container_bag_path
        print(
            f"[slot {slot.index}] launched {run_id} (pid={proc.pid}, "
            f"GZ_PARTITION={gz_partition}, ROS_DOMAIN_ID={ros_domain}"
            + (f", cpus={slot.cpus}" if slot.cpus else "")
            + (f", timeout={slot.timeout_sec:.0f}s" if slot.timeout_sec else "")
            + ")"
        )

    def reap_slot(self, slot: Slot, exit_code: int, timed_out: bool = False) -> None:
        assert (
            slot.run_id is not None
            and slot.yaml_path is not None
            and slot.start_ts is not None
            and slot.proc is not None
        )
        # ALWAYS sweep the run's process group, clean exit included. On a
        # normal watchdog-terminated shutdown, ros2 launch's SIGTERM
        # escalation kills the `ruby gz` wrapper but the real `gz sim`
        # grandchild survives, reparents to PID 1, and keeps stepping its
        # world at a full core forever. Over a long sweep those orphans
        # stack up run after run — a large part of the v2 campaign's
        # load spiral (RTF collapse + DDS meltdown were downstream of
        # it). The leader is already dead, but the process GROUP id
        # lives as long as any member does; one SIGKILL to it reaps
        # every straggler. ProcessLookupError = nothing survived: the
        # good case.
        try:
            os.killpg(slot.proc.pid, signal.SIGKILL)
            print(
                f"[slot {slot.index}] {slot.run_id} swept surviving "
                f"processes from group {slot.proc.pid}"
            )
        except ProcessLookupError:
            pass
        end_ts = time.time()
        duration = end_ts - slot.start_ts
        # The run watchdog (watchdog:=true) writes its verdict next to
        # the bag before shutting the launch down; surface it in the
        # status CSV so analysis can skip aborted runs without opening
        # a single bag. Absent file = legacy run or wall-clock kill.
        watchdog_verdict = ""
        verdict_file: Optional[Path] = None
        if slot.host_bag_path is not None:
            verdict_file = slot.host_bag_path.parent / "run_verdict.json"
            if verdict_file.is_file():
                try:
                    watchdog_verdict = str(
                        json.loads(verdict_file.read_text()).get("verdict", "")
                    )
                except (OSError, ValueError):
                    watchdog_verdict = ""
        attempt = slot.attempt
        row = StatusRow(
            run_id=slot.run_id,
            slot=slot.index,
            start_ts=datetime.fromtimestamp(slot.start_ts, tz=timezone.utc).isoformat(),
            end_ts=datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            exit_code=exit_code,
            yaml_path=str(slot.yaml_path),
            duration_sec=round(duration, 2),
            timed_out=timed_out,
            verdict=watchdog_verdict,
            attempt=attempt,
        )
        self.status.append(row)
        with self.status_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    row.run_id,
                    row.slot,
                    row.start_ts,
                    row.end_ts,
                    row.exit_code,
                    row.yaml_path,
                    row.duration_sec,
                    row.timed_out,
                    row.verdict,
                    row.attempt,
                ]
            )
        if slot.log_file is not None:
            slot.log_file.close()
        if timed_out:
            verdict = f"TIMEOUT (killed, exit {exit_code})"
        elif exit_code == 0:
            verdict = "OK"
        else:
            verdict = f"FAIL (exit {exit_code})"
        if watchdog_verdict:
            verdict += f" [{watchdog_verdict}]"
        print(f"[slot {slot.index}] {slot.run_id} {verdict} in {duration:.1f}s")

        bag_present = slot.host_bag_path is not None and any(
            slot.host_bag_path.glob("*.mcap*")
        )
        retry = (
            not self._shutdown
            and attempt < self.max_retries
            and should_retry(
                verdict=watchdog_verdict,
                exit_code=exit_code,
                timed_out=timed_out,
                bag_present=bag_present,
                record=self.record,
                watchdog=self._watchdog_in_play,
            )
        )
        if retry:
            self._requeue_for_retry(slot, watchdog_verdict, verdict_file)
        elif slot.host_bag_path is not None and slot.container_bag_path is not None:
            self._finalize_bag(
                slot.index,
                slot.run_id,
                slot.host_bag_path,
                slot.container_bag_path,
                slot.cpus,
            )
        slot.proc = None
        slot.run_id = None
        slot.yaml_path = None
        slot.log_file = None
        slot.start_ts = None
        slot.timeout_sec = None
        slot.host_bag_path = None
        slot.container_bag_path = None

    def _requeue_for_retry(
        self,
        slot: Slot,
        verdict: str,
        verdict_file: Optional[Path],
    ) -> None:
        """Park the failed attempt's artifacts and requeue the run.

        The failed bag directory is renamed ``raw.failed<N>`` (kept for
        forensics, cheap — aborted runs die in ~2 min) and the stale
        verdict file removed so the retry can't be misread. The run goes
        to the BACK of the queue: the rest of the sweep drains first,
        keeping a persistent failure from monopolizing a slot.
        """
        assert slot.run_id is not None and slot.yaml_path is not None
        attempt = slot.attempt
        if verdict_file is not None and verdict_file.is_file():
            try:
                verdict_file.unlink()
            except OSError:
                pass
        if slot.host_bag_path is not None and slot.host_bag_path.is_dir():
            parked = slot.host_bag_path.with_name(f"raw.failed{attempt}")
            try:
                if parked.exists():
                    shutil.rmtree(parked)
                slot.host_bag_path.rename(parked)
            except OSError as exc:
                print(
                    f"[slot {slot.index}] {slot.run_id} could not park failed "
                    f"bag ({exc}); removing it instead"
                )
                shutil.rmtree(slot.host_bag_path, ignore_errors=True)
        self.queue.append((slot.run_id, slot.yaml_path, attempt + 1))
        print(
            f"[slot {slot.index}] {slot.run_id} retrying "
            f"(attempt {attempt + 1}/{self.max_retries}, cause: "
            f"{verdict or 'no verdict'})"
        )

    def _finalize_bag(
        self,
        slot_index: int,
        run_id: str,
        host_bag_dir: Path,
        container_bag_dir: str,
        cpus: Optional[str],
    ) -> None:
        """Reconstruct + compress a freshly-recorded bag via apptainer.

        Recording uses `bag_compression:=none`, so even a SIGKILL during
        timeout teardown only ever strips the last few in-flight
        messages — the .mcap remains parseable. The bag is either
        already finalized (clean exit) or missing metadata.yaml
        (SIGKILL). In both cases we rebuild metadata, zstd every .mcap
        in place, and then patch metadata to declare `compression_mode:
        FILE, compression_format: zstd` so downstream `ros2 bag info /
        play` sees the bag exactly as if it had been recorded with the
        old `--compression-mode file` flag.

        Reindex (`ros2 bag reindex`) and compression (via libzstd
        through ctypes) both run inside the SIF via apptainer_exec.sh,
        because the typical sweep host has no ROS and the SIF has no
        `zstd` CLI -- only libzstd1 from the rosbag2 compression plugin.
        The host-side step is just the metadata.yaml YAML patch.
        """
        prefix = f"[slot {slot_index}] {run_id}"
        if not host_bag_dir.is_dir():
            print(f"{prefix} bag dir missing: {host_bag_dir}")
            return
        if not sorted(host_bag_dir.glob("*.mcap")):
            print(f"{prefix} no .mcap in {host_bag_dir}")
            return

        # Single apptainer invocation: reindex (if needed) + compress.
        # Single quotes on PYEND make the heredoc literal so $-vars in
        # the Python don't get expanded by bash before Python sees them.
        script = (
            "set -e\n"
            f'cd "{container_bag_dir}"\n'
            "if [ ! -f metadata.yaml ]; then\n"
            "  ros2 bag reindex . -s mcap\n"
            "fi\n"
            "python3 - <<'PYEND'\n"
            f"{_FINALIZE_PYTHON}"
            "PYEND\n"
        )
        cmd = [str(APPTAINER_EXEC)]
        if cpus:
            cmd += ["--cpus", cpus]
        cmd += [str(self.sif), "bash", "-c", script]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(
                f"{prefix} apptainer finalize failed (exit {proc.returncode}); leaving bag as-is"
            )
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            return

        metadata_path = host_bag_dir / "metadata.yaml"
        compressed = sorted(host_bag_dir.glob("*.mcap.zstd"))
        if not metadata_path.is_file() or not compressed:
            print(
                f"{prefix} finalize completed but expected outputs missing in {host_bag_dir}"
            )
            return
        _mark_metadata_compressed(metadata_path, [c.name for c in compressed])
        print(f"{prefix} finalized {len(compressed)} mcap(s) in {host_bag_dir}")

    def _kill_slot(self, slot: Slot) -> int:
        """SIGTERM the slot's entire process group; SIGKILL if it lingers.

        The group covers apptainer + entrypoint + ros2 launch + Gazebo +
        every node, because launch_slot started the chain with
        start_new_session=True. Signalling the group is the only way to
        guarantee gz sim doesn't survive as an orphan.
        """
        assert slot.proc is not None
        pgid = slot.proc.pid  # equals pgid because we used setsid in the child
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                # Group is already gone (race with poll(), or kernel reaped
                # the last member). Nothing more to do.
                break
            try:
                return slot.proc.wait(timeout=KILL_GRACE_SEC)
            except subprocess.TimeoutExpired:
                continue
        return slot.proc.wait()

    def run(self) -> int:
        total = len(self.queue) + sum(1 for s in self.slots if s.proc is not None)
        print(
            f"sweep '{self.sweep_name}': {total} runs, "
            f"{len(self.slots)} concurrent slots, output -> {self.sweep_dir}"
        )

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        while not self._shutdown:
            # Reap finished slots first so their slot opens for the next run.
            for slot in self.slots:
                if slot.proc is None:
                    continue
                rc = slot.proc.poll()
                if rc is not None:
                    self.reap_slot(slot, rc)

            # Enforce per-run wall-clock budget. The sawtooth/trim launches
            # keep the controller stack alive after the mission completes,
            # so without this they'd never voluntarily exit and the slot
            # would hang forever. The budget is per-slot: scaled to the
            # run's sampled mission when known, else --per-run-timeout.
            now = time.time()
            for slot in self.slots:
                if (
                    slot.proc is None
                    or slot.start_ts is None
                    or slot.timeout_sec is None
                ):
                    continue
                elapsed = now - slot.start_ts
                if elapsed < slot.timeout_sec:
                    continue
                print(
                    f"[slot {slot.index}] {slot.run_id} hit timeout "
                    f"({elapsed:.1f}s >= {slot.timeout_sec:.1f}s); terminating"
                )
                rc = self._kill_slot(slot)
                self.reap_slot(slot, rc, timed_out=True)

            # Refill idle slots from the queue, at most one launch per
            # stagger window: N containers importing Python + running DDS
            # discovery at the same instant is exactly the stampede that
            # melted Fast DDS reader creation in the v2 campaign.
            for slot in self.slots:
                if slot.proc is None and self.queue:
                    if time.time() - self._last_launch_ts < self.stagger_sec:
                        break
                    run_id, yaml_path, attempt = self.queue.pop(0)
                    self.launch_slot(slot, run_id, yaml_path, attempt)
                    self._last_launch_ts = time.time()

            if not self.queue and all(s.proc is None for s in self.slots):
                break
            time.sleep(POLL_INTERVAL_SEC)

        if self._shutdown:
            self._terminate_all()

        ok = sum(1 for r in self.status if r.exit_code == 0 and not r.timed_out)
        timed_out = sum(1 for r in self.status if r.timed_out)
        failed = len(self.status) - ok - timed_out
        print(
            f"sweep done: {len(self.status)} runs recorded — "
            f"{ok} ok, {timed_out} timed out, {failed} failed. "
            f"Status: {self.status_path}"
        )
        return 0 if failed == 0 and timed_out == 0 and not self._shutdown else 1

    def _handle_signal(self, signum, frame) -> None:
        print(f"\nreceived signal {signum}; terminating active runs", file=sys.stderr)
        self._shutdown = True

    def _terminate_all(self) -> None:
        for slot in self.slots:
            if slot.proc is None:
                continue
            rc = self._kill_slot(slot)
            self.reap_slot(slot, rc)


def preflight_validate(sif: Path, scenarios_dir: Path, first_yaml: Path) -> None:
    """Run the first scenario through load_scenario inside the SIF.

    A schema typo in a sweep spec produces N identical-shape YAMLs, all
    of which will fail load_scenario the same way. Catch it once with a
    short-running exec before queueing the sweep proper.
    """
    scenario_in_container = f"/ros2_ws/scenarios/{first_yaml.name}"
    code = (
        "from py_pkg.scenarios.loader import load_scenario; "
        f"load_scenario('{scenario_in_container}')"
    )
    cmd = [
        str(APPTAINER_EXEC),
        "--scenarios-dir",
        str(scenarios_dir),
        str(sif),
        "python3",
        "-c",
        code,
    ]
    print(f"pre-flight: validating {first_yaml.name} via load_scenario ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"pre-flight failed (exit {proc.returncode}); refusing to queue the sweep"
        )
    print("pre-flight ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scenarios-dir",
        required=True,
        type=Path,
        help="Directory of scenario YAMLs (e.g. lhs_sample.py output).",
    )
    ap.add_argument("--sif", required=True, type=Path, help="Path to nautilus_sim.sif.")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent apptainer runs (default: 2).",
    )
    ap.add_argument(
        "--cpu-budget",
        default="",
        help="CPU list (e.g. '0-31' or '0,2,4-7') to slice across slots via taskset. "
        "Omit to let the OS schedule.",
    )
    ap.add_argument(
        "--launch",
        default=DEFAULT_LAUNCH,
        help=f"ros2 launch file under nautilus_hal/ (default: {DEFAULT_LAUNCH}).",
    )
    ap.add_argument(
        "--launch-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed verbatim to ros2 launch (e.g. target_pressure_pa:=147150.0). "
        "Must come last on the command line.",
    )
    ap.add_argument(
        "--ros-domain-base",
        type=int,
        default=10,
        help="Slot i gets ROS_DOMAIN_ID = base + i (default base: 10). Keep "
        "base + concurrency - 1 <= 100: Fast DDS ports are 7400 + 250*domain, "
        "and domains >= 102 land in the Linux ephemeral port range where "
        "transient sockets cause sporadic participant-creation failures "
        "(the v2 campaign's 64 slots at base 50 put slots 52-63 there).",
    )
    ap.add_argument(
        "--sim-data-dir",
        type=Path,
        default=Path("./sim_data"),
        help="Where bags + sweep_status.csv land (default: ./sim_data).",
    )
    ap.add_argument(
        "--sweep-name",
        help="Defaults to the scenarios-dir basename; doubles as sampler_id "
        "inside the launch (controls the bag output sub-dir).",
    )
    ap.add_argument(
        "--no-record",
        action="store_true",
        help="Skip record:=true; sims will not record bags.",
    )
    ap.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the load_scenario pre-flight check (saves ~5s; not recommended).",
    )
    ap.add_argument(
        "--per-run-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Wall-clock budget per run. The sawtooth/trim launches keep the "
        "controller stack alive after the mission completes, so without a "
        "timeout the slot would hang forever. Pick this as roughly "
        "n_oscillations * single-cycle-wall-time with some headroom. Timed-out "
        "runs are recorded with timed_out=true in sweep_status.csv. When the "
        "manifest carries mission values, this acts as a CAP on the per-run "
        "scaled budget — a cap below any run's scaled budget is refused "
        "unless --allow-cap-below-scaled is given.",
    )
    ap.add_argument(
        "--allow-cap-below-scaled",
        action="store_true",
        help="Permit a --per-run-timeout below some runs' scaled mission "
        "budgets (they WILL be killed mid-mission). The v2 campaign lost "
        "~40%% of its runs to exactly this; refuse it by default.",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=2,
        metavar="N",
        help="Retry a run up to N times when it ends in an infrastructure "
        "verdict (abort_init/abort_bad_start/abort_floater/abort_sinker), "
        "crashes without a verdict, or leaves no bag. 0 disables (default: 2).",
    )
    ap.add_argument(
        "--stagger-sec",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="Minimum spacing between slot launches, so N containers never "
        "start their Python/DDS bringup at the same instant (default: 15).",
    )
    args = ap.parse_args(argv)

    if args.per_run_timeout is not None and args.per_run_timeout <= 0:
        raise SystemExit("--per-run-timeout must be positive")

    if not APPTAINER_EXEC.is_file():
        raise SystemExit(f"apptainer_exec.sh missing at {APPTAINER_EXEC}")
    if not args.sif.is_file():
        raise SystemExit(f"SIF not found: {args.sif}")
    if not args.scenarios_dir.is_dir():
        raise SystemExit(f"scenarios-dir not a directory: {args.scenarios_dir}")

    yamls = sorted(args.scenarios_dir.glob("lhs_*.yaml"))
    if not yamls:
        raise SystemExit(f"no lhs_*.yaml files found in {args.scenarios_dir}")

    sweep_name = args.sweep_name or args.scenarios_dir.name
    queue = [(p.stem, p, 0) for p in yamls]

    if domain_window_collides_with_ephemeral(args.ros_domain_base, args.concurrency):
        raise SystemExit(
            f"--ros-domain-base {args.ros_domain_base} with {args.concurrency} "
            f"slots maps Fast DDS ports into the Linux ephemeral port range "
            f"(domains >= 102): transient sockets will sporadically break "
            f"participant creation. Use --ros-domain-base "
            f"{max(0, max_safe_domain_base(args.concurrency))} or lower."
        )

    if args.cpu_budget:
        budget = parse_cpu_list(args.cpu_budget)
        slot_cpus = slice_cpus(budget, args.concurrency)
    else:
        slot_cpus = [None] * args.concurrency
    slots = [Slot(index=i, cpus=cpus) for i, cpus in enumerate(slot_cpus)]

    if not args.no_preflight:
        preflight_validate(args.sif.resolve(), args.scenarios_dir.resolve(), yamls[0])

    mission_args = load_mission_args(args.scenarios_dir)
    if mission_args:
        print(
            f"manifest.json provides per-run mission args for "
            f"{len(mission_args)} runs (e.g. {next(iter(mission_args.values()))})"
        )

    offenders = truncating_cap_runs(mission_args, args.per_run_timeout)
    if offenders:
        worst_id, worst_scaled = offenders[0]
        msg = (
            f"--per-run-timeout {args.per_run_timeout:.0f}s caps BELOW the "
            f"scaled mission budget of {len(offenders)} run(s) "
            f"(worst: {worst_id} needs ~{worst_scaled:.0f}s) — those runs "
            f"would be killed mid-mission and their bags truncated."
        )
        if not args.allow_cap_below_scaled:
            raise SystemExit(
                msg + " Raise the cap (or drop the flag to use per-run "
                "scaled budgets), or pass --allow-cap-below-scaled to "
                "truncate anyway."
            )
        print(f"WARNING: {msg} Proceeding (--allow-cap-below-scaled).")

    runner = SweepRunner(
        sif=args.sif.resolve(),
        scenarios_dir=args.scenarios_dir.resolve(),
        sweep_name=sweep_name,
        sim_data_dir=args.sim_data_dir.resolve(),
        launch_file=args.launch,
        extra_launch_args=list(args.launch_args),
        ros_domain_base=args.ros_domain_base,
        record=not args.no_record,
        per_run_timeout=args.per_run_timeout,
        slots=slots,
        queue=queue,
        mission_args=mission_args,
        max_retries=max(0, args.max_retries),
        stagger_sec=max(0.0, args.stagger_sec),
    )
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
