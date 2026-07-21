"""Mission profile contract, plus the pieces every profile builds it from.

A mission turns time-since-start into a `POSITION_TARGET` Pose. The
executor (`pathfinding_node`) owns the clock and the publisher; the
mission only answers `reference(t) -> Pose` and `is_done(t) -> bool`.
"""

from dataclasses import dataclass
from typing import Protocol

from geometry_msgs.msg import Pose

from py_pkg.math_utils import rpy_to_quaternion


def depth_pitch_pose(depth_pa: float, pitch_rad: float) -> Pose:
    """The setpoint encoding every profile emits: gauge Pa in `position.z`,
    a level-roll/zero-yaw pitch quaternion in `orientation`."""
    pose = Pose()
    pose.position.z = depth_pa
    qx, qy, qz, qw = rpy_to_quaternion(0.0, pitch_rad, 0.0)
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


class DwellTimer:
    """Bounded hold at a station, stamped on first use and never restarted.

    `SawtoothMission` and `StaircaseMission` share this: the hold runs
    `dwell_s` from the FIRST `expired()` call after entering the band —
    deliberately unlike `SurfaceMission`'s restart-on-bob dwell — so every
    hold is bounded and a sweep's wall-clock budget stays computable.
    Callers freeze their own state machine while holding, so a pressure bob
    can neither restart the hold nor advance the mission.
    """

    def __init__(self, dwell_s: float = 0.0) -> None:
        self.dwell_s = dwell_s
        self._start_t: float | None = None

    def reset(self) -> None:
        """Forget any stamp, so the next `expired()` starts a fresh hold."""
        self._start_t = None

    def expired(self, mission_t: float) -> bool:
        """True once `dwell_s` has elapsed since the first call of this hold.

        Lazy-stamps on that first call, so `dwell_s <= 0` expires on it —
        a staircase with no dwell advances a leg per tick, as it did before
        dwell existed.
        """
        if self._start_t is None:
            self._start_t = mission_t
        return mission_t - self._start_t >= self.dwell_s


@dataclass
class MissionState:
    """Snapshot of glider state + operator-supplied parameters at start.

    `pose` is the last estimator pose (used by missions that hold horizontal
    position). The remaining fields mirror `nautilus_msgs/MissionCommand`;
    each mission consumes the fields it needs and ignores the rest.
    """

    pose: Pose | None = None
    target_pressure_pa: float = 0.0  # TRIM_AND_NEUTRAL_BUOYANCY hold-depth
    angle_rad: float = 0.0  # SAWTOOTH glide pitch magnitude
    n_resurfaces: int = 0  # SAWTOOTH termination count
    dwell_s: float = 0.0  # SAWTOOTH/STAIRCASE hold time at depth (0 = no hold)
    n_steps: int = 1  # STAIRCASE descent step count


class MissionProfile(Protocol):
    """Strategy interface for mission profiles."""

    def start(self, state: MissionState) -> None:
        """Capture initial conditions and operator params at mission start."""

    def update(self, current_pressure_pa: float) -> None:
        """Feed observed gauge pressure for closed-loop missions.

        The executor calls this every tick before `reference` / `is_done`.
        Open-loop missions can leave it as a no-op.
        """

    def reference(self, mission_t: float) -> Pose | None:
        """Setpoint at `mission_t` seconds since start.

        `position.z` MUST be gauge Pa. `orientation` encodes target
        roll/pitch for the ACU (yaw is unused).

        Returning `None` means "no setpoint this tick": the executor
        publishes nothing, so the controllers hold their last target
        (the SURFACE/SAWTOOTH missions decline to command between phases
        this way). The controllers are silenced only by /command=false stop.
        """

    def is_done(self, mission_t: float) -> bool:
        """True once the mission has run to completion."""
