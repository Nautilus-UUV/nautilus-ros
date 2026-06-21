"""Mission profile contract.

A mission turns time-since-start into a `POSITION_TARGET` Pose. The
executor (`pathfinding_node`) owns the clock and the publisher; the
mission only answers `reference(t) -> Pose` and `is_done(t) -> bool`.
"""

from dataclasses import dataclass
from typing import Protocol

from geometry_msgs.msg import Pose


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
