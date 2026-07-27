"""Mission profile contract, plus the pieces every profile builds it from.

A mission turns time-since-start into a `POSITION_TARGET` Pose. The
executor (`pathfinding_node`) owns the clock and the publisher; the
mission only answers `reference(t) -> Pose` and `is_done(t) -> bool`.
"""

from dataclasses import dataclass
from typing import Protocol

from geometry_msgs.msg import Pose

from py_pkg.math_utils import rpy_to_quaternion
from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M

# Arrival bands -- the mission-contract half of the bang-bang loop. A leg holds
# one sign of depth error until the vehicle enters one of these bands; entering
# it is what turns the mission and so what reverses the BCU. They live here
# rather than in any one profile because sawtooth, staircase and surface all
# turn on the same boundaries, and a profile importing another profile's
# private constant meant retuning one silently retuned the others.
#
# 0.8 m water column (lake-analysis convention: depth > 0.8 m = diving), wider
# than weather-driven atmospheric drift and sea-state noise. The Pa value
# tracks the physics-layer water density, so a salt-water override moves it.
#
# These must stay comfortably above `DepthSpec.deadband_pa` (the BCU's only
# idle condition): if the pump went idle before the vehicle reached a band, the
# leg could never complete.
_SURFACE_THRESHOLD_M = 0.8

# Gauge pressure below this counts as "surfaced" (gates final completion).
SURFACE_THRESHOLD_PA = _SURFACE_THRESHOLD_M * WATER_PRESSURE_GRADIENT_PA_PER_M
# Within this of the deep extremum counts as "at depth". Equal by symmetry.
DESCEND_TOLERANCE_PA = SURFACE_THRESHOLD_PA
# Within this of the shallow extremum counts as "at the shallow turn". Also
# equal by symmetry, so a shallow extremum of 0 makes the shallow turn and the
# surface coincide -- which is what makes the legacy sawtooth profile fall out.
SHALLOW_TOLERANCE_PA = SURFACE_THRESHOLD_PA


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


@dataclass
class MissionState:
    """Snapshot of glider state + operator-supplied parameters at start.

    `pose` is the last estimator pose (used by missions that hold horizontal
    position). The remaining fields mirror `nautilus_msgs/MissionCommand`;
    each mission consumes the fields it needs and ignores the rest.
    """

    pose: Pose | None = None
    # TRIM_AND_NEUTRAL_BUOYANCY target depth; SAWTOOTH deep extremum (gauge Pa).
    target_pressure_pa: float = 0.0
    shallow_pressure_pa: float = 0.0  # SAWTOOTH shallow extremum (0 => surface)
    angle_rad: float = 0.0  # SAWTOOTH glide pitch magnitude
    n_resurfaces: int = 0  # SAWTOOTH dive count before the final surfacing
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

    def reference(self, mission_t: float) -> Pose:
        """Setpoint at `mission_t` seconds since start.

        `position.z` MUST be gauge Pa. `orientation` encodes target
        roll/pitch for the ACU (yaw is unused).

        Every tick of a running mission yields a setpoint -- a profile has no
        way to abstain. Two things silence a controller, both of them events
        rather than an absence of setpoints: an operator /command=false, and
        this mission's own completion on /mission/complete.
        """

    def is_done(self, mission_t: float) -> bool:
        """True once the mission has run to completion."""
