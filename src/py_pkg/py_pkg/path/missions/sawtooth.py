"""Event-driven sawtooth glide between the surface and an operator-set depth.

Each cycle is two legs: descend (pitch = -angle_rad) until close to the
operator-supplied `target_pressure_pa`, then ascend (pitch = +angle_rad)
until back at the surface. Termination is by *observed* resurface count,
not elapsed time -- pathfinding feeds the latest gauge pressure into
`update` each tick and a hysteresis state machine flips legs on real
threshold crossings.

Mission parameters supplied by the operator via `MissionCommand`:
    - `target_pressure_pa`: deep extremum of the dive (gauge Pa).
    - `angle_rad`: glide pitch magnitude.
    - `n_resurfaces`: number of resurface events before completion.

Tolerances:
    - SURFACE_THRESHOLD_PA: gauge pressure below this counts as "surfaced".
      Set wider than weather-driven atmospheric drift (~5 kPa) and sea-state
      noise.
    - DESCEND_TOLERANCE_PA: gauge pressure within this of `target_pa` counts
      as "at depth". Equal to the surface threshold by symmetry.

At each leg boundary the ACU flips pitch hard from -angle to +angle. Real
ACUs slew at finite speed and the hard step can stall briefly; if that
shows up in sim, smooth the leg-end with a tanh blend on `reference`.
"""

from geometry_msgs.msg import Pose

from py_pkg.math_utils import rpy_to_quaternion

from .profile import MissionState

SURFACE_THRESHOLD_PA = 5_000.0    # ~0.5 m water column
DESCEND_TOLERANCE_PA = 5_000.0    # ~0.5 m above the target


class SawtoothMission:
    def __init__(self) -> None:
        self._target_pa: float = 0.0
        self._angle_rad: float = 0.0
        self._n_resurfaces: int = 0
        self._descending: bool = True
        self._resurface_count: int = 0

    def start(self, state: MissionState) -> None:
        self._target_pa = state.target_pressure_pa
        self._angle_rad = state.angle_rad
        self._n_resurfaces = state.n_resurfaces
        self._descending = True
        self._resurface_count = 0

    def update(self, current_pressure_pa: float) -> None:
        if self._descending:
            if current_pressure_pa >= self._target_pa - DESCEND_TOLERANCE_PA:
                self._descending = False
        else:
            if current_pressure_pa <= SURFACE_THRESHOLD_PA:
                self._descending = True
                self._resurface_count += 1

    def reference(self, mission_t: float) -> Pose:
        if self._descending:
            depth_pa = self._target_pa
            pitch_rad = -self._angle_rad
        else:
            depth_pa = 0.0
            pitch_rad = +self._angle_rad

        pose = Pose()
        pose.position.z = depth_pa
        qx, qy, qz, qw = rpy_to_quaternion(0.0, pitch_rad, 0.0)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def is_done(self, mission_t: float) -> bool:
        return self._resurface_count >= self._n_resurfaces
