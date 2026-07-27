"""Drive the glider to the surface, then self-terminate.

Setpoint is constant: gauge pressure 0 Pa, identity attitude. The
mission watches `EXTERNAL_PRESSURE` (fed via `update`) and declares
itself done once the glider has been at the surface for
`DWELL_AT_SURFACE_S` seconds without interruption — a transient bob
above the threshold from sea-state noise restarts the dwell timer
rather than aborting the mission.

Operator parameters from `MissionCommand` are all ignored — this
mission has a single, fixed goal.
"""

from geometry_msgs.msg import Pose

from .profile import SURFACE_THRESHOLD_PA, MissionState

DWELL_AT_SURFACE_S = 10.0


class SurfaceMission:
    def __init__(self) -> None:
        self._at_surface: bool = False
        self._dwell_started_t: float | None = None

    def start(self, state: MissionState) -> None:
        self._at_surface = False
        self._dwell_started_t = None

    def update(self, current_pressure_pa: float) -> None:
        self._at_surface = current_pressure_pa <= SURFACE_THRESHOLD_PA

    def reference(self, mission_t: float) -> Pose:
        pose = Pose()
        pose.position.z = 0.0
        # Identity quaternion -> roll = pitch = yaw = 0.
        pose.orientation.w = 1.0
        return pose

    def is_done(self, mission_t: float) -> bool:
        if not self._at_surface:
            self._dwell_started_t = None
            return False
        if self._dwell_started_t is None:
            self._dwell_started_t = mission_t
            return False
        return (mission_t - self._dwell_started_t) >= DWELL_AT_SURFACE_S
