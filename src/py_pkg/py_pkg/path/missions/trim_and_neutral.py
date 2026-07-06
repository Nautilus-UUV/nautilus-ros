"""Hold operator-specified depth, drive pitch & roll to zero.

The hold-depth is supplied by the operator via `MissionCommand`'s
`target_pressure_pa` (gauge Pa). The cascaded depth controller drives
BCU to neutral buoyancy at that depth; the ACU drives mass to zero
attitude (horizontal trim).

The mission ends itself once it has settled at the goal: the depth is
within `NEAR_GOAL_M` of the target AND has moved less than
`STATIONARY_RANGE_M` peak-to-peak over the last `SETTLING_WINDOW_S`. Both
conditions must hold -- "near the goal" alone could still be drifting, and
"stationary" alone could be an off-target equilibrium. (It also still ends
on an external `stop`/`abort`.)
"""

from geometry_msgs.msg import Pose

from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M

from .profile import MissionState
from .settling import DepthSettlingMonitor

NEAR_GOAL_M = 1.0  # within this of the commanded depth
STATIONARY_RANGE_M = 0.5  # peak-to-peak depth motion allowed...
SETTLING_WINDOW_S = 10.0  # ...over this trailing window

NEAR_GOAL_PA = NEAR_GOAL_M * WATER_PRESSURE_GRADIENT_PA_PER_M
STATIONARY_RANGE_PA = STATIONARY_RANGE_M * WATER_PRESSURE_GRADIENT_PA_PER_M


class TrimAndNeutralBuoyancyMission:
    def __init__(self) -> None:
        self._target_pa: float = 0.0
        self._x: float = 0.0
        self._y: float = 0.0
        self._latest_pressure_pa: float | None = None
        self._settling = DepthSettlingMonitor(SETTLING_WINDOW_S, STATIONARY_RANGE_PA)

    def start(self, state: MissionState) -> None:
        self._target_pa = state.target_pressure_pa
        if state.pose is not None:
            self._x = state.pose.position.x
            self._y = state.pose.position.y
        self._latest_pressure_pa = None
        self._settling.reset()

    def update(self, current_pressure_pa: float) -> None:
        # Stash the latest depth; is_done feeds it to the settling window
        # (that's where the mission clock is available).
        self._latest_pressure_pa = current_pressure_pa

    def reference(self, mission_t: float) -> Pose:
        pose = Pose()
        pose.position.x = self._x
        pose.position.y = self._y
        pose.position.z = self._target_pa
        # Identity quaternion -> roll = pitch = yaw = 0.
        pose.orientation.w = 1.0
        return pose

    def is_done(self, mission_t: float) -> bool:
        if self._latest_pressure_pa is None:
            return False
        self._settling.update(mission_t, self._latest_pressure_pa)
        near_goal = abs(self._latest_pressure_pa - self._target_pa) <= NEAR_GOAL_PA
        return near_goal and self._settling.stationary()
