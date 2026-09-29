"""Descend to an operator-specified depth, drive pitch & roll to zero.

The target depth is supplied by the operator via `MissionCommand`'s
`target_pressure_pa` (gauge Pa). The BCU drives the vehicle down to it;
the ACU drives mass to zero attitude (horizontal trim).

The mission ends once the depth is within `NEAR_GOAL_M` of the target.
Under the bang-bang BCU there is no neutral-buoyancy hold to converge to
-- the bladder is at a tank rail for the whole descent and the vehicle
coasts through the target -- so arrival IS the completion condition. The
settling monitor this used to run (peak-to-peak depth quiet over a
trailing window) could never latch against a controller with no idle
equilibrium, and is gone. (It also still ends on an external
`stop`/`abort`.)
"""

from geometry_msgs.msg import Pose

from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M

from .profile import MissionState

# 0.5 m: the arrival band, and the widest miss the depth assertions
# downstream will accept. Comfortably above the BCU's deadband_pa
# (2000 Pa, ~0.2 m), so the mission always finishes on arrival rather
# than on the controller going quiet.
NEAR_GOAL_M = 0.5
NEAR_GOAL_PA = NEAR_GOAL_M * WATER_PRESSURE_GRADIENT_PA_PER_M


class TrimAndNeutralBuoyancyMission:
    def __init__(self) -> None:
        self._target_pa: float = 0.0
        self._x: float = 0.0
        self._y: float = 0.0
        self._latest_pressure_pa: float | None = None

    def start(self, state: MissionState) -> None:
        self._target_pa = state.target_pressure_pa
        if state.pose is not None:
            self._x = state.pose.position.x
            self._y = state.pose.position.y
        self._latest_pressure_pa = None

    def update(self, current_pressure_pa: float) -> None:
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
        return abs(self._latest_pressure_pa - self._target_pa) <= NEAR_GOAL_PA
