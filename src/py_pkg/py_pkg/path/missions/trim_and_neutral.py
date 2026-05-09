"""Hold operator-specified depth, drive pitch & roll to zero.

The hold-depth is supplied by the operator via `MissionCommand`'s
`target_pressure_pa` (gauge Pa). The cascaded depth controller drives
BCU to neutral buoyancy at that depth; the ACU drives mass to zero
attitude (horizontal trim). Ends only on external `stop` / `abort`.
"""

from geometry_msgs.msg import Pose

from .profile import MissionState


class TrimAndNeutralBuoyancyMission:
    def __init__(self) -> None:
        self._target_pa: float = 0.0
        self._x: float = 0.0
        self._y: float = 0.0

    def start(self, state: MissionState) -> None:
        self._target_pa = state.target_pressure_pa
        if state.pose is not None:
            self._x = state.pose.position.x
            self._y = state.pose.position.y

    def update(self, current_pressure_pa: float) -> None:
        # Open-loop hold; nothing to observe.
        pass

    def reference(self, mission_t: float) -> Pose:
        pose = Pose()
        pose.position.x = self._x
        pose.position.y = self._y
        pose.position.z = self._target_pa
        # Identity quaternion -> roll = pitch = yaw = 0.
        pose.orientation.w = 1.0
        return pose

    def is_done(self, mission_t: float) -> bool:
        return False
