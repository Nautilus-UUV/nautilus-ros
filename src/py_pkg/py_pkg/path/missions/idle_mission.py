"""Idle / Hold Mission.

Holds the vehicle at its starting depth and level attitude indefinitely.
Does not self-terminate — remains active until an operator sends a stop command.
"""

from geometry_msgs.msg import Pose
from .profile import MissionState


class IdleMission:
    def __init__(self) -> None:
        self._x: float = 0.0
        self._y: float = 0.0
        self._z_pa: float = 0.0

    def start(self, state: MissionState) -> None:
        """Capture the current pose so the vehicle holds its starting position/depth."""
        if state.pose is not None:
            self._x = state.pose.position.x
            self._y = state.pose.position.y
            self._z_pa = state.pose.position.z
        else:
            self._x = 0.0
            self._y = 0.0
            self._z_pa = 0.0

    def update(self, current_pressure_pa: float) -> None:
        """No state transitions needed."""
        pass

    def reference(self, mission_t: float) -> Pose:
        """Continuously return the captured starting depth and level orientation."""
        pose = Pose()
        pose.position.x = self._x
        pose.position.y = self._y
        pose.position.z = self._z_pa
        pose.orientation.w = 1.0  # Identity quaternion = 0 pitch/roll/yaw
        return pose

    def is_done(self, mission_t: float) -> bool:
        """Never self-terminates."""
        return False