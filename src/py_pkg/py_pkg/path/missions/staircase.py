"""Stepped descent through evenly spaced depths, then surface.

Step targets are `target_pressure_pa * i / n` for i = 1..n, followed by
a final ascent leg to the surface. Each down-leg descends at pitch
-angle_rad until the observed gauge pressure enters the arrival band
(`DESCEND_TOLERANCE_PA`, shared in `profile.py`), then advances to the next
step. The final leg ascends at +angle_rad; the mission is done once the
glider reads at the surface (`SURFACE_THRESHOLD_PA`). `n_steps = 1`
degenerates to a single dive followed by a surfacing.

Each step advance is a turn for the bang-bang BCU: the new target sits
below the vehicle, so the depth error keeps one sign until the next
arrival. The last step and the surface leg are the two sign flips.

Mission parameters supplied by the operator via `MissionCommand`:
    - `target_pressure_pa`: deepest step of the ladder (gauge Pa).
    - `angle_rad`: glide pitch magnitude on transit legs.
    - `n_steps`: number of descent steps (values < 1 are treated as 1).
"""

from geometry_msgs.msg import Pose

from .profile import (
    DESCEND_TOLERANCE_PA,
    SURFACE_THRESHOLD_PA,
    MissionState,
    depth_pitch_pose,
)


class StaircaseMission:
    def __init__(self) -> None:
        self._targets_pa: list[float] = [0.0]
        self._angle_rad: float = 0.0
        self._leg: int = 0
        self._done: bool = False

    def start(self, state: MissionState) -> None:
        n = max(1, state.n_steps)
        self._targets_pa = [state.target_pressure_pa * i / n for i in range(1, n + 1)]
        self._targets_pa.append(0.0)  # final surface leg
        self._angle_rad = state.angle_rad
        self._leg = 0
        self._done = False

    def update(self, current_pressure_pa: float) -> None:
        if self._done:
            return
        if self._leg == len(self._targets_pa) - 1:
            if current_pressure_pa <= SURFACE_THRESHOLD_PA:
                self._done = True
        elif current_pressure_pa >= self._targets_pa[self._leg] - DESCEND_TOLERANCE_PA:
            self._leg += 1

    def reference(self, mission_t: float) -> Pose:
        depth_pa = self._targets_pa[self._leg]
        if self._leg == len(self._targets_pa) - 1:
            pitch_rad = +self._angle_rad  # final ascent to the surface
        else:
            pitch_rad = -self._angle_rad
        return depth_pitch_pose(depth_pa, pitch_rad)

    def is_done(self, mission_t: float) -> bool:
        return self._done
