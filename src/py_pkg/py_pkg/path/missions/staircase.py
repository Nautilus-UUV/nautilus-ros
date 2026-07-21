"""Stepped descent through evenly spaced depths, hold at each, then surface.

Step targets are `target_pressure_pa * i / n` for i = 1..n, followed by
a final ascent leg to the surface. Each down-leg descends at pitch
-angle_rad until the observed gauge pressure enters the arrival band
(`DESCEND_TOLERANCE_PA`, shared with sawtooth), then station-keeps level
(pitch 0) for `dwell_s` seconds before advancing. The final leg ascends
at +angle_rad; the mission is done once the glider reads at the surface
(`SURFACE_THRESHOLD_PA`). `n_steps = 1` degenerates to dive -> hold ->
surface: the station-keep profile.

Mission parameters supplied by the operator via `MissionCommand`:
    - `target_pressure_pa`: deepest step of the ladder (gauge Pa).
    - `angle_rad`: glide pitch magnitude on transit legs.
    - `dwell_s`: hold time at each step (seconds).
    - `n_steps`: number of descent steps (values < 1 are treated as 1).

As in sawtooth's dwell, the per-step timer runs from the FIRST band
entry with NO restart on a bob — deliberately unlike `SurfaceMission` —
so every hold is bounded at `dwell_s` and a sweep's wall-clock budget
stays computable.
"""

from geometry_msgs.msg import Pose

from .profile import DwellTimer, MissionState, depth_pitch_pose
from .sawtooth import DESCEND_TOLERANCE_PA, SURFACE_THRESHOLD_PA


class StaircaseMission:
    def __init__(self) -> None:
        self._targets_pa: list[float] = [0.0]
        self._angle_rad: float = 0.0
        self._dwell = DwellTimer()
        self._leg: int = 0
        self._arrived: bool = False
        self._done: bool = False

    def start(self, state: MissionState) -> None:
        n = max(1, state.n_steps)
        self._targets_pa = [state.target_pressure_pa * i / n for i in range(1, n + 1)]
        self._targets_pa.append(0.0)  # final surface leg
        self._angle_rad = state.angle_rad
        self._dwell = DwellTimer(state.dwell_s)
        self._leg = 0
        self._arrived = False
        self._done = False

    def update(self, current_pressure_pa: float) -> None:
        if self._done or self._arrived:
            # Holding at a step freezes the state machine, exactly like
            # sawtooth's dwell; `reference` advances legs on its own clock.
            return
        if self._leg == len(self._targets_pa) - 1:
            if current_pressure_pa <= SURFACE_THRESHOLD_PA:
                self._done = True
        else:
            step_pa = self._targets_pa[self._leg]
            if current_pressure_pa >= step_pa - DESCEND_TOLERANCE_PA:
                self._arrived = True

    def reference(self, mission_t: float) -> Pose:
        if self._arrived and self._dwell.expired(mission_t):
            self._leg += 1
            self._arrived = False
            self._dwell.reset()

        depth_pa = self._targets_pa[self._leg]
        if self._arrived:
            pitch_rad = 0.0  # level station-keep at the current step
        elif self._leg == len(self._targets_pa) - 1:
            pitch_rad = +self._angle_rad  # final ascent to the surface
        else:
            pitch_rad = -self._angle_rad
        return depth_pitch_pose(depth_pa, pitch_rad)

    def is_done(self, mission_t: float) -> bool:
        return self._done
