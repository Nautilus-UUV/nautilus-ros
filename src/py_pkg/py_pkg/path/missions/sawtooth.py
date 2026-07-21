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
    - `dwell_s`: level station-keep at the deep extremum before each
      ascend leg (seconds). 0 (the default) flips legs immediately —
      the historical sawtooth behavior, bit-for-bit.

While dwelling the state machine is frozen: a pressure bob out of the
band can neither restart the hold nor count as a resurface. The dwell
timer runs from the FIRST band entry — deliberately unlike
`SurfaceMission`'s restart-on-bob dwell — so every hold is bounded at
`dwell_s` and a sweep's wall-clock budget stays computable.

Tolerances:
    - SURFACE_THRESHOLD_PA: gauge pressure below this counts as "surfaced".
      0.8 m of lake water — the same dive boundary the 2026-06-24 lake
      analysis uses (depth > 0.8 m = diving), wider than weather-driven
      atmospheric drift and sea-state noise.
    - DESCEND_TOLERANCE_PA: gauge pressure within this of `target_pa` counts
      as "at depth". Equal to the surface threshold by symmetry.

At each leg boundary the ACU flips pitch hard from -angle to +angle. Real
ACUs slew at finite speed and the hard step can stall briefly; if that
shows up in sim, smooth the leg-end with a tanh blend on `reference`.
"""

from geometry_msgs.msg import Pose

from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M

from .profile import DwellTimer, MissionState, depth_pitch_pose

# 0.8 m water column (lake-analysis convention); the Pa value tracks the
# physics-layer water density so a salt-water override moves it too.
_SURFACE_THRESHOLD_M = 0.8
SURFACE_THRESHOLD_PA = _SURFACE_THRESHOLD_M * WATER_PRESSURE_GRADIENT_PA_PER_M
DESCEND_TOLERANCE_PA = SURFACE_THRESHOLD_PA  # "at depth" band, by symmetry


class SawtoothMission:
    def __init__(self) -> None:
        self._target_pa: float = 0.0
        self._angle_rad: float = 0.0
        self._n_resurfaces: int = 0
        self._dwell = DwellTimer()
        self._descending: bool = True
        self._resurface_count: int = 0
        self._dwelling: bool = False

    def start(self, state: MissionState) -> None:
        self._target_pa = state.target_pressure_pa
        self._angle_rad = state.angle_rad
        self._n_resurfaces = state.n_resurfaces
        self._dwell = DwellTimer(state.dwell_s)
        self._descending = True
        self._resurface_count = 0
        self._dwelling = False

    def update(self, current_pressure_pa: float) -> None:
        if self._dwelling:
            # Dwell freezes the state machine: a bob out of the band must
            # neither flip a leg nor count as a resurface. `reference` ends
            # the dwell on its own clock.
            return
        if self._descending:
            if current_pressure_pa >= self._target_pa - DESCEND_TOLERANCE_PA:
                if self._dwell.dwell_s <= 0.0:
                    self._descending = False
                else:
                    self._dwelling = True
        else:
            if current_pressure_pa <= SURFACE_THRESHOLD_PA:
                self._descending = True
                self._resurface_count += 1

    def reference(self, mission_t: float) -> Pose:
        if self._dwelling and self._dwell.expired(mission_t):
            self._dwelling = False
            self._descending = False
            self._dwell.reset()

        if self._dwelling:
            depth_pa = self._target_pa
            pitch_rad = 0.0  # level station-keep at the deep extremum
        elif self._descending:
            depth_pa = self._target_pa
            pitch_rad = -self._angle_rad
        else:
            depth_pa = 0.0
            pitch_rad = +self._angle_rad
        return depth_pitch_pose(depth_pa, pitch_rad)

    def is_done(self, mission_t: float) -> bool:
        return self._resurface_count >= self._n_resurfaces
