"""Event-driven sawtooth glide between two operator-set pressures.

The vehicle dives to the deep extremum (`target_pressure_pa`) at
pitch = -angle_rad, then climbs to the shallow extremum
(`shallow_pressure_pa`) at pitch = +angle_rad, and repeats. One *dive*
to the deep extremum is one oscillation; after `n_oscillations` dives
the final climb targets the surface (gauge 0 Pa) and the mission ends
once the vehicle is back at the surface.

A `shallow_pressure_pa` of 0 reproduces the legacy profile -- every
climb goes all the way to the surface -- so callers that only set the
deep extremum keep their old behaviour.

Termination is by *observed* pressure, not elapsed time: pathfinding
feeds the latest gauge pressure into `update` each tick and a hysteresis
state machine flips legs on real threshold crossings.

Mission parameters supplied by the operator via `MissionCommand`:
    - `target_pressure_pa`: deep extremum of the glide (gauge Pa).
    - `shallow_pressure_pa`: shallow extremum of the glide (gauge Pa).
    - `angle_rad`: glide pitch magnitude.
    - `n_oscillations`: number of dives before the final surfacing.

Tolerances:
    - SURFACE_THRESHOLD_PA: gauge pressure below this counts as
      "surfaced" (gates the final completion). Set wider than
      weather-driven atmospheric drift (~5 kPa) and sea-state noise.
    - DESCEND_TOLERANCE_PA: gauge pressure within this of the deep
      extremum counts as "at depth".
    - SHALLOW_TOLERANCE_PA: gauge pressure within this of the shallow
      extremum counts as "at the shallow turn". Equal to the surface
      threshold by symmetry, so a shallow extremum of 0 makes the
      shallow turn and the surface coincide (the legacy profile).

At each leg boundary the ACU flips pitch hard from -angle to +angle.
Real ACUs slew at finite speed and the hard step can stall briefly; if
that shows up in sim, smooth the leg-end with a tanh blend on
`reference`.
"""

from geometry_msgs.msg import Pose

from py_pkg.math_utils import rpy_to_quaternion

from .profile import MissionState

SURFACE_THRESHOLD_PA = 5_000.0  # ~0.5 m water column
DESCEND_TOLERANCE_PA = 5_000.0  # ~0.5 m above the deep extremum
SHALLOW_TOLERANCE_PA = 5_000.0  # ~0.5 m below the shallow extremum


class SawtoothMission:
    def __init__(self) -> None:
        self._deep_pa: float = 0.0
        self._shallow_pa: float = 0.0
        self._angle_rad: float = 0.0
        self._n_oscillations: int = 0
        self._descending: bool = True
        self._dive_count: int = 0
        self._surfacing: bool = False
        self._done: bool = False

    def start(self, state: MissionState) -> None:
        self._deep_pa = state.target_pressure_pa
        # Keep the shallow turn strictly above the deep extremum and at or
        # below the surface; an inverted/degenerate request falls back to the
        # legacy "climb to the surface between dives" profile.
        shallow = state.shallow_pressure_pa
        if not (0.0 <= shallow < self._deep_pa):
            shallow = 0.0
        self._shallow_pa = shallow
        self._angle_rad = state.angle_rad
        self._n_oscillations = state.n_oscillations
        self._dive_count = 0
        # 0 oscillations -> nothing to glide; go straight to the surfacing
        # ascent (surfacing is an ascend leg, so descending must be False).
        self._surfacing = self._n_oscillations <= 0
        self._descending = not self._surfacing
        self._done = False

    def update(self, current_pressure_pa: float) -> None:
        if self._done:
            return
        if self._surfacing:
            # Final ascent: complete once we're back at the surface.
            if current_pressure_pa <= SURFACE_THRESHOLD_PA:
                self._done = True
            return
        if self._descending:
            if current_pressure_pa >= self._deep_pa - DESCEND_TOLERANCE_PA:
                # Reached the deep extremum: one dive done.
                self._dive_count += 1
                self._descending = False
                if self._dive_count >= self._n_oscillations:
                    self._surfacing = True
        else:
            # Climbing toward the shallow turn; dive again once we reach it.
            if current_pressure_pa <= self._shallow_pa + SHALLOW_TOLERANCE_PA:
                self._descending = True

    def reference(self, mission_t: float) -> Pose:
        if self._descending:
            depth_pa = self._deep_pa
            pitch_rad = -self._angle_rad
        else:
            depth_pa = 0.0 if self._surfacing else self._shallow_pa
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
        return self._done
