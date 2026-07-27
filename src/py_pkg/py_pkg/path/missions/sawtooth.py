"""Event-driven sawtooth glide between two operator-set pressures.

The vehicle dives to the deep extremum (`target_pressure_pa`) at
pitch = -angle_rad, then climbs to the shallow extremum
(`shallow_pressure_pa`) at pitch = +angle_rad, and repeats. One *dive*
to the deep extremum is one oscillation; after `n_resurfaces` dives the
final climb targets the surface (gauge 0 Pa) and the mission ends once
the vehicle is back at the surface.

A `shallow_pressure_pa` of 0 reproduces the legacy profile -- every
climb goes all the way to the surface, so every turn IS a resurface and
`n_resurfaces` keeps its original meaning. Callers that only set the
deep extremum keep their old behaviour bit-for-bit.

Termination is by *observed* pressure, not elapsed time: pathfinding
feeds the latest gauge pressure into `update` each tick and a hysteresis
state machine flips legs on real threshold crossings.

The leg flip is what reverses the bang-bang BCU. Each leg's target sits
on the far side of the vehicle, so the depth error keeps one sign for
the whole leg and the pump runs one way until either the tank rails or
this state machine turns -- at which point the error's sign flips and the
pump reverses. The arrival bands below are therefore also the controller's
turnaround points; they are much wider than its `deadband_pa`, so a leg
always ends on a turn rather than on the deadband.

Mission parameters supplied by the operator via `MissionCommand`:
    - `target_pressure_pa`: deep extremum of the glide (gauge Pa).
    - `shallow_pressure_pa`: shallow extremum of the glide (gauge Pa).
      0 => climb to the surface between dives.
    - `angle_rad`: glide pitch magnitude.
    - `n_resurfaces`: number of dives to the deep extremum before the
      final surfacing.

Tolerances (SURFACE_THRESHOLD_PA / DESCEND_TOLERANCE_PA /
SHALLOW_TOLERANCE_PA) are the shared arrival bands defined in `profile.py`,
where staircase and surface read them from too.

At each leg boundary the ACU flips pitch hard from -angle to +angle.
Real ACUs slew at finite speed and the hard step can stall briefly; if
that shows up in sim, smooth the leg-end with a tanh blend on
`reference`.
"""

from geometry_msgs.msg import Pose

from .profile import (
    DESCEND_TOLERANCE_PA,
    SHALLOW_TOLERANCE_PA,
    SURFACE_THRESHOLD_PA,
    MissionState,
    depth_pitch_pose,
)


class SawtoothMission:
    def __init__(self) -> None:
        self._deep_pa: float = 0.0
        self._shallow_pa: float = 0.0
        self._angle_rad: float = 0.0
        self._n_resurfaces: int = 0
        self._descending: bool = True
        self._dive_count: int = 0
        self._done: bool = False

    @property
    def _surfacing(self) -> bool:
        """On the final ascent: climbing, with every requested dive spent.

        Derived rather than stored -- it is exactly "not descending and out of
        dives" in every reachable state, and a stored copy could only ever
        drift out of step with the pair it is computed from.
        """
        return not self._descending and self._dive_count >= self._n_resurfaces

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
        self._n_resurfaces = state.n_resurfaces
        self._dive_count = 0
        # 0 dives -> nothing to glide; go straight to the surfacing ascent
        # (surfacing is an ascend leg, so descending must be False).
        self._descending = self._n_resurfaces > 0
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
                # Reached the deep extremum: one dive done. Turn -- either
                # climb to the shallow turn, or begin the final ascent if
                # that was the last dive.
                # `_surfacing` follows from these two: once the dive count has
                # caught up with the request, "climbing" IS the final ascent.
                self._dive_count += 1
                self._descending = False
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
        return depth_pitch_pose(depth_pa, pitch_rad)

    def is_done(self, mission_t: float) -> bool:
        return self._done
