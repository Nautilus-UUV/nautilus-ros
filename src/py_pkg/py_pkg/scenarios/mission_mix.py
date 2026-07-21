"""Per-run mission-profile sampling for sweep campaigns.

Pure logic, no ROS, stdlib RNG only (same conventions as
``anomaly.py``): the host-side sampler (``scripts/lhs_sample.py``)
consumes this to assign every run of a sweep a mission profile — plain
sawtooth, dwell sawtooth, staircase, or station-keep — and to draw that
run's mission parameters (target pressure, leg count, dwell) from
authored bands.

Resolved OUTSIDE the LHS matrix, exactly like the anomaly mix: the mix
adds zero LHS dimensions, and its drawn values are emitted as flat
``mission.*`` paths — the same MISSION_PREFIX launch-dimension keys the
sampler already records — so they ride the manifest's per-sample
``values`` into per-run launch args (``run_sweep.py::load_mission_args``
strips the prefix) and are never written into the scenario YAML.

Design contracts (each Tier-1 tested):

- **Stratified exact counts**: ``assign_profiles`` splits ``n`` runs
  into largest-remainder-rounded profile counts and shuffles them with
  a dedicated seeded stream — a 1024-run sweep at 0.25/0.30/0.25/0.20
  carries exactly 256/307/256/205, deterministically.
- **Decoupling**: the profile stream (``"mission_profile_assignment"``)
  and the per-run parameter streams (``"mission_params:<idx>"``) are
  derive_seed children of the sweep seed, independent of the LHS matrix
  and of the anomaly streams. Editing one profile's bands never
  reshuffles assignments or other profiles' drawn values.
- **Fixed draw order** inside ``draw_mission`` (target pressure, then
  leg count, then dwell), so same seed + same mix config => identical
  missions forever.

``expected_mission_duration_s`` is the deliberately optimistic no-safety
duration estimate that fault-onset placement scales into an onset time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from pydantic import model_validator

from py_pkg.physics import WATER_PRESSURE_GRADIENT_PA_PER_M

from .seed import derive_seed
from .spec._shared import StrictModel
from .stratify import check_weights, stratified_assignment, stratified_counts

# Kept explicit and ORDER-LOCKED: this tuple's order feeds both the
# largest-remainder tie-break and the seeded profile shuffle, so it must
# never be reordered (test_profile_tuple_order_is_load_bearing).
MISSION_PROFILES = ("sawtooth_plain", "sawtooth_dwell", "staircase", "station_keep")

# expected_mission_duration_s leg speeds: the no-safety speeds
# run_sweep.py budgets with, used here WITHOUT its 2x safety factor or
# bringup allowance, i.e. as the fastest a run could conceivably move
# through its mission. (The Pa/m gradient comes from physics.py, so a
# fluid-density override reaches this estimate too.)
OPTIMISTIC_DESCENT_MPS = 0.10
OPTIMISTIC_ASCENT_MPS = 0.035


class MissionBand(StrictModel):
    """Closed band [low, high]; low == high pins a fixed value.

    A dedicated band rather than ``anomaly.Band``: mission knobs need
    the sweep ``Dimension``'s ``integer`` flooring (and never anomaly's
    ``signed`` magnitude flip), so reusing ``Band`` would carry the
    wrong semantics across modules.
    """

    low: float
    high: float
    integer: bool = False

    @model_validator(mode="after")
    def _check_order(self) -> "MissionBand":
        if self.high < self.low:
            raise ValueError(f"band high ({self.high}) must be >= low ({self.low})")
        return self

    def draw(self, rng: random.Random) -> float | int:
        # One rng.random() per draw regardless of `integer`, so flipping
        # the flag never desynchronizes later draws on the same stream.
        v = self.low + rng.random() * (self.high - self.low)
        # `integer: true` floors with u < 1 keeping the high edge
        # exclusive — exactly Dimension.scale in scripts/lhs_sample.py,
        # so {low: 2, high: 3} always draws 2 and {low: 1, high: 4}
        # draws {1, 2, 3} uniformly, whether the band rides the LHS
        # matrix or this mix.
        if self.integer:
            return int(math.floor(v))
        return float(v)


class MissionProfileSpec(StrictModel):
    """One profile's launch-arg bands.

    ``mission_id`` is the profile's fixed launch-side identifier (never
    drawn). Exactly one of ``n_oscillations`` / ``n_steps`` carries the
    leg count — sawtooth-shaped profiles count oscillations, staircase
    profiles count steps. ``dwell_s`` accepts a plain scalar (fixed
    dwell, no RNG consumed) or a band.
    """

    mission_id: int
    target_pressure_pa: MissionBand
    n_oscillations: Optional[MissionBand] = None
    n_steps: Optional[MissionBand] = None
    dwell_s: MissionBand | float = 0.0

    @model_validator(mode="after")
    def _check_leg_count(self) -> "MissionProfileSpec":
        if (self.n_oscillations is None) == (self.n_steps is None):
            raise ValueError(
                "mission profile needs exactly one of n_oscillations / n_steps"
            )
        return self


class MissionMixSpec(StrictModel):
    """The per-run profile mix of a sweep (e.g. 0.25/0.30/0.25/0.20)."""

    weights: dict[str, float]
    sawtooth_plain: Optional[MissionProfileSpec] = None
    sawtooth_dwell: Optional[MissionProfileSpec] = None
    staircase: Optional[MissionProfileSpec] = None
    station_keep: Optional[MissionProfileSpec] = None

    @model_validator(mode="after")
    def _check_weights_and_blocks(self) -> "MissionMixSpec":
        check_weights(
            self.weights, MISSION_PROFILES, block="mission_mix", noun="profiles"
        )
        for profile in MISSION_PROFILES:
            if self.weights.get(profile, 0.0) > 0.0 and getattr(self, profile) is None:
                raise ValueError(
                    f"mission_mix: profile {profile!r} has weight > 0 but no "
                    "config block"
                )
        return self


@dataclass(frozen=True)
class MissionAssignment:
    """One run's resolved profile + drawn launch values.

    ``values`` keys are the FLAT ``mission.*`` paths the sampler
    manifest records (``mission.mission_id``,
    ``mission.target_pressure_pa``, ``mission.n_oscillations`` OR
    ``mission.n_steps``, ``mission.dwell_s``) — ``run_sweep.py`` strips
    the prefix to make launch args.
    """

    profile: str
    values: dict[str, float | int]

    def record(self) -> dict:
        """Manifest record — profile label + the flat mission.* values."""
        return {"profile": self.profile, **self.values}


def profile_counts(mix: MissionMixSpec, n: int) -> dict[str, int]:
    """Exact per-profile counts (`stratify.stratified_counts`).

    Ties break in fixed MISSION_PROFILES order — sawtooth_plain, the
    first profile, takes the tie, the same role ``anomaly.class_counts``
    gives ``nominal``.
    """
    return stratified_counts(mix.weights, MISSION_PROFILES, n)


def assign_profiles(mix: MissionMixSpec, parent_seed: int, n: int) -> list[str]:
    """Stratified profile assignment: exact counts, seeded permutation."""
    return stratified_assignment(
        mix.weights,
        MISSION_PROFILES,
        parent_seed,
        "mission_profile_assignment",
        n,
    )


def draw_mission(
    mix: MissionMixSpec, parent_seed: int, idx: int, profile: str
) -> MissionAssignment:
    """Parameter draw for run ``idx`` given its assigned profile.

    Every run gets its own derived RNG stream
    (``"mission_params:<idx>"``, the mission analogue of anomaly's
    ``"anomaly_severity:<idx>"``). The draw order is FIXED —
    target_pressure_pa, then the leg count (n_oscillations or n_steps),
    then dwell_s — a determinism contract: same seed + same mix config
    => identical missions forever, and a dwell-band edit never moves
    the earlier draws.
    """
    if profile not in MISSION_PROFILES:
        raise ValueError(f"unknown mission profile {profile!r}")
    spec = _require(getattr(mix, profile), profile)

    rng = random.Random(derive_seed(parent_seed, f"mission_params:{idx}"))

    values: dict[str, float | int] = {
        "mission.mission_id": spec.mission_id,
        "mission.target_pressure_pa": spec.target_pressure_pa.draw(rng),
    }
    if spec.n_oscillations is not None:
        values["mission.n_oscillations"] = spec.n_oscillations.draw(rng)
    else:
        values["mission.n_steps"] = spec.n_steps.draw(rng)
    values["mission.dwell_s"] = (
        spec.dwell_s.draw(rng)
        if isinstance(spec.dwell_s, MissionBand)
        else float(spec.dwell_s)
    )
    return MissionAssignment(profile, values)


def _require(block: Optional[MissionProfileSpec], profile: str) -> MissionProfileSpec:
    if block is None:
        raise ValueError(f"mission profile {profile!r} assigned but mix has no block")
    return block


def expected_mission_duration_s(values: dict) -> float:
    """Optimistic no-safety duration of one drawn mission, in seconds.

    Round-trip travel at the fastest leg speeds plus the commanded
    dwells — no bringup, no safety factor, no controller settling, no
    surfacing overhead. Optimistic speeds are CORRECT here, not a bug:
    fault-onset placement computes ``onset_s = frac * expected`` with
    ``frac <= 0.6``, and a real run is never faster than this estimate,
    so the onset always lands inside the actual run.

    ``values`` is a MissionAssignment's flat ``mission.*`` dict; a
    missing leg count means one round trip, a staircase's ``n_steps``
    counts its dwells.
    """
    depth_m = (
        float(values["mission.target_pressure_pa"]) / WATER_PRESSURE_GRADIENT_PA_PER_M
    )
    n_osc = int(values.get("mission.n_oscillations", 1))
    n_dwells = int(values.get("mission.n_steps", n_osc))
    travel = (
        n_osc * depth_m * (1.0 / OPTIMISTIC_DESCENT_MPS + 1.0 / OPTIMISTIC_ASCENT_MPS)
    )
    return travel + n_dwells * float(values.get("mission.dwell_s", 0.0))
