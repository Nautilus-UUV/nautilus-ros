"""Correlated buoyancy derivation for sampled sweeps.

Past sweeps sampled ``rig.hydrodynamics.fluid_density`` independently of
the neutral-buoyancy trim, so most runs spawned as permanent sinkers or
floaters and were filtered *after* burning full sim cost. This module
closes that hole: the sampler draws (fluid density, neutral-volume
target) and *derives* the trim masses so every emitted plant is
oscillation-viable by construction.

Physics (kg-equivalent, g cancels throughout):

    1000 * V_STATIC + rho_fluid * V_bladder  =  M_FIXED + m_bow + m_stern + m_bladder

The static displacement (three collision boxes) is priced at the *world*
water density — the world buoyancy plugin's ``<default_density>`` is
hardcoded in ``dave_ocean_waves.world`` — while the bladder term is
priced at the *model*-level BuoyancyEngine ``<fluid_density>``, which is
what a scenario's ``rig.hydrodynamics.fluid_density`` templates.

The bow/stern split is a level-float condition: the total mass moment
about x must equal the buoyancy centroid at the spawn volume times the
total mass, so the vehicle floats level at rest (CoM_x = CoB_x => no
static pitch couple, regardless of net buoyancy magnitude).

Pure functions only — no numpy, no I/O. The Tier 3 parity test
(``test/sim/test_buoyancy_budget_parity.py``) locks every constant below
to the canonical SDF/world files, so a silent model edit trips a test
instead of skewing a sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from py_pkg.robot_specs import BLADDER_VOLUME_M3

from .spec.rig import HydrodynamicsSpec

# ---------------------------------------------------------------------------
# Constants — provenance: glider_nautilus/model.sdf + dave_ocean_waves.world
# ---------------------------------------------------------------------------

# World buoyancy plugin <default_density> (dave_ocean_waves.world, graded
# buoyancy block). Prices the static collision-box displacement. NOT
# templated by scenarios — hardcoded in the world file. Numerically equal
# to physics.WATER_DENSITY_KG_M3 (both encode the fresh-water lake
# calibration); kept as its own name because this one mirrors the world
# file specifically and is locked to it by the Tier 3 parity test.
WORLD_WATER_DENSITY_KGM3 = 1000.0

# The three collision boxes the world graded-buoyancy plugin displaces
# (the model's only collision geometries — the parity test counts them).
# base_link_collision <size>, box centred at the hull origin (x = 0).
V_HULL_BOX_M3 = 1.007723 * 0.232517 * 0.232517  # = 5.4481693e-2
# bow_trimming_foam_collision <size>, on trimming_bow (link x = +0.65).
V_BOW_FOAM_M3 = 0.065 * 0.18 * 0.18  # = 2.106e-3
# stern_trimming_foam_collision <size>, on trimming_stern (link x = -0.70).
V_STERN_FOAM_M3 = 0.015 * 0.18 * 0.18  # = 4.86e-4

# Total static (bladder-independent) displaced volume.
V_STATIC_M3 = V_HULL_BOX_M3 + V_BOW_FOAM_M3 + V_STERN_FOAM_M3  # = 5.7073693e-2

# Link x-poses of the trim masses / foam boxes (model.sdf <pose> first field).
X_BOW_M = 0.65
X_STERN_M = -0.70
X_BLADDER_M = 0.6

# Sum of the fixed (non-trim) link masses, in model.sdf document order.
# LeftFin/RightFin/TopRudder carry the implicit SDF-default 1.0 kg
# inertial (no <inertial> element). Trim masses (bow/stern/bladder) are
# deliberately excluded — they are what this module derives.
M_FIXED_KG = (
    44.637  # base_link
    + 1.0  # LeftFin_link (implicit SDF-default inertial)
    + 1.0  # RightFin_link (implicit SDF-default inertial)
    + 1.0  # TopRudder_link (implicit SDF-default inertial)
    + 2.1825  # oil_weight_link
    + 0.015  # imu_link
    + 0.001  # battery_link
    + 1.83  # acu_tilt_link
    + 4.069  # acu_roll_link
    + 0.001  # navsat_link
)  # = 55.7355

# Nominal (canonical-SDF) trim anchors. Everything below self-calibrates
# from these at import time, so the derivation reproduces the canonical
# trim masses exactly at the nominal point. Sourced from the spec layer's
# defaults — HydrodynamicsSpec mirrors model.sdf verbatim (locked by the
# Jinja render parity test) — so a canonical re-trim is a single edit.
_HYDRO_NOMINAL = HydrodynamicsSpec()
NOMINAL_TRIM_BOW_KG = _HYDRO_NOMINAL.trim_mass_bow  # trimming_bow <mass>
NOMINAL_TRIM_STERN_KG = _HYDRO_NOMINAL.trim_mass_stern  # trimming_stern <mass>
NOMINAL_TRIM_BLADDER_KG = _HYDRO_NOMINAL.trim_mass_bladder  # bladder_link <mass>

# BuoyancyEngine plugin (model.sdf): <default_volume> spawn fill and the
# hard <max_volume> clamp the engine enforces regardless of scenario
# (the same SDF value robot_specs mirrors as the mechanical capacity).
NOMINAL_SPAWN_VOLUME_M3 = _HYDRO_NOMINAL.bladder_spawn_volume_m3
SDF_BLADDER_MAX_VOLUME_M3 = BLADDER_VOLUME_M3

# Default viability margins (m^3 of bladder authority past neutral).
DEFAULT_MIN_DIVE_MARGIN_M3 = 2.0e-4
DEFAULT_MIN_CLIMB_MARGIN_M3 = 1.5e-4


def net_buoyancy_kg(
    fluid_density: float,
    bladder_volume_m3: float,
    trim_bow: float,
    trim_stern: float,
    trim_bladder: float = NOMINAL_TRIM_BLADDER_KG,
) -> float:
    """Net upward force in kg-equivalent (> 0 floats, < 0 sinks).

    Static displacement is priced at the world water density; the
    bladder term at the model-level `fluid_density` (see module
    docstring for why the two differ).
    """
    displaced_kg = (
        WORLD_WATER_DENSITY_KGM3 * V_STATIC_M3 + fluid_density * bladder_volume_m3
    )
    mass_kg = M_FIXED_KG + trim_bow + trim_stern + trim_bladder
    return displaced_kg - mass_kg


def neutral_volume_m3(
    fluid_density: float,
    trim_bow: float,
    trim_stern: float,
    trim_bladder: float = NOMINAL_TRIM_BLADDER_KG,
) -> float:
    """Bladder volume at which the vehicle is exactly neutral."""
    return (
        M_FIXED_KG
        + trim_bow
        + trim_stern
        + trim_bladder
        - WORLD_WATER_DENSITY_KGM3 * V_STATIC_M3
    ) / fluid_density


def buoyancy_centroid_x(fluid_density: float, bladder_volume_m3: float) -> float:
    """x-position of the total buoyancy centroid at a given bladder fill.

    The hull box is centred at x = 0 and contributes no moment; the two
    foam boxes ride their links at X_BOW_M / X_STERN_M, and the bladder
    force acts at X_BLADDER_M.
    """
    moment_kgm = (
        WORLD_WATER_DENSITY_KGM3
        * (V_BOW_FOAM_M3 * X_BOW_M + V_STERN_FOAM_M3 * X_STERN_M)
        + fluid_density * bladder_volume_m3 * X_BLADDER_M
    )
    total_kg = (
        WORLD_WATER_DENSITY_KGM3 * V_STATIC_M3 + fluid_density * bladder_volume_m3
    )
    return moment_kgm / total_kg


# Neutral bladder volume of the canonical SDF at nominal density
# (= 2.0963072e-3 m^3, matching the lake fit's 2.097e-3 to ~1 mL).
NOMINAL_NEUTRAL_VOLUME_M3 = neutral_volume_m3(
    WORLD_WATER_DENSITY_KGM3, NOMINAL_TRIM_BOW_KG, NOMINAL_TRIM_STERN_KG
)

# The canonical SDF spawns slightly buoyant: <default_volume> minus the
# nominal neutral volume (= 1.0369e-4 m^3). The default "offset" spawn
# policy preserves this gentle pre-dive surface float at any sampled
# (density, neutral-volume) point.
SPAWN_OFFSET_M3 = NOMINAL_SPAWN_VOLUME_M3 - NOMINAL_NEUTRAL_VOLUME_M3

_NOMINAL_TOTAL_MASS_KG = (
    M_FIXED_KG + NOMINAL_TRIM_BOW_KG + NOMINAL_TRIM_STERN_KG + NOMINAL_TRIM_BLADDER_KG
)  # = 59.1700, matching the model.sdf total-mass comment

# Fixed-mass moment about x, self-calibrated from the nominal anchor:
# at nominal the canonical trims float the vehicle level at the spawn
# volume, so x_B(1000, 0.0022) * 59.170 equals the total mass moment;
# subtracting the known trim moments leaves the fixed links' share
# (= 0.4854912 kg*m). Self-calibrating keeps this consistent with the
# anchors above by construction.
M_FIXED_MOMENT_KGM = (
    buoyancy_centroid_x(WORLD_WATER_DENSITY_KGM3, NOMINAL_SPAWN_VOLUME_M3)
    * _NOMINAL_TOTAL_MASS_KG
) - (
    NOMINAL_TRIM_BOW_KG * X_BOW_M
    + NOMINAL_TRIM_STERN_KG * X_STERN_M
    + NOMINAL_TRIM_BLADDER_KG * X_BLADDER_M
)


@dataclass(frozen=True)
class DerivedBuoyancy:
    """Trim masses + spawn fill derived for one sampled plant point."""

    trim_mass_bow: float
    trim_mass_stern: float
    bladder_spawn_volume_m3: float
    neutral_volume_m3: float


@dataclass(frozen=True)
class Viability:
    """Outcome of the oscillation-viability check for a derived plant."""

    ok: bool
    dive_margin_m3: float
    climb_margin_m3: float
    reasons: tuple[str, ...]


def derive_trim_masses(
    fluid_density: float,
    neutral_volume_m3: float,
    *,
    spawn_volume_m3: float | None = None,
    spawn_offset_m3: float = SPAWN_OFFSET_M3,
    trim_bladder: float = NOMINAL_TRIM_BLADDER_KG,
) -> DerivedBuoyancy:
    """Derive (bow, stern) trim masses hitting a neutral-volume target.

    Two constraints, two unknowns:

    1. Neutral buoyancy at ``neutral_volume_m3`` fixes the trim-mass
       *sum* S (module-docstring balance).
    2. Level float at the spawn volume fixes the bow/stern *split*:
       CoM_x = buoyancy centroid at spawn, so the vehicle rests level.

    ``spawn_volume_m3``, when given, wins over ``spawn_offset_m3`` —
    needed for a bladder-full-float spawn policy. Raises ValueError on
    non-positive inputs or non-positive derived masses (a target the
    fixed geometry cannot reach).
    """
    if fluid_density <= 0.0:
        raise ValueError(f"fluid_density must be > 0, got {fluid_density}")
    if neutral_volume_m3 <= 0.0:
        raise ValueError(f"neutral_volume_m3 must be > 0, got {neutral_volume_m3}")

    if spawn_volume_m3 is None:
        spawn_volume_m3 = neutral_volume_m3 + spawn_offset_m3
    if spawn_volume_m3 <= 0.0:
        raise ValueError(f"spawn volume must be > 0, got {spawn_volume_m3}")

    # Constraint 1: trim-mass sum from the neutral condition.
    trim_sum = (
        WORLD_WATER_DENSITY_KGM3 * V_STATIC_M3
        + fluid_density * neutral_volume_m3
        - M_FIXED_KG
        - trim_bladder
    )

    # Constraint 2: level float at spawn — total mass moment equals
    # x_B(spawn) * m_total; peel off the fixed and bladder shares to get
    # the bow/stern pair's moment, then solve the 2x2 by elimination.
    x_b = buoyancy_centroid_x(fluid_density, spawn_volume_m3)
    total_mass = M_FIXED_KG + trim_sum + trim_bladder
    pair_moment = x_b * total_mass - M_FIXED_MOMENT_KGM - trim_bladder * X_BLADDER_M
    trim_stern = (X_BOW_M * trim_sum - pair_moment) / (X_BOW_M - X_STERN_M)
    trim_bow = trim_sum - trim_stern

    if trim_bow <= 0.0 or trim_stern <= 0.0:
        raise ValueError(
            f"derived trim masses non-positive (bow={trim_bow:.4f} kg, "
            f"stern={trim_stern:.4f} kg) at fluid_density={fluid_density}, "
            f"neutral_volume_m3={neutral_volume_m3}, "
            f"spawn_volume_m3={spawn_volume_m3}; target unreachable with "
            "the fixed hull geometry"
        )

    return DerivedBuoyancy(
        trim_mass_bow=trim_bow,
        trim_mass_stern=trim_stern,
        bladder_spawn_volume_m3=spawn_volume_m3,
        neutral_volume_m3=neutral_volume_m3,
    )


def check_viability(
    derived: DerivedBuoyancy,
    bladder_min_m3: float,
    bladder_max_m3: float,
    *,
    min_dive_margin_m3: float = DEFAULT_MIN_DIVE_MARGIN_M3,
    min_climb_margin_m3: float = DEFAULT_MIN_CLIMB_MARGIN_M3,
) -> Viability:
    """Check a derived plant can both dive and climb within the clamps.

    The neutral volume must sit at least the dive margin above the
    bridge's bladder floor (pumping down from neutral must produce real
    negative buoyancy) and the climb margin below the ceiling; the spawn
    fill must be inside the operating clamps, and the ceiling itself
    must respect the SDF's hard ``<max_volume>`` (the engine silently
    clamps above it, which would fake climb authority).

    No depth-reachability check is needed: the sim has no
    depth-dependent density, so any dive margin holds at every depth —
    2.0e-4 m^3 of authority already yields >= 0.067 m/s terminal
    descent with the lake-fitted heave drag.
    """
    dive_margin = derived.neutral_volume_m3 - bladder_min_m3
    climb_margin = bladder_max_m3 - derived.neutral_volume_m3
    spawn = derived.bladder_spawn_volume_m3

    reasons: list[str] = []
    if dive_margin < min_dive_margin_m3:
        reasons.append(
            f"dive margin {dive_margin:.3e} m^3 < required "
            f"{min_dive_margin_m3:.3e} (neutral {derived.neutral_volume_m3:.4e} "
            f"vs bladder_min {bladder_min_m3:.4e})"
        )
    if climb_margin < min_climb_margin_m3:
        reasons.append(
            f"climb margin {climb_margin:.3e} m^3 < required "
            f"{min_climb_margin_m3:.3e} (neutral {derived.neutral_volume_m3:.4e} "
            f"vs bladder_max {bladder_max_m3:.4e})"
        )
    if not bladder_min_m3 < spawn <= bladder_max_m3:
        reasons.append(
            f"spawn volume {spawn:.4e} m^3 outside operating clamps "
            f"({bladder_min_m3:.4e}, {bladder_max_m3:.4e}]"
        )
    if bladder_max_m3 > SDF_BLADDER_MAX_VOLUME_M3:
        reasons.append(
            f"bladder_max {bladder_max_m3:.4e} m^3 exceeds the SDF hard "
            f"<max_volume> clamp {SDF_BLADDER_MAX_VOLUME_M3:.4e}"
        )

    return Viability(
        ok=not reasons,
        dive_margin_m3=dive_margin,
        climb_margin_m3=climb_margin,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Lake-fit heave anchors (2026-06-24 calibration)
# ---------------------------------------------------------------------------

# The heave fit's own force convention (heave_calibration_targets.py uses
# rho * g = 1000 * 9.8) — deliberately not physics.py's 9.806 gradient.
LAKE_FIT_RHO_G = 1000.0 * 9.8

# Neutral bladder volume the lake heave fit anchored on
# (heave_calibration_fit.json). Deliberately its own constant, NOT
# NOMINAL_NEUTRAL_VOLUME_M3 above (the canonical-SDF derivation, ~1 mL
# apart): tests that reproduce the lake analysis must use the fit's own
# anchor.
LAKE_FIT_NEUTRAL_VOLUME_M3 = 2.097481e-3


def terminal_heave_speed_mps(force_n: float, retain_fraction: float = 1.0) -> float:
    """Terminal vertical speed on the fitted quadratic heave-drag curve.

    Solves ``d1*v + d2*v^2 = force_n`` for ``v >= 0``, with d1/d2 taken
    from the HydrodynamicsSpec defaults — those ARE the adopted lake fit,
    and the SDF parity test ties them to the canonical model.sdf, so a
    re-fit moves every consumer of this curve automatically.
    ``retain_fraction`` scales both drag terms: pass the spec's
    ``ascent_relief.retain_fraction`` for ascent legs, where the
    HeaveAugmentPlugin leaves only that fraction of the heave drag in
    play.
    """
    d1 = retain_fraction * -_HYDRO_NOMINAL.drag_zW
    d2 = retain_fraction * -_HYDRO_NOMINAL.drag_zWabsW
    return (-d1 + math.sqrt(d1**2 + 4.0 * d2 * force_n)) / (2.0 * d2)
