"""Tier 1 guard for scripts/sweeps/train_validation_mix_v4.yaml.

Loads the shipped v4 train/validation sweep spec BY PATH (same repo walk
as test_train_mix_v3_spec.py) and locks it as a DIFF against the v3 spec
it evolves: v4 is v3 + the A2b nominal rate-coverage knobs, so exactly
two things may move — the seed, and two NEW dimensions sampling the
HeaveAugmentPlugin knobs (ascent drag relief + entry-momentum peak).
Everything else (physics dimensions, anomaly mix, mission_mix, buoyancy
derivation, onset blocks) must be byte-identical, because everything
else about v3 passed acceptance and the freeze list says so.

The load-bearing tests here are the two band-coverage checks: they
re-derive, from the live HydrodynamicsSpec defaults (which the parity
test ties to the canonical model.sdf), that the sampled bands actually
reach the real dives' rate envelope — the entire point of the campaign.
A re-fit of the nominal k moves those guards automatically.
"""

from __future__ import annotations

from py_pkg.scenarios.buoyancy import (
    LAKE_FIT_NEUTRAL_VOLUME_M3,
    LAKE_FIT_RHO_G,
    terminal_heave_speed_mps,
)
from py_pkg.scenarios.spec.rig import HydrodynamicsSpec

from _sweep_specs import lhs_sample, load_sweep_spec, needs_scripts, run_sweep

_V4_PATH, _V4 = load_sweep_spec("train_validation_mix_v4.yaml")
_, _V3 = load_sweep_spec("train_validation_mix_v3.yaml")
DIMS = {d["path"]: d for d in _V4["dimensions"]}
V3_DIMS = {d["path"]: d for d in _V3["dimensions"]}

K_PATH = "rig.hydrodynamics.ascent_relief.retain_fraction"
ENTRY_PATH = "rig.hydrodynamics.entry.peak_speed_mps"

# Dive 4's ascent-rail buoyancy force: V_b 2.465e-3 vs the fit's V_n.
F_DIVE4_ASCENT_N = LAKE_FIT_RHO_G * (2.465e-3 - LAKE_FIT_NEUTRAL_VOLUME_M3)


# ---------------------------------------------------------------------------
# Scale: identical run count, fresh seed
# ---------------------------------------------------------------------------


def test_scale_unchanged_and_seed_fresh():
    assert int(_V4["n_samples"]) == int(_V3["n_samples"]) == 2048
    # The dimension list changed, so the LHS matrix is a new draw
    # regardless; the fresh seed just makes that explicit.
    assert _V4["seed"] != _V3["seed"]
    assert _V4["base_scenario"] == _V3["base_scenario"]
    assert _V4["sampling_mode"] == _V3["sampling_mode"]
    assert _V4["isotropic_jitter_sigma"] == _V3["isotropic_jitter_sigma"]


# ---------------------------------------------------------------------------
# Exactly two new dimensions; every shared dimension byte-equal
# ---------------------------------------------------------------------------


def test_exactly_the_two_rate_coverage_dimensions_are_new():
    assert set(DIMS) == set(V3_DIMS) | {K_PATH, ENTRY_PATH}
    for path in V3_DIMS:
        assert DIMS[path] == V3_DIMS[path], path


def test_the_frozen_blocks_are_v3_byte_equal():
    # The freeze list: anomaly severities, mission mix, and the buoyancy
    # derivation all passed v3 acceptance and are the cross-campaign
    # comparison anchors.
    assert _V4["anomaly_mix"] == _V3["anomaly_mix"]
    assert _V4["mission_mix"] == _V3["mission_mix"]
    assert _V4["buoyancy_derivation"] == _V3["buoyancy_derivation"]


# ---------------------------------------------------------------------------
# Band coverage: the reason v4 exists
# ---------------------------------------------------------------------------


def test_relief_band_covers_the_real_ascent_envelope():
    """A2b needs ascent-rate coverage past 0.12 m/s (real max 0.115)."""
    band = DIMS[K_PATH]
    # k > 1 would ADD drag; 1.0 keeps the symmetric v3 plant as the top
    # end of the sampled population (continuity with v3).
    assert band["high"] == 1.0
    assert band["distribution"] == "uniform"
    # At the band floor, dive 4's own rail force must ascend >= 0.12 and
    # the campaign's strongest corner (fullest bladder_max, shallowest
    # neutral) must clear 0.13 with margin.
    assert (
        terminal_heave_speed_mps(F_DIVE4_ASCENT_N, retain_fraction=band["low"]) >= 0.12
    )
    f_max = LAKE_FIT_RHO_G * (
        DIMS["rig.plant.bladder_max_m3"]["high"]
        - DIMS["derive.neutral_volume_m3"]["low"]
    )
    assert terminal_heave_speed_mps(f_max, retain_fraction=band["low"]) >= 0.13
    # The adopted nominal fit sits strictly inside the sampled band.
    k_nominal = HydrodynamicsSpec().ascent_relief.retain_fraction
    assert band["low"] < k_nominal < band["high"]


def test_entry_band_covers_the_real_entry_peaks():
    """A2b needs entry coverage of the real 0.20-0.24 m/s descents."""
    band = DIMS[ENTRY_PATH]
    assert band["distribution"] == "uniform"
    # Ceiling: cover +0.26 with margin, but stay well below the dive-1
    # plunge (0.43 m/s) reserved for a future kinematic-disturbance
    # fault archetype — the nominal band must not poison its severity
    # floor.
    assert 0.26 <= band["high"] <= 0.30
    # Floor at/below the steady descent terminal: the bottom of the band
    # preserves v3-like gentle entries (no visible transient).
    assert band["low"] <= 0.12
    # The adopted nominal (mean of the dive-2/4 measured peaks) sits
    # strictly inside.
    peak_nominal = HydrodynamicsSpec().entry.peak_speed_mps
    assert band["low"] < peak_nominal < band["high"]
    # And the real measured peaks themselves are covered.
    for real_peak in (0.225, 0.184):
        assert band["low"] < real_peak < band["high"]


# ---------------------------------------------------------------------------
# Sampler + runner integration (scripts/ imports)
# ---------------------------------------------------------------------------


@needs_scripts
def test_worst_corner_fits_the_recommended_timeout():
    # Both new knobs only speed legs up, so the v3 budget arithmetic
    # stands; the deepest sawtooth must still fit the header's 18000 s.
    worst = {"target_pressure_pa": 392240.0, "n_oscillations": 2}
    assert run_sweep._scaled_timeout(worst, None) <= 18000.0


@needs_scripts
def test_v4_spec_loads_through_the_sampler():
    spec = lhs_sample.SweepSpec.load(_V4_PATH)
    assert spec.n_samples == 2048
    assert spec.mission_mix is not None
    assert spec.anomaly_mix is not None
    # The new paths are dotted scenario overlays (not bare PhysicsKnobs
    # names, not mission.*/derive.* pseudo-dimensions), so the sampler
    # treats them as plain per-run scenario fields.
    paths = {d.path for d in spec.dimensions}
    assert {K_PATH, ENTRY_PATH} <= paths
    # The tank-cushion pin survives Dimension validation.
    pin = next(d for d in spec.dimensions if d.path == "rig.plant.tank_air_volume_m3")
    assert pin.low == pin.high == 0.0
