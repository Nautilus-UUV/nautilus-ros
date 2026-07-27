"""Tier 1 guard for scripts/sweeps/train_validation_mix_v3.yaml.

Loads the shipped v3 train/validation sweep spec BY PATH (same repo walk
as test_train_mix_v2_spec.py) and locks it as a DIFF against the v2 spec
it evolves: v3 is the full-yield re-run, so exactly four things may move
— the run count / mix weights, the bcu_pump severe tails, the
biofouling drag + added-mass ceilings, and the seed. Everything else
(physics dimensions, mission_mix, sensor bands, onset blocks, buoyancy
derivation) must be byte-identical, because the point of the re-run is
that the v2 design was validated on water.

The load-bearing test here is
``test_biofouling_mass_stays_inside_the_derivation_ceiling``: the
sampler HARD-FAILS generation on a non-viable draw (it never resamples),
so a fouling mass past the worst-corner climb margin aborts a 2048-run
sweep after minutes of work. The bound is arithmetic, so assert it.
"""

from __future__ import annotations

import pytest
from py_pkg.scenarios.anomaly import AnomalyMixSpec, class_counts
from py_pkg.scenarios.mission_mix import MissionMixSpec, profile_counts

from _sweep_specs import lhs_sample, load_sweep_spec, needs_scripts, run_sweep

_V3_PATH, _V3 = load_sweep_spec("train_validation_mix_v3.yaml")
_, _V2 = load_sweep_spec("train_validation_mix_v2.yaml")
MIX = AnomalyMixSpec.model_validate(_V3["anomaly_mix"])
MISSION = MissionMixSpec.model_validate(_V3["mission_mix"])
DIMS = {d["path"]: d for d in _V3["dimensions"]}
V2_DIMS = {d["path"]: d for d in _V2["dimensions"]}

N = int(_V3["n_samples"])


# ---------------------------------------------------------------------------
# Scale: 2x the runs, 2x the fault share
# ---------------------------------------------------------------------------


def test_run_count_doubles_and_the_seed_is_fresh():
    assert N == 2 * int(_V2["n_samples"]) == 2048
    # n_samples changed, so the LHS matrix is a new draw regardless; the
    # fresh seed just makes that explicit rather than implicit.
    assert _V3["seed"] != _V2["seed"]
    assert _V3["base_scenario"] == _V2["base_scenario"]
    assert _V3["sampling_mode"] == _V2["sampling_mode"]
    assert _V3["isotropic_jitter_sigma"] == _V2["isotropic_jitter_sigma"]


def test_anomaly_weights_and_stratified_counts_at_2048():
    assert MIX.weights == {
        "nominal": 0.70,
        "bcu_pump": 0.10,
        "sensor": 0.10,
        "biofouling": 0.10,
    }
    # comms stays dropped (unobservable at the 1 Hz record throttle).
    assert "comms" not in MIX.weights
    # 0.70 x 2048 = 1433.6 and 0.10 x 2048 = 204.8: the three 0.8
    # remainders outrank nominal's 0.6, so every fault class rounds up.
    assert class_counts(MIX, N) == {
        "nominal": 1433,
        "bcu_pump": 205,
        "sensor": 205,
        "comms": 0,
        "biofouling": 205,
    }
    # The binding constraint on v2's retrain was 29 usable fault runs.
    # Even at a pessimistic yield this clears it by an order of
    # magnitude — and the nominal pool still grows against v2's 871.
    assert class_counts(MIX, N)["nominal"] > class_counts(
        AnomalyMixSpec.model_validate(_V2["anomaly_mix"]), int(_V2["n_samples"])
    )["nominal"]


# ---------------------------------------------------------------------------
# bcu_pump: same faint end, re-extended severe tail
# ---------------------------------------------------------------------------


def test_pump_faint_end_is_unchanged_from_v2():
    # Only the severe tails move: the faint end must stay inside the
    # envelope so the class still spans invisible -> unmistakable.
    v2_pump = _V2["anomaly_mix"]["bcu_pump"]
    assert MIX.bcu_pump.effectiveness.high == v2_pump["effectiveness"]["high"]
    assert MIX.bcu_pump.response_delay_s.low == v2_pump["response_delay_s"]["low"]
    assert MIX.bcu_pump.slew_rpm_per_s.high == v2_pump["slew_rpm_per_s"]["high"]


def test_pump_severe_tails_extend_past_v2():
    v2_pump = _V2["anomaly_mix"]["bcu_pump"]
    assert MIX.bcu_pump.effectiveness.low < v2_pump["effectiveness"]["low"]
    assert MIX.bcu_pump.response_delay_s.high > v2_pump["response_delay_s"]["high"]
    assert MIX.bcu_pump.slew_rpm_per_s.low < v2_pump["slew_rpm_per_s"]["low"]


def test_pump_severities_still_straddle_the_flow_envelope():
    vpr = DIMS["rig.plant.volume_per_rev_m3"]
    eff = MIX.bcu_pump.effectiveness
    # 1.0 exactly would read as fault-free to the Scenario label validator.
    assert eff.high < 1.0
    # Faint end INSIDE the envelope: a max-effectiveness pump on the
    # fastest nominal plant still outflows the slowest nominal plant.
    assert eff.high * vpr["high"] > vpr["low"]
    # Severe end OUTSIDE, and by a wider margin than v2's 0.605 vs 0.90:
    # v2's whole band sat inside the observable footprint (supervised
    # AUC 0.514), which is what this re-tune exists to fix.
    assert eff.low * vpr["high"] < vpr["low"]
    assert eff.low * vpr["high"] < 0.5 * vpr["low"]


def test_pump_delay_and_slew_bands_still_straddle_their_envelopes():
    delay_env = DIMS["rig.plant.pump_response_delay_s"]
    delay = MIX.bcu_pump.response_delay_s
    assert delay_env["low"] < delay.low < delay_env["high"] < delay.high
    slew_env = DIMS["rig.plant.pump_slew_rpm_per_s"]
    slew = MIX.bcu_pump.slew_rpm_per_s
    assert slew.low < slew_env["low"] < slew.high < slew_env["high"]


def test_worst_pump_plant_still_dives_before_the_floater_deadline():
    """The severe tail must not manufacture false `abort_floater`s.

    run_watchdog.launch.py's floater deadline is a fixed 360 s from
    arming, and its comment sized it on a 0.55-effectiveness pump.
    v3 goes to 0.35 with 8 s of dead time and a 60 rpm/s ramp, so
    re-derive the worst corner: time to pump the spawn fill (bladder_max)
    down past the neutral volume, which is when the vehicle starts
    sinking at all.
    """
    eff = MIX.bcu_pump.effectiveness.low
    delay_s = MIX.bcu_pump.response_delay_s.high
    slew = MIX.bcu_pump.slew_rpm_per_s.low
    vpr = DIMS["rig.plant.volume_per_rev_m3"]["low"]  # slowest plant
    # Worst corner: fullest spawn, shallowest neutral target.
    swing_m3 = (
        DIMS["rig.plant.bladder_max_m3"]["high"] - DIMS["derive.neutral_volume_m3"]["low"]
    )
    rail_rpm = 3000.0
    rail_flow = (rail_rpm / 60.0) * vpr * eff  # m^3/s at the command rail
    ramp_s = rail_rpm / slew
    ramp_volume = 0.5 * rail_flow * ramp_s  # triangular ramp-up
    if ramp_volume >= swing_m3:
        # Still ramping when it crosses neutral: V(t) = rail_flow t^2 / 2 ramp_s.
        t_neutral = delay_s + (2.0 * swing_m3 * ramp_s / rail_flow) ** 0.5
    else:
        t_neutral = delay_s + ramp_s + (swing_m3 - ramp_volume) / rail_flow
    # Leave the rest of the 360 s base for the sink to MIN_DIVE_M.
    assert t_neutral < 200.0, f"worst-corner time to neutral {t_neutral:.0f} s"


# ---------------------------------------------------------------------------
# biofouling: ceilings re-extended, mass pinned by the derivation
# ---------------------------------------------------------------------------


def test_biofouling_floors_hold_and_ceilings_extend_past_v2():
    v2_bio = _V2["anomaly_mix"]["biofouling"]
    bio = _V3["anomaly_mix"]["biofouling"]
    # Floors stay outside the envelope's worst multiplicative extremes
    # (heave drag ~ x1.36, added mass ~ x1.21 incl. 3-sigma jitter).
    assert bio["drag_mult"]["low"] == v2_bio["drag_mult"]["low"] > 1.36
    assert bio["added_mass_mult"]["low"] == v2_bio["added_mass_mult"]["low"] > 1.21
    assert bio["drag_mult"]["high"] > v2_bio["drag_mult"]["high"]
    assert bio["added_mass_mult"]["high"] > v2_bio["added_mass_mult"]["high"]
    # Unchanged by contract: fouling is baked into the SDF at spawn.
    assert "onset" not in bio
    assert bio["min_climb_margin_m3"] == v2_bio["min_climb_margin_m3"]


def test_biofouling_mass_stays_inside_the_derivation_ceiling():
    """The sampler HARD-FAILS (never resamples) on a non-viable draw.

    check_viability requires
    ``bladder_max - (neutral + m/rho) >= min_climb_margin_m3``; take the
    worst corner of every band involved. This is why v3's fouling mass
    could not follow the drag ceiling upward — 0.20 kg aborts the sweep.
    """
    bio = MIX.biofouling
    bladder_max_low = DIMS["rig.plant.bladder_max_m3"]["low"]
    neutral_high = DIMS["derive.neutral_volume_m3"]["high"]
    rho_low = DIMS["rig.hydrodynamics.fluid_density"]["low"]

    worst_margin = bladder_max_low - (neutral_high + bio.fouling_mass_kg.high / rho_low)
    assert worst_margin >= bio.min_climb_margin_m3, (
        f"worst-corner climb margin {worst_margin:.3e} m^3 < required "
        f"{bio.min_climb_margin_m3:.3e} — generation would hard-fail"
    )
    # And the floor still pushes every fouled run past the envelope
    # ceiling: the neutral-volume band is 8.0e-5 m^3 wide (== 0.080 kg).
    assert bio.fouling_mass_kg.low / rho_low > 8.0e-5


# ---------------------------------------------------------------------------
# Everything else is v2 verbatim
# ---------------------------------------------------------------------------


def test_sensor_block_is_v2_byte_equal():
    # The one class v2's retrain found genuinely observable — it is the
    # cross-campaign comparison anchor, bands AND onset.
    assert _V3["anomaly_mix"]["sensor"] == _V2["anomaly_mix"]["sensor"]


def test_pump_onset_block_is_v2_byte_equal():
    assert _V3["anomaly_mix"]["bcu_pump"]["onset"] == _V2["anomaly_mix"]["bcu_pump"]["onset"]


def test_mission_mix_and_buoyancy_derivation_are_v2_byte_equal():
    assert _V3["mission_mix"] == _V2["mission_mix"]
    assert _V3["buoyancy_derivation"] == _V2["buoyancy_derivation"]


def test_mission_profile_counts_at_2048():
    assert sum(MISSION.weights.values()) == pytest.approx(1.0)
    # 0.55 x 2048 = 1126.4 and 0.45 x 2048 = 921.6: staircase's 0.6
    # remainder takes the leftover run.
    assert profile_counts(MISSION, N) == {
        "sawtooth_plain": 1126,
        "staircase": 922,
    }


def test_dimensions_are_v2_byte_equal():
    # The physics envelope is the thing v2 validated on water; a re-run
    # that moved it would not be a re-run.
    assert set(DIMS) == set(V2_DIMS)
    for path in DIMS:
        assert DIMS[path] == V2_DIMS[path], path


# ---------------------------------------------------------------------------
# Sampler + runner integration (scripts/ imports)
# ---------------------------------------------------------------------------


@needs_scripts
def test_worst_corner_fits_the_recommended_timeout():
    # The deepest sawtooth (40 m, 2 cycles) must fit the header's
    # recommended --per-run-timeout of 18000 s.
    worst = {"target_pressure_pa": 392240.0, "n_oscillations": 2}
    assert run_sweep._scaled_timeout(worst, None) <= 18000.0


@needs_scripts
def test_v3_spec_loads_through_the_sampler():
    spec = lhs_sample.SweepSpec.load(_V3_PATH)
    assert spec.n_samples == N
    assert spec.mission_mix is not None
    assert spec.anomaly_mix is not None
    # The pin dimension survives Dimension validation (uniform-only
    # low == high) and scales to its constant.
    pin = next(d for d in spec.dimensions if d.path == "rig.plant.tank_air_volume_m3")
    assert pin.low == pin.high == 0.0
