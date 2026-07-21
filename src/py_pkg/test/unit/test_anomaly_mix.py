"""Tier 1 for the per-run anomaly-mix sampling (scenarios/anomaly.py).

Locks the validation-sweep contracts: stratified exact class counts,
seeded determinism, stream decoupling (band edits never reshuffle
assignments or sibling classes), apply_anomaly writing exactly the
schema-consistent fields, the biofouling buoyancy interplay
(fouling mass -> neutral shift -> relaxed climb margin), and the
append-only fault-onset draws (an ``onset:`` block never moves a
seed's severities; non-immediate onsets land as validating
``schedule:`` blocks, drift/stuck forced to step).
"""

from __future__ import annotations

import copy
from collections import Counter

import pytest
import yaml
from py_pkg.scenarios import buoyancy
from py_pkg.scenarios.anomaly import (
    ANOMALY_CLASSES,
    FOULING_ADDED_MASS_SLOTS,
    FOULING_DRAG_SLOTS,
    AnomalyMixSpec,
    Band,
    OnsetMix,
    apply_anomaly,
    assign_classes,
    class_counts,
    draw_assignment,
    fouled_neutral_volume,
)
from py_pkg.scenarios.spec.rig import HydrodynamicsSpec
from py_pkg.scenarios.spec.scenario import Scenario


def _mix_dict() -> dict:
    return {
        "weights": {
            "nominal": 0.80,
            "bcu_pump": 0.05,
            "sensor": 0.05,
            "comms": 0.05,
            "biofouling": 0.05,
        },
        "bcu_pump": {
            "effectiveness": {"low": 0.30, "high": 0.75},
            "response_delay_s": {"low": 2.5, "high": 5.0},
            "slew_rpm_per_s": {"low": 100.0, "high": 250.0},
        },
        "sensor": {
            "channels": ["external_pressure", "tank_pressure"],
            "archetypes": ["bias", "drift", "stuck", "dropout"],
            "bands": {
                "external_pressure": {
                    "bias": {"low": 1000.0, "high": 10000.0, "signed": True},
                    "drift": {"low": 1.0, "high": 10.0, "signed": True},
                    "dropout": {"low": 0.2, "high": 0.8},
                },
                "tank_pressure": {
                    "bias": {"low": 5000.0, "high": 20000.0, "signed": True},
                    "drift": {"low": 5.0, "high": 50.0, "signed": True},
                    "dropout": {"low": 0.2, "high": 0.8},
                },
            },
        },
        "comms": {"drop_prob": {"low": 0.30, "high": 0.80}},
        "biofouling": {
            "drag_mult": {"low": 1.6, "high": 2.5},
            "added_mass_mult": {"low": 1.3, "high": 1.5},
            "fouling_mass_kg": {"low": 0.09, "high": 0.18},
            "min_climb_margin_m3": 3.0e-5,
        },
    }


def _mix(**edits) -> AnomalyMixSpec:
    d = _mix_dict()
    d.update(edits)
    return AnomalyMixSpec.model_validate(d)


MIX = _mix()


def _pump_onset_dict(p_immediate: float = 0.5) -> dict:
    return {
        "p_immediate": p_immediate,
        "onset_frac": {"low": 0.05, "high": 0.60},
        "shape_weights": {"step": 0.5, "ramp": 0.3, "intermittent": 0.2},
        "ramp_s": {"low": 30.0, "high": 300.0},
        "period_s": {"low": 20.0, "high": 120.0},
        "duty": {"low": 0.3, "high": 0.7},
    }


def _sensor_onset_dict(p_immediate: float = 0.5) -> dict:
    # No ramp weight: drift/stuck accept only step schedules, so the
    # sensor class never authors a ramp band (mirrors the v2 spec).
    return {
        "p_immediate": p_immediate,
        "onset_frac": {"low": 0.05, "high": 0.60},
        "shape_weights": {"step": 0.8, "intermittent": 0.2},
        "period_s": {"low": 20.0, "high": 120.0},
        "duty": {"low": 0.3, "high": 0.7},
    }


def _onset_mix(p_immediate: float = 0.5) -> AnomalyMixSpec:
    d = _mix_dict()
    d["bcu_pump"]["onset"] = _pump_onset_dict(p_immediate)
    d["sensor"]["onset"] = _sensor_onset_dict(p_immediate)
    return AnomalyMixSpec.model_validate(d)


# Any positive stand-in for the run's expected mission duration.
DURATION_S = 2000.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mix_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="sum"):
        _mix(weights={"nominal": 0.9, "bcu_pump": 0.05})
    with pytest.raises(ValueError, match="unknown"):
        _mix(weights={"nominal": 0.5, "gremlins": 0.5})
    with pytest.raises(ValueError, match="config block"):
        AnomalyMixSpec.model_validate({"weights": {"nominal": 0.9, "comms": 0.1}})
    bad_sensor = _mix_dict()
    del bad_sensor["sensor"]["bands"]["tank_pressure"]["drift"]
    with pytest.raises(ValueError, match="missing band"):
        AnomalyMixSpec.model_validate(bad_sensor)
    with pytest.raises(ValueError, match="exceed 1.0"):
        _mix(
            biofouling={
                "drag_mult": {"low": 0.9, "high": 2.0},
                "added_mass_mult": {"low": 1.3, "high": 1.5},
                "fouling_mass_kg": {"low": 0.09, "high": 0.18},
            }
        )
    with pytest.raises(ValueError, match="inside"):
        _mix(comms={"drop_prob": {"low": 0.5, "high": 1.0}})


def test_band_semantics():
    with pytest.raises(ValueError):
        Band(low=2.0, high=1.0)
    # low == high pins a fixed severity — the "one canonical value" switch.
    import random

    fixed = Band(low=0.5, high=0.5)
    assert fixed.draw(random.Random(1)) == 0.5
    signed = Band(low=1.0, high=1.0, signed=True)
    draws = {signed.draw(random.Random(s)) for s in range(20)}
    assert draws == {1.0, -1.0}


# ---------------------------------------------------------------------------
# Stratified assignment
# ---------------------------------------------------------------------------


def test_class_counts_exact_320_and_20():
    assert class_counts(MIX, 320) == {
        "nominal": 256,
        "bcu_pump": 16,
        "sensor": 16,
        "comms": 16,
        "biofouling": 16,
    }
    assert class_counts(MIX, 20) == {
        "nominal": 16,
        "bcu_pump": 1,
        "sensor": 1,
        "comms": 1,
        "biofouling": 1,
    }


@pytest.mark.parametrize("n", [1, 3, 7, 100, 256, 319])
def test_class_counts_always_sum_to_n(n):
    counts = class_counts(MIX, n)
    assert sum(counts.values()) == n
    assert all(c >= 0 for c in counts.values())


def test_assign_classes_is_a_seeded_permutation_of_the_counts():
    a = assign_classes(MIX, parent_seed=813, n=320)
    b = assign_classes(MIX, parent_seed=813, n=320)
    c = assign_classes(MIX, parent_seed=814, n=320)
    assert a == b  # deterministic
    assert a != c  # seed moves the shuffle
    assert Counter(a) == class_counts(MIX, 320)
    assert Counter(c) == class_counts(MIX, 320)  # counts invariant


# ---------------------------------------------------------------------------
# Severity draws + decoupling
# ---------------------------------------------------------------------------


def test_draw_assignment_deterministic_per_run():
    a = draw_assignment(MIX, 813, 42, "bcu_pump")
    b = draw_assignment(MIX, 813, 42, "bcu_pump")
    c = draw_assignment(MIX, 813, 43, "bcu_pump")
    assert a == b
    assert a.severity != c.severity  # per-run stream
    assert 0.30 <= a.severity["effectiveness"] <= 0.75
    assert 2.5 <= a.severity["response_delay_s"] <= 5.0
    assert 100.0 <= a.severity["slew_rpm_per_s"] <= 250.0


def test_sensor_draw_stays_inside_the_declared_space():
    seen_archetypes = set()
    for idx in range(200):
        a = draw_assignment(MIX, 1, idx, "sensor")
        assert a.channel in ("external_pressure", "tank_pressure")
        assert a.archetype in ("bias", "drift", "stuck", "dropout")
        seen_archetypes.add(a.archetype)
        if a.archetype == "stuck":
            assert a.severity == {}
        elif a.archetype == "dropout":
            assert 0.2 <= a.severity["drop_prob"] <= 0.8
        else:
            band = MIX.sensor.bands[a.channel][a.archetype]
            assert band.low <= abs(a.severity["magnitude"]) <= band.high
    assert seen_archetypes == {"bias", "drift", "stuck", "dropout"}


def test_band_edits_do_not_reshuffle_or_leak_across_classes():
    edited = _mix_dict()
    edited["sensor"]["bands"]["tank_pressure"]["bias"] = {
        "low": 7777.0,
        "high": 8888.0,
        "signed": True,
    }
    mix_b = AnomalyMixSpec.model_validate(edited)

    # Same weights -> identical stratified assignment.
    assert assign_classes(MIX, 813, 320) == assign_classes(mix_b, 813, 320)
    # Non-sensor severities are untouched by a sensor band edit.
    for idx in (3, 57, 200):
        for cls in ("nominal", "bcu_pump", "comms", "biofouling"):
            assert draw_assignment(MIX, 813, idx, cls) == draw_assignment(
                mix_b, 813, idx, cls
            )


# ---------------------------------------------------------------------------
# apply_anomaly
# ---------------------------------------------------------------------------


def _base_scenario() -> dict:
    # Physics-mode-shaped base: hydro block present, defaults elsewhere.
    return {
        "seed": 4,
        "rig": {"hydrodynamics": HydrodynamicsSpec().model_dump()},
    }


def test_nominal_assignment_is_a_byte_identical_noop():
    scenario = _base_scenario()
    before = yaml.safe_dump(scenario, sort_keys=False)
    apply_anomaly(scenario, draw_assignment(MIX, 813, 0, "nominal"))
    assert yaml.safe_dump(scenario, sort_keys=False) == before


@pytest.mark.parametrize("cls", ["bcu_pump", "sensor", "comms"])
def test_fault_classes_write_schema_consistent_scenarios(cls):
    scenario = _base_scenario()
    assignment = draw_assignment(MIX, 813, 11, cls)
    apply_anomaly(scenario, assignment)

    # The emitted dict must pass the full Scenario validation — including
    # the label<->faults consistency validator.
    scen = Scenario.model_validate(scenario)
    assert scen.anomaly.anomaly_class == cls

    faults = scenario["rig"]["faults"]
    if cls == "bcu_pump":
        assert (
            faults["bcu_pump"]["effectiveness"] == assignment.severity["effectiveness"]
        )
        assert (
            scenario["rig"]["plant"]["pump_response_delay_s"]
            == assignment.severity["response_delay_s"]
        )
    elif cls == "comms":
        assert faults["comms"]["drop_prob"] == assignment.severity["drop_prob"]
    else:
        channel = faults["sensors"][assignment.channel]
        assert channel["kind"] == assignment.archetype
        assert scen.anomaly.channel == assignment.channel
        assert scen.anomaly.archetype == assignment.archetype


def test_biofouling_scales_exactly_the_fouling_slots():
    scenario = _base_scenario()
    clean = HydrodynamicsSpec().model_dump()
    assignment = draw_assignment(MIX, 813, 29, "biofouling")
    apply_anomaly(scenario, assignment)

    fouled = scenario["rig"]["hydrodynamics"]
    drag_k = assignment.severity["drag_mult"]
    mass_k = assignment.severity["added_mass_mult"]
    for slot in FOULING_DRAG_SLOTS:
        assert fouled[slot] == pytest.approx(clean[slot] * drag_k)
    for slot in FOULING_ADDED_MASS_SLOTS:
        assert fouled[slot] == pytest.approx(clean[slot] * mass_k)
    # Everything else in the hydro block is untouched — the derivation
    # owns density/trim/spawn; fins and stalls are not fouling surfaces.
    touched = set(FOULING_DRAG_SLOTS) | set(FOULING_ADDED_MASS_SLOTS)
    for key, value in clean.items():
        if key not in touched:
            assert fouled[key] == value, key
    # Label is trusted for biofouling and validates as a full Scenario.
    scen = Scenario.model_validate(scenario)
    assert scen.anomaly.anomaly_class == "biofouling"


def test_biofouling_without_hydro_block_raises():
    with pytest.raises(ValueError, match="hydrodynamics"):
        apply_anomaly({"seed": 1}, draw_assignment(MIX, 813, 29, "biofouling"))


# ---------------------------------------------------------------------------
# Fault onsets
# ---------------------------------------------------------------------------


def test_onset_mix_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="p_immediate"):
        OnsetMix.model_validate({**_pump_onset_dict(), "p_immediate": 1.5})
    with pytest.raises(ValueError, match="onset_frac"):
        OnsetMix.model_validate(
            {**_pump_onset_dict(), "onset_frac": {"low": 0.05, "high": 1.2}}
        )
    with pytest.raises(ValueError, match="sum"):
        OnsetMix.model_validate(
            {**_pump_onset_dict(), "shape_weights": {"step": 0.5, "ramp": 0.4}}
        )
    # A weighted shape without its param band is unresolvable...
    missing_ramp = _pump_onset_dict()
    del missing_ramp["ramp_s"]
    with pytest.raises(ValueError, match="no ramp_s band"):
        OnsetMix.model_validate(missing_ramp)
    # ...and a param band for a zero-weight shape is a dead config.
    with pytest.raises(ValueError, match="no weight"):
        OnsetMix.model_validate(
            {**_sensor_onset_dict(), "ramp_s": {"low": 30.0, "high": 300.0}}
        )


def test_biofouling_mix_rejects_an_onset_block():
    # Fouling is baked into the SDF at spawn — it cannot switch on
    # mid-run, so BiofoulingMix has no onset field (StrictModel forbids).
    d = _mix_dict()
    d["biofouling"]["onset"] = _sensor_onset_dict()
    with pytest.raises(ValueError):
        AnomalyMixSpec.model_validate(d)


def test_onset_block_is_append_only_for_severities():
    # The load-bearing v2 contract: a given seed must produce identical
    # severity values with and without the onset block.
    onset_mix = _onset_mix()
    for idx in range(60):
        for cls in ("bcu_pump", "sensor"):
            plain = draw_assignment(MIX, 813, idx, cls)
            with_onset = draw_assignment(
                onset_mix, 813, idx, cls, expected_duration_s=DURATION_S
            )
            assert with_onset.severity == plain.severity
            assert with_onset.channel == plain.channel
            assert with_onset.archetype == plain.archetype


def test_p_immediate_boundary_semantics():
    always = _onset_mix(p_immediate=1.0)
    never = _onset_mix(p_immediate=0.0)
    for idx in range(40):
        for cls in ("bcu_pump", "sensor"):
            a = draw_assignment(always, 813, idx, cls, expected_duration_s=DURATION_S)
            assert a.onset == {"immediate": True}
            b = draw_assignment(never, 813, idx, cls, expected_duration_s=DURATION_S)
            assert "immediate" not in b.onset
            assert 0.05 <= b.onset["onset_frac"] <= 0.60
            assert b.onset["onset_s"] == pytest.approx(
                b.onset["onset_frac"] * DURATION_S
            )
            assert b.onset["shape"] in ("step", "ramp", "intermittent")


def test_non_immediate_onset_without_duration_raises():
    with pytest.raises(ValueError, match="expected_duration_s"):
        draw_assignment(_onset_mix(p_immediate=0.0), 813, 0, "bcu_pump")


def test_drawn_onset_lands_in_the_labeled_fault_block():
    never = _onset_mix(p_immediate=0.0)
    for idx in range(30):
        for cls in ("bcu_pump", "sensor"):
            scenario = _base_scenario()
            assignment = draw_assignment(
                never, 813, idx, cls, expected_duration_s=DURATION_S
            )
            apply_anomaly(scenario, assignment)
            faults = scenario["rig"]["faults"]
            if cls == "bcu_pump":
                schedule = faults["bcu_pump"]["schedule"]
            else:
                schedule = faults["sensors"][assignment.channel]["schedule"]
            assert schedule["onset_s"] == pytest.approx(assignment.onset["onset_s"])
            assert schedule["shape"] == assignment.onset["shape"]
            # Shape params travel iff the shape needs them (the spec
            # validators reject e.g. a ramp_s on a step).
            if schedule["shape"] == "ramp":
                assert schedule["ramp_s"] > 0.0
                assert "period_s" not in schedule
            elif schedule["shape"] == "intermittent":
                assert schedule["period_s"] > 0.0
                assert 0.0 < schedule["duty"] < 1.0
                assert "ramp_s" not in schedule
            else:
                assert set(schedule) == {"onset_s", "shape"}
            # The generated dict validates through the full Scenario
            # schema — including the FaultScheduleSpec shape validators.
            scen = Scenario.model_validate(scenario)
            assert scen.anomaly.anomaly_class == cls


def test_immediate_onset_keeps_the_v1_byte_identical_yaml():
    always = _onset_mix(p_immediate=1.0)
    for idx in (0, 7, 19):
        for cls in ("bcu_pump", "sensor"):
            plain, with_onset = _base_scenario(), _base_scenario()
            apply_anomaly(plain, draw_assignment(MIX, 813, idx, cls))
            apply_anomaly(
                with_onset,
                draw_assignment(always, 813, idx, cls, expected_duration_s=DURATION_S),
            )
            assert yaml.safe_dump(with_onset, sort_keys=False) == yaml.safe_dump(
                plain, sort_keys=False
            )


def test_drift_and_stuck_archetypes_force_step_schedules():
    # All weight on intermittent so the force is guaranteed to engage.
    d = _mix_dict()
    d["sensor"]["onset"] = {
        "p_immediate": 0.0,
        "onset_frac": {"low": 0.2, "high": 0.4},
        "shape_weights": {"intermittent": 1.0},
        "period_s": {"low": 20.0, "high": 120.0},
        "duty": {"low": 0.3, "high": 0.7},
    }
    mix = AnomalyMixSpec.model_validate(d)
    seen = set()
    for idx in range(80):
        a = draw_assignment(mix, 813, idx, "sensor", expected_duration_s=DURATION_S)
        seen.add(a.archetype)
        if a.archetype in ("drift", "stuck"):
            assert a.onset["shape"] == "step"
            assert "period_s" not in a.onset and "duty" not in a.onset
        else:
            assert a.onset["shape"] == "intermittent"
        # Whatever the archetype, the emitted scenario must validate.
        scenario = _base_scenario()
        apply_anomaly(scenario, a)
        Scenario.model_validate(scenario)
    assert {"drift", "stuck"} <= seen


def test_record_carries_the_onset():
    never = _onset_mix(p_immediate=0.0)
    rec = draw_assignment(
        never, 813, 3, "bcu_pump", expected_duration_s=DURATION_S
    ).record()
    assert rec["onset"]["onset_s"] == pytest.approx(
        rec["onset"]["onset_frac"] * DURATION_S
    )
    assert rec["onset"]["shape"] in ("step", "ramp", "intermittent")
    # An immediate draw is still recorded explicitly...
    always = _onset_mix(p_immediate=1.0)
    rec = draw_assignment(
        always, 813, 3, "bcu_pump", expected_duration_s=DURATION_S
    ).record()
    assert rec["onset"] == {"immediate": True}
    # ...while onset-free (v1) mixes keep the v1 record shape exactly.
    assert "onset" not in draw_assignment(MIX, 813, 3, "bcu_pump").record()


# ---------------------------------------------------------------------------
# Biofouling buoyancy interplay
# ---------------------------------------------------------------------------


def test_fouled_neutral_volume_math():
    assert fouled_neutral_volume(2.0e-3, 0.1, 1000.0) == pytest.approx(2.1e-3)
    assert fouled_neutral_volume(2.0e-3, 0.0, 1000.0) == 2.0e-3
    with pytest.raises(ValueError):
        fouled_neutral_volume(2.0e-3, -0.1, 1000.0)
    with pytest.raises(ValueError):
        fouled_neutral_volume(2.0e-3, 0.1, 0.0)


def test_worst_corner_fouled_run_needs_the_relaxed_climb_margin():
    # Worst sweep corner: envelope-max neutral volume + band-max fouling
    # mass + band-min bladder ceiling (validation_mix_v1 values).
    v_n = fouled_neutral_volume(2.135e-3, 0.18, 1000.0)
    bladder_min, bladder_max = 0.0008, 0.00235
    derived = buoyancy.derive_trim_masses(1000.0, v_n, spawn_volume_m3=bladder_max)
    relaxed = buoyancy.check_viability(
        derived,
        bladder_min,
        bladder_max,
        min_climb_margin_m3=3.0e-5,
    )
    assert relaxed.ok, relaxed.reasons
    # The nominal margin would reject it — the relaxed margin is
    # load-bearing, i.e. a barely-climbing vehicle IS the signature.
    nominal_margin = buoyancy.check_viability(derived, bladder_min, bladder_max)
    assert not nominal_margin.ok


def test_over_cap_fouling_mass_hard_fails_viability():
    # Fouling beyond the authored cap eats the whole climb margin.
    v_n = fouled_neutral_volume(2.135e-3, 0.30, 1000.0)
    derived = buoyancy.derive_trim_masses(1000.0, v_n, spawn_volume_m3=0.00235)
    viability = buoyancy.check_viability(
        derived, 0.0008, 0.00235, min_climb_margin_m3=3.0e-5
    )
    assert not viability.ok


def test_classes_tuple_is_the_schema_vocabulary():
    # The label Literal, the mix classes, and this module must agree.
    from typing import get_args

    from py_pkg.scenarios.spec.scenario import AnomalyLabelSpec

    label_classes = set(
        get_args(AnomalyLabelSpec.model_fields["anomaly_class"].annotation)
    )
    assert set(ANOMALY_CLASSES) == label_classes
