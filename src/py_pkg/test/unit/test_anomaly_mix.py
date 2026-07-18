"""Tier 1 for the per-run anomaly-mix sampling (scenarios/anomaly.py).

Locks the validation-sweep contracts: stratified exact class counts,
seeded determinism, stream decoupling (band edits never reshuffle
assignments or sibling classes), apply_anomaly writing exactly the
schema-consistent fields, and the biofouling buoyancy interplay
(fouling mass -> neutral shift -> relaxed climb margin).
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mix_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="sum"):
        _mix(weights={"nominal": 0.9, "bcu_pump": 0.05})
    with pytest.raises(ValueError, match="unknown"):
        _mix(weights={"nominal": 0.5, "gremlins": 0.5})
    with pytest.raises(ValueError, match="config block"):
        AnomalyMixSpec.model_validate(
            {"weights": {"nominal": 0.9, "comms": 0.1}}
        )
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
        assert faults["bcu_pump"]["effectiveness"] == assignment.severity[
            "effectiveness"
        ]
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
    derived = buoyancy.derive_trim_masses(
        1000.0, v_n, spawn_volume_m3=bladder_max
    )
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
