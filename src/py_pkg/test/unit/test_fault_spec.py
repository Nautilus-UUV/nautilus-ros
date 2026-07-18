"""Tier 1 for the persistent-fault schema (rig.faults + the anomaly label).

Locks the contracts of the per-run anomaly-injection design:
- defaults mean "no fault", so nominal scenarios stay valid and inert;
- per-kind validators reject nonsensical severity combinations;
- the retired Poisson-ladder schema (faults.bcu_rpm) fails loudly at
  load — the intended migration forcing function;
- Scenario.anomaly round-trips and is validated against rig.faults, so
  a bag's label stream can never claim "nominal" while a runtime fault
  is configured.
"""

from __future__ import annotations

from typing import get_args

import pytest
from py_pkg.scenarios import load_scenario
from py_pkg.scenarios.spec.rig import (
    BcuPumpFaultSpec,
    CommsFaultSpec,
    FaultsSpec,
    RigScenario,
    SensorFaultKind,
    SensorFaultSpec,
    SensorFaultsSpec,
)
from py_pkg.scenarios.spec.scenario import AnomalyLabelSpec, Scenario


def test_defaults_are_no_fault():
    f = FaultsSpec()
    assert f.bcu_pump.effectiveness == 1.0
    assert f.sensors.external_pressure == SensorFaultSpec()
    assert f.sensors.tank_pressure == SensorFaultSpec()
    assert f.sensors.external_pressure.kind == "none"
    assert f.comms.drop_prob == 0.0
    s = Scenario()
    assert s.anomaly == AnomalyLabelSpec()
    assert s.anomaly.anomaly_class == "nominal"


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0000001, 2.0])
def test_pump_effectiveness_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        BcuPumpFaultSpec(effectiveness=bad)


@pytest.mark.parametrize("ok", [1.0, 0.75, 0.3, 1e-6])
def test_pump_effectiveness_accepts_in_range(ok):
    assert BcuPumpFaultSpec(effectiveness=ok).effectiveness == ok


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "bias", "magnitude": 5000.0},
        {"kind": "bias", "magnitude": -1000.0},
        {"kind": "drift", "magnitude": 5.0},
        {"kind": "drift", "magnitude": -1.0},
        {"kind": "stuck"},
        {"kind": "dropout", "drop_prob": 0.5},
        {"kind": "dropout", "drop_prob": 1.0},
        {"kind": "none"},
    ],
)
def test_sensor_fault_valid_combinations(fields):
    SensorFaultSpec(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "bias"},  # bias needs a magnitude
        {"kind": "bias", "magnitude": 0.0},
        {"kind": "bias", "magnitude": 100.0, "drop_prob": 0.5},
        {"kind": "drift"},  # drift needs a magnitude
        {"kind": "drift", "magnitude": 1.0, "drop_prob": 0.1},
        {"kind": "stuck", "magnitude": 3.0},  # stuck takes no severity
        {"kind": "stuck", "drop_prob": 0.2},
        {"kind": "dropout"},  # dropout needs drop_prob in (0, 1]
        {"kind": "dropout", "drop_prob": 0.0},
        {"kind": "dropout", "drop_prob": 1.5},
        {"kind": "dropout", "drop_prob": 0.5, "magnitude": 1.0},
        {"kind": "none", "magnitude": 1.0},  # none must stay all-default
        {"kind": "none", "drop_prob": 0.1},
        {"kind": "spikes", "magnitude": 1.0},  # not an implemented archetype
    ],
)
def test_sensor_fault_invalid_combinations_raise(fields):
    with pytest.raises(ValueError):
        SensorFaultSpec(**fields)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.0])
def test_comms_drop_prob_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        CommsFaultSpec(drop_prob=bad)


def test_stale_poisson_ladder_yaml_raises():
    # The pre-migration schema must fail loudly, not silently no-op.
    with pytest.raises(ValueError, match="bcu_rpm"):
        RigScenario.model_validate({"faults": {"bcu_rpm": {"mttf_sec": 60.0}}})


def test_label_literals_mirror_the_fault_schema():
    # channel values == the SensorFaultsSpec channels (plus the unset "");
    # archetype values == the implemented kinds minus "none" (plus "").
    chan = set(get_args(AnomalyLabelSpec.model_fields["channel"].annotation))
    assert chan == {""} | set(SensorFaultsSpec.model_fields)
    arch = set(get_args(AnomalyLabelSpec.model_fields["archetype"].annotation))
    assert arch == {""} | (set(get_args(SensorFaultKind)) - {"none"})


def test_label_sensor_requires_channel_and_archetype():
    with pytest.raises(ValueError):
        AnomalyLabelSpec(anomaly_class="sensor")
    with pytest.raises(ValueError):
        AnomalyLabelSpec(anomaly_class="sensor", channel="tank_pressure")
    with pytest.raises(ValueError):
        AnomalyLabelSpec(anomaly_class="nominal", channel="tank_pressure")
    AnomalyLabelSpec(
        anomaly_class="sensor", channel="tank_pressure", archetype="stuck"
    )


def _scenario(faults: dict, anomaly: dict | None = None) -> Scenario:
    data: dict = {"rig": {"faults": faults}}
    if anomaly is not None:
        data["anomaly"] = anomaly
    return Scenario.model_validate(data)


def test_configured_fault_requires_matching_label():
    with pytest.raises(ValueError, match="bcu_pump"):
        _scenario({"bcu_pump": {"effectiveness": 0.5}})
    with pytest.raises(ValueError, match="sensor"):
        _scenario({"sensors": {"tank_pressure": {"kind": "stuck"}}})
    with pytest.raises(ValueError, match="comms"):
        _scenario({"comms": {"drop_prob": 0.4}})


def test_label_requires_matching_fault():
    with pytest.raises(ValueError):
        _scenario({}, {"anomaly_class": "bcu_pump"})
    with pytest.raises(ValueError):
        _scenario(
            {},
            {
                "anomaly_class": "sensor",
                "channel": "tank_pressure",
                "archetype": "stuck",
            },
        )
    with pytest.raises(ValueError):
        _scenario({}, {"anomaly_class": "comms"})


def test_consistent_label_and_fault_pass():
    s = _scenario(
        {"bcu_pump": {"effectiveness": 0.5}}, {"anomaly_class": "bcu_pump"}
    )
    assert s.anomaly.anomaly_class == "bcu_pump"
    s = _scenario(
        {"sensors": {"external_pressure": {"kind": "bias", "magnitude": 3000.0}}},
        {
            "anomaly_class": "sensor",
            "channel": "external_pressure",
            "archetype": "bias",
        },
    )
    assert s.rig.faults.sensors.external_pressure.magnitude == 3000.0
    _scenario({"comms": {"drop_prob": 0.4}}, {"anomaly_class": "comms"})


def test_sensor_label_channel_and_archetype_must_match():
    fault = {"sensors": {"tank_pressure": {"kind": "stuck"}}}
    with pytest.raises(ValueError, match="channel"):
        _scenario(
            fault,
            {
                "anomaly_class": "sensor",
                "channel": "external_pressure",
                "archetype": "stuck",
            },
        )
    with pytest.raises(ValueError, match="archetype"):
        _scenario(
            fault,
            {
                "anomaly_class": "sensor",
                "channel": "tank_pressure",
                "archetype": "bias",
            },
        )


def test_sensor_label_requires_exactly_one_faulted_channel():
    with pytest.raises(ValueError, match="exactly one"):
        _scenario(
            {
                "sensors": {
                    "tank_pressure": {"kind": "stuck"},
                    "external_pressure": {"kind": "stuck"},
                }
            },
            {
                "anomaly_class": "sensor",
                "channel": "tank_pressure",
                "archetype": "stuck",
            },
        )


def test_biofouling_label_is_trusted_but_excludes_runtime_faults():
    # Pure hydro overlay: no rig.faults marker, label trusted as authored.
    s = _scenario({}, {"anomaly_class": "biofouling"})
    assert s.anomaly.anomaly_class == "biofouling"
    # But it may not coexist with a configured runtime fault.
    with pytest.raises(ValueError):
        _scenario(
            {"bcu_pump": {"effectiveness": 0.5}}, {"anomaly_class": "biofouling"}
        )


def test_baseline_library_yaml_is_a_labeled_pump_fault(library_scenario_path):
    scen = load_scenario(library_scenario_path("baseline.yaml"))
    assert scen.rig.faults.bcu_pump.effectiveness == 0.6
    assert scen.anomaly.anomaly_class == "bcu_pump"


@pytest.mark.parametrize(
    "library_yaml",
    ["nominal.yaml", "nominal_with_hydrodynamics.yaml", "suggested_lhs_0071.yaml"],
)
def test_no_fault_library_yamls_stay_nominal(library_scenario_path, library_yaml):
    scen = load_scenario(library_scenario_path(library_yaml))
    assert scen.rig.faults == FaultsSpec()
    assert scen.anomaly.anomaly_class == "nominal"
