"""Tier 1 for the fault wire-param emission (compile.py).

Locks the scenario -> bridge contract for the persistent-fault design:
nominal compiles to inert values, each fault lands on exactly the right
bridge, the retired ladder params are gone, and every stochastic fault
stream gets its own pairwise-distinct derived seed.
"""

from __future__ import annotations

from py_pkg.scenarios import derive_seed
from py_pkg.scenarios.compile import (
    params_for_anomaly_label,
    params_for_bcu_bridge,
    params_for_external_sensor_bridge,
    params_for_imu_bridge,
)
from py_pkg.scenarios.spec.rig import RigScenario
from py_pkg.scenarios.spec.scenario import Scenario


def test_nominal_emission_is_inert_and_ladder_free():
    rig = RigScenario()
    bcu = params_for_bcu_bridge(rig, parent_seed=3)
    ext = params_for_external_sensor_bridge(rig, parent_seed=3)
    imu = params_for_imu_bridge(rig, parent_seed=3)

    assert bcu["fault_effectiveness"] == 1.0
    assert bcu["tank_fault_kind"] == "none"
    assert bcu["tank_fault_magnitude"] == 0.0
    assert bcu["tank_fault_drop_prob"] == 0.0
    assert ext["fault_kind"] == "none"
    assert ext["fault_magnitude"] == 0.0
    assert ext["fault_drop_prob"] == 0.0
    for params in (bcu, ext, imu):
        assert params["comms_drop_prob"] == 0.0

    # The Poisson-ladder wire params are gone.
    for retired in ("fault_mttf_sec", "fault_num_levels", "rng_seed"):
        assert retired not in bcu


def test_fault_routing_lands_on_the_right_bridge():
    rig = RigScenario.model_validate(
        {
            "faults": {
                "sensors": {
                    "tank_pressure": {"kind": "drift", "magnitude": 25.0},
                    "external_pressure": {"kind": "dropout", "drop_prob": 0.6},
                }
            }
        }
    )
    bcu = params_for_bcu_bridge(rig)
    ext = params_for_external_sensor_bridge(rig)
    assert bcu["tank_fault_kind"] == "drift"
    assert bcu["tank_fault_magnitude"] == 25.0
    assert ext["fault_kind"] == "dropout"
    assert ext["fault_drop_prob"] == 0.6

    rig = RigScenario.model_validate(
        {"faults": {"bcu_pump": {"effectiveness": 0.4}, "comms": {"drop_prob": 0.3}}}
    )
    bcu = params_for_bcu_bridge(rig)
    ext = params_for_external_sensor_bridge(rig)
    imu = params_for_imu_bridge(rig)
    assert bcu["fault_effectiveness"] == 0.4
    assert bcu["comms_drop_prob"] == 0.3
    assert ext["comms_drop_prob"] == 0.3
    assert imu["comms_drop_prob"] == 0.3


def test_fault_seed_streams_are_derived_and_pairwise_distinct():
    parent = 42
    bcu = params_for_bcu_bridge(RigScenario(), parent_seed=parent)
    ext = params_for_external_sensor_bridge(RigScenario(), parent_seed=parent)
    imu = params_for_imu_bridge(RigScenario(), parent_seed=parent)

    assert bcu["tank_fault_seed"] == derive_seed(parent, "tank_pressure_fault")
    assert bcu["comms_seed"] == derive_seed(parent, "bcu_comms_drop")
    assert ext["fault_seed"] == derive_seed(parent, "external_pressure_fault")
    assert ext["comms_seed"] == derive_seed(parent, "external_pressure_comms_drop")
    assert imu["comms_seed"] == derive_seed(parent, "imu_comms_drop")

    seeds = [
        bcu["tank_fault_seed"],
        bcu["tank_noise_seed"],
        bcu["comms_seed"],
        ext["fault_seed"],
        ext["noise_seed"],
        ext["comms_seed"],
        imu["noise_seed"],
        imu["comms_seed"],
    ]
    assert len(set(seeds)) == len(seeds)
    for s in seeds:
        assert 0 <= s < 2**63


def test_anomaly_label_params_pass_through():
    scen = Scenario.model_validate(
        {
            "rig": {
                "faults": {
                    "sensors": {
                        "tank_pressure": {"kind": "bias", "magnitude": 8000.0}
                    }
                }
            },
            "anomaly": {
                "anomaly_class": "sensor",
                "channel": "tank_pressure",
                "archetype": "bias",
            },
        }
    )
    assert params_for_anomaly_label(scen) == {
        "anomaly_class": "sensor",
        "channel": "tank_pressure",
        "archetype": "bias",
    }
    assert params_for_anomaly_label(Scenario()) == {
        "anomaly_class": "nominal",
        "channel": "",
        "archetype": "",
    }
