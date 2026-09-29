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
    params_for_auto_mission,
    params_for_bcu_bridge,
    params_for_external_sensor_bridge,
    params_for_imu_bridge,
)
from py_pkg.scenarios.spec.rig import PlantSpec, RigScenario
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
    # The labeled class's schedule rides along under schedule_*; here the
    # bias fault carries the default (step at t=0) envelope.
    default_schedule = {
        "schedule_onset_s": 0.0,
        "schedule_shape": "step",
        "schedule_ramp_s": 0.0,
        "schedule_period_s": 0.0,
        "schedule_duty": 0.5,
    }
    assert params_for_anomaly_label(scen) == {
        "anomaly_class": "sensor",
        "channel": "tank_pressure",
        "archetype": "bias",
        **default_schedule,
    }
    assert params_for_anomaly_label(Scenario()) == {
        "anomaly_class": "nominal",
        "channel": "",
        "archetype": "",
        **default_schedule,
    }


# ---------------------------------------------------------------------------
# Schedule + pump-overshoot + auto-mission wire params (v2)
# ---------------------------------------------------------------------------

_DEFAULT_SCHEDULE_KEYS = ("onset_s", "shape", "ramp_s", "period_s", "duty")


def _default_schedule(prefix: str) -> dict:
    return {
        f"{prefix}onset_s": 0.0,
        f"{prefix}shape": "step",
        f"{prefix}ramp_s": 0.0,
        f"{prefix}period_s": 0.0,
        f"{prefix}duty": 0.5,
    }


def test_bcu_bridge_emits_overshoot_and_both_schedule_blocks():
    bcu = params_for_bcu_bridge(RigScenario())
    # Pump overshoot rides alongside the other pump-transient knobs.
    assert bcu["pump_overshoot_frac"] == 0.0
    # The pump's own schedule under the `fault_` prefix (defaults inert).
    for k, v in _default_schedule("fault_").items():
        assert bcu[k] == v
    # PLUS the tank sensor channel's schedule under `tank_fault_`.
    for k, v in _default_schedule("tank_fault_").items():
        assert bcu[k] == v


def test_external_sensor_bridge_emits_schedule_block():
    ext = params_for_external_sensor_bridge(RigScenario())
    for k, v in _default_schedule("fault_").items():
        assert ext[k] == v


def test_scheduled_pump_fault_round_trips_into_params():
    rig = RigScenario.model_validate(
        {
            "faults": {
                "bcu_pump": {
                    "effectiveness": 0.6,
                    "schedule": {"shape": "ramp", "onset_s": 30.0, "ramp_s": 60.0},
                }
            }
        }
    )
    bcu = params_for_bcu_bridge(rig)
    assert bcu["fault_effectiveness"] == 0.6
    assert bcu["fault_shape"] == "ramp"
    assert bcu["fault_onset_s"] == 30.0
    assert bcu["fault_ramp_s"] == 60.0
    assert bcu["fault_period_s"] == 0.0
    assert bcu["fault_duty"] == 0.5


def test_pump_overshoot_frac_round_trips_into_params():
    rig = RigScenario.model_validate({"plant": {"pump_overshoot_frac": 0.0377}})
    assert params_for_bcu_bridge(rig)["pump_overshoot_frac"] == 0.0377


def test_anomaly_label_picks_pump_schedule_for_pump_labeled_run():
    scen = Scenario.model_validate(
        {
            "rig": {
                "faults": {
                    "bcu_pump": {
                        "effectiveness": 0.5,
                        "schedule": {"shape": "step", "onset_s": 20.0},
                    }
                }
            },
            "anomaly": {"anomaly_class": "bcu_pump"},
        }
    )
    label = params_for_anomaly_label(scen)
    assert label["schedule_shape"] == "step"
    assert label["schedule_onset_s"] == 20.0


def test_anomaly_label_picks_labeled_sensor_channel_schedule():
    scen = Scenario.model_validate(
        {
            "rig": {
                "faults": {
                    "sensors": {
                        "tank_pressure": {
                            "kind": "drift",
                            "magnitude": 25.0,
                            "schedule": {"shape": "step", "onset_s": 12.0},
                        }
                    }
                }
            },
            "anomaly": {
                "anomaly_class": "sensor",
                "channel": "tank_pressure",
                "archetype": "drift",
            },
        }
    )
    label = params_for_anomaly_label(scen)
    assert label["schedule_onset_s"] == 12.0
    assert label["schedule_shape"] == "step"


def test_auto_mission_carries_plant_endpoints():
    # Defaults mirror PlantSpec.
    default = params_for_auto_mission(Scenario())
    assert default["dive_init_tank_empty_pa"] == PlantSpec().tank_pressure_empty_pa
    assert default["dive_init_tank_full_pa"] == PlantSpec().tank_pressure_full_pa

    # Sampled plant truth is carried verbatim (not the defaults) -- the sim
    # surrogate for the operator's pre-dive tank measurement.
    scen = Scenario.model_validate(
        {
            "rig": {
                "plant": {
                    "tank_pressure_empty_pa": 95_000.0,
                    "tank_pressure_full_pa": 185_000.0,
                }
            }
        }
    )
    sampled = params_for_auto_mission(scen)
    assert sampled["dive_init_tank_empty_pa"] == 95_000.0
    assert sampled["dive_init_tank_full_pa"] == 185_000.0
    assert sampled["dive_init_tank_empty_pa"] != PlantSpec().tank_pressure_empty_pa
