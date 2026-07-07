"""Tier 1 tests for NoiseSpec + its compile-time wiring.

Locks three contracts: (1) the spec defaults ARE the lake-fitted values
(change-detection against noise_characterization.json's provenance),
(2) the `enabled` gate compiles to all-zero wire params so bridges never
see a boolean, (3) each noise stream draws from its own derived seed,
disjoint from the fault stream.
"""

import pytest
import yaml
from py_pkg.scenarios.compile import (
    params_for_bcu_bridge,
    params_for_external_sensor_bridge,
    params_for_imu_bridge,
)
from py_pkg.scenarios.loader import load_scenario
from py_pkg.scenarios.seed import derive_seed
from py_pkg.scenarios.spec.rig import NoiseSpec
from py_pkg.scenarios.spec.scenario import Scenario

# Adopted values from UG-anomaly_detection/lake_test_jun24/investigation/
# noise_characterization.json (2026-06-24 lake test). If a re-fit changes
# them, update NoiseSpec defaults, the library YAMLs, and this table
# together.
FITTED_ACCEL_SIGMA = (0.01215, 0.02372, 0.009544)
FITTED_GYRO_SIGMA = (0.006541, 0.001033, 0.0008779)
FITTED_EXTERNAL = {"sigma_pa": 0.0, "quantization_pa": 100.0}
FITTED_TANK = {"sigma_pa": 353.0, "quantization_pa": 600.0}


def test_defaults_are_the_lake_fitted_values():
    n = NoiseSpec()
    assert n.enabled is True
    assert n.imu.accel_sigma_mps2 == FITTED_ACCEL_SIGMA
    assert n.imu.gyro_sigma_rads == FITTED_GYRO_SIGMA
    assert n.external_pressure.sigma_pa == FITTED_EXTERNAL["sigma_pa"]
    assert n.external_pressure.quantization_pa == FITTED_EXTERNAL["quantization_pa"]
    assert n.tank_pressure.sigma_pa == FITTED_TANK["sigma_pa"]
    assert n.tank_pressure.quantization_pa == FITTED_TANK["quantization_pa"]


def test_noise_yaml_round_trip_and_strictness():
    scen = Scenario(
        **yaml.safe_load(
            """
seed: 3
rig:
  noise:
    enabled: true
    imu:
      accel_sigma_mps2: [0.01, 0.02, 0.03]
      gyro_sigma_rads: [0.001, 0.002, 0.003]
    external_pressure:
      sigma_pa: 5.0
      quantization_pa: 50.0
    tank_pressure:
      sigma_pa: 100.0
      quantization_pa: 300.0
"""
        )
    )
    assert scen.rig.noise.imu.accel_sigma_mps2 == (0.01, 0.02, 0.03)
    assert scen.rig.noise.tank_pressure.quantization_pa == 300.0

    with pytest.raises(Exception):
        Scenario(**yaml.safe_load("rig:\n  noise:\n    not_a_field: 1\n"))


def test_enabled_false_compiles_to_all_zero_params():
    scen = Scenario(**yaml.safe_load("rig:\n  noise:\n    enabled: false\n"))
    imu = params_for_imu_bridge(scen.rig, parent_seed=scen.seed)
    ext = params_for_external_sensor_bridge(scen.rig, parent_seed=scen.seed)
    bcu = params_for_bcu_bridge(scen.rig, parent_seed=scen.seed)
    assert imu["noise_accel_sigma"] == [0.0, 0.0, 0.0]
    assert imu["noise_gyro_sigma"] == [0.0, 0.0, 0.0]
    assert ext["noise_sigma_pa"] == 0.0 and ext["noise_quantization_pa"] == 0.0
    assert bcu["tank_noise_sigma_pa"] == 0.0
    assert bcu["tank_noise_quantization_pa"] == 0.0


def test_enabled_true_compiles_fitted_values_and_seeds():
    scen = Scenario(**yaml.safe_load("seed: 7\n"))
    imu = params_for_imu_bridge(scen.rig, parent_seed=7)
    ext = params_for_external_sensor_bridge(scen.rig, parent_seed=7)
    bcu = params_for_bcu_bridge(scen.rig, parent_seed=7)
    assert imu["noise_accel_sigma"] == list(FITTED_ACCEL_SIGMA)
    assert imu["noise_gyro_sigma"] == list(FITTED_GYRO_SIGMA)
    assert ext["noise_quantization_pa"] == 100.0
    assert bcu["tank_noise_sigma_pa"] == 353.0
    assert imu["noise_seed"] == derive_seed(7, "imu_noise")
    assert ext["noise_seed"] == derive_seed(7, "external_pressure_noise")
    assert bcu["tank_noise_seed"] == derive_seed(7, "tank_pressure_noise")


def test_noise_seed_streams_are_pairwise_distinct_and_int64():
    ids = [
        "bcu_rpm_fault",
        "imu_noise",
        "external_pressure_noise",
        "tank_pressure_noise",
    ]
    seeds = [derive_seed(0, i) for i in ids]
    assert len(set(seeds)) == len(seeds)
    for s in seeds:
        assert 0 <= s < 2**63


# All three library scenarios restate the noise block explicitly (they
# are fully-explicit references, like the plant block that
# test_scenario_robot_specs_parity guards) — this keeps every copy
# locked to the spec defaults so a re-fit can't leave one behind.
@pytest.mark.parametrize(
    "library_yaml",
    ["nominal.yaml", "baseline.yaml", "nominal_with_hydrodynamics.yaml"],
)
def test_installed_library_yamls_are_noise_on_at_fitted_values(
    library_scenario_path, library_yaml
):
    scen = load_scenario(library_scenario_path(library_yaml))
    assert scen.rig.noise.enabled is True
    assert scen.rig.noise == NoiseSpec()
