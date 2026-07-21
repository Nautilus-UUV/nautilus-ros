"""Tier 1 guard for scripts/sweeps/train_validation_mix_v2.yaml.

Loads the shipped v2 train/validation sweep spec BY PATH (same repo
walk as test_anomaly_envelope_separation.py) and locks its authored
facts against the v1 spec it evolves: the pump class now STRADDLES the
nominal envelope (faint end indistinguishable on purpose, severe end
outside — inverting v1's disjoint-band philosophy for that class only),
sensor/biofouling bands stay v1 byte-equal, comms is dropped, the
mission dims become a mission_mix, the free tank cushion becomes a pin,
pump overshoot joins the matrix, and the worst mission corner fits the
header's recommended --per-run-timeout. Sampler-side loading (the pin
dimension, the mission_mix/mission.* two-authority check) is exercised
through scripts/lhs_sample.py itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from py_pkg.scenarios.anomaly import AnomalyMixSpec, class_counts
from py_pkg.scenarios.mission_mix import MissionMixSpec, profile_counts

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
_V2_PATH = _SCRIPTS_DIR / "sweeps" / "train_validation_mix_v2.yaml"
_V1_PATH = _SCRIPTS_DIR / "sweeps" / "train_validation_mix_v1.yaml"
for _p in (_V2_PATH, _V1_PATH):
    if not _p.is_file():  # pragma: no cover — repo-layout guard
        pytest.skip(f"sweep spec not found at {_p}", allow_module_level=True)

_V2 = yaml.safe_load(_V2_PATH.read_text())
_V1 = yaml.safe_load(_V1_PATH.read_text())
MIX = AnomalyMixSpec.model_validate(_V2["anomaly_mix"])
MISSION = MissionMixSpec.model_validate(_V2["mission_mix"])
DIMS = {d["path"]: d for d in _V2["dimensions"]}
V1_DIMS = {d["path"]: d for d in _V1["dimensions"]}

# The sweep scripts are plain modules two dirs above py_pkg, not a
# package — import them the way run_sweep imports its own siblings.
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import lhs_sample  # noqa: E402
    import run_sweep  # noqa: E402

    _SCRIPTS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover — apt python3 ships numpy/scipy
    lhs_sample = run_sweep = None
    _SCRIPTS_IMPORT_ERROR = exc

needs_scripts = pytest.mark.skipif(
    _SCRIPTS_IMPORT_ERROR is not None,
    reason=f"scripts import failed (numpy/scipy missing?): {_SCRIPTS_IMPORT_ERROR}",
)


# ---------------------------------------------------------------------------
# Pump straddle (the class that inverts v1's separation philosophy)
# ---------------------------------------------------------------------------


def test_pump_severities_straddle_the_flow_envelope():
    vpr = DIMS["rig.plant.volume_per_rev_m3"]
    eff = MIX.bcu_pump.effectiveness
    # 1.0 exactly would read as fault-free to the Scenario label validator.
    assert eff.high < 1.0
    # Faint end INSIDE the envelope: a max-effectiveness pump on the
    # fastest nominal plant still outflows the slowest nominal plant.
    assert eff.high * vpr["high"] > vpr["low"]
    # Severe end OUTSIDE: min effectiveness exits even the fastest plant.
    assert eff.low * vpr["high"] < vpr["low"]


def test_pump_delay_band_straddles_the_envelope_high_edge():
    env = DIMS["rig.plant.pump_response_delay_s"]
    band = MIX.bcu_pump.response_delay_s
    assert env["low"] < band.low < env["high"] < band.high


def test_pump_slew_band_straddles_the_envelope_low_edge():
    env = DIMS["rig.plant.pump_slew_rpm_per_s"]
    band = MIX.bcu_pump.slew_rpm_per_s
    assert band.low < env["low"] < band.high < env["high"]


# ---------------------------------------------------------------------------
# v1 continuity: sensor + biofouling untouched, comms dropped
# ---------------------------------------------------------------------------


def test_sensor_bands_and_biofouling_block_are_v1_byte_equal():
    v2_sensor = dict(_V2["anomaly_mix"]["sensor"])
    v2_sensor.pop("onset")  # additive in v2; the bands must not move
    assert v2_sensor == _V1["anomaly_mix"]["sensor"]
    assert _V2["anomaly_mix"]["biofouling"] == _V1["anomaly_mix"]["biofouling"]


def test_anomaly_weights_and_stratified_counts_at_1024():
    assert MIX.weights == {
        "nominal": 0.85,
        "bcu_pump": 0.05,
        "sensor": 0.05,
        "biofouling": 0.05,
    }
    assert "comms" not in MIX.weights
    # 1024 x 0.85 = 870.4: largest-remainder floors everything and
    # nominal's 0.4 remainder beats the three 0.2s — 871/51/51/51.
    assert class_counts(MIX, int(_V2["n_samples"])) == {
        "nominal": 871,
        "bcu_pump": 51,
        "sensor": 51,
        "comms": 0,
        "biofouling": 51,
    }


# ---------------------------------------------------------------------------
# Mission mix
# ---------------------------------------------------------------------------


def test_mission_mix_weights_and_profile_counts_at_1024():
    assert sum(MISSION.weights.values()) == pytest.approx(1.0)
    # 0.30 x 1024 = 307.2 and 0.20 x 1024 = 204.8: station_keep's 0.8
    # remainder takes the leftover run.
    assert profile_counts(MISSION, int(_V2["n_samples"])) == {
        "sawtooth_plain": 256,
        "sawtooth_dwell": 307,
        "staircase": 256,
        "station_keep": 205,
    }


def test_onset_blocks_are_sane():
    for cls in ("bcu_pump", "sensor"):
        onset = getattr(MIX, cls).onset
        assert onset is not None, cls
        assert 0.0 <= onset.p_immediate <= 1.0
        # >= 40% of the expected mission stays post-onset.
        assert onset.onset_frac.high <= 0.6
    # drift/stuck accept only step schedules, so the sensor class never
    # weights a ramp.
    assert "ramp" not in MIX.sensor.onset.shape_weights
    # Biofouling is baked into the SDF at spawn — no onset by contract.
    assert "onset" not in _V2["anomaly_mix"]["biofouling"]


# ---------------------------------------------------------------------------
# Dimension contract against v1
# ---------------------------------------------------------------------------


def test_dimension_paths_are_v1_minus_mission_plus_overshoot():
    dropped = {"mission.target_pressure_pa", "mission.n_oscillations"}
    added = {"rig.plant.pump_overshoot_frac"}
    assert set(DIMS) == (set(V1_DIMS) - dropped) | added
    # v1's free tank-cushion dim survives only as a pin: 0.0 on the wire
    # = derive the gas-law cushion through both calibrated endpoints.
    pin = DIMS["rig.plant.tank_air_volume_m3"]
    assert pin["distribution"] == "uniform"
    assert pin["low"] == pin["high"] == 0.0
    # Every other shared band is numerically identical to v1's.
    for path in set(DIMS) & set(V1_DIMS) - {"rig.plant.tank_air_volume_m3"}:
        assert DIMS[path] == V1_DIMS[path], path


def test_pump_overshoot_dim_spans_zero_to_the_lake_anchor_margin():
    dim = DIMS["rig.plant.pump_overshoot_frac"]
    # 0 keeps the plain delay+slew plant in-distribution; the ceiling
    # clears the 0.038 lake anchor (3113 vs 3000 rpm).
    assert dim["distribution"] == "uniform"
    assert dim["low"] == 0.0
    assert dim["high"] > 0.038


# ---------------------------------------------------------------------------
# Sampler + runner integration (scripts/ imports)
# ---------------------------------------------------------------------------


@needs_scripts
def test_worst_corner_fits_the_recommended_timeout():
    # The deepest dwelled sawtooth (40 m, 2 cycles, 600 s dwells) must
    # fit the header's recommended --per-run-timeout of 18000 s.
    worst = {"target_pressure_pa": 392240.0, "n_oscillations": 2, "dwell_s": 600.0}
    assert run_sweep._scaled_timeout(worst, None) <= 18000.0


@needs_scripts
def test_v2_spec_loads_through_the_sampler():
    spec = lhs_sample.SweepSpec.load(_V2_PATH)
    assert spec.mission_mix is not None
    assert spec.anomaly_mix is not None
    # The pin dimension survives Dimension validation (uniform-only
    # low == high) and scales to its constant.
    pin = next(d for d in spec.dimensions if d.path == "rig.plant.tank_air_volume_m3")
    assert pin.low == pin.high == 0.0


@needs_scripts
def test_mission_mix_and_mission_dims_are_two_authorities(tmp_path):
    raw = yaml.safe_load(_V2_PATH.read_text())
    raw["dimensions"].append(
        {
            "path": "mission.target_pressure_pa",
            "distribution": "uniform",
            "low": 29418.0,
            "high": 392240.0,
        }
    )
    conflicted = tmp_path / "conflicted.yaml"
    conflicted.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="mission_mix"):
        lhs_sample.SweepSpec.load(conflicted)
