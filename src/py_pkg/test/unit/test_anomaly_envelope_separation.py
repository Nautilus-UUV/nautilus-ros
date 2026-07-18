"""Tier 1: anomaly severity bands sit strictly outside the normality envelope.

Loads the shipped validation sweep spec (scripts/sweeps/
validation_mix_v1.yaml) and asserts every anomalous band's near edge
clears the nominal envelope's worst-case extreme — LHS band corners
composed with the 3-sigma isotropic jitter factor e^(3*0.05) where the
observable is jittered. The point: an anomaly detector's validation
label must never be contradicted by a nominal run that legitimately
sampled the same parameter value.

NoiseSpec is imported live, so a lake-noise re-fit that erodes a
sensor-fault margin trips this test too. If a band or an envelope
dimension changes, update the sweep spec and this file together.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from py_pkg.robot_specs import VOLUME_PER_REV_M3
from py_pkg.scenarios.anomaly import AnomalyMixSpec, class_counts
from py_pkg.scenarios.spec.rig import NoiseSpec, PhysicsKnobs

_SPEC_PATH = (
    Path(__file__).resolve().parents[4] / "scripts" / "sweeps" / "validation_mix_v1.yaml"
)
if not _SPEC_PATH.is_file():  # pragma: no cover — repo-layout guard
    pytest.skip(
        f"validation sweep spec not found at {_SPEC_PATH}", allow_module_level=True
    )

_RAW = yaml.safe_load(_SPEC_PATH.read_text())
MIX = AnomalyMixSpec.model_validate(_RAW["anomaly_mix"])
DIMS = {d["path"]: d for d in _RAW["dimensions"]}
# 3-sigma multiplicative jitter factor for jittered hydro observables.
JITTER3 = math.exp(3.0 * float(_RAW["isotropic_jitter_sigma"]))
KNOBS = PhysicsKnobs()
NOISE = NoiseSpec()


def test_mix_is_the_locked_80_5x4_split():
    assert MIX.weights == {
        "nominal": 0.80,
        "bcu_pump": 0.05,
        "sensor": 0.05,
        "comms": 0.05,
        "biofouling": 0.05,
    }
    n = int(_RAW["n_samples"])
    counts = class_counts(MIX, n)
    assert counts["nominal"] == int(0.80 * n)
    assert all(counts[c] == int(0.05 * n) for c in counts if c != "nominal")


def test_sensor_scope_is_the_locked_4_archetypes_2_channels():
    assert set(MIX.sensor.channels) == {"external_pressure", "tank_pressure"}
    assert set(MIX.sensor.archetypes) == {"bias", "drift", "stuck", "dropout"}


def test_pump_effectiveness_clears_the_flow_envelope():
    # Envelope flow spread comes from volume_per_rev alone (effectiveness
    # is exactly 1.0 nominal): worst-slow nominal flow = vpr.low, worst-
    # fast anomalous flow = eff.high * vpr.high. They must not overlap.
    vpr = DIMS["rig.plant.volume_per_rev_m3"]
    eff = MIX.bcu_pump.effectiveness
    assert eff.high * vpr["high"] < vpr["low"]
    # Envelope brackets nominal (sanity that the ratio math above means
    # what it says).
    assert vpr["low"] < VOLUME_PER_REV_M3 < vpr["high"]
    # Floor keeps a full bladder stroke feasible inside a 4800 s run.
    assert eff.low >= 0.30


def test_pump_transient_bands_are_fully_disjoint():
    delay = DIMS["rig.plant.pump_response_delay_s"]
    slew = DIMS["rig.plant.pump_slew_rpm_per_s"]
    assert MIX.bcu_pump.response_delay_s.low > delay["high"]
    assert MIX.bcu_pump.slew_rpm_per_s.high < slew["low"]


def test_external_pressure_fault_floors():
    bands = MIX.sensor.bands["external_pressure"]
    # Nominal channel carries zero bias/drift; the only artifact is the
    # quantization half-step.
    half_step = NOISE.external_pressure.quantization_pa / 2.0
    assert bands["bias"].low >= 20.0 * half_step
    assert bands["drift"].low > 0.0
    assert bands["dropout"].low >= 0.2


def test_tank_pressure_fault_floors():
    bands = MIX.sensor.bands["tank_pressure"]
    # Worst nominal excursion of a single reading: 3 sigma of the fitted
    # Gaussian + quantization chain (Sheppard variance for the comb).
    tank = NOISE.tank_pressure
    three_sigma = 3.0 * math.sqrt(
        tank.sigma_pa**2 + tank.quantization_pa**2 / 12.0
    )
    assert bands["bias"].low >= 4.0 * three_sigma
    assert bands["drift"].low > 0.0
    assert bands["dropout"].low >= 0.2


def test_comms_floor_clears_the_zero_point_mass():
    # Nominal runs never drop a frame (rates fixed, drop prob exactly 0);
    # the floor itself is the separation margin.
    assert MIX.comms.drop_prob.low >= 0.30
    assert MIX.comms.drop_prob.high < 1.0  # link degraded, not severed


def test_biofouling_drag_floor_clears_the_envelope_extreme():
    # Worst nominal multiplicative extreme on the heave-drag slots:
    # cross-flow knob at band max (drag ∝ C_d_c) times the geometric
    # contributions (∝ sqrt(L), sqrt(nabla) for Y_v/Z_w strip terms),
    # times the 3-sigma isotropic jitter.
    worst = (
        (DIMS["C_d_c"]["high"] / KNOBS.C_d_c)
        * math.sqrt(DIMS["L"]["high"] / KNOBS.L)
        * math.sqrt(DIMS["nabla"]["high"] / KNOBS.nabla)
        * JITTER3
    )
    assert MIX.biofouling.drag_mult.low > worst


def test_biofouling_added_mass_floor_clears_the_envelope_extreme():
    # Closed-form knob worst on the added-mass diagonals is ~ +4%
    # (nabla +1% body share plus the fin c_f^2*b_f term on its ~23%
    # share — see the hydro sampling plan §12); jitter dominates.
    worst = 1.04 * JITTER3
    assert MIX.biofouling.added_mass_mult.low > worst


def test_biofouling_mass_floor_and_cap():
    vn = DIMS["derive.neutral_volume_m3"]
    rho = DIMS["rig.hydrodynamics.fluid_density"]
    bladder_max = DIMS["rig.plant.bladder_max_m3"]

    # Floor: even in the densest water, the smallest fouling mass shifts
    # the neutral volume past the entire nominal band's width — every
    # fouled run's true neutral sits above the envelope ceiling.
    assert MIX.biofouling.fouling_mass_kg.low / rho["high"] > vn["high"] - vn["low"]

    # Cap: at the worst corner (max neutral, max fouling, lightest
    # water, smallest bladder ceiling) the run still keeps the relaxed
    # climb margin — the derivation's HARD-FAIL stays an authoring
    # error, never a sampling outcome.
    worst_neutral = vn["high"] + MIX.biofouling.fouling_mass_kg.high / rho["low"]
    assert (
        worst_neutral
        <= bladder_max["low"] - MIX.biofouling.min_climb_margin_m3
    )
