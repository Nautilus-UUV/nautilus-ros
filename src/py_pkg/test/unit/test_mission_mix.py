"""Tier 1 for the per-run mission-mix sampling (scenarios/mission_mix.py).

Locks the sweep-campaign contracts: stratified exact profile counts
(largest-remainder, sawtooth_plain tie-break), seeded determinism,
stream decoupling (band edits never reshuffle assignments or sibling
profiles), integer-band flooring, the flat ``mission.*`` value keys that
ride the sampler manifest into launch args, and the optimistic
``expected_mission_duration_s`` arithmetic fault-onset placement scales.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
from py_pkg.scenarios.mission_mix import (
    MISSION_PROFILES,
    OPTIMISTIC_ASCENT_MPS,
    OPTIMISTIC_DESCENT_MPS,
    MissionAssignment,
    MissionBand,
    MissionMixSpec,
    MissionProfileSpec,
    assign_profiles,
    draw_mission,
    expected_mission_duration_s,
    profile_counts,
)


def _mix_dict() -> dict:
    return {
        "weights": {
            "sawtooth_plain": 0.25,
            "sawtooth_dwell": 0.30,
            "staircase": 0.25,
            "station_keep": 0.20,
        },
        "sawtooth_plain": {
            "mission_id": 1,
            "target_pressure_pa": {"low": 49030.0, "high": 98060.0},
            "n_oscillations": {"low": 2, "high": 5, "integer": True},
        },
        "sawtooth_dwell": {
            "mission_id": 1,
            "target_pressure_pa": {"low": 49030.0, "high": 98060.0},
            "n_oscillations": {"low": 2, "high": 4, "integer": True},
            "dwell_s": {"low": 20.0, "high": 60.0},
        },
        "staircase": {
            "mission_id": 2,
            "target_pressure_pa": {"low": 58836.0, "high": 98060.0},
            "n_steps": {"low": 3, "high": 6, "integer": True},
            "dwell_s": {"low": 30.0, "high": 90.0},
        },
        "station_keep": {
            "mission_id": 3,
            "target_pressure_pa": {"low": 49030.0, "high": 78448.0},
            "n_oscillations": {"low": 1, "high": 2, "integer": True},
            "dwell_s": 300.0,
        },
    }


def _mix(**edits) -> MissionMixSpec:
    d = _mix_dict()
    d.update(edits)
    return MissionMixSpec.model_validate(d)


MIX = _mix()

# The flat manifest/launch keys each profile must emit (run_sweep.py
# strips the "mission." prefix to make launch args).
_EXPECTED_KEYS = {
    "sawtooth_plain": {
        "mission.mission_id",
        "mission.target_pressure_pa",
        "mission.n_oscillations",
        "mission.dwell_s",
    },
    "sawtooth_dwell": {
        "mission.mission_id",
        "mission.target_pressure_pa",
        "mission.n_oscillations",
        "mission.dwell_s",
    },
    "staircase": {
        "mission.mission_id",
        "mission.target_pressure_pa",
        "mission.n_steps",
        "mission.dwell_s",
    },
    "station_keep": {
        "mission.mission_id",
        "mission.target_pressure_pa",
        "mission.n_oscillations",
        "mission.dwell_s",
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mix_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="sum"):
        _mix(
            weights={
                "sawtooth_plain": 0.5,
                "sawtooth_dwell": 0.3,
                "staircase": 0.3,
                "station_keep": 0.2,
            }
        )
    with pytest.raises(ValueError, match="unknown"):
        _mix(weights={"sawtooth_plain": 0.5, "loiter": 0.5})
    with pytest.raises(ValueError, match="config block"):
        MissionMixSpec.model_validate(
            {
                "weights": {"sawtooth_plain": 0.5, "staircase": 0.5},
                "sawtooth_plain": _mix_dict()["sawtooth_plain"],
            }
        )


def test_profile_spec_needs_exactly_one_leg_count():
    band = {"low": 49030.0, "high": 98060.0}
    count = {"low": 2, "high": 5, "integer": True}
    with pytest.raises(ValueError, match="exactly one"):
        MissionProfileSpec.model_validate({"mission_id": 1, "target_pressure_pa": band})
    with pytest.raises(ValueError, match="exactly one"):
        MissionProfileSpec.model_validate(
            {
                "mission_id": 1,
                "target_pressure_pa": band,
                "n_oscillations": count,
                "n_steps": count,
            }
        )


def test_band_semantics():
    with pytest.raises(ValueError):
        MissionBand(low=2.0, high=1.0)
    # low == high pins a fixed value — the "one canonical value" switch.
    fixed = MissionBand(low=0.5, high=0.5)
    assert fixed.draw(random.Random(1)) == 0.5
    # Integer bands floor with the high edge exclusive (Dimension.scale
    # semantics): [2, 3] always draws exactly 2.
    two = MissionBand(low=2, high=3, integer=True)
    assert {two.draw(random.Random(s)) for s in range(50)} == {2}
    spread = MissionBand(low=1, high=4, integer=True)
    draws = {spread.draw(random.Random(s)) for s in range(200)}
    assert draws == {1, 2, 3}
    assert all(isinstance(d, int) for d in draws)


# ---------------------------------------------------------------------------
# Stratified assignment
# ---------------------------------------------------------------------------


def test_profile_counts_exact_1024_and_32():
    # Largest remainder by hand at n=1024: quotas 256.0/307.2/256.0/204.8
    # floor to 256/307/256/204 (sum 1023); the one leftover goes to the
    # largest remainder, station_keep (0.8).
    assert profile_counts(MIX, 1024) == {
        "sawtooth_plain": 256,
        "sawtooth_dwell": 307,
        "staircase": 256,
        "station_keep": 205,
    }
    # Truncated n=32: quotas 8.0/9.6/8.0/6.4 floor to 8/9/8/6 (sum 31);
    # the leftover goes to sawtooth_dwell (0.6).
    assert profile_counts(MIX, 32) == {
        "sawtooth_plain": 8,
        "sawtooth_dwell": 10,
        "staircase": 8,
        "station_keep": 6,
    }


def test_profile_tuple_order_is_load_bearing():
    # Equal weights, n=2: every remainder ties at 0.5, so the two
    # leftovers land on the FIRST two profiles of MISSION_PROFILES —
    # sawtooth_plain takes the tie (anomaly.py gives nominal this role).
    equal = _mix(
        weights={
            "sawtooth_plain": 0.25,
            "sawtooth_dwell": 0.25,
            "staircase": 0.25,
            "station_keep": 0.25,
        }
    )
    assert profile_counts(equal, 2) == {
        "sawtooth_plain": 1,
        "sawtooth_dwell": 1,
        "staircase": 0,
        "station_keep": 0,
    }
    assert MISSION_PROFILES == (
        "sawtooth_plain",
        "sawtooth_dwell",
        "staircase",
        "station_keep",
    )


@pytest.mark.parametrize("n", [1, 3, 7, 100, 256, 1023])
def test_profile_counts_always_sum_to_n(n):
    counts = profile_counts(MIX, n)
    assert sum(counts.values()) == n
    assert all(c >= 0 for c in counts.values())


def test_assign_profiles_is_a_seeded_permutation_of_the_counts():
    a = assign_profiles(MIX, parent_seed=813, n=1024)
    b = assign_profiles(MIX, parent_seed=813, n=1024)
    c = assign_profiles(MIX, parent_seed=814, n=1024)
    assert a == b  # deterministic
    assert a != c  # seed moves the shuffle
    assert Counter(a) == profile_counts(MIX, 1024)
    assert Counter(c) == profile_counts(MIX, 1024)  # counts invariant


# ---------------------------------------------------------------------------
# Parameter draws + decoupling
# ---------------------------------------------------------------------------


def test_draw_mission_deterministic_per_run():
    a = draw_mission(MIX, 813, 42, "sawtooth_dwell")
    b = draw_mission(MIX, 813, 42, "sawtooth_dwell")
    c = draw_mission(MIX, 813, 43, "sawtooth_dwell")
    assert a == b
    assert a.values != c.values  # per-run stream
    assert 49030.0 <= a.values["mission.target_pressure_pa"] <= 98060.0
    assert a.values["mission.n_oscillations"] in (2, 3)
    assert 20.0 <= a.values["mission.dwell_s"] <= 60.0


def test_draw_mission_rejects_unknown_or_blockless_profiles():
    with pytest.raises(ValueError, match="unknown mission profile"):
        draw_mission(MIX, 813, 0, "loiter")
    no_staircase = _mix_dict()
    no_staircase["weights"] = {"sawtooth_plain": 1.0}
    del no_staircase["staircase"]
    mix = MissionMixSpec.model_validate(no_staircase)
    with pytest.raises(ValueError, match="no block"):
        draw_mission(mix, 813, 0, "staircase")


def test_emitted_values_are_exactly_the_flat_mission_keys():
    for profile in MISSION_PROFILES:
        for idx in (0, 17, 99):
            a = draw_mission(MIX, 813, idx, profile)
            assert set(a.values) == _EXPECTED_KEYS[profile], profile
            assert all(k.startswith("mission.") for k in a.values)
            assert isinstance(a.values["mission.mission_id"], int)
            assert isinstance(a.values["mission.target_pressure_pa"], float)
            assert isinstance(a.values["mission.dwell_s"], float)
    # Fixed identifiers pass through undrawn; fixed scalar dwell too.
    keep = draw_mission(MIX, 813, 3, "station_keep")
    assert keep.values["mission.mission_id"] == 3
    assert keep.values["mission.dwell_s"] == 300.0
    assert keep.values["mission.n_oscillations"] == 1  # [1, 2) floors to 1
    stairs = draw_mission(MIX, 813, 3, "staircase")
    assert stairs.values["mission.mission_id"] == 2
    assert stairs.values["mission.n_steps"] in (3, 4, 5)


def test_record_carries_profile_and_values():
    a = draw_mission(MIX, 813, 7, "staircase")
    assert a.record() == {"profile": "staircase", **a.values}
    # And the dataclass shape itself is what the sampler stores.
    assert a == MissionAssignment("staircase", dict(a.values))


def test_band_edits_do_not_reshuffle_or_leak_across_profiles():
    edited = _mix_dict()
    edited["sawtooth_dwell"]["dwell_s"] = {"low": 111.0, "high": 222.0}
    mix_b = MissionMixSpec.model_validate(edited)

    # Same weights -> identical stratified assignment.
    assert assign_profiles(MIX, 813, 1024) == assign_profiles(mix_b, 813, 1024)
    # Other profiles' draws are untouched by a sawtooth_dwell band edit.
    for idx in (3, 57, 200):
        for profile in ("sawtooth_plain", "staircase", "station_keep"):
            assert draw_mission(MIX, 813, idx, profile) == draw_mission(
                mix_b, 813, idx, profile
            )
        # Fixed draw order: the edited dwell band draws LAST, so the
        # edited profile's own earlier draws don't move either.
        a = draw_mission(MIX, 813, idx, "sawtooth_dwell")
        b = draw_mission(mix_b, 813, idx, "sawtooth_dwell")
        assert (
            a.values["mission.target_pressure_pa"]
            == b.values["mission.target_pressure_pa"]
        )
        assert a.values["mission.n_oscillations"] == b.values["mission.n_oscillations"]
        assert 111.0 <= b.values["mission.dwell_s"] <= 222.0


# ---------------------------------------------------------------------------
# expected_mission_duration_s
# ---------------------------------------------------------------------------


def test_expected_duration_arithmetic():
    round_trip_per_m = 1.0 / OPTIMISTIC_DESCENT_MPS + 1.0 / OPTIMISTIC_ASCENT_MPS
    # Sawtooth: 98060 Pa = 10 m, 3 oscillations, no dwell.
    sawtooth = {
        "mission.mission_id": 1,
        "mission.target_pressure_pa": 98060.0,
        "mission.n_oscillations": 3,
        "mission.dwell_s": 0.0,
    }
    assert expected_mission_duration_s(sawtooth) == pytest.approx(
        3 * 10.0 * round_trip_per_m
    )
    # Staircase: 49030 Pa = 5 m, one round trip, 4 dwells of 60 s.
    staircase = {
        "mission.mission_id": 2,
        "mission.target_pressure_pa": 49030.0,
        "mission.n_steps": 4,
        "mission.dwell_s": 60.0,
    }
    assert expected_mission_duration_s(staircase) == pytest.approx(
        5.0 * round_trip_per_m + 4 * 60.0
    )
    # Dwell sawtooth: n_dwells falls back to n_oscillations.
    dwell = {
        "mission.target_pressure_pa": 98060.0,
        "mission.n_oscillations": 2,
        "mission.dwell_s": 30.0,
    }
    assert expected_mission_duration_s(dwell) == pytest.approx(
        2 * 10.0 * round_trip_per_m + 2 * 30.0
    )
    # Bare target: one round trip, one (zero-length) dwell.
    assert expected_mission_duration_s(
        {"mission.target_pressure_pa": 98060.0}
    ) == pytest.approx(10.0 * round_trip_per_m)


def test_expected_duration_bounds_every_drawn_mission_below():
    # The estimate is positive and scales with what was drawn — a sanity
    # sweep across all profiles/streams (onset placement divides by it).
    for profile in MISSION_PROFILES:
        for idx in range(20):
            a = draw_mission(MIX, 813, idx, profile)
            assert expected_mission_duration_s(a.values) > 0.0
