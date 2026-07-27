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
            "sawtooth_plain": 0.55,
            "staircase": 0.45,
        },
        "sawtooth_plain": {
            "mission_id": 1,
            "target_pressure_pa": {"low": 49030.0, "high": 98060.0},
            "n_oscillations": {"low": 2, "high": 5, "integer": True},
        },
        "staircase": {
            "mission_id": 3,
            "target_pressure_pa": {"low": 58836.0, "high": 98060.0},
            "n_steps": {"low": 3, "high": 6, "integer": True},
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
    },
    "staircase": {
        "mission.mission_id",
        "mission.target_pressure_pa",
        "mission.n_steps",
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mix_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="sum"):
        _mix(weights={"sawtooth_plain": 0.5, "staircase": 0.3})
    with pytest.raises(ValueError, match="unknown"):
        _mix(weights={"sawtooth_plain": 0.5, "loiter": 0.5})
    with pytest.raises(ValueError, match="config block"):
        MissionMixSpec.model_validate(
            {
                "weights": {"sawtooth_plain": 0.5, "staircase": 0.5},
                "sawtooth_plain": _mix_dict()["sawtooth_plain"],
            }
        )


def test_dwell_is_no_longer_a_mission_knob():
    # The bang-bang BCU has no hold, so a leftover `dwell_s:` in an old
    # sweep spec must fail loudly at load rather than be silently ignored.
    stale = _mix_dict()
    stale["staircase"]["dwell_s"] = {"low": 30.0, "high": 90.0}
    with pytest.raises(ValueError, match="dwell_s"):
        MissionMixSpec.model_validate(stale)


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
    # Largest remainder by hand at n=1024: quotas 563.2/460.8 floor to
    # 563/460 (sum 1023); the one leftover goes to the largest remainder,
    # staircase (0.8).
    assert profile_counts(MIX, 1024) == {
        "sawtooth_plain": 563,
        "staircase": 461,
    }
    # Truncated n=32: quotas 17.6/14.4 floor to 17/14 (sum 31); the
    # leftover goes to sawtooth_plain (0.6).
    assert profile_counts(MIX, 32) == {
        "sawtooth_plain": 18,
        "staircase": 14,
    }


def test_profile_tuple_order_is_load_bearing():
    # Equal weights, n=1: both remainders tie at 0.5, so the leftover
    # lands on the FIRST profile of MISSION_PROFILES — sawtooth_plain
    # takes the tie (anomaly.py gives nominal this role).
    equal = _mix(weights={"sawtooth_plain": 0.5, "staircase": 0.5})
    assert profile_counts(equal, 1) == {"sawtooth_plain": 1, "staircase": 0}
    assert MISSION_PROFILES == ("sawtooth_plain", "staircase")


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
    a = draw_mission(MIX, 813, 42, "sawtooth_plain")
    b = draw_mission(MIX, 813, 42, "sawtooth_plain")
    c = draw_mission(MIX, 813, 43, "sawtooth_plain")
    assert a == b
    assert a.values != c.values  # per-run stream
    assert 49030.0 <= a.values["mission.target_pressure_pa"] <= 98060.0
    assert a.values["mission.n_oscillations"] in (2, 3, 4)


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
    # Fixed identifiers pass through undrawn.
    stairs = draw_mission(MIX, 813, 3, "staircase")
    assert stairs.values["mission.mission_id"] == 3
    assert stairs.values["mission.n_steps"] in (3, 4, 5)


def test_record_carries_profile_and_values():
    a = draw_mission(MIX, 813, 7, "staircase")
    assert a.record() == {"profile": "staircase", **a.values}
    # And the dataclass shape itself is what the sampler stores.
    assert a == MissionAssignment("staircase", dict(a.values))


def test_band_edits_do_not_reshuffle_or_leak_across_profiles():
    edited = _mix_dict()
    edited["staircase"]["n_steps"] = {"low": 2, "high": 3, "integer": True}
    mix_b = MissionMixSpec.model_validate(edited)

    # Same weights -> identical stratified assignment.
    assert assign_profiles(MIX, 813, 1024) == assign_profiles(mix_b, 813, 1024)
    for idx in (3, 57, 200):
        # The other profile's draws are untouched by a staircase band edit.
        assert draw_mission(MIX, 813, idx, "sawtooth_plain") == draw_mission(
            mix_b, 813, idx, "sawtooth_plain"
        )
        # Fixed draw order: target pressure draws FIRST, so the edited
        # profile's own earlier draw doesn't move either.
        a = draw_mission(MIX, 813, idx, "staircase")
        b = draw_mission(mix_b, 813, idx, "staircase")
        assert (
            a.values["mission.target_pressure_pa"]
            == b.values["mission.target_pressure_pa"]
        )
        assert b.values["mission.n_steps"] == 2


# ---------------------------------------------------------------------------
# expected_mission_duration_s
# ---------------------------------------------------------------------------


def test_expected_duration_arithmetic():
    round_trip_per_m = 1.0 / OPTIMISTIC_DESCENT_MPS + 1.0 / OPTIMISTIC_ASCENT_MPS
    # Sawtooth: 98060 Pa = 10 m, 3 oscillations.
    sawtooth = {
        "mission.mission_id": 1,
        "mission.target_pressure_pa": 98060.0,
        "mission.n_oscillations": 3,
    }
    assert expected_mission_duration_s(sawtooth) == pytest.approx(
        3 * 10.0 * round_trip_per_m
    )
    # Staircase: 49030 Pa = 5 m; ONE round trip however many steps it
    # descends through, matching run_sweep.py's timeout budget.
    staircase = {
        "mission.mission_id": 3,
        "mission.target_pressure_pa": 49030.0,
        "mission.n_steps": 4,
    }
    assert expected_mission_duration_s(staircase) == pytest.approx(
        5.0 * round_trip_per_m
    )
    # Bare target: one round trip.
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
