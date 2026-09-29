"""Tier 1: HeaveAugmentPlugin scenario-spec surface.

The plugin itself is C++ (gtest-covered in dave_gz_model_plugins); these
tests lock the py_pkg side of the contract: the spec defaults ARE the
adopted lake calibration (the render-parity Tier 3 test asserts the same
values against the canonical model.sdf), the validators reject
out-of-band values, partial authoring fills defaults, and the forward
map leaves the new blocks alone.
"""

import pytest
from pydantic import ValidationError

from py_pkg.scenarios.compile import forward_map
from py_pkg.scenarios.spec.rig import (
    AscentDragReliefSpec,
    EntryMomentumSpec,
    HydrodynamicsSpec,
    PhysicsKnobs,
    RigScenario,
)


class TestAdoptedCalibrationDefaults:
    """The recalibrated-nominal constants, fitted to lake dives 2/4."""

    def test_ascent_relief_default_is_the_two_point_fit(self):
        assert AscentDragReliefSpec().retain_fraction == 0.66

    def test_entry_defaults_are_the_dive_2_4_entry_profile(self):
        entry = EntryMomentumSpec()
        assert entry.peak_speed_mps == 0.205
        assert entry.rise_time_s == 20.0
        assert entry.decay_time_s == 25.0

    def test_hydrodynamics_spec_carries_both_blocks_by_default(self):
        hydro = HydrodynamicsSpec()
        assert hydro.ascent_relief == AscentDragReliefSpec()
        assert hydro.entry == EntryMomentumSpec()

    def test_entry_transient_spent_before_the_railed_steady_window(self):
        # The lake-velocity calibration measures a bladder-railed steady
        # descent window from ~75 s after descent onset. By
        # rise + 2 * decay the reference hump is down to e^-2 of peak
        # (~0.03 m/s at nominal) — below the steady descent speed, so
        # the one-sided servo is already inert; keep that point near the
        # window's onset so the transient can't leak into the measurand.
        entry = EntryMomentumSpec()
        assert entry.rise_time_s + 2.0 * entry.decay_time_s <= 90.0


class TestValidators:
    def test_retain_fraction_rejects_zero_and_above_one(self):
        with pytest.raises(ValidationError):
            AscentDragReliefSpec(retain_fraction=0.0)
        with pytest.raises(ValidationError):
            AscentDragReliefSpec(retain_fraction=1.2)

    def test_retain_fraction_one_is_the_valid_no_op(self):
        assert AscentDragReliefSpec(retain_fraction=1.0).retain_fraction == 1.0

    def test_peak_speed_zero_and_negative_are_valid_disable_values(self):
        assert EntryMomentumSpec(peak_speed_mps=0.0).peak_speed_mps == 0.0
        assert EntryMomentumSpec(peak_speed_mps=-1.0).peak_speed_mps == -1.0

    def test_entry_times_must_be_positive(self):
        with pytest.raises(ValidationError):
            EntryMomentumSpec(rise_time_s=0.0)
        with pytest.raises(ValidationError):
            EntryMomentumSpec(decay_time_s=-5.0)

    def test_extra_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            AscentDragReliefSpec(retain_fractoin=0.5)
        with pytest.raises(ValidationError):
            EntryMomentumSpec(peak_speed=0.2)


class TestPartialAuthoring:
    def test_scenario_authoring_only_the_entry_knob_fills_defaults(self):
        rig = RigScenario.model_validate(
            {"hydrodynamics": {"entry": {"peak_speed_mps": 0.24}}}
        )
        assert rig.hydrodynamics.entry.peak_speed_mps == 0.24
        assert rig.hydrodynamics.entry.rise_time_s == 20.0
        assert rig.hydrodynamics.ascent_relief.retain_fraction == 0.66
        # The rest of the block stays canonical.
        assert rig.hydrodynamics.drag_zW == -52.2

    def test_absent_hydrodynamics_block_still_means_canonical_sdf(self):
        assert RigScenario().hydrodynamics is None


class TestForwardMapUntouched:
    def test_forward_map_leaves_the_new_blocks_at_spec_defaults(self):
        hydro = forward_map(PhysicsKnobs())
        assert hydro.ascent_relief == AscentDragReliefSpec()
        assert hydro.entry == EntryMomentumSpec()

    def test_model_dump_round_trips_through_validation(self):
        # The LHS sampler dumps the spec into the scenario YAML and the
        # loader re-validates it; the new nested blocks must survive.
        dumped = forward_map(PhysicsKnobs()).model_dump()
        assert dumped["ascent_relief"] == {"retain_fraction": 0.66}
        assert dumped["entry"] == {
            "peak_speed_mps": 0.205,
            "rise_time_s": 20.0,
            "decay_time_s": 25.0,
        }
        assert HydrodynamicsSpec.model_validate(dumped) == forward_map(PhysicsKnobs())
