"""Tier 1 unit tests for the correlated buoyancy derivation.

Pure logic — no rclpy, no sim, no file I/O. Pins the closed-form
derivation to the canonical-SDF nominal anchor (rho = 1000,
V_n = 2.0963072e-3 -> bow 3.157 / stern 0.2765 / spawn 0.0022), then
exercises the (density, neutral-volume) envelope the sampler draws
from: round-trip neutrality, the level-trim moment invariant,
monotonicity, positivity, and the viability gate's edge cases.

The constants themselves are locked to model.sdf / the world file by
the Tier 3 parity test (test/sim/test_buoyancy_budget_parity.py).
"""

import pytest
from py_pkg.scenarios import buoyancy

# Wide sampling band: +/- ~11% density around fresh water x the trim
# envelope both real ballast configs sit in.
RHO_GRID = (877.9, 950.0, 1000.0, 1050.0, 1111.8)
VN_GRID = (2.0e-3, 2.1e-3, 2.2e-3)


class TestNominalAnchor:
    """derive(nominal) must reproduce the canonical SDF trims exactly —
    the self-calibration contract everything else builds on."""

    def test_nominal_neutral_volume_matches_sdf_budget(self):
        assert buoyancy.NOMINAL_NEUTRAL_VOLUME_M3 == pytest.approx(
            2.0963072e-3, abs=1e-9
        )

    def test_nominal_derive_reproduces_canonical_trims(self):
        d = buoyancy.derive_trim_masses(1000.0, buoyancy.NOMINAL_NEUTRAL_VOLUME_M3)
        assert d.trim_mass_bow == pytest.approx(3.157, abs=1e-6)
        assert d.trim_mass_stern == pytest.approx(0.2765, abs=1e-6)

    def test_nominal_spawn_is_sdf_default_volume_under_offset_policy(self):
        d = buoyancy.derive_trim_masses(1000.0, buoyancy.NOMINAL_NEUTRAL_VOLUME_M3)
        assert d.bladder_spawn_volume_m3 == pytest.approx(0.0022, abs=1e-9)

    def test_neutral_volume_inverse_of_derivation(self):
        assert buoyancy.neutral_volume_m3(1000.0, 3.157, 0.2765) == pytest.approx(
            buoyancy.NOMINAL_NEUTRAL_VOLUME_M3, abs=1e-12
        )


class TestRoundTripNeutrality:
    """Derived trims must be exactly neutral at the requested volume —
    the whole point of correlating the draw."""

    @pytest.mark.parametrize("rho", RHO_GRID)
    @pytest.mark.parametrize("vn", VN_GRID)
    def test_net_buoyancy_zero_at_derived_neutral(self, rho, vn):
        d = buoyancy.derive_trim_masses(rho, vn)
        net = buoyancy.net_buoyancy_kg(
            rho, d.neutral_volume_m3, d.trim_mass_bow, d.trim_mass_stern
        )
        assert net == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("rho", RHO_GRID)
    @pytest.mark.parametrize("vn", VN_GRID)
    def test_spawn_is_positively_buoyant(self, rho, vn):
        # Offset policy spawns above neutral: a gentle surface float,
        # never a sinker, at every sampled plant point.
        d = buoyancy.derive_trim_masses(rho, vn)
        net = buoyancy.net_buoyancy_kg(
            rho, d.bladder_spawn_volume_m3, d.trim_mass_bow, d.trim_mass_stern
        )
        assert net > 0.0


class TestLevelTrimMomentInvariant:
    """The split is a level-float condition: total mass moment equals
    the buoyancy centroid at spawn times the total mass."""

    @pytest.mark.parametrize("rho", RHO_GRID)
    @pytest.mark.parametrize("vn", VN_GRID)
    def test_mass_moment_matches_buoyancy_centroid(self, rho, vn):
        d = buoyancy.derive_trim_masses(rho, vn)
        total_mass = (
            buoyancy.M_FIXED_KG
            + d.trim_mass_bow
            + d.trim_mass_stern
            + buoyancy.NOMINAL_TRIM_BLADDER_KG
        )
        mass_moment = (
            buoyancy.M_FIXED_MOMENT_KGM
            + d.trim_mass_bow * buoyancy.X_BOW_M
            + d.trim_mass_stern * buoyancy.X_STERN_M
            + buoyancy.NOMINAL_TRIM_BLADDER_KG * buoyancy.X_BLADDER_M
        )
        x_b = buoyancy.buoyancy_centroid_x(rho, d.bladder_spawn_volume_m3)
        assert mass_moment == pytest.approx(x_b * total_mass, abs=1e-9)


class TestMonotonicity:
    """Denser water / bigger neutral target both displace more, so the
    trim-mass sum must rise with each."""

    def test_trim_sum_increases_with_density(self):
        sums = []
        for rho in RHO_GRID:
            d = buoyancy.derive_trim_masses(rho, 2.1e-3)
            sums.append(d.trim_mass_bow + d.trim_mass_stern)
        assert sums == sorted(sums)
        assert sums[0] < sums[-1]

    def test_trim_sum_increases_with_neutral_volume(self):
        sums = []
        for vn in VN_GRID:
            d = buoyancy.derive_trim_masses(1000.0, vn)
            sums.append(d.trim_mass_bow + d.trim_mass_stern)
        assert sums == sorted(sums)
        assert sums[0] < sums[-1]


class TestDerivationDomain:
    @pytest.mark.parametrize("rho", RHO_GRID)
    @pytest.mark.parametrize("vn", VN_GRID)
    def test_masses_positive_across_wide_band(self, rho, vn):
        d = buoyancy.derive_trim_masses(rho, vn)
        assert d.trim_mass_bow > 0.0
        assert d.trim_mass_stern > 0.0

    def test_explicit_spawn_volume_wins_over_offset(self):
        # Bladder-full-float spawn policy: the caller pins the spawn
        # fill; the offset must not be applied on top.
        d = buoyancy.derive_trim_masses(
            1000.0, 2.1e-3, spawn_volume_m3=0.0025, spawn_offset_m3=1.0
        )
        assert d.bladder_spawn_volume_m3 == 0.0025

    def test_custom_spawn_offset_honored(self):
        d = buoyancy.derive_trim_masses(1000.0, 2.1e-3, spawn_offset_m3=2.0e-4)
        assert d.bladder_spawn_volume_m3 == pytest.approx(2.3e-3, abs=1e-12)

    @pytest.mark.parametrize("rho", [0.0, -1000.0])
    def test_absurd_density_raises(self, rho):
        with pytest.raises(ValueError):
            buoyancy.derive_trim_masses(rho, 2.1e-3)

    def test_non_positive_neutral_volume_raises(self):
        with pytest.raises(ValueError):
            buoyancy.derive_trim_masses(1000.0, 0.0)


def _derived(vn=2.1e-3, spawn=2.2e-3):
    return buoyancy.DerivedBuoyancy(
        trim_mass_bow=3.157,
        trim_mass_stern=0.2765,
        bladder_spawn_volume_m3=spawn,
        neutral_volume_m3=vn,
    )


class TestViability:
    """Edge semantics of the gate: exact margins pass, an epsilon under
    fails, and the reasons tuple names every violated check."""

    def test_nominal_clamps_are_viable(self):
        v = buoyancy.check_viability(_derived(), 0.001, 0.00245)
        assert v.ok
        assert v.reasons == ()
        assert v.dive_margin_m3 == pytest.approx(1.1e-3, abs=1e-12)
        assert v.climb_margin_m3 == pytest.approx(3.5e-4, abs=1e-12)

    def test_exact_dive_margin_passes(self):
        # neutral - min == exactly the required margin -> viable. The
        # required margin is computed with the same subtraction the
        # checker performs so the comparison is bitwise-exact.
        vn, bladder_min = 2.1e-3, 1.9e-3
        v = buoyancy.check_viability(
            _derived(vn=vn), bladder_min, 0.00245, min_dive_margin_m3=vn - bladder_min
        )
        assert v.ok

    def test_dive_margin_below_minimum_fails(self):
        v = buoyancy.check_viability(_derived(vn=2.1e-3), 2.1e-3 - 1.9e-4, 0.00245)
        assert not v.ok
        assert any("dive margin" in r for r in v.reasons)

    def test_exact_climb_margin_passes(self):
        vn, bladder_max = 2.1e-3, 2.25e-3
        v = buoyancy.check_viability(
            _derived(vn=vn, spawn=2.2e-3),
            0.001,
            bladder_max,
            min_climb_margin_m3=bladder_max - vn,
        )
        assert v.ok

    def test_climb_margin_below_minimum_fails(self):
        v = buoyancy.check_viability(
            _derived(vn=2.1e-3, spawn=2.2e-3), 0.001, 2.1e-3 + 1.4e-4
        )
        assert not v.ok
        assert any("climb margin" in r for r in v.reasons)

    def test_spawn_above_bladder_max_fails(self):
        v = buoyancy.check_viability(_derived(spawn=2.46e-3), 0.001, 0.00245)
        assert not v.ok
        assert any("spawn" in r for r in v.reasons)

    def test_spawn_equal_to_bladder_max_passes(self):
        v = buoyancy.check_viability(_derived(spawn=0.00245), 0.001, 0.00245)
        assert v.ok

    def test_spawn_at_bladder_min_fails(self):
        # Strict lower bound: spawning parked on the floor leaves zero
        # immediate dive authority.
        v = buoyancy.check_viability(_derived(spawn=0.001), 0.001, 0.00245)
        assert not v.ok
        assert any("spawn" in r for r in v.reasons)

    def test_bladder_max_above_sdf_hard_clamp_fails(self):
        v = buoyancy.check_viability(_derived(), 0.001, 0.00251)
        assert not v.ok
        assert any("max_volume" in r for r in v.reasons)

    def test_margins_reported_even_on_failure(self):
        v = buoyancy.check_viability(_derived(vn=1.05e-3), 0.001, 0.00245)
        assert not v.ok
        assert v.dive_margin_m3 == pytest.approx(5.0e-5, abs=1e-12)

    def test_tighter_margins_flip_a_nominal_pass(self):
        d = _derived()
        assert buoyancy.check_viability(d, 0.001, 0.00245).ok
        v = buoyancy.check_viability(d, 0.001, 0.00245, min_climb_margin_m3=4.0e-4)
        assert not v.ok
