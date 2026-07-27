"""Tier 1 unit tests for the pure conversion functions used by BCU control.

These functions are pure math (no ROS, no hardware) and form the
input/output edges of the BCU control pipeline:
  - gauge_pressure_pa: external pressure sensor (abs Pa) -> gauge (Pa)
  - depth_to_pressure_pa: depth (m) -> absolute pressure (Pa)
  - pressure_to_depth: absolute pressure (Pa) -> depth (m), legacy/log helper
"""

import pytest
from py_pkg.physics import (
    ATMOSPHERIC_PRESSURE_PA,
    WATER_PRESSURE_GRADIENT_PA_PER_M,
    SurfaceReference,
    depth_to_pressure_pa,
    gauge_pressure_pa,
    pressure_to_depth,
)


class TestGaugePressurePa:
    """gauge_pressure_pa(P_abs) = P_abs - P_atm."""

    def test_atmospheric_is_zero_gauge(self):
        assert gauge_pressure_pa(ATMOSPHERIC_PRESSURE_PA) == pytest.approx(0.0)

    def test_overpressure_is_positive(self):
        assert gauge_pressure_pa(ATMOSPHERIC_PRESSURE_PA + 5e4) == pytest.approx(5e4)

    def test_below_atmospheric_is_negative(self):
        assert gauge_pressure_pa(0.5 * ATMOSPHERIC_PRESSURE_PA) < 0

    def test_custom_atmospheric_shifts_zero(self):
        assert gauge_pressure_pa(1.5e5, atmospheric_pa=1.5e5) == pytest.approx(0.0)


class TestDepthToPressurePa:
    """depth_to_pressure_pa(d) = P_atm + d * rho * g."""

    def test_zero_depth_is_atmospheric(self):
        assert depth_to_pressure_pa(0.0) == pytest.approx(ATMOSPHERIC_PRESSURE_PA)

    def test_round_trip_through_pressure_to_depth(self):
        for d in (1.0, 5.5, 30.0, 70.0):
            assert pressure_to_depth(depth_to_pressure_pa(d)) == pytest.approx(d)

    def test_uses_water_pressure_gradient(self):
        # 10 m → P_atm + 10 * gradient
        expected = ATMOSPHERIC_PRESSURE_PA + 10.0 * WATER_PRESSURE_GRADIENT_PA_PER_M
        assert depth_to_pressure_pa(10.0) == pytest.approx(expected)

    def test_density_override(self):
        # Fresh water gives less pressure rise than salt water for the same depth.
        salt = depth_to_pressure_pa(10.0, density=1025)
        fresh = depth_to_pressure_pa(10.0, density=1000)
        assert salt > fresh


class TestPressureToDepth:
    """pressure_to_depth(P) = (P - P_atm) / (rho * g)."""

    def test_atmospheric_pressure_is_surface(self):
        assert pressure_to_depth(ATMOSPHERIC_PRESSURE_PA) == pytest.approx(0.0)

    def test_one_bar_overpressure_in_fresh_water(self):
        # 1e5 Pa above atm / (1000 kg/m^3 * 9.806 m/s^2) ≈ 10.198 m
        depth = pressure_to_depth(ATMOSPHERIC_PRESSURE_PA + 1e5, density=1000)
        assert depth == pytest.approx(10.198, abs=0.01)

    def test_one_bar_overpressure_in_salt_water(self):
        # 1e5 Pa above atm / (1025 kg/m^3 * 9.806 m/s^2) ≈ 9.949 m
        depth = pressure_to_depth(ATMOSPHERIC_PRESSURE_PA + 1e5, density=1025)
        assert depth == pytest.approx(9.949, abs=0.01)

    def test_below_atmospheric_returns_negative(self):
        assert pressure_to_depth(0.5 * ATMOSPHERIC_PRESSURE_PA) < 0

    def test_denser_fluid_gives_shallower_depth(self):
        depth_fresh = pressure_to_depth(2e5, density=1000)
        depth_salt = pressure_to_depth(2e5, density=1025)
        assert depth_salt < depth_fresh

    def test_custom_atmospheric_shifts_zero(self):
        assert pressure_to_depth(1.5e5, atmospheric_pa=1.5e5) == pytest.approx(0.0)


class TestSurfaceReference:
    """The gauge reference a node converts against: standard atmosphere
    until the operator registers the real surface pressure."""

    def test_fallback_is_standard_atmosphere(self):
        ref = SurfaceReference()
        assert ref.reference_pa == pytest.approx(ATMOSPHERIC_PRESSURE_PA)
        assert ref.gauge(ATMOSPHERIC_PRESSURE_PA) == pytest.approx(0.0)

    def test_registered_surface_shifts_zero(self):
        ref = SurfaceReference()
        assert ref.register(98_000.0) is True
        assert ref.reference_pa == pytest.approx(98_000.0)
        # The actual surface now reads 0 gauge instead of -3.3 kPa.
        assert ref.gauge(98_000.0) == pytest.approx(0.0)
        assert ref.gauge(99_000.0) == pytest.approx(1_000.0)

    def test_non_positive_registration_is_rejected(self):
        # A half-filled DiveInit decodes missing fields as 0.0; adopting
        # that would shift the whole gauge frame by ~101 kPa.
        ref = SurfaceReference()
        assert ref.register(0.0) is False
        assert ref.register(-5.0) is False
        assert ref.reference_pa == pytest.approx(ATMOSPHERIC_PRESSURE_PA)

    def test_rejection_keeps_previous_registration(self):
        ref = SurfaceReference()
        ref.register(98_000.0)
        assert ref.register(0.0) is False
        assert ref.reference_pa == pytest.approx(98_000.0)

    def test_re_registration_overwrites(self):
        ref = SurfaceReference()
        ref.register(98_000.0)
        assert ref.register(103_000.0) is True
        assert ref.reference_pa == pytest.approx(103_000.0)
