"""Tier 1 unit tests for the pure conversion functions used by BCU control.

These functions are pure math (no ROS, no hardware) and form the
input/output edges of the BCU control pipeline:
  - pressure_to_depth: external pressure sensor (Pa) -> depth (m)
  - q_to_rpm: bladder flow ratio (1/s) -> motor RPM
"""

import pytest
from py_pkg.physics import (
    ATMOSPHERIC_PRESSURE_PA,
    pressure_to_depth,
    q_to_rpm,
)

BLADDER_VOLUME_M3 = 0.002275  # from depth_config.init_buoyancy_engine["tank_volume"]


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


class TestQToRpm:
    """q (flow ratio, 1/s) maps linearly to motor RPM."""

    def test_zero_q_is_zero_rpm(self):
        assert q_to_rpm(0, BLADDER_VOLUME_M3) == 0.0

    def test_signs_are_preserved(self):
        assert q_to_rpm(0.1, BLADDER_VOLUME_M3) > 0
        assert q_to_rpm(-0.1, BLADDER_VOLUME_M3) < 0

    def test_rpm_scales_linearly_with_q(self):
        rpm_a = q_to_rpm(0.05, BLADDER_VOLUME_M3)
        rpm_b = q_to_rpm(0.10, BLADDER_VOLUME_M3)
        assert rpm_b == pytest.approx(2 * rpm_a)

    def test_rpm_scales_linearly_with_volume(self):
        rpm_small = q_to_rpm(0.05, 0.001)
        rpm_large = q_to_rpm(0.05, 0.002)
        assert rpm_large == pytest.approx(2 * rpm_small)
