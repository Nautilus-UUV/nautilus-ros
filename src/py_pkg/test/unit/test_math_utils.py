"""Tier 1 unit tests for py_pkg.math_utils.

Pure math (no ROS, no hardware). Pins down the contracts of the
quaternion <-> Euler conversions and the saturating clamp helper that
the BCU/ACU/depth nodes consume.
"""

import math

import numpy as np
import pytest
from py_pkg.math_utils import (
    clamp,
    quaternion_to_roll_pitch,
    quaternion_to_yaw,
    rpy_to_quaternion,
)

pi = np.pi


class TestQuaternionToRollPitch:
    """ZYX Tait-Bryan extraction of (roll, pitch) from (qx, qy, qz, qw)."""

    def test_identity_quaternion_is_zero(self):
        roll, pitch = quaternion_to_roll_pitch(0.0, 0.0, 0.0, 1.0)
        assert roll == pytest.approx(0.0)
        assert pitch == pytest.approx(0.0)

    def test_pure_roll_90(self):
        # q = (sin(45deg), 0, 0, cos(45deg)) -> roll = pi/2
        s, c = np.sin(pi / 4), np.cos(pi / 4)
        roll, pitch = quaternion_to_roll_pitch(s, 0.0, 0.0, c)
        assert roll == pytest.approx(pi / 2)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    def test_pure_pitch_45(self):
        # q = (0, sin(22.5deg), 0, cos(22.5deg)) -> pitch = pi/4
        s, c = np.sin(pi / 8), np.cos(pi / 8)
        roll, pitch = quaternion_to_roll_pitch(0.0, s, 0.0, c)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(pi / 4)

    def test_gimbal_lock_pitch_clamped_to_half_pi(self):
        # qy = qw = sqrt(2)/2, others zero -> sinp = 1 → pitch = +pi/2
        s = np.sqrt(2) / 2
        _, pitch = quaternion_to_roll_pitch(0.0, s, 0.0, s)
        assert pitch == pytest.approx(pi / 2)

    def test_returns_tuple_of_two(self):
        result = quaternion_to_roll_pitch(0.0, 0.0, 0.0, 1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestQuaternionToYaw:
    """ZYX Tait-Bryan extraction of yaw from (qx, qy, qz, qw)."""

    def test_identity_quaternion_yields_zero_yaw(self):
        assert quaternion_to_yaw(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        "yaw_deg",
        [-170.0, -90.0, -45.0, -1.0, 1.0, 45.0, 90.0, 170.0],
    )
    def test_round_trip_via_rpy_to_quaternion(self, yaw_deg):
        yaw = math.radians(yaw_deg)
        qx, qy, qz, qw = rpy_to_quaternion(0.0, 0.0, yaw)
        recovered = quaternion_to_yaw(qx, qy, qz, qw)
        assert recovered == pytest.approx(yaw, abs=1e-9)

    def test_pitch_does_not_leak_into_extracted_yaw(self):
        qx, qy, qz, qw = rpy_to_quaternion(0.0, math.radians(40.0), 0.0)
        assert quaternion_to_yaw(qx, qy, qz, qw) == pytest.approx(0.0, abs=1e-9)


class TestRpyToQuaternion:
    """(roll, pitch, yaw) -> (qx, qy, qz, qw), inverse of the extractors."""

    def test_zero_rpy_is_identity(self):
        qx, qy, qz, qw = rpy_to_quaternion(0.0, 0.0, 0.0)
        assert (qx, qy, qz) == (
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(0.0),
        )
        assert qw == pytest.approx(1.0)

    def test_pitch_only_has_zero_qx_and_qz(self):
        # roll=0, yaw=0 -> qx and qz vanish, only qy + qw are non-zero.
        qx, qy, qz, _ = rpy_to_quaternion(0.0, math.radians(35.0), 0.0)
        assert qx == pytest.approx(0.0, abs=1e-12)
        assert qz == pytest.approx(0.0, abs=1e-12)
        assert qy != pytest.approx(0.0)

    def test_pitch_round_trips_through_quaternion_to_roll_pitch(self):
        # Ensures the planner's pose orientations decode to the input pitch
        # under the math_utils extractor that downstream nodes use.
        for pitch_deg in (-35.0, -10.0, 5.0, 35.0):
            qx, qy, qz, qw = rpy_to_quaternion(0.0, math.radians(pitch_deg), 0.0)
            roll, pitch = quaternion_to_roll_pitch(qx, qy, qz, qw)
            assert roll == pytest.approx(0.0, abs=1e-9)
            assert math.degrees(pitch) == pytest.approx(pitch_deg, abs=1e-6)


class TestClamp:
    """clamp(val, lo, hi) — saturates val into [lo, hi]."""

    def test_in_range_passes_through(self):
        assert clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_above_max_saturates_high(self):
        assert clamp(2.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_below_min_saturates_low(self):
        assert clamp(-1.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_at_min_returns_min(self):
        assert clamp(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_at_max_returns_max(self):
        assert clamp(1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_equal_bounds_returns_that_bound(self):
        # lo == hi → val is forced to that single value
        assert clamp(0.7, 0.5, 0.5) == pytest.approx(0.5)
        assert clamp(0.3, 0.5, 0.5) == pytest.approx(0.5)
