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
    deadband_snap,
    gravity_to_roll_pitch,
    quaternion_to_roll_pitch,
    quaternion_to_yaw,
    rpy_to_quaternion,
    span_band_guards,
)

from _attitude_helpers import specific_force as _specific_force

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


class TestGravityToRollPitch:
    """(roll, pitch) inferred from the NED specific-force (gravity) vector.

    Must agree with rpy_to_quaternion / quaternion_to_roll_pitch so the estimator
    feeds the ACU roll PID the same sign convention it expects.
    """

    def test_level_reads_zero(self):
        # NED z points down, so the level gravity reaction is -g on body z.
        roll, pitch = gravity_to_roll_pitch(0.0, 0.0, -9.806)
        assert roll == pytest.approx(0.0)
        assert pitch == pytest.approx(0.0)

    @pytest.mark.parametrize("roll_deg", [-40.0, -10.0, 5.0, 30.0, 60.0])
    def test_pure_roll_recovered(self, roll_deg):
        roll = math.radians(roll_deg)
        ax, ay, az = _specific_force(roll, 0.0, 0.0)
        r, p = gravity_to_roll_pitch(ax, ay, az)
        assert r == pytest.approx(roll, abs=1e-9)
        assert p == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("pitch_deg", [-50.0, -15.0, 10.0, 45.0])
    def test_pure_pitch_recovered(self, pitch_deg):
        pitch = math.radians(pitch_deg)
        ax, ay, az = _specific_force(0.0, pitch, 0.0)
        r, p = gravity_to_roll_pitch(ax, ay, az)
        assert r == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(pitch, abs=1e-9)

    @pytest.mark.parametrize("roll_deg", [-30.0, 0.0, 25.0])
    @pytest.mark.parametrize("pitch_deg", [-40.0, 0.0, 35.0])
    def test_combined_round_trip(self, roll_deg, pitch_deg):
        roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
        ax, ay, az = _specific_force(roll, pitch, 0.0)
        r, p = gravity_to_roll_pitch(ax, ay, az)
        assert r == pytest.approx(roll, abs=1e-9)
        assert p == pytest.approx(pitch, abs=1e-9)

    @pytest.mark.parametrize("yaw_deg", [0.0, 37.0, 123.0, -90.0])
    def test_yaw_does_not_affect_result(self, yaw_deg):
        # Gravity is yaw-blind: spinning about the vertical can't move the vector.
        roll, pitch = math.radians(20.0), math.radians(-15.0)
        ax, ay, az = _specific_force(roll, pitch, math.radians(yaw_deg))
        r, p = gravity_to_roll_pitch(ax, ay, az)
        assert r == pytest.approx(roll, abs=1e-9)
        assert p == pytest.approx(pitch, abs=1e-9)

    def test_magnitude_independent(self):
        # atan2 ratios -> the vector's length (g) is irrelevant.
        base = gravity_to_roll_pitch(1.0, -2.0, 9.0)
        scaled = gravity_to_roll_pitch(5.0, -10.0, 45.0)
        assert scaled[0] == pytest.approx(base[0])
        assert scaled[1] == pytest.approx(base[1])


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


class TestDeadbandSnap:
    """deadband_snap(val, zero_below, snap_to, limit) — the BCU pump deadband.

    Tuned to the nominal scenario: zero_below=500, snap_to=1000, limit=4000.
    """

    ZERO_BELOW = 500
    SNAP_TO = 1000
    LIMIT = 4000

    def _snap(self, val):
        return deadband_snap(val, self.ZERO_BELOW, self.SNAP_TO, self.LIMIT)

    def test_zero_passes_through(self):
        assert self._snap(0.0) == 0.0

    @pytest.mark.parametrize("val", [-499.0, -1.0, 1.0, 250.0, 499.0])
    def test_below_zero_threshold_suppressed_to_zero(self, val):
        assert self._snap(val) == 0.0

    @pytest.mark.parametrize(
        "val,expected",
        [
            (500.0, 1000.0),   # boundary snaps up, not down
            (750.0, 1000.0),
            (999.0, 1000.0),
            (-500.0, -1000.0),
            (-750.0, -1000.0),
        ],
    )
    def test_in_deadband_snaps_up_to_edge(self, val, expected):
        assert self._snap(val) == pytest.approx(expected)

    @pytest.mark.parametrize("val", [1000.0, -1000.0, 2500.0, -2500.0, 4000.0, -4000.0])
    def test_above_edge_passes_through(self, val):
        assert self._snap(val) == pytest.approx(val)

    @pytest.mark.parametrize(
        "val,expected", [(5000.0, 4000.0), (-5000.0, -4000.0), (1e9, 4000.0)]
    )
    def test_beyond_limit_saturates(self, val, expected):
        assert self._snap(val) == pytest.approx(expected)

    def test_sign_is_preserved(self):
        assert self._snap(700.0) > 0
        assert self._snap(-700.0) < 0

    @pytest.mark.parametrize("val", [-9000.0, -3000.0, -750.0, -10.0, 0.0, 10.0, 3000.0])
    def test_zero_thresholds_reduce_to_plain_clamp(self, val):
        # zero_below == snap_to == 0 → no deadband, just a ±limit clamp.
        # This is the dataclass default, so a bare DepthSpec() is unchanged.
        assert deadband_snap(val, 0, 0, self.LIMIT) == pytest.approx(
            clamp(val, -self.LIMIT, self.LIMIT)
        )


class TestSpanBandGuards:
    """span_band_guards(lo, hi, band) — insets both ends by band·span.

    Shared by the depth controller's tank clamp and the lifeguard blow
    stand-down. Pinned to the BCU tank endpoints (empty=70k, full=150k,
    span=80k); at band=0.10 the guards land at 78k / 142k.
    """

    EMPTY = 70_000.0
    FULL = 150_000.0

    def test_ten_percent_of_span(self):
        low, high = span_band_guards(self.EMPTY, self.FULL, 0.10)
        assert low == pytest.approx(78_000.0)
        assert high == pytest.approx(142_000.0)

    def test_zero_band_returns_endpoints(self):
        assert span_band_guards(self.EMPTY, self.FULL, 0.0) == (
            pytest.approx(self.EMPTY),
            pytest.approx(self.FULL),
        )

    def test_margin_is_symmetric_in_absolute_pa(self):
        # band·span is inset equally from each end -- the basis is the span,
        # not the (different-magnitude) endpoint values.
        low, high = span_band_guards(self.EMPTY, self.FULL, 0.25)
        span = self.FULL - self.EMPTY
        assert low - self.EMPTY == pytest.approx(self.FULL - high)
        assert low - self.EMPTY == pytest.approx(0.25 * span)
