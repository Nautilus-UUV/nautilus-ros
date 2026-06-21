"""Tier 1 unit tests for the STM IMU conversions.

Pure math (no ROS, no hardware): raw int16 sensor counts -> SI, plus the
sensor -> NED body-frame axis/sign remap. These tests are the executable record
of the firmware sign table -- if a real axis comes out wrong at bring-up and an
entry in IMU_ACCEL_AXIS_MAP / IMU_GYRO_AXIS_MAP gets flipped, the matching
assertion below moves with it.

  - accel_counts_to_mps2: count -> m/s^2 (±6 g full scale)
  - gyro_counts_to_rads:  count -> rad/s (±2000 °/s full scale)
  - imu_counts_to_body:   six counts -> SI accel + gyro in the NED body frame
"""

import math

import pytest
from py_pkg.physics import (
    GRAVITY_M_S2,
    accel_counts_to_mps2,
    gyro_counts_to_rads,
    imu_counts_to_body,
)

# +1 g lands at full-scale/6 counts: 32768 / 6.
ONE_G_COUNTS = round(32768 / 6)


class TestAccelScale:
    """±6 g full scale: count 32768 == 6 g, linear and signed about zero."""

    def test_full_scale_is_six_g(self):
        assert accel_counts_to_mps2(32768) == pytest.approx(6.0 * GRAVITY_M_S2)

    def test_one_g_count(self):
        assert accel_counts_to_mps2(ONE_G_COUNTS) == pytest.approx(GRAVITY_M_S2, abs=0.01)

    def test_zero(self):
        assert accel_counts_to_mps2(0) == 0.0

    def test_sign_preserved(self):
        assert accel_counts_to_mps2(-1234) == pytest.approx(-accel_counts_to_mps2(1234))


class TestGyroScale:
    """±2000 °/s full scale: count 32768 == 2000 °/s expressed in rad/s."""

    def test_full_scale_is_two_thousand_dps(self):
        assert gyro_counts_to_rads(32768) == pytest.approx(2000.0 * math.pi / 180.0)

    def test_zero(self):
        assert gyro_counts_to_rads(0) == 0.0

    def test_sign_preserved(self):
        assert gyro_counts_to_rads(-500) == pytest.approx(-gyro_counts_to_rads(500))


def _body(ax=0, ay=0, az=0, gx=0, gy=0, gz=0):
    """imu_counts_to_body with named sensor-frame counts, defaulting to zero."""
    return imu_counts_to_body(ax, ay, az, gx, gy, gz)


class TestAxisRemapAccel:
    """Documented accel directions land on the right NED body axis.

    Sensor table: +ay = forward, +az = up. The +ax axis was confirmed at
    bring-up to point body-RIGHT, which IS +y in the NED body frame (x forward,
    y right, z DOWN), so it maps to POSITIVE body y. +az (sensor up) maps to
    NEGATIVE body z, because NED z points down.
    """

    def test_forward_accel_is_body_x(self):
        accel, _ = _body(ay=1000)
        assert accel[0] > 0
        assert accel[1] == 0 and accel[2] == 0
        # Body x magnitude is just the scaled forward count.
        assert accel[0] == pytest.approx(accel_counts_to_mps2(1000))

    def test_sensor_ax_is_positive_body_y(self):
        # +ax (sensor) points body-right, which is +y in NED.
        accel, _ = _body(ax=1000)
        assert accel[1] > 0
        assert accel[0] == 0 and accel[2] == 0
        assert accel[1] == pytest.approx(accel_counts_to_mps2(1000))

    def test_up_accel_is_negative_body_z(self):
        # Sensor +az points up; NED body z points down, so up -> negative body z.
        accel, _ = _body(az=1000)
        assert accel[2] < 0
        assert accel[0] == 0 and accel[1] == 0

    def test_static_level_reads_minus_g_on_z(self):
        # IMU sitting level: +1 g of specific force up -> body a_z ≈ -9.81 (NED).
        accel, _ = _body(az=ONE_G_COUNTS)
        assert accel[2] == pytest.approx(-GRAVITY_M_S2, abs=0.01)
        assert accel[0] == 0 and accel[1] == 0


class TestAxisRemapGyro:
    """Documented rotations land on the right NED body rate + sign.

    Sensor table: +gx = pitch up, +gy = roll right, +gz = yaw left. In NED
    (x forward, y right, z down) positive pitch about +y is nose-UP, so a
    documented pitch-up reads POSITIVE on body w_y. Roll-right stays positive on
    w_x; yaw-left reads NEGATIVE on w_z (positive yaw about NED +z is yaw-right).
    """

    def test_roll_right_is_positive_body_x(self):
        _, gyro = _body(gy=1000)
        assert gyro[0] > 0
        assert gyro[1] == 0 and gyro[2] == 0

    def test_pitch_up_is_positive_body_y(self):
        _, gyro = _body(gx=1000)
        assert gyro[1] > 0
        assert gyro[0] == 0 and gyro[2] == 0

    def test_yaw_left_is_negative_body_z(self):
        _, gyro = _body(gz=1000)
        assert gyro[2] < 0
        assert gyro[0] == 0 and gyro[1] == 0
