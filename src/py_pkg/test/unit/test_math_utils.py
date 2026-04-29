"""Tier 1 unit tests for py_pkg.math_utils.

Pure math (no ROS, no hardware). Pins down the contracts of the Vector
class, the Euler/quaternion conversions, and the scalar helpers (lerp,
clamp, clamp_mag, sign) that are consumed by the BCU/ACU/depth nodes.
"""

import numpy as np
import pytest
from py_pkg.math_utils import (
    Vector,
    clamp,
    clamp_mag,
    euler_to_direction,
    euler_to_rotation_matrix,
    lerp,
    pi,
    quaternion_to_roll_pitch,
    sign,
)


class TestVectorConstruction:
    """Vector(x, y, z) — defaults and accessors."""

    def test_default_is_zero_vector(self):
        v = Vector()
        assert v.x() == 0
        assert v.y() == 0
        assert v.z() == 0

    def test_explicit_components(self):
        v = Vector(1.0, 2.0, 3.0)
        assert v.x() == pytest.approx(1.0)
        assert v.y() == pytest.approx(2.0)
        assert v.z() == pytest.approx(3.0)

    def test_accessor_set_mutates_component(self):
        v = Vector(1.0, 2.0, 3.0)
        v.x(set=10.0)
        v.y(set=20.0)
        v.z(set=30.0)
        assert v.x() == pytest.approx(10.0)
        assert v.y() == pytest.approx(20.0)
        assert v.z() == pytest.approx(30.0)

    def test_str_contains_components(self):
        s = str(Vector(1.0, 2.0, 3.0))
        assert "1.0" in s and "2.0" in s and "3.0" in s


class TestVectorMagnitude:
    """magnitude() = ||vec||_2."""

    def test_zero_vector_has_zero_magnitude(self):
        assert Vector(0, 0, 0).magnitude() == pytest.approx(0.0)

    def test_unit_axis_has_unit_magnitude(self):
        assert Vector(1, 0, 0).magnitude() == pytest.approx(1.0)
        assert Vector(0, 1, 0).magnitude() == pytest.approx(1.0)
        assert Vector(0, 0, 1).magnitude() == pytest.approx(1.0)

    def test_3_4_5_triple(self):
        assert Vector(3.0, 4.0, 0.0).magnitude() == pytest.approx(5.0)

    def test_magnitude_returns_python_float(self):
        assert isinstance(Vector(1, 2, 3).magnitude(), float)


class TestVectorNormalized:
    """normalized() returns the unit vector; zero-vector returns zero."""

    def test_normalized_has_unit_length(self):
        n = Vector(3.0, 4.0, 0.0).normalized()
        assert n.magnitude() == pytest.approx(1.0)

    def test_normalized_preserves_direction(self):
        n = Vector(3.0, 4.0, 0.0).normalized()
        assert n.x() == pytest.approx(0.6)
        assert n.y() == pytest.approx(0.8)
        assert n.z() == pytest.approx(0.0)

    def test_zero_vector_normalized_is_zero(self):
        n = Vector(0, 0, 0).normalized()
        assert n.x() == 0 and n.y() == 0 and n.z() == 0

    def test_returns_new_vector(self):
        v = Vector(3.0, 4.0, 0.0)
        n = v.normalized()
        assert isinstance(n, Vector)
        assert n is not v


class TestVectorDotCross:
    """Hand-computed dot/cross results."""

    def test_dot_orthogonal_axes_is_zero(self):
        assert Vector(1, 0, 0).dot(Vector(0, 1, 0)) == pytest.approx(0.0)

    def test_dot_parallel_is_product_of_magnitudes(self):
        assert Vector(2, 0, 0).dot(Vector(3, 0, 0)) == pytest.approx(6.0)

    def test_dot_general(self):
        # (1,2,3) . (4,5,6) = 4 + 10 + 18 = 32
        assert Vector(1, 2, 3).dot(Vector(4, 5, 6)) == pytest.approx(32.0)

    def test_cross_x_y_is_z(self):
        c = Vector(1, 0, 0).cross(Vector(0, 1, 0))
        assert c.x() == pytest.approx(0.0)
        assert c.y() == pytest.approx(0.0)
        assert c.z() == pytest.approx(1.0)

    def test_cross_anticommutative(self):
        a, b = Vector(1, 2, 3), Vector(4, 5, 6)
        ab = a.cross(b)
        ba = b.cross(a)
        assert ab.x() == pytest.approx(-ba.x())
        assert ab.y() == pytest.approx(-ba.y())
        assert ab.z() == pytest.approx(-ba.z())

    def test_cross_with_self_is_zero(self):
        c = Vector(1, 2, 3).cross(Vector(1, 2, 3))
        assert c.x() == pytest.approx(0.0)
        assert c.y() == pytest.approx(0.0)
        assert c.z() == pytest.approx(0.0)


class TestVectorOperators:
    """__add__/__sub__/__mul__/__truediv__/__neg__ return new Vectors."""

    def test_add_componentwise(self):
        r = Vector(1, 2, 3) + Vector(4, 5, 6)
        assert isinstance(r, Vector)
        assert r.x() == pytest.approx(5)
        assert r.y() == pytest.approx(7)
        assert r.z() == pytest.approx(9)

    def test_sub_componentwise(self):
        r = Vector(4, 5, 6) - Vector(1, 2, 3)
        assert isinstance(r, Vector)
        assert r.x() == pytest.approx(3)
        assert r.y() == pytest.approx(3)
        assert r.z() == pytest.approx(3)

    def test_mul_by_scalar(self):
        r = Vector(1, 2, 3) * 2.0
        assert isinstance(r, Vector)
        assert r.x() == pytest.approx(2)
        assert r.y() == pytest.approx(4)
        assert r.z() == pytest.approx(6)

    def test_mul_by_zero_gives_zero(self):
        r = Vector(1, 2, 3) * 0
        assert r.x() == pytest.approx(0)
        assert r.y() == pytest.approx(0)
        assert r.z() == pytest.approx(0)

    def test_truediv_by_scalar(self):
        r = Vector(2, 4, 6) / 2.0
        assert isinstance(r, Vector)
        assert r.x() == pytest.approx(1)
        assert r.y() == pytest.approx(2)
        assert r.z() == pytest.approx(3)

    def test_neg_negates_each_component(self):
        r = -Vector(1, -2, 3)
        assert isinstance(r, Vector)
        assert r.x() == pytest.approx(-1)
        assert r.y() == pytest.approx(2)
        assert r.z() == pytest.approx(-3)

    def test_operators_do_not_mutate_operands(self):
        a = Vector(1, 2, 3)
        b = Vector(4, 5, 6)
        _ = a + b
        _ = a - b
        _ = a * 5
        _ = -a
        assert (a.x(), a.y(), a.z()) == (1, 2, 3)
        assert (b.x(), b.y(), b.z()) == (4, 5, 6)


class TestVectorRotate:
    """Rotation around an axis using Rodrigues' formula."""

    def test_rotate_x_axis_90_around_z_gives_y(self):
        rotated = Vector(1, 0, 0).rotate(pi / 2, Vector(0, 0, 1))
        assert rotated.x() == pytest.approx(0.0, abs=1e-9)
        assert rotated.y() == pytest.approx(1.0)
        assert rotated.z() == pytest.approx(0.0, abs=1e-9)

    def test_rotate_y_axis_90_around_x_gives_z(self):
        rotated = Vector(0, 1, 0).rotate(pi / 2, Vector(1, 0, 0))
        assert rotated.x() == pytest.approx(0.0, abs=1e-9)
        assert rotated.y() == pytest.approx(0.0, abs=1e-9)
        assert rotated.z() == pytest.approx(1.0)

    def test_rotate_zero_angle_is_identity(self):
        rotated = Vector(1, 2, 3).rotate(0.0, Vector(0, 0, 1))
        assert rotated.x() == pytest.approx(1.0)
        assert rotated.y() == pytest.approx(2.0)
        assert rotated.z() == pytest.approx(3.0)

    def test_rotate_uses_unnormalized_axis(self):
        # rotate() normalizes the axis internally — same result as unit axis
        rotated_a = Vector(1, 0, 0).rotate(pi / 2, Vector(0, 0, 5))
        rotated_b = Vector(1, 0, 0).rotate(pi / 2, Vector(0, 0, 1))
        assert rotated_a.x() == pytest.approx(rotated_b.x(), abs=1e-9)
        assert rotated_a.y() == pytest.approx(rotated_b.y(), abs=1e-9)
        assert rotated_a.z() == pytest.approx(rotated_b.z(), abs=1e-9)


class TestVectorModuloLoop:
    """Modulo and looping semantics."""

    def test_modulo_componentwise(self):
        r = Vector(7, 5, 3).modulo(4)
        assert r.x() == pytest.approx(3)
        assert r.y() == pytest.approx(1)
        assert r.z() == pytest.approx(3)

    def test_loop_keeps_in_range_value(self):
        r = Vector(1.5, 1.5, 1.5).loop(0.0, 3.0)
        assert r.x() == pytest.approx(1.5)
        assert r.y() == pytest.approx(1.5)
        assert r.z() == pytest.approx(1.5)

    def test_loop_wraps_above_high(self):
        # value 4.0, range [0, 3): wraps to 1.0
        r = Vector(4.0, 4.0, 4.0).loop(0.0, 3.0)
        assert r.x() == pytest.approx(1.0)
        assert r.y() == pytest.approx(1.0)
        assert r.z() == pytest.approx(1.0)

    def test_loop_wraps_pi_to_minus_pi_range(self):
        # 1.5*pi in [-pi, pi) wraps to -0.5*pi
        r = Vector(1.5 * pi, 0, 0).loop(-pi, pi)
        assert r.x() == pytest.approx(-0.5 * pi)


class TestEulerToDirection:
    """euler_to_direction(roll, pitch, yaw) -> unit-length forward vector."""

    def test_zero_angles_points_along_x(self):
        v = euler_to_direction(0, 0, 0)
        assert v.x() == pytest.approx(1.0)
        assert v.y() == pytest.approx(0.0)
        assert v.z() == pytest.approx(0.0)

    def test_yaw_90_points_along_y(self):
        v = euler_to_direction(0, 0, pi / 2)
        assert v.x() == pytest.approx(0.0, abs=1e-9)
        assert v.y() == pytest.approx(1.0)
        assert v.z() == pytest.approx(0.0)

    def test_pitch_up_points_along_z(self):
        v = euler_to_direction(0, pi / 2, 0)
        assert v.x() == pytest.approx(0.0, abs=1e-9)
        assert v.y() == pytest.approx(0.0, abs=1e-9)
        assert v.z() == pytest.approx(1.0)

    def test_unit_length(self):
        # Direction vectors should be unit length for non-degenerate angles
        v = euler_to_direction(0.3, 0.4, 0.5)
        assert v.magnitude() == pytest.approx(1.0)

    def test_roll_does_not_affect_direction(self):
        # Function ignores roll — verify by varying roll alone
        a = euler_to_direction(0.0, 0.2, 0.3)
        b = euler_to_direction(1.0, 0.2, 0.3)
        assert a.x() == pytest.approx(b.x())
        assert a.y() == pytest.approx(b.y())
        assert a.z() == pytest.approx(b.z())


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


class TestEulerToRotationMatrix:
    """Rotation matrix from Vector(roll, pitch, yaw)."""

    def test_zero_angles_returns_identity(self):
        R = euler_to_rotation_matrix(Vector(0, 0, 0))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_returns_3x3_ndarray(self):
        R = euler_to_rotation_matrix(Vector(0.1, 0.2, 0.3))
        assert isinstance(R, np.ndarray)
        assert R.shape == (3, 3)

    def test_is_orthogonal(self):
        R = euler_to_rotation_matrix(Vector(0.3, 0.4, 0.5))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_determinant_is_one(self):
        R = euler_to_rotation_matrix(Vector(0.3, 0.4, 0.5))
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_pure_yaw_rotates_x_to_y(self):
        R = euler_to_rotation_matrix(Vector(0, 0, pi / 2))
        rotated = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-9)


class TestLerp:
    """Linear interpolation."""

    def test_t_zero_returns_start(self):
        assert lerp(2.0, 8.0, 0.0) == pytest.approx(2.0)

    def test_t_one_returns_end(self):
        assert lerp(2.0, 8.0, 1.0) == pytest.approx(8.0)

    def test_t_half_is_midpoint(self):
        assert lerp(2.0, 8.0, 0.5) == pytest.approx(5.0)

    def test_extrapolates_beyond_one(self):
        # lerp does not clamp t — caller's responsibility
        assert lerp(0.0, 10.0, 2.0) == pytest.approx(20.0)

    def test_extrapolates_below_zero(self):
        assert lerp(0.0, 10.0, -1.0) == pytest.approx(-10.0)

    def test_equal_endpoints_constant(self):
        assert lerp(5.0, 5.0, 0.3) == pytest.approx(5.0)


class TestClamp:
    """clamp(val, min, max) — saturates val into [min, max]."""

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
        # min == max → val is forced to that single value
        assert clamp(0.7, 0.5, 0.5) == pytest.approx(0.5)
        assert clamp(0.3, 0.5, 0.5) == pytest.approx(0.5)

    def test_inverted_bounds_min_greater_than_max(self):
        # Existing behavior: with min > max, the val > max branch fires first,
        # so any value >= max is pulled down to max — even values that are
        # below min. This encodes the actual implementation, not a desired
        # contract. Callers should not pass min > max.
        assert clamp(10.0, 5.0, 1.0) == pytest.approx(1.0)
        assert clamp(3.0, 5.0, 1.0) == pytest.approx(1.0)
        # Only values strictly below max can hit the val < min branch and
        # be promoted to min.
        assert clamp(0.0, 5.0, 1.0) == pytest.approx(5.0)


class TestClampMag:
    """clamp_mag(val, max_mag) — symmetric saturation around zero."""

    def test_in_range_passes_through(self):
        assert clamp_mag(0.5, 1.0) == pytest.approx(0.5)
        assert clamp_mag(-0.5, 1.0) == pytest.approx(-0.5)

    def test_above_max_clamps_to_positive_max(self):
        assert clamp_mag(5.0, 1.0) == pytest.approx(1.0)

    def test_below_neg_max_clamps_to_negative_max(self):
        assert clamp_mag(-5.0, 1.0) == pytest.approx(-1.0)

    def test_zero_max_mag_clamps_everything_to_zero(self):
        assert clamp_mag(3.0, 0.0) == pytest.approx(0.0)
        assert clamp_mag(-3.0, 0.0) == pytest.approx(0.0)

    def test_at_boundary_passes_through(self):
        assert clamp_mag(1.0, 1.0) == pytest.approx(1.0)
        assert clamp_mag(-1.0, 1.0) == pytest.approx(-1.0)


class TestSign:
    """sign(val) returns 1, 0, or -1."""

    def test_positive_returns_one(self):
        assert sign(3.5) == 1
        assert sign(1) == 1

    def test_zero_returns_zero(self):
        assert sign(0) == 0
        assert sign(0.0) == 0

    def test_negative_returns_minus_one(self):
        assert sign(-3.5) == -1
        assert sign(-1) == -1

    def test_returns_int(self):
        assert isinstance(sign(2.0), int)
        assert isinstance(sign(0.0), int)
        assert isinstance(sign(-2.0), int)


class TestPiConstant:
    """pi is re-exported from numpy."""

    def test_pi_matches_numpy(self):
        assert pi == pytest.approx(np.pi)
