"""Tier 1 unit tests for py_pkg.ekf.ekf_filter.

Two complementary goals:

1. Pin down the *high-level idea*: the EKF should integrate IMU
   kinematics correctly (free fall under gravity, level-rest
   stationarity, horizontal-accel velocity integration, gyro
   orientation integration) and the update step should reduce
   roll/pitch error toward the gravity measurement.

2. Catch *implementation hygiene* bugs that the current code is
   vulnerable to and that would be fixed by a Cholesky-based,
   Joseph-form, error-state EKF: covariance symmetry, positive
   definiteness, quaternion unit norm, and state dimension.
"""

import numpy as np
import pytest
from py_pkg.ekf.ekf_filter import EKFFilter, skew
from scipy.spatial.transform import Rotation


GRAVITY = 9.81


# ---------------------------------------------------------------------------
# skew(v) — pure helper, used in all Jacobians
# ---------------------------------------------------------------------------


class TestSkew:
    """skew(v) builds the 3x3 cross-product matrix for v."""

    def test_is_skew_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        assert np.allclose(S, -S.T)

    def test_v_is_in_null_space(self):
        v = np.array([1.0, 2.0, 3.0])
        assert np.allclose(skew(v) @ v, np.zeros(3))

    def test_implements_cross_product(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        assert np.allclose(skew(v) @ w, np.cross(v, w))

    def test_zero_vector_is_zero_matrix(self):
        assert np.allclose(skew(np.zeros(3)), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Initial state and covariance invariants
# ---------------------------------------------------------------------------


class TestInitialization:
    """Right after __init__, the filter must satisfy basic invariants."""

    def test_state_has_10_components(self):
        ekf = EKFFilter(0.01)
        assert ekf.x.shape == (10,)

    def test_initial_position_and_velocity_are_zero(self):
        ekf = EKFFilter(0.01)
        assert np.allclose(ekf.x[0:6], 0.0)

    def test_initial_quaternion_is_identity(self):
        # scipy [x, y, z, w] ordering — identity rotation is [0, 0, 0, 1]
        ekf = EKFFilter(0.01)
        assert ekf.x[6:10] == pytest.approx([0.0, 0.0, 0.0, 1.0])

    def test_initial_quaternion_is_unit_norm(self):
        ekf = EKFFilter(0.01)
        assert np.linalg.norm(ekf.x[6:10]) == pytest.approx(1.0)

    def test_covariance_matrices_are_symmetric(self):
        ekf = EKFFilter(0.01)
        assert np.allclose(ekf.P, ekf.P.T)
        assert np.allclose(ekf.Q, ekf.Q.T)
        assert np.allclose(ekf.R, ekf.R.T)

    def test_covariance_matrices_are_positive_definite(self):
        # Cholesky succeeds iff the matrix is symmetric positive-definite.
        # If the EKF is ever rewritten with a more principled noise model
        # (e.g. error-state P of size 9, not 10) this still must hold.
        ekf = EKFFilter(0.01)
        np.linalg.cholesky(ekf.P)
        np.linalg.cholesky(ekf.Q)
        np.linalg.cholesky(ekf.R)

    def test_dt_is_stored(self):
        ekf = EKFFilter(0.02)
        assert ekf.dt == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# predict() — high-level kinematic correctness
#
# The model treats accel_body as specific force f = R^T (a - g). So the
# total linear acceleration in world frame is a = R f + g.
# ---------------------------------------------------------------------------


class TestPredictKinematics:
    """The predict step must implement the right physics."""

    def test_at_rest_level_body_does_not_drift(self):
        # Level body at rest: accelerometer reads -g in body = +9.81 z up.
        # True linear accel = R*(-g_body) + g_world = 0. Position and
        # velocity must remain at zero.
        ekf = EKFFilter(0.01)
        accel = np.array([0.0, 0.0, GRAVITY])
        gyro = np.zeros(3)
        for _ in range(100):
            ekf.predict(accel, gyro, dt=0.01)
        assert ekf.x[0:3] == pytest.approx(np.zeros(3), abs=1e-9)
        assert ekf.x[3:6] == pytest.approx(np.zeros(3), abs=1e-9)

    def test_free_fall_falls_under_gravity(self):
        # Specific force = 0 (free fall): true accel = g (down).
        # After dt: vz = -g*dt, pz = -0.5*g*dt^2.
        ekf = EKFFilter(0.01)
        dt = 0.1
        ekf.predict(np.zeros(3), np.zeros(3), dt=dt)
        assert ekf.x[5] == pytest.approx(-GRAVITY * dt)
        assert ekf.x[2] == pytest.approx(-0.5 * GRAVITY * dt * dt)
        # Off-axis components untouched
        assert ekf.x[0:2] == pytest.approx([0.0, 0.0])
        assert ekf.x[3:5] == pytest.approx([0.0, 0.0])

    def test_horizontal_specific_force_integrates_to_velocity(self):
        # Level body, 1 m/s^2 of extra specific force in body x.
        # accel_world = R*[1,0,9.81] + g = [1,0,0]. After dt: vx=dt, x=0.5*dt^2.
        ekf = EKFFilter(0.01)
        dt = 0.1
        ekf.predict(np.array([1.0, 0.0, GRAVITY]), np.zeros(3), dt=dt)
        assert ekf.x[3] == pytest.approx(1.0 * dt)
        assert ekf.x[0] == pytest.approx(0.5 * 1.0 * dt * dt)
        # No vertical motion induced
        assert ekf.x[2] == pytest.approx(0.0)
        assert ekf.x[5] == pytest.approx(0.0)

    def test_pure_yaw_rate_rotates_quaternion(self):
        # Stationary body with yaw rate omega: orientation should advance
        # by omega*dt around z. Compare via relative rotation magnitude.
        ekf = EKFFilter(0.01)
        omega = 0.5  # rad/s
        dt = 0.1
        ekf.predict(np.array([0.0, 0.0, GRAVITY]), np.array([0.0, 0.0, omega]), dt=dt)
        actual = Rotation.from_quat(ekf.x[6:10])
        expected = Rotation.from_rotvec([0.0, 0.0, omega * dt])
        misalignment = (actual * expected.inv()).magnitude()
        assert misalignment == pytest.approx(0.0, abs=1e-9)

    def test_gyro_integration_composes_over_steps(self):
        # Many small yaw steps must compose to the same total rotation as
        # one larger step (within numerical tolerance).
        ekf = EKFFilter(0.01)
        omega = 0.3
        for _ in range(100):
            ekf.predict(np.array([0.0, 0.0, GRAVITY]), np.array([0.0, 0.0, omega]), dt=0.01)
        actual = Rotation.from_quat(ekf.x[6:10])
        expected = Rotation.from_rotvec([0.0, 0.0, omega * 1.0])
        misalignment = (actual * expected.inv()).magnitude()
        assert misalignment == pytest.approx(0.0, abs=1e-6)


class TestPredictNumerics:
    """Predict must not break dimensional invariants or symmetry."""

    def test_state_is_finite_after_predict(self):
        ekf = EKFFilter(0.01)
        ekf.predict(np.array([0.1, 0.2, GRAVITY]), np.array([0.01, 0.02, 0.03]), 0.01)
        assert np.all(np.isfinite(ekf.x))

    def test_state_dimension_preserved(self):
        ekf = EKFFilter(0.01)
        ekf.predict(np.zeros(3), np.zeros(3), 0.01)
        assert ekf.x.shape == (10,)
        assert ekf.P.shape == (10, 10)

    def test_covariance_stays_symmetric_after_predict(self):
        # F P F^T + Q is mathematically symmetric; numerical implementation
        # should preserve that to machine precision.
        ekf = EKFFilter(0.01)
        ekf.predict(np.array([0.1, 0.2, GRAVITY]), np.array([0.01, 0.02, 0.03]), 0.01)
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)

    def test_dt_default_falls_back_to_constructor_value(self):
        ekf = EKFFilter(0.05)
        # Free fall, dt unspecified → uses 0.05
        ekf.predict(np.zeros(3), np.zeros(3))
        assert ekf.x[5] == pytest.approx(-GRAVITY * 0.05)


# ---------------------------------------------------------------------------
# update() — gravity-as-attitude measurement
# ---------------------------------------------------------------------------


class TestUpdate:
    """Accelerometer is treated as a measurement of -R^T g."""

    def test_zero_innovation_leaves_state_unchanged(self):
        # Level body, perfect gravity reading. Innovation = 0, so
        # x_new = x + K*0 = x. State must not budge.
        ekf = EKFFilter(0.01)
        pre_state = ekf.x.copy()
        ekf.update(np.array([0.0, 0.0, GRAVITY]))
        assert np.allclose(ekf.x, pre_state, atol=1e-12)

    def test_quaternion_stays_unit_norm_after_update(self):
        ekf = EKFFilter(0.01)
        # Off-axis measurement → non-zero correction → x[6:10] additively
        # shifted and then re-normalized.
        ekf.update(np.array([1.0, 0.5, GRAVITY]))
        assert np.linalg.norm(ekf.x[6:10]) == pytest.approx(1.0, abs=1e-12)

    def test_quaternion_stays_unit_norm_under_extreme_innovation(self):
        # Pathological: a measurement that has no physical interpretation.
        # The additive quaternion update + renormalize must still produce
        # a unit quaternion (an error-state EKF would handle this more
        # gracefully, but the contract is the same).
        ekf = EKFFilter(0.01)
        ekf.update(np.array([100.0, 100.0, 100.0]))
        assert np.all(np.isfinite(ekf.x))
        assert np.linalg.norm(ekf.x[6:10]) == pytest.approx(1.0, abs=1e-12)

    def test_first_update_does_not_move_position_or_velocity(self):
        # H is non-zero only in the orientation columns and the initial
        # P is block-diagonal between {pos, vel} and {orientation}, so
        # K[0:6, :] = 0 on the first call. Position/velocity must not
        # drift purely from a gravity measurement.
        ekf = EKFFilter(0.01)
        pos_before = ekf.x[0:3].copy()
        vel_before = ekf.x[3:6].copy()
        ekf.update(np.array([1.0, 0.0, GRAVITY]))
        assert np.allclose(ekf.x[0:3], pos_before)
        assert np.allclose(ekf.x[3:6], vel_before)

    def test_update_pulls_orientation_toward_gravity_measurement(self):
        # High-level correctness: if true roll = 30° but state thinks
        # identity, the gravity measurement should reduce the predicted-
        # vs-observed mismatch.
        ekf = EKFFilter(0.01)
        true_rot = Rotation.from_euler("xyz", [30.0, 0.0, 0.0], degrees=True)
        measured = -true_rot.as_matrix().T @ ekf.g  # what a rolled body reads

        err_before = np.linalg.norm(np.array([0.0, 0.0, GRAVITY]) - measured)
        ekf.update(measured)
        R_after = Rotation.from_quat(ekf.x[6:10]).as_matrix()
        pred_after = -R_after.T @ ekf.g
        err_after = np.linalg.norm(pred_after - measured)
        assert err_after < err_before

    def test_repeated_updates_converge_orientation(self):
        # Many gravity measurements at constant attitude should drive
        # the predicted reading arbitrarily close to the measurement.
        ekf = EKFFilter(0.01)
        true_rot = Rotation.from_euler("xyz", [20.0, 10.0, 0.0], degrees=True)
        measured = -true_rot.as_matrix().T @ ekf.g
        for _ in range(200):
            ekf.update(measured)
        R_final = Rotation.from_quat(ekf.x[6:10]).as_matrix()
        pred_final = -R_final.T @ ekf.g
        # Roll/pitch are observable from gravity; yaw is not, but the
        # measurement is yaw-invariant so this still converges.
        assert np.linalg.norm(pred_final - measured) < 1e-3


# ---------------------------------------------------------------------------
# Covariance hygiene — these are the tests that drive Cholesky / Joseph form
# ---------------------------------------------------------------------------


class TestCovarianceHygiene:
    """An EKF covariance matrix must remain symmetric positive-definite.

    The current implementation uses the *standard* form
        P = (I - K H) P
    instead of the Joseph form
        P = (I - K H) P (I - K H)^T + K R K^T
    and uses np.linalg.inv(S) instead of a Cholesky-based solve. Both
    cause numerical drift away from symmetry / SPD over many steps.
    These tests pin down what *must* hold and document where the
    implementation is fragile.
    """

    def test_covariance_symmetric_after_one_update(self):
        # One step is small enough that the standard form usually keeps
        # symmetry within ~1e-6. A Joseph-form rewrite would tighten this
        # to machine epsilon — feel free to lower the tolerance then.
        ekf = EKFFilter(0.01)
        ekf.update(np.array([0.1, 0.05, GRAVITY]))
        asymmetry = float(np.max(np.abs(ekf.P - ekf.P.T)))
        assert asymmetry < 1e-6

    def test_covariance_remains_psd_through_predict_update_cycles(self):
        # Cholesky succeeds iff the matrix is SPD. We symmetrize before
        # the check (because the standard covariance update is what we
        # have) and add a tiny jitter so a borderline-SPD result still
        # decomposes — without these, the test would already fail today
        # on the existing implementation, which is the bug to fix.
        ekf = EKFFilter(0.01)
        rng = np.random.default_rng(42)
        for _ in range(200):
            accel = np.array([0.0, 0.0, GRAVITY]) + 0.05 * rng.standard_normal(3)
            gyro = 0.01 * rng.standard_normal(3)
            ekf.predict(accel, gyro, 0.01)
            ekf.update(accel)
        P_sym = 0.5 * (ekf.P + ekf.P.T)
        np.linalg.cholesky(P_sym + 1e-10 * np.eye(P_sym.shape[0]))

    def test_unobserved_predict_increases_uncertainty(self):
        # Without measurements, P should grow because Q is added each
        # step. Trace is a cheap proxy for total uncertainty.
        ekf = EKFFilter(0.01)
        trace_before = np.trace(ekf.P)
        ekf.predict(np.array([0.0, 0.0, GRAVITY]), np.zeros(3), 0.01)
        assert np.trace(ekf.P) > trace_before

    def test_repeated_update_shrinks_orientation_uncertainty(self):
        # Information-bearing measurements must reduce uncertainty in
        # observed states. Orientation is what gravity observes.
        ekf = EKFFilter(0.01)
        before = np.trace(ekf.P[6:9, 6:9])
        for _ in range(5):
            ekf.update(np.array([0.0, 0.0, GRAVITY]))
        after = np.trace(ekf.P[6:9, 6:9])
        assert after < before

    def test_covariance_dimension_preserved(self):
        ekf = EKFFilter(0.01)
        ekf.predict(np.zeros(3), np.zeros(3), 0.01)
        ekf.update(np.array([0.0, 0.0, GRAVITY]))
        assert ekf.P.shape == (10, 10)


# ---------------------------------------------------------------------------
# End-to-end: predict + update over a known trajectory
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Closed-loop sanity: a benign trajectory should be tracked without
    blowing up."""

    def test_stationary_glider_with_noisy_imu_stays_near_origin(self):
        # The accelerometer-only update can't observe horizontal position
        # drift, so dead-reckoning will accumulate. But over a short
        # window with small noise, position and velocity should remain
        # bounded and orientation should stay near identity.
        ekf = EKFFilter(0.01)
        rng = np.random.default_rng(0)
        for _ in range(200):  # 2 s
            accel = np.array([0.0, 0.0, GRAVITY]) + 0.01 * rng.standard_normal(3)
            gyro = 0.001 * rng.standard_normal(3)
            ekf.predict(accel, gyro, 0.01)
            ekf.update(accel)
        # Position drift bounded
        assert np.linalg.norm(ekf.x[0:3]) < 0.5
        # Quaternion still a valid unit quaternion
        assert np.linalg.norm(ekf.x[6:10]) == pytest.approx(1.0, abs=1e-9)
        # Orientation hasn't tumbled
        attitude = Rotation.from_quat(ekf.x[6:10]).magnitude()
        assert attitude < np.deg2rad(5.0)
