import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    """Return the 3x3 skew-symmetric matrix for a 3-vector v.

    Used in the linearized Jacobians to express cross-product operations
    in matrix form for small-angle rotations.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


class EKF_Filter:
    def __init__(self, dt):
        """Initialize the EKF filter.

        Args:
            dt (float): Time step between updates.
        """
        self.dt = dt
        # State vector: [px, py, pz, vx, vy, vz, qx, qy, qz, qw]
        self.x = np.zeros(10)
        self.x[9] = 1.0  # quaternion as [x, y, z, w]

        # Covariances
        self.P = np.eye(10)
        self.P[0:6, 0:6] *= 1.0   # position & velocity uncertainty
        self.P[6:10, 6:10] *= 0.1  # orientation uncertainty

        # Process noise
        self.Q = np.eye(10)
        self.Q[0:3, 0:3] *= 0.1         # position process noise
        self.Q[3:6, 3:6] *= 0.5         # velocity process noise
        self.Q[6:10, 6:10] *= 0.01      # orientation process noise

        # Measurement noise (accelerometer only)
        # We use the accelerometer as an attitude (gravity) measurement in update()
        # Gyroscopes are used as inputs for prediction (and later can be part of
        # a measurement model once bias states are added).
        self.R = np.eye(3)
        self.R[0:3, 0:3] *= 0.1         # accelerometer measurement noise

        # Gravity (world frame)
        self.g = np.array([0.0, 0.0, -9.81])

    def predict(self, accel_body, gyro_body, dt=None):
        """EKF Prediction step.

        Args:
            accel_body (np.ndarray): Measured acceleration in body frame [ax, ay, az].
            gyro_body (np.ndarray): Measured angular velocity in body frame [wx, wy, wz].
            dt (float, optional): Time step in seconds. Uses self.dt if not provided.
        """
        if dt is None:
            dt = self.dt

        # Unpack state
        pos = self.x[0:3]
        vel = self.x[3:6]
        rot = Rotation.from_quat(self.x[6:10])

        # Transform acceleration from body to world
        accel_world = rot.apply(accel_body)

        # Compensate gravity (accelerometer measures specific force a - g so we must add g to get the actual linear acceleration)
        accel_world = accel_world + self.g

        # Kinematic updates
        pos_new = pos + vel * dt + 0.5 * accel_world * dt * dt
        vel_new = vel + accel_world * dt

        # Orientation update via angular velocity integration (body-frame increment)
        delta_rot = Rotation.from_rotvec(gyro_body * dt)
        q_new = (rot * delta_rot).as_quat()

        # Update state
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:10] = q_new

        # State transition Jacobian (simplified)
        F = np.eye(10)

        # Position uncertainty grows because velocity is uncertain.
        F[0:3, 3:6] = np.eye(3) * dt

        # Describes how errors in orientation propagate to velocity.
        # Small-angle approximation: orientation errors map to velocity
        # errors via the skew of the world-acceleration (accel_world).
        F[3:6, 6:9] = -skew(accel_world) * dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measured_accel):
        """EKF Update step using accelerometer only (attitude correction).

        We treat the accelerometer as a measurement of the gravity vector
        expressed in the body frame under the assumption that translational accelerations are small during the
        update window. This corrects roll/pitch robustly for low-dynamics.
        """
        # Current orientation as a rotation matrix (body -> world)
        R_mat = Rotation.from_quat(self.x[6:10]).as_matrix()

        # Measurement: accelerometer reading in body frame
        z = measured_accel

        # Predicted accel in body if only gravity acts: h = -R^T g
        expected_accel_body = - R_mat.T @ self.g

        # Measurement Jacobian H (3 x state_dim)
        H = np.zeros((3, self.P.shape[0]))
        H[0:3, 6:9] = -skew(R_mat.T @ self.g)

        # Innovation
        y = z - expected_accel_body

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update (apply additive correction). For the quaternion we
        # add the small correction to the quaternion components and then
        # re-normalize (practical, but an error-state/multiplicative EKF is
        # more rigorous for larger corrections).
        self.x = self.x + K @ y
        q = self.x[6:10]
        self.x[6:10] = q / np.linalg.norm(q)

        # Covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
