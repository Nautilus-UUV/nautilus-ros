import numpy as np


def normalize_quaternion(q):
    """Normalize a quaternion.

    Args:
        q (np.ndarray): A 4-element array representing the quaternion [x, y, z, w].

    Returns:
        np.ndarray: A normalized (unit) quaternion.
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length quaternion")
    return q / norm


def quaternion_multiply(q1, q2):
    """Multiply two quaternions for subsequent rotations.

    Args:
        q1 (np.ndarray): First quaternion [x, y, z, w].
        q2 (np.ndarray): Second quaternion [x, y, z, w].

    Returns:
        np.ndarray: The product quaternion.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w])


def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix.

    Args:
        q (np.ndarray): Quaternion [x, y, z, w].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    x, y, z, w = q
    # Precompute repeated terms
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(y2 + z2), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(x2 + z2), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(x2 + y2)]
    ])
    return R


def rotate_vector(q, v):
    """Rotate a vector from body frame to world frame using quaternion.

    Args:
        q (np.ndarray): Quaternion representing rotation [x, y, z, w].
        v (np.ndarray): Vector to rotate [x, y, z].

    Returns:
        np.ndarray: Rotated vector in world frame.
    """
    R = quaternion_to_rotation_matrix(q)
    return R @ v


def skew(v):
    """Return the 3x3 skew-symmetric matrix for a 3-vector v.

    This is useful for converting cross-product operations into matrix form
    and appears in linearized Jacobians for small-angle rotations.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def quaternion_integrate(q, omega, dt):
    """Integrate angular velocity to update quaternion orientation.

    Args:
        q (np.ndarray): Current orientation quaternion [x, y, z, w].
        omega (np.ndarray): Angular velocity vector [wx, wy, wz].
        dt (float): Time step.

    Returns:
        np.ndarray: Updated orientation quaternion.
    """
    wx, wy, wz = omega
    norm_omega = np.linalg.norm(omega)
    if norm_omega > 0:
        theta = norm_omega * dt                     # angle of rotation
        axis = omega / norm_omega                   # axis of rotation
        sin_half_theta = np.sin(theta / 2.0)
        delta_q = np.array([                        # delta quaternion (change in orientation)
            axis[0] * sin_half_theta,
            axis[1] * sin_half_theta,
            axis[2] * sin_half_theta,
            np.cos(theta / 2.0)
        ])
    else:
        delta_q = np.array([0.0, 0.0, 0.0, 1.0])

    q_new = quaternion_multiply(q, delta_q)         # apply the small rotation
    return normalize_quaternion(q_new)


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
        q = self.x[6:10]

        # Transform acceleration from body to world
        accel_world = rotate_vector(q, accel_body)

        # Compensate gravity (accelerometer measures specific force a - g so we must add g to get the actual linear acceleration)
        accel_world = accel_world + self.g

        # Kinematic updates
        pos_new = pos + vel * dt + 0.5 * accel_world * dt * dt
        vel_new = vel + accel_world * dt

        # Orientation update via angular velocity integration and subsequent normalization
        q_new = quaternion_integrate(q, gyro_body, dt)
        q_new = normalize_quaternion(q_new)

        # Update state
        self.x[0:3] = pos_new
        self.x[3:6] = vel_new
        self.x[6:10] = q_new

        # State transition Jacobian (simplified)
        F = np.eye(10)

        # Position uncertainty grows because velocity is uncertain.
        F[0:3, 3:6] = np.eye(3) * dt

        # Describes how errors in orientation propagate to velocity.
        # Use a robust small-angle approximation: orientation errors map to
        # velocity errors via the skew of the world-acceleration (accel_world).
        try:
            F[3:6, 6:9] = -skew(accel_world) * dt
        except Exception:
            # If any shape issue occurs, leave cross-term zero (conservative)
            pass

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measured_accel):
        """EKF Update step using accelerometer only (attitude correction).

        We treat the accelerometer as a measurement of the gravity vector
        expressed in the body frame under the assumption that translational accelerations are small during the
        update window. This corrects roll/pitch robustly for low-dynamics.
        """
        # Current orientation (quaternion)
        q = self.x[6:10]
        R_mat = quaternion_to_rotation_matrix(q)

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
        self.x[6:10] = normalize_quaternion(self.x[6:10])

        # Covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
