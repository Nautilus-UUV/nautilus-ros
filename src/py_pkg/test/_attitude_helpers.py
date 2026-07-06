"""Shared gravity/attitude test fixtures for the Tier 1 + Tier 2 suites.

The Tier 1 ``gravity_to_roll_pitch`` tests (``unit/test_math_utils.py``) and the
Tier 2 ``AttitudeNode`` tests (``node/test_attitude_node.py``) both need to build
the body-frame gravity reaction a vehicle at a given (roll, pitch, yaw) would
read. Defined once here -- imported from the test root, which the root
``conftest.py`` puts on ``sys.path`` for both tiers -- so the two suites can't
drift onto different rotation maths or a different ``g``.
"""

import numpy as np
from py_pkg.math_utils import rpy_to_quaternion
from py_pkg.physics import GRAVITY_M_S2


def rot_matrix(qx, qy, qz, qw):
    """Body->world rotation matrix from a quaternion (x, y, z, w)."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def specific_force(roll, pitch, yaw=0.0, g=GRAVITY_M_S2):
    """Body-frame gravity reaction at (roll, pitch, yaw), via rpy_to_quaternion.

    Ties the tests to the project's own quaternion convention. The frame is NED
    (x fwd, y right, z DOWN), so world "up" is -Z: the gravity reaction is
    ``(0, 0, -g)`` expressed in the body frame, which the estimator's
    gravity_to_roll_pitch inverts exactly.
    """
    R = rot_matrix(*rpy_to_quaternion(roll, pitch, yaw))  # body -> world
    return R.T @ np.array([0.0, 0.0, -g])                 # world up (-Z in NED) in body
