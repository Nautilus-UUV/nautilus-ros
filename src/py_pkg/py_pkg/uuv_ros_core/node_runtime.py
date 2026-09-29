"""Shared spin/shutdown runner for py_pkg nodes.

Every node's ``main()`` repeats the same rclpy boilerplate: spin, swallow the
SIGINT/SIGTERM shutdown, then destroy the node and ``try_shutdown``. Centralizing
it here means the teardown-race fix below lands fleet-wide in one place.

The subtlety this guards against: a timer or subscription callback can touch an
rclpy handle (almost always a ``publish()``) inside the SIGINT teardown window,
just after the context has begun shutting down. That raises an ``RCLError`` -- a
plain ``RuntimeError`` subclass, *not* an ``ExternalShutdownException`` -- which
escapes ``rclpy.spin()``. With the old ``except (KeyboardInterrupt,
ExternalShutdownException)`` the process then exits 1 instead of 0, and
launch_testing's ``assertExitCodes`` flakes on the Tier 3 sim tests (it expects
0 / -2 / -15). It's a race, so it surfaces on whichever node loses it on a given
run rather than any one node in particular.

We swallow such an exception *only once shutdown is already underway*
(``not rclpy.ok()``); anything raised while the context is still healthy
re-raises, so genuine mid-run faults still fail loudly.
"""

import rclpy
from rclpy.executors import ExternalShutdownException


def now_s(node) -> float:
    """Current node-clock time in seconds.

    The one home for the ``nanoseconds / 1e9`` conversion every node needs.
    """
    return node.get_clock().now().nanoseconds / 1e9


def spin_node(node) -> None:
    """Spin ``node`` until shutdown, then tear it down cleanly.

    Drop-in for the per-node ``try: rclpy.spin(node) ... finally:`` block.
    Assumes ``rclpy.init()`` has already run -- nodes call it themselves so they
    can pass their own init args and construct the node first.
    """
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A callback raced the SIGINT teardown (e.g. publish() after the context
        # began shutting down). Benign once we're on the way down; re-raise if
        # the context is still up so real faults aren't masked.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
