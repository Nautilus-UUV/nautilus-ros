"""Shared Tier 2 test scaffolding: node-under-test + tester node + executor.

Importable home for the generic in-process rclpy harness pieces used by both
this repo's ``test/node/`` conftest and the dave repo's ``nautilus_hal``
test tree (which already depends on installed ``py_pkg``). Keeping one copy
matters: the lifecycle details below (executor remove-before-destroy
ordering, double-destroy guard, ``__getattr__`` recursion refusal, per-PID
domain isolation) are exactly the kind of scaffolding that drifts silently
when duplicated.

Deliberately imports no pytest — the fixtures wrapping these stay in each
conftest (they differ: autouse here, opt-in in nautilus_hal).
"""

import os
import time

from rclpy.executors import SingleThreadedExecutor


def isolated_ros_domain_id() -> int:
    """Per-PID ROS_DOMAIN_ID. Tests use production topic names, so any
    sibling rclpy participant on the default domain (stray `ros2` CLI,
    leftover Gazebo, parallel pytest) would inject traffic into the
    harness and corrupt assertions."""
    return (os.getpid() % 101) + 1


class NodeHarness:
    """Generic Tier 2 harness: node-under-test + tester node + executor.

    Owns lifecycle (executor add/remove, destroy_node, executor shutdown).
    Subclasses or callers supply the node-under-test and a tester node;
    the tester node owns the test-side publishers/subscribers.
    """

    def __init__(self, node_under_test_cls, tester_cls):
        # Both arguments are zero-arg callables: a Node subclass, or a
        # factory lambda when the node needs constructor arguments.
        self.node = node_under_test_cls()
        self.tester = tester_cls()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.tester)
        # Set by destroy_node_under_test so shutdown doesn't destroy it twice.
        self._node_destroyed = False

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the tester node.

        Only consulted for names not found normally, so members defined
        on a subclass (e.g. transforming properties) still win. Private
        and dunder names are refused so pickling and pytest introspection
        get a clean AttributeError instead of recursing into a
        half-constructed harness.
        """
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        tester = self.__dict__.get("tester")
        if tester is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(tester, name)

    def spin_for(self, duration_s: float, slice_s: float = 0.02) -> None:
        """Spin the executor for at least ``duration_s`` of wall time."""
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=slice_s)

    def spin_until(self, predicate, timeout: float = 2.0, slice_s: float = 0.02):
        """Spin until ``predicate()`` is truthy or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.executor.spin_once(timeout_sec=slice_s)
        if not predicate():
            raise TimeoutError(f"predicate did not become true within {timeout}s")

    def destroy_node_under_test(self) -> None:
        """Tear the node-under-test down mid-test, leaving the tester spinning.

        For nodes that publish a parking value from ``destroy_node`` (bcu_node's
        safe-stop, acu_node's neutral): the tester has to outlive the node to
        observe what it emitted on the way out, so the usual ``shutdown()``
        teardown can't exercise it. Spin after calling this to collect.
        """
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self._node_destroyed = True

    def shutdown(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.executor.remove_node(self.tester)
        finally:
            if not self._node_destroyed:
                self.node.destroy_node()
            self.tester.destroy_node()
            self.executor.shutdown()
