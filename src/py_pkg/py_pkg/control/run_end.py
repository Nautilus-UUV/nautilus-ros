"""The two events that end a run, wired once for every controller.

A run ends two ways, kept distinct on the wire because they mean different
things to an operator: ``/command=false`` is the OPERATOR aborting, and
``/mission/complete`` is ``pathfinding`` reporting the mission finished. Every
controller collapses both to the same action -- drop the target, park the wire,
go silent -- so the polarity of each surface is decided here instead of being
hand-copied into each node.

Only the *dispatch* is shared. What a node does on a stop stays on the node
(``bcu_node`` emits a safe-stop burst, ``acu_node`` emits one neutral), which is
the split that matters: the bodies genuinely differ, the trigger does not.

Adding a third run-end surface -- a liveness verdict, a lifeguard engage, an
abort distinct from stop -- is one edit here rather than one per controller.
"""

from py_pkg.uuv_ros_core import UUVTopics, create_subscription_for_topic


def subscribe_run_end(node, on_stop) -> None:
    """Call ``on_stop(reason)`` on an operator abort or a mission completion.

    ``on_stop`` must be idempotent: the UI sends ``/command=false`` before every
    manual command, and a completion can land on a stack the operator already
    stopped. Both nodes get that for free by edge-triggering on their own
    ``target_pressure_pa is None`` sentinel.
    """

    def _on_command(msg) -> None:
        # /command=true (start) is a no-op for a controller -- it waits for a
        # POSITION_TARGET rather than acting on the start edge.
        if not bool(msg.data):
            on_stop("operator stop")

    def _on_mission_complete(msg) -> None:
        # Latched completion event. Bool(false) is not a thing anyone publishes
        # here; ignore it if it ever is.
        if bool(msg.data):
            on_stop("mission complete")

    create_subscription_for_topic(node, UUVTopics.COMMAND, _on_command)
    create_subscription_for_topic(
        node, UUVTopics.MISSION_COMPLETE, _on_mission_complete
    )
