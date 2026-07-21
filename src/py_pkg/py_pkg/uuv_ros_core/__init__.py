"""
UUV ROS Utils - Essential interface management.

This package provides:
- UUVTopics: Topic name constants
- UUVQoS: QoS profiles for underwater operations
- TOPIC_MESSAGE_MAP: Topic to message type mappings
- Simple helper functions (optional)

Example:
-------
    from uuv_ros_core import UUVTopics, UUVQoS

    self.publisher = self.create_publisher(
        Float32, UUVTopics.BCU_FLOW_RATE, UUVQoS.CONTROL
    )

"""

# Topic names are dependency-free, so importing this package never requires
# a sourced ROS env on its own.
from .topics import UUVTopics

# Everything else binds rclpy or nautilus_msgs. Deferred to first attribute
# access (PEP 562) so the host-side analysis tools can read topic names out
# of the registry — the single source of truth — instead of restating them
# as literals. Runtime nodes touch these immediately and are unaffected.
_LAZY = {
    "UUVQoS": ".qos_profiles",
    "TOPIC_QOS_MAP": ".qos_profiles",
    "TOPIC_MESSAGE_MAP": ".message_types",
    "create_publisher_for_topic": ".node_factory",
    "create_subscription_for_topic": ".node_factory",
    "now_s": ".node_runtime",
    "spin_node": ".node_runtime",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module, __name__), name)
    globals()[name] = value  # bind, so the indirection is paid once
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__version__ = "1.0.0"

__all__ = [
    # Core constants
    "UUVTopics",
    "UUVQoS",
    "TOPIC_QOS_MAP",
    "TOPIC_MESSAGE_MAP",
    # Optional utilities
    "create_publisher_for_topic",
    "create_subscription_for_topic",
    "now_s",
    "spin_node",
]
