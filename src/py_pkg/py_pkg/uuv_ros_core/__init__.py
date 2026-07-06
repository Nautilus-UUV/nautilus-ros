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

# Core constants
from .topics import UUVTopics
from .qos_profiles import UUVQoS, TOPIC_QOS_MAP
from .message_types import TOPIC_MESSAGE_MAP

# Optional utilities
from .node_factory import create_publisher_for_topic, create_subscription_for_topic
from .node_runtime import now_s, spin_node

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
