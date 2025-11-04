"""
UUV ROS Utils - Essential interface management.

This package provides:
- UUVTopics: Topic name constants
- UUVQoS: QoS profiles for underwater operations
- TOPIC_MESSAGE_MAP: Topic to message type mappings
- Simple helper functions (optional)

Example:
-------
    from uuv_ros_utils import UUVTopics, UUVQoS

    self.publisher = self.create_publisher(
        Float32, UUVTopics.BCU_FLOW_RATE, UUVQoS.CONTROL
    )

"""

# Core constants
from .topics import UUVTopics
from .qos_profiles import UUVQoS, TOPIC_QOS_MAP
from .message_types import TOPIC_MESSAGE_MAP

# Optional utilities
from .namespaces import add_namespace, remove_namespace
from .node_factory import create_publisher_for_topic, create_subscription_for_topic

__version__ = "1.0.0"

__all__ = [
    # Core constants
    "UUVTopics",
    "UUVQoS",
    "TOPIC_QOS_MAP",
    "TOPIC_MESSAGE_MAP",
    # Optional utilities
    "add_namespace",
    "remove_namespace",
    "create_publisher_for_topic",
    "create_subscription_for_topic",
]
