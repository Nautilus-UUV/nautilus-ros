"""
Nautilus ROS Utils - Essential interface management for Polaris UUV.

This package provides:
- PolarisTopics: Topic name constants
- PolarisQoS: QoS profiles for underwater operations
- TOPIC_MESSAGE_MAP: Topic to message type mappings
- Simple helper functions (optional)

Example:
    from polarisutils import PolarisTopics, PolarisQoS

    self.publisher = self.create_publisher(
        Float32, PolarisTopics.BCU_FLOW_RATE, PolarisQoS.CONTROL
    )
"""

# Core constants
from .topics import PolarisTopics
from .qos_profiles import PolarisQoS, TOPIC_QOS_MAP
from .message_types import TOPIC_MESSAGE_MAP

# Optional utilities
from .namespaces import add_namespace, remove_namespace
from .node_factory import create_publisher_for_topic, create_subscription_for_topic

__version__ = "1.0.0"

__all__ = [
    # Core constants
    "PolarisTopics",
    "PolarisQoS",
    "TOPIC_QOS_MAP",
    "TOPIC_MESSAGE_MAP",

    # Optional utilities
    "add_namespace",
    "remove_namespace",
    "create_publisher_for_topic",
    "create_subscription_for_topic",
]