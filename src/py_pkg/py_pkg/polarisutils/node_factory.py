"""
Simple helper functions for creating publishers and subscribers.

These are optional convenience functions that use the topic constants,
message types, and QoS profiles. You can still use regular ROS2 methods.

Example:
    from polarisutils.node_factory import create_publisher_for_topic
    from polarisutils.topics import PolarisTopics

    # Convenience function
    pub = create_publisher_for_topic(self, PolarisTopics.BCU_FLOW_RATE)

    # Still works - standard ROS2
    pub = self.create_publisher(Float32, PolarisTopics.BCU_FLOW_RATE, 10)
"""

from .topics import PolarisTopics
from .message_types import TOPIC_MESSAGE_MAP
from .qos_profiles import TOPIC_QOS_MAP, PolarisQoS


def create_publisher_for_topic(node, topic: str, **kwargs):
    """
    Create a publisher with automatic message type and QoS selection.

    Args:
        node: The ROS2 node
        topic: Topic constant from PolarisTopics
        **kwargs: Additional arguments passed to create_publisher
    """
    msg_type = TOPIC_MESSAGE_MAP[topic]
    qos = TOPIC_QOS_MAP.get(topic, PolarisQoS.CONTROL)
    return node.create_publisher(msg_type, topic, qos, **kwargs)


def create_subscription_for_topic(node, topic: str, callback, **kwargs):
    """
    Create a subscription with automatic message type and QoS selection.

    Args:
        node: The ROS2 node
        topic: Topic constant from PolarisTopics
        callback: Callback function
        **kwargs: Additional arguments passed to create_subscription
    """
    msg_type = TOPIC_MESSAGE_MAP[topic]
    qos = TOPIC_QOS_MAP.get(topic, PolarisQoS.CONTROL)
    return node.create_subscription(msg_type, topic, callback, qos, **kwargs)