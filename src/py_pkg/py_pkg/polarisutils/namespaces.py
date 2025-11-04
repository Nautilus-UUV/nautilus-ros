"""
Simple namespace utilities for Polaris UUV system.

Example:
    from polarisutils.namespaces import add_namespace
    from polarisutils.topics import PolarisTopics

    topic = add_namespace(PolarisTopics.BCU_FLOW_RATE, "polaris")
    # Result: "/polaris/bcu/flow_rate"
"""


def add_namespace(topic: str, namespace: str = "polaris") -> str:
    """Add namespace to a topic."""
    return f"/{namespace}{topic}"


def remove_namespace(topic: str) -> str:
    """Remove namespace from a topic (get base topic)."""
    if topic.count('/') >= 2:
        parts = topic.split('/', 2)
        return '/' + parts[2]
    return topic