# uuv_ros_utils

ROS2 interface management system - providing centralized topic definitions, QoS profiles, and non-intrusive utilities.

## Usage Comparison

### Publishers

```python
# ❌ BEFORE: Hard-coded strings, manual QoS, easy typos
self.flow_pub = self.create_publisher(Float32, "/bcu/flow_rate", 10)
self.leak_sub = self.create_subscription(UInt8MultiArray, "/internal/leak", self.callback, 10)

# ✅ GOOD: Constants + QoS profiles
from uuv_ros_utils import UUVTopics, UUVQoS
self.flow_pub = self.create_publisher(Float32, UUVTopics.BCU_FLOW_RATE, UUVQoS.CONTROL)
self.leak_sub = self.create_subscription(UInt8MultiArray, UUVTopics.INTERNAL_LEAK, self.callback, UUVQoS.SAFETY_CRITICAL)

# ⭐ BEST: Automatic everything
from uuv_ros_utils import create_publisher_for_topic, create_subscription_for_topic
self.flow_pub = create_publisher_for_topic(self, UUVTopics.BCU_FLOW_RATE)
self.leak_sub = create_subscription_for_topic(self, UUVTopics.INTERNAL_LEAK, self.callback)
```

### Complete Example

```python
import rclpy
from rclpy.node import Node
from uuv_ros_utils import UUVTopics, UUVQoS, create_publisher_for_topic, create_subscription_for_topic

class BCUNode(Node):
    def __init__(self):
        super().__init__("bcu_node")

        # Old way: 3 lines, prone to typos, inconsistent QoS
        # self.flow_pub = self.create_publisher(Float32, "/bcu/flow_rate", 10)
        # self.pressure_sub = self.create_subscription(Int32, "/bcu/pressure", self.pressure_cb, 10)
        # self.leak_sub = self.create_subscription(UInt8MultiArray, "/internal/leak", self.leak_cb, 10)

        # New way: 3 lines, typo-proof, consistent QoS
        self.flow_pub = create_publisher_for_topic(self, UUVTopics.BCU_FLOW_RATE)
        self.pressure_sub = create_subscription_for_topic(self, UUVTopics.BCU_PRESSURE, self.pressure_cb)
        self.leak_sub = create_subscription_for_topic(self, UUVTopics.INTERNAL_LEAK, self.leak_cb)

    def pressure_cb(self, msg): pass
    def leak_cb(self, msg): pass
```

## Reference

- [Topics](topics.py)
- [Message types](message_types.py)
- [QoS profiles](qos_profiles.py)