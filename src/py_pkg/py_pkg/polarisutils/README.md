# PolarisUtils

ROS2 interface management system for Polaris UUV - providing centralized topic definitions, QoS profiles, and non-intrusive utilities.

## Installation

**Prerequisites**: [ROS2](https://docs.ros.org/en/humble/Installation.html) (Humble or later)

```bash
# Clone into your ROS2 workspace src directory
cd ~/your_ros_workspace/src
git clone https://github.com/Nautilus-UUV/polarisutils.git
```

```bash
# Build the package
cd ~/your_ros_workspace
colcon build --packages-select polarisutils
source install/setup.bash
```

> **Note**: This package includes C++ build configuration for potential future [micro-ROS](https://micro.ros.org/) integration with embedded systems. Currently, only Python utilities are implemented.

## Usage Comparison

### Publishers

```python
# ❌ BEFORE: Hard-coded strings, manual QoS, easy typos
self.flow_pub = self.create_publisher(Float32, "/bcu/flow_rate", 10)
self.leak_sub = self.create_subscription(UInt8MultiArray, "/internal/leak", self.callback, 10)

# ✅ GOOD: Constants + QoS profiles
from polarisutils import PolarisTopics, PolarisQoS
self.flow_pub = self.create_publisher(Float32, PolarisTopics.BCU_FLOW_RATE, PolarisQoS.CONTROL)
self.leak_sub = self.create_subscription(UInt8MultiArray, PolarisTopics.INTERNAL_LEAK, self.callback, PolarisQoS.SAFETY_CRITICAL)

# ⭐ BEST: Automatic everything
from polarisutils import create_publisher_for_topic, create_subscription_for_topic
self.flow_pub = create_publisher_for_topic(self, PolarisTopics.BCU_FLOW_RATE)
self.leak_sub = create_subscription_for_topic(self, PolarisTopics.INTERNAL_LEAK, self.callback)
```

### Complete Example

```python
import rclpy
from rclpy.node import Node
from polarisutils import PolarisTopics, PolarisQoS, create_publisher_for_topic, create_subscription_for_topic

class BCUNode(Node):
    def __init__(self):
        super().__init__("bcu_node")

        # Old way: 3 lines, prone to typos, inconsistent QoS
        # self.flow_pub = self.create_publisher(Float32, "/bcu/flow_rate", 10)
        # self.pressure_sub = self.create_subscription(Int32, "/bcu/pressure", self.pressure_cb, 10)
        # self.leak_sub = self.create_subscription(UInt8MultiArray, "/internal/leak", self.leak_cb, 10)

        # New way: 3 lines, typo-proof, consistent QoS
        self.flow_pub = create_publisher_for_topic(self, PolarisTopics.BCU_FLOW_RATE)
        self.pressure_sub = create_subscription_for_topic(self, PolarisTopics.BCU_PRESSURE, self.pressure_cb)
        self.leak_sub = create_subscription_for_topic(self, PolarisTopics.INTERNAL_LEAK, self.leak_cb)

    def pressure_cb(self, msg): pass
    def leak_cb(self, msg): pass
```

## Topic Reference

| Topic Constant | Topic Name | Message Type | QoS Profile |
|----------------|------------|--------------|-------------|
| `INTERNAL_TEMPERATURE` | `/internal/temperature` | `Temperature` | `CONTROL` |
| `INTERNAL_PRESSURE` | `/internal/pressure` | `Int32` | `SAFETY_CRITICAL` |
| `INTERNAL_LEAK` | `/internal/leak` | `UInt8MultiArray` | `SAFETY_CRITICAL` |
| `INTERNAL_HUMIDITY` | `/internal/humidity` | `Float32` | `CONTROL` |
| `EXTERNAL_TEMPERATURE` | `/external/temperature` | `Temperature` | `CONTROL` |
| `EXTERNAL_PRESSURE` | `/external/pressure` | `Int32` | `SAFETY_CRITICAL` |
| `BCU_PRESSURE` | `/bcu/pressure` | `Int32` | `CONTROL` |
| `BCU_FLOW_RATE` | `/bcu/flow_rate` | `Float32` | `CONTROL` |
| `BCU_RPM` | `/bcu/rpm` | `Int32` | `CONTROL` |
| `ACU_TILT` | `/acu/tilt` | `Float32` | `CONTROL` |
| `ACU_ROLL` | `/acu/roll` | `Float32` | `CONTROL` |
| `ACU_TILT_STEPS` | `/acu/tilt/steps` | `Int32` | `CONTROL` |
| `ACU_ROLL_STEPS` | `/acu/roll/steps` | `Int32` | `CONTROL` |
| `IMU_LEFT` | `/imu/left` | `Imu` | `SENSOR_STREAM` |
| `IMU_RIGHT` | `/imu/right` | `Imu` | `SENSOR_STREAM` |
| `IMU_FILTERED_LEFT` | `/imu/filtered/left` | `Imu` | `CONTROL` |
| `IMU_FILTERED_RIGHT` | `/imu/filtered/right` | `Imu` | `CONTROL` |
| `POSITION_TARGET` | `/position/target` | `Point` | `CONTROL` |
| `POSITION_ESTIMATION` | `/position/estimation` | `Point` | `CONTROL` |
| `PATH` | `/path` | `Float32MultiArray` | `COMMAND` |
| `COMMAND` | `/command` | `String` | `COMMAND` |
| `CAN_OUT` | `/can/out` | `Frame` | `CONTROL` |

> [!NOTE]
> Might be outdated, always check documentation
