# Python Development Setup

## Prerequisites

- **ROS2 Humble**: Follow the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- **Python 3.8+**: Included with ROS2 Humble

## Setup

1. **Source ROS2 installation**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Install colcon build tools** (if not available)
   ```bash
   sudo apt update
   sudo apt install python3-colcon-common-extensions
   ```

3. **Build the workspace**
   ```bash
   colcon build --packages-select py_pkg
   ```

4. **Source the workspace**
   ```bash
   source install/setup.bash
   ```

## Running Python Nodes

```bash
ros2 run py_pkg YOUR_NODE_NAME
```

## Using polarisutils

The `polarisutils` library provides centralized topic definitions and utilities:

```python
import rclpy
from rclpy.node import Node
from polarisutils import PolarisTopics, create_publisher_for_topic

class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        # Create publishers with automatic message types and QoS
        self.flow_pub = create_publisher_for_topic(self, PolarisTopics.BCU_FLOW_RATE)
```

See [polarisutils documentation](../src/py_pkg/py_pkg/polarisutils/README.md) for complete usage examples.

## Development Workflow

1. **Create new nodes** in `src/py_pkg/py_pkg/`
2. **Add entry points** in `src/py_pkg/setup.py`
3. **Build and test**:
   ```bash
   colcon build --packages-select py_pkg
   source install/setup.bash
   ros2 run py_pkg your_new_node
   ```

## Testing

Run Python package tests:
```bash
colcon test --packages-select py_pkg
colcon test-result --verbose
```

## Debugging

**List available topics:**
```bash
ros2 topic list
```

**Monitor topic data:**
```bash
ros2 topic echo /topic_name
```

**Publish test messages:**
```bash
ros2 topic pub /topic_name std_msgs/String 'data: Hello World' -1
```