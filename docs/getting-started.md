# Getting Started

## Prerequisites

- **Ubuntu 22.04 LTS** (recommended)
- **ROS2 Humble**: Follow the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- **Python 3.8+**: Included with ROS2 Humble

## Setup

1. **Source ROS2 installation**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Install colcon build tools** (if not available)
   ```bash
   # Default ros distro: humble
   bash scripts/install-deps.sh

   # Or, use your own (humble and jazzy tested)
   ROS_DISTRO=jazzy bash scripts/install-deps.sh
   ```

3. **Build the workspace**
   ```bash
   # Build all packages
   colcon build

   # Or build specific packages
   colcon build --packages-select py_pkg
   ```

4. **Source the workspace**
   ```bash
   source install/setup.bash
   ```

## Running Nodes

```bash
# Python nodes
ros2 run py_pkg YOUR_NODE_NAME
```

## Python Development

### Using uuv_ros_core

The `uuv_ros_core` library provides centralized topic definitions and utilities:

```python
import rclpy
from rclpy.node import Node
from uuv_ros_core import UUVTopics, create_publisher_for_topic

class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        # Create publishers with automatic message types and QoS
        self.flow_pub = create_publisher_for_topic(self, UUVTopics.BCU_FLOW_RATE)
```

See [uuv_ros_core documentation](../src/py_pkg/py_pkg/uuv_ros_core/README.md) for complete usage examples.

### Creating Python Nodes

1. **Create new nodes** in `src/py_pkg/py_pkg/`
2. **Add entry points** in `src/py_pkg/setup.py`:
   ```python
   entry_points={
       'console_scripts': [
           'your_node = py_pkg.your_node:main',
       ],
   },
   ```
3. **Test locally**:
   ```bash
   colcon build --packages-select py_pkg
   source install/setup.bash
   ros2 run py_pkg your_node
   ```

## Adding Nodes/Packages

### Example: Adding a "Sensor Monitor" Feature

### Python Package
1. **Package Structure**: Add your package following this structure:
   ```
   uuv-ros/
   ├── README.md
   └── src/py_pkg/py_pkg/
       ├── __init__.py
       └── sensor_monitor/              # Package name
           ├── __init__.py              # Makes it a Python package
           ├── README.md                # Package documentation
           ├── monitor_node.py          # Main ROS2 node
           ├── alert_system.py          # Helper module
           └── examples/
               └── basic_usage.py
   ```

2. **Main Package Integration**: Update `src/py_pkg/py_pkg/__init__.py` to import your package:
   ```python
   from . import sensor_monitor
   ```

3. **Node integration**: Add an entry to `console_scripts` in `src/py_pkg/setup.py`:
   ```python
   entry_points={
       'console_scripts': [
           'sensor_monitor_node = py_pkg.sensor_monitor.monitor_node:main',
       ],
   },
   ```

4. **Run the node**:
   ```bash
   ros2 run py_pkg sensor_monitor_node
   ```

### Adding Dependencies
If you need additional ROS2 packages, add them to `package.xml`.

## Testing

```bash
# Test specific package
colcon test --packages-select py_pkg

# View test results
colcon test-result --verbose
```

For the Tier 3 simulation tests and the unbounded `trim_sim.launch.py`
GUI workflow, see [`running_sim.md`](running_sim.md).

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