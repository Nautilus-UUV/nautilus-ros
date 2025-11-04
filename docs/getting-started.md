# Getting Started

## Prerequisites

- **Ubuntu 22.04 LTS** (recommended)
- **ROS2 Humble**: Follow the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- **Python 3.8+**: Included with ROS2 Humble
- **CMake 3.8+**: For C++ development

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
   # Build all packages
   colcon build

   # Or build specific packages
   colcon build --packages-select py_pkg
   colcon build --packages-select cpp_pkg
   ```

4. **Source the workspace**
   ```bash
   source install/setup.bash
   ```

## Running Nodes

```bash
# Python nodes
ros2 run py_pkg YOUR_NODE_NAME

# C++ nodes
ros2 run cpp_pkg YOUR_NODE_NAME
```

## Python Development

### Using polarisutils

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

## C++ Development

### Creating C++ Nodes

1. **Add source files** to `src/cpp_pkg/src/`
2. **Update CMakeLists.txt**:
   ```cmake
   add_executable(your_node src/your_node.cpp)
   ament_target_dependencies(your_node rclcpp std_msgs)
   install(TARGETS your_node DESTINATION lib/${PROJECT_NAME})
   ```
3. **Basic node structure**:
   ```cpp
   #include "rclcpp/rclcpp.hpp"
   #include "std_msgs/msg/string.hpp"

   class MyNode : public rclcpp::Node
   {
   public:
     MyNode() : Node("my_node")
     {
       publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
     }

   private:
     rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
   };

   int main(int argc, char * argv[])
   {
     rclcpp::init(argc, argv);
     rclcpp::spin(std::make_shared<MyNode>());
     rclcpp::shutdown();
     return 0;
   }
   ```

4. **Test locally**:
   ```bash
   colcon build --packages-select cpp_pkg
   source install/setup.bash
   ros2 run cpp_pkg your_node
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

### C++ Package
1. **Package Structure**: Add your package following this structure:
   ```
   uuv-ros/
   ├── README.md
   └── src/cpp_pkg/
       ├── src/
       │   └── sensor_monitor/          # Package name
       │       ├── monitor_node.cpp     # Main ROS2 node
       │       ├── alert_system.cpp     # Helper implementation
       │       └── examples/
       │           └── basic_usage.cpp
       ├── include/cpp_pkg/
       │   └── sensor_monitor/
       │       ├── monitor_node.hpp     # Node header
       │       └── alert_system.hpp     # Helper headers
       └── README.md                    # Package documentation
   ```

2. **CMakeLists.txt Integration**: Register your executable in `src/cpp_pkg/CMakeLists.txt`:
   ```cmake
   # Add executable
   add_executable(sensor_monitor_node src/sensor_monitor/monitor_node.cpp src/sensor_monitor/alert_system.cpp)

   # Link dependencies
   ament_target_dependencies(sensor_monitor_node rclcpp std_msgs geometry_msgs sensor_msgs)

   # Install executable
   install(TARGETS sensor_monitor_node DESTINATION lib/${PROJECT_NAME})

   # Add include directories
   target_include_directories(sensor_monitor_node PUBLIC
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
     $<INSTALL_INTERFACE:include>)
   ```

3. **Run the node**:
   ```bash
   ros2 run cpp_pkg sensor_monitor_node
   ```

### Adding Dependencies
If you need additional ROS2 packages, add them to `package.xml` and (for C++) `CMakeLists.txt`.

## Testing

```bash
# Test specific package
colcon test --packages-select py_pkg
colcon test --packages-select cpp_pkg

# View test results
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