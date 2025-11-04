# C++ Development Setup

## Prerequisites

- **ROS2 Humble**: Follow the [official installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- **CMake 3.8+**: Included with ROS2 development tools
- **C++14 compiler**: GCC or Clang

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
   colcon build --packages-select cpp_pkg
   ```

4. **Source the workspace**
   ```bash
   source install/setup.bash
   ```

## Running C++ Nodes

```bash
ros2 run cpp_pkg YOUR_NODE_NAME
```

## Creating C++ Nodes

1. **Add source files** to `src/cpp_pkg/src/`
2. **Update CMakeLists.txt** to include your executables:
   ```cmake
   add_executable(my_node src/my_node.cpp)
   ament_target_dependencies(my_node rclcpp std_msgs)

   install(TARGETS
     my_node
     DESTINATION lib/${PROJECT_NAME}
   )
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

## Development Workflow

1. **Write C++ nodes** in `src/cpp_pkg/src/`
2. **Update CMakeLists.txt** with new executables
3. **Build and test**:
   ```bash
   colcon build --packages-select cpp_pkg
   source install/setup.bash
   ros2 run cpp_pkg your_new_node
   ```

## Testing

Run C++ package tests:
```bash
colcon test --packages-select cpp_pkg
colcon test-result --verbose
```

## Available Dependencies

The package is configured with:
- `rclcpp` - ROS2 C++ client library
- `std_msgs` - Standard message types
- `geometry_msgs` - Geometry message types
- `sensor_msgs` - Sensor message types

Add additional dependencies in `package.xml` and `CMakeLists.txt` as needed.

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