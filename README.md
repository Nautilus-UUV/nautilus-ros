# UUV - ROS2 Stack

ROS2 navigation and control system for Nautilus's autonomous underwater vehicle, designed to run on RaspberryPi as part of the distributed software architecture.

## [Getting Started](docs/getting-started.md)

All about prerequisites, setup, development, package's architecture, testing and debugging.

## Packages

### `py_pkg` - Python ROS2 Package
Contains Python-based ROS2 nodes and auxiliary libraries:
- **[uuv_ros_core](src/py_pkg/py_pkg/uuv_ros_core/)** - Centralized topic definitions, QoS profiles, and node utilities
- Path planning, EKF filtering, PID control, MQTT communication

### `cpp_pkg` - C++ ROS2 Package
Prepared for C++ ROS2 nodes.

## [Contributing](docs/CONTRIBUTING.md)

Follow our contribution guidelines for development standards and workflows.

