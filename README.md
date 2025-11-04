# uuv-ROS

This repository is the basis for the HS25 ROS stack setup, which should run on the RaspberryPi.

## [Getting Started](docs/getting-started.md)
- Commands for setup and ROS tips

## [Contributing](docs/CONTRIBUTING.md)
- Follow our contribution guide for conventions to follow

## Content:

### Nodes
- 

### Others
- **[polarisutils](src/py_pkg/py_pkg/polarisutils/)** - Single source of truth for UUV's topics, QoS profiles, message types and node creation helper functions
- **[CI](.github/workflows/ci.yml)** - Continuos integration check running a docker container with `ROS2 Humble` on `Ubuntu:latest`

