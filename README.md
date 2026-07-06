# UUV - ROS2 Stack

ROS2 navigation and control system for Nautilus's autonomous underwater vehicle, designed to run on RaspberryPie.

## [Getting Started](docs/getting-started.md)
All about prerequisites, setup, development, package's architecture, testing and debugging.

## Running the simulation

- [`docs/running_sim.md`](docs/running_sim.md): native colcon + Gazebo flow.
- [`docs/running_in_apptainer.md`](docs/running_in_apptainer.md): the containerized flow for parameter sweeps on a multi-CPU server.

## Packages

### `py_pkg` - Python ROS2 Package
Contains the enitre control stack in Python:
- **[uuv_ros_core](src/py_pkg/py_pkg/uuv_ros_core/)** - Centralized topic definitions, QoS profiles, and node utilities
- Path planning, state estimation, PID control, MQTT communication


## Testing

Tests live in `src/py_pkg/test/`, organised in four tiers of increasing realism and cost:

| Tier | Location | What it exercises | Needs |
|---|---|---|---|
| 1 - unit | `test/unit/` | pure logic: PID, depth/ACU control system, physics, math | nothing |
| 2 - node | `test/node/` | in-process `rclpy` nodes with stubbed I/O via a `NodeHarness` | `rclpy` only |
| 3 - sim | `test/sim/` | end-to-end through the HAL into Gazebo, via `launch_testing`


Tiers 3 are marker-gated (`@pytest.mark.sim`) and excluded from default runs by `pytest.ini`.

Run from `src/py_pkg/`:

```bash
source /opt/ros/jazzy/setup.bash
pytest test/unit/ -v          # Tier 1
pytest test/node/ -v          # Tier 2
pytest test/                  # Tier 1 + 2 (sim filtered out)
pytest -m sim test/sim/ -v    # Tier 3 - requires built DAVE workspace
```

### Tier 3 tests with the Digital Twin

To run integration testing of the whole system, follow the [nautilus-dave](https://github.com/Nautilus-UUV/nautilus-dave/tree/dev) repository Installation instructions, then the control can be tested in tandem with our simulator.


```bash
source /opt/ros/jazzy/setup.bash
source /home/girji/dave_ws/install/setup.bash    # nautilus_hal + dave_demos

SIM_GUI=1 pytest -m sim test/sim/YOUR_TEST.py -v -s
```

`SIM_GUI` is an env variable that enables the simulation frontend.


## [Contributing](docs/CONTRIBUTING.md)

Follow our contribution guidelines for development standards and workflows.

