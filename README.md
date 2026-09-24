
ROS2 navigation and control system for Nautilus's autonomous glider (UG).

# Development

Create a folder `~/nautilus_ws/`. Here you will clone all repositories related to the UG.

## Setup

To complete the setup you need to follow README instructions of our 3 repositories:

1. Control Stack (this README)
2. Digital Twin: ([Nautilus-UUV/dave](https://github.com/Nautilus-UUV/dave))
3. Pilot UI ([nautilus-command-bridge-frontend](https://github.com/Nautilus-UUV/nautilus-command-bridge-frontend))

### Steps

1. Install ROS 2 Jazzy or Humble following the [official guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
2. Clone repository:
```bash
mkdir -p ~/nautilus_ws/src
cd ~/nautilus_ws/src
git clone -b dev git@github.com:Nautilus-UUV/nautilus-ros.git
```
3. Create a virtual environment:
```bash
cd ~/nautilus_ws
/usr/bin/python3 -m venv --system-site-packages .venv
touch .venv/COLCON_IGNORE
source .venv/bin/activate
pip install "pydantic>=2"
```
4. Resolve dependencies and build:
```bash
cd ~/nautilus_ws
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
sudo rosdep init                # first time on this machine only
rosdep update
rosdep install --from-paths src --ignore-src -y --skip-keys "protobuf"
python -m colcon build --symlink-install
source install/setup.bash
```
5. Verify the setup (Tier 1 + 2 tests):
```bash
cd ~/nautilus_ws/src/nautilus-ros/src/py_pkg
python -m pytest test/ -v
```

Every new shell:
```bash
cd ~/nautilus_ws
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```


**(!) At this point you must continue by following the documentation from the other repositories:**
1. <del>Control Stack (this README)</del>
2. Digital Twin: ([Nautilus-UUV/dave](https://github.com/Nautilus-UUV/dave))
3. Pilot UI ([nautilus-command-bridge-frontend](https://github.com/Nautilus-UUV/nautilus-command-bridge-frontend))


## Control with Digital Twin

When developing code and testing it is crucial to first test your changes on the digital twin. At this point we assume all 3 repositories are setup.

1. Rebuild:
```bash
cd ~/nautilus_ws
python -m colcon build --symlink-install
source install/setup.bash
```

2. Start the control, digital twin, and Pilot UI (each in a different terminal):
```bash
# control and digital twin
cd ~/nautilus_ws
ros2 launch nautilus_hal trim_sim.launch.py headless:=false

# Pilot UI
cd ~/nautilus_ws/nautilus-command-bridge-frontend
npm run dev

# MQTT
cd ~/nautilus_ws/nautilus-command-bridge-frontend
mosquitto -c ./mosquitto/mosquitto.conf -v
```
In the UI: press Initialize, then send a mission.

If Gazebo starts but the glider does not move, run `export GZ_IP=127.0.0.1` before launching (needed behind a VPN).


# Deployment



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

