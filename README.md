
ROS2 navigation and control system for Nautilus's autonomous glider (UG).

# Development

Create a folder `~/nautilus_ws/`. Here you will clone all repositories related to the UG.

## Setup

To complete the setup you need to follow README instructions of our 3 repositories:

1. Control Stack (this README)
2. Digital Twin: ([Nautilus-UUV/nautilus-dave](https://github.com/Nautilus-UUV/nautilus-dave))
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
2. Digital Twin: ([Nautilus-UUV/nautilus-dave](https://github.com/Nautilus-UUV/nautilus-dave))
3. Pilot UI ([nautilus-command-bridge-frontend](https://github.com/Nautilus-UUV/nautilus-command-bridge-frontend))


## Control with Digital Twin

When developing code and testing it is crucial to first test your changes on the digital twin. At this point we assume all 3 repositories are setup.

1. Rebuild:
```bash
cd ~/nautilus_ws
python -m colcon build --symlink-install
source install/setup.bash
```

If rebuilding fails, try removing the deprecated folders first and try again:
```bash
cd ~/nautilus_ws
rm -rf install/ log/ build/
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

## Automatic Testing

We test the code at various levels. When modifying or developing code, only change tests that relate to the unit you work on.

| Tier | Folder | What it tests |
|---|---|---|
| 1 | `test/unit/` | Pure logic |
| 2 | `test/node/` | Single ROS nodes with stubbed inputs |
| 3 | `test/sim/` | Full stack in the digital twin (slow, each test starts Gazebo) |

1. Run Tier 1 + 2:
```bash
cd ~/nautilus_ws/src/nautilus-ros/src/py_pkg
python -m pytest test/ -v
```
2. Run Tier 3:
```bash
cd ~/nautilus_ws/src/nautilus-ros/src/py_pkg
python -m pytest -m sim test/sim/ -v
```
If tests are failing make sure the transport layer does not fail due to a VPN. If you use a VPN set the enviroment variable `GZ_IP=127.0.0.1`.

3. Run a single Tier 3 test with the Gazebo window:
```bash
SIM_GUI=1 python -m pytest -m sim test/sim/test_trim_neutral_sim.py -v -s
```

## Contributing Workflow

### Create a Feature Branch

```bash
cd ~/nautilus_ws/src/nautilus-ros

# Make sure you're on the dev branch
git checkout dev

# Pull the latest changes
git pull origin dev

# Create your feature branch
git checkout -b github_username/feature_name
```

### Make Your Changes

Edit files in the appropriate locations:

- **Controllers (BCU, ACU)**: `src/py_pkg/py_pkg/control/`
- **Estimators**: `src/py_pkg/py_pkg/imu_prefilter/`, `src/py_pkg/py_pkg/attitude/`
- **Missions**: `src/py_pkg/py_pkg/path/missions/`
- **Topics, QoS, message types**: `src/py_pkg/py_pkg/uuv_ros_core/` (never hardcode topic strings in a node)
- **Message definitions**: `src/nautilus_msgs/msg/`
- **Scenarios**: `src/py_pkg/py_pkg/scenarios/library/`
- **New nodes**: add a `console_scripts` entry in `src/py_pkg/setup.py`

```
nautilus-ros/
├── src/
│   ├── nautilus_msgs/
│   │   └── msg/                    # Shared message types
│   └── py_pkg/
│       ├── launch/                 # Control stack + mission autostart launches
│       ├── py_pkg/
│       │   ├── imu_prefilter/      # Raw IMU prefilter
│       │   ├── attitude/           # State estimator (/position/estimation)
│       │   ├── control/            # BCU + ACU controllers
│       │   ├── path/               # Mission dispatcher + mission profiles
│       │   ├── scenarios/          # Scenario YAML schema + library
│       │   ├── uuv_ros_core/       # Topic / QoS / message type registry
│       │   └── ...                 # mqtt, liveness, stm_com, debug
│       ├── test/                   # Tier 1-3 tests
│       └── setup.py                # Node entry points
├── scripts/                        # Monte Carlo sweep tooling
├── docs/                           # Sim and sweep guides
└── README.md                       # This file
```

### Test Your Changes

```bash
# Rebuild the workspace
cd ~/nautilus_ws
python -m colcon build --symlink-install

# Source the workspace
source install/setup.bash

# Run Tier 1 + 2 tests
cd ~/nautilus_ws/src/nautilus-ros/src/py_pkg
python -m pytest test/ -v

# Run the Tier 3 tests related to your change (see Automatic Testing)
python -m pytest -m sim test/sim/<test_file>.py -v

# Try it on the digital twin (see Control with Digital Twin)
ros2 launch nautilus_hal trim_sim.launch.py headless:=false
```

If you changed a `.msg` file, rebuild `nautilus_msgs` and re-source before running anything that imports it.

### Open a Pull Request

Commit using [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat(bcu): ...`, `fix(attitude): ...`), format the files you touched with Black, push your branch, and open a pull request into `dev` (not `main`).


