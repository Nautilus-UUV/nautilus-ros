# Running the Nautilus simulation

**Important:** build the workspace with the `dave_ws` as root following the [nautilus-dave](https://github.com/Nautilus-UUV/nautilus-dave/tree/dev) repository documentation. 



## Running Interactive Missions with Dave

Build the workspace
```bash
cd /home/$USER/dave_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Trim and Neutral Mission



```bash
ros2 launch nautilus_hal trim_sim.launch.py \
    headless:=false \
    mission_autostart:=true \
    target_pressure_pa:=65332.0
```

- `scenario:=<path>` selects the scenario YAML that drives gains, plant,
  bridge publish rates, and fault injection. Defaults to the
  `library/nominal.yaml` that is considered the best overall configuration of our setup:



#### Re-firing mid-run


```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 0, target_pressure_pa: 60295.0, angle_rad: 0.0, n_resurfaces: 0}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/Bool "{data: true}"
```


### Sawtooth Mission

Also sometimes referre as a YoYo path. It is descrbied by the UUV gliding up and down in between a predefined tuple of pressures.

```bash
ros2 launch nautilus_hal sawtooth_sim.launch.py \
    headless:=false \
    mission_autostart:=true \
    target_pressure_pa:=147150.0 \
    angle_rad:=0.6109 \
    n_resurfaces:=1
```

#### Re-firing mid-run

```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 1, target_pressure_pa: 147150.0, angle_rad: 0.6109, n_resurfaces: 2}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/Bool "{data: true}"
```


### Surface Mission Profile

```bash
ros2 launch nautilus_hal surface_sim.launch.py \
    headless:=false \
    mission_autostart:=true
```

#### Re-firing mid-run

```bash
ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /path nautilus_msgs/msg/MissionCommand \
    "{mission_id: 2, target_pressure_pa: 0.0, angle_rad: 0.0, n_resurfaces: 0}"

ros2 topic pub --once \
    --qos-reliability reliable --qos-durability transient_local \
    /command std_msgs/msg/Bool "{data: true}"
```


> To stop any running mission and reset the stack to its clean initial state
> (no RPM, valves closed, no mission loaded), publish a Bool `false` on
> `/command`:
>
> ```bash
> ros2 topic pub --once /command std_msgs/msg/Bool "{data: false}"
> ```

### Inspecting state

```bash
source install/setup.bash

# Ground-truth pose (privileged sim-only — bridged out of Gazebo by
# dave_robot_models/config/glider_nautilus/robot_config.py):
ros2 topic echo /model/glider_nautilus/odometry --once

# What the controller is tracking + commanding:
ros2 topic echo /position/target
ros2 topic echo /external/pressure
ros2 topic echo /bcu/rpm
ros2 topic echo /bcu/volume
ros2 topic echo /acu/pitch
ros2 topic echo /acu/roll

# What the estimator reports:
ros2 topic echo /position/estimation
```

## Dive Simulation with the Frontend

Even better you can run it with the official mission laptop setup.

For this, both the [nautilus-dave](https://github.com/Nautilus-UUV/nautilus-dave/tree/dev) repository and [nautilus-command-bridge-frontend](https://github.com/Nautilus-UUV/nautilus-command-bridge-frontend) repository must be setup.

Then, an interactive session as close as to real-life can be initialized as follows:

### UI Connection

#### Mission Laptop

Go to the `nautilus-command-bridge-frontend` folder and:

```bash
mosquitto -c ./mosquitto/mosquitto.conf -v
```

```bash
npm run dev
```

#### Main Board Simulation

Build the `dave_ws` workspace as before and:

Start the HAL:
```bash
ros2 launch nautilus_hal bridge.launch.py
```

Start the simulation:
```bash
ros2 launch dave_demos dave_robot.launch.py \
    namespace:=glider_nautilus world_name:=dave_ocean_waves \
    z:=-5 roll:=3.141592653589793 yaw:=1.5707963267948966 \
    paused:=false headless:=false
```

Start control:
```bash
ros2 launch py_pkg control_stack.launch.py
```
