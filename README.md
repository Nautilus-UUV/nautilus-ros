# uuv-ROS

This repository is the basis for the HS25 ROS stack setup, which should run on the RaspberryPi. When working with this repository, do not commit your changes to the main branch directly, instead create your own branch and open a pull request when needed.

## Current packages:

- **[polarisutils](src/py_pkg/py_pkg/polarisutils/)** - Single source of truth for UUV's topics, QoS profiles, message types and node creation helper functions

## Setup

In order to work with this repository, you need to install ROS2 humble following [this](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) instruction.
In order to run a node do the following:
1. source the ROS installation
```bash
source /opt/ros/humble/setup.bash
```
2. build packages
```bash
colcon build
```
3. source the workspace
```bash
source install/setup.bash
```

4. depending on the language used for your node, either run
```
ros2 run py_pkg YOUR_NODE_NAME
ros2 run cpp_pkg YOUR_NODE_NAME
```

if you get an error that colcon is not available, run
```bash
sudo apt update
sudo apt install python3-colcon-common-extensions
```
and then verify using
```bash
which colcon
```
the result should be something like `/usr/bin/colcon`

## Working with topics in the command line

Manual publishing of messages or inspecting the trafic might be usefull for debugging. For this purpose you can use the following commands (keep in mind that you need to source ROS and the workspace before):

```bash
ros2 topic list
ros2 topic pub /my_topic std_msgs/String 'data: Hello World'
ros2 topic pub /my_topic std_msgs/String 'data: Hello World' -1
ros2 topic echo /my_topic
```

## Adding Nodes

Nodes can either be written in C++ or Python. Example nodes are provided in the `usage_example` branch

### Python

In the folder `src/py_pkg/py_pkg` add `<YOUR_NODE>.py` with a main function, then add an entry to `console_scripts` in `src/py_pkg/setup.py`
```Python
entry_points={
    'console_scripts': [
        ...,
        '<YOUR_NODE> = py_pkg.<YOUR_NODE>:main',
        ...
    ],
},
```


### C++

In the folder `src/cpp_pkg/src` add `<YOUR_NODE>.cpp` and register it in `src/cpp_pkg/CMAkeLists.txt` using
```CMAKE
add_executable(status_subscriber src/<YOUR_NODE>.cpp)
ament_target_dependencies(<YOUR_NODE> rclcpp std_msgs)
install(TARGETS <YOUR_NODE> DESTINATION lib/${PROJECT_NAME})
```

## Continuous Intergration (CI)

Currently there is a CI pipeline that will build the project inside a docker container running `ROS2 Humble` on `Ubuntu:latest`. If the build fails or the smoke test does not pass, your PR or Push will be flagged.

