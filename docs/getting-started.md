# Getting Started
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

## Debugging
### Working with topics in the command line

Manual publishing of messages or inspecting the trafic might be usefull for debugging. For this purpose you can use the following commands (keep in mind that you need to source ROS and the workspace before):

```bash
ros2 topic list
ros2 topic pub /my_topic std_msgs/String 'data: Hello World'
ros2 topic pub /my_topic std_msgs/String 'data: Hello World' -1
ros2 topic echo /my_topic
```