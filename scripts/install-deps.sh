#!/bin/bash
set -eo pipefail

# ROS2 system dependencies
ROS_DISTRO=${ROS_DISTRO:-humble}

# Logic: Check if running as root. If not, use sudo.
if [ "$EUID" -ne 0 ]; then
  SUDO="sudo"
  echo "Running with sudo rights..."
else
  SUDO=""
fi

$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  python3-colcon-common-extensions \
  ros-$ROS_DISTRO-ros2cli \
  ros-$ROS_DISTRO-std-msgs \
  ros-$ROS_DISTRO-sensor-msgs \
  ros-$ROS_DISTRO-geometry-msgs \
  ros-$ROS_DISTRO-can-msgs