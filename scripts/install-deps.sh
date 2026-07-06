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

# Tooling that isn't a package <depend>: colcon for building, rosdep for
# resolving every package.xml below, pip for the pydantic step further down.
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pip \
  ros-$ROS_DISTRO-ros2cli

# `py_pkg/scenarios/spec/_shared.py` uses pydantic v2-only API (`ConfigDict`,
# `model_dump`). On Ubuntu Jammy (which `ros:humble` runs on) apt's
# `python3-pydantic` ships v1, so we install v2 via pip 
$SUDO env PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install "pydantic>=2"

# rosdep is pre-initialized in the ros: base images; guard for local runs.
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  $SUDO rosdep init
fi
rosdep update

# Resolve every <depend> under src/ against the running ROS_DISTRO.
# --skip-keys protobuf matches the workspace Dockerfile: rosdep's protobuf
# resolution conflicts with the ros-gz stack and protobuf is installed by
# other means.
rosdep install \
  --from-paths src \
  --ignore-src \
  -y \
  --rosdistro "$ROS_DISTRO" \
  --skip-keys "protobuf"
