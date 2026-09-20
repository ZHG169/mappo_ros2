#!/usr/bin/env bash
set -e

# Make ROS 2 available to the container's startup command, including
# non-interactive commands launched through `docker compose run`.
source /opt/ros/jazzy/setup.bash

# Source the local overlay only after it has been built with colcon.
ROS2_WS="${ROS2_WS:-/home/ncrl/mappo_ros2}"
if [ -f "${ROS2_WS}/install/setup.bash" ]; then
    source "${ROS2_WS}/install/setup.bash"
fi

exec "$@"
