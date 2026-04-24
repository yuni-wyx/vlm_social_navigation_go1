#!/bin/bash
# Replay a fixed bag file with local roscore + rviz
# Usage: ./replay.sh <bag_file>
# Works standalone - no robot or network needed.

BAG="${1:-$(ls -t /home/cairo/dog_ws/streaming/bags/*_fixed.bag 2>/dev/null | head -1)}"
if [ ! -f "$BAG" ]; then
    echo "ERROR: No bag file found. Usage: $0 <bag_file>"
    exit 1
fi

echo "Replaying: $BAG"

# Kill anything existing
killall -9 rosmaster roscore rviz rosbag 2>/dev/null
sleep 2

# Force local ROS - override any existing env
source /opt/ros/noetic/setup.bash
unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=127.0.0.1

# Start roscore and wait until it's ready
roscore &
echo "Waiting for roscore..."
until rostopic list > /dev/null 2>&1; do sleep 1; done
echo "roscore ready"

# Set sim time
rosparam set use_sim_time true

# Start rviz
rosrun rviz rviz -d /home/cairo/dog_ws/streaming/go1_rviz.rviz &
sleep 3

# Play bag
echo "Playing bag... Ctrl+C to stop."
rosbag play --clock --wait "$BAG"

# Cleanup
echo "Playback finished. Killing processes..."
killall rosmaster roscore rviz 2>/dev/null
