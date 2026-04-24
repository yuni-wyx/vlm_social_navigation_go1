#!/bin/bash
# Run RTAB-Map 3D mapping on a Go1 bag file
# Usage: ./run.sh [path/to/fixed_bag.bag]
# Works standalone — no robot needed.

BAG="${1:-$(ls -t /home/cairo/dog_ws/streaming/bags/*_fixed.bag 2>/dev/null | head -1)}"
if [ ! -f "$BAG" ]; then
    echo "ERROR: No bag file found. Usage: $0 <fixed_bag.bag>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/output"

echo "============================================"
echo "  RTAB-Map 3D Mapping from Go1 Bag"
echo "  Bag: $BAG"
echo "============================================"

# Kill existing ROS processes
killall -9 rosmaster roscore rtabmap rtabmap_viz rosbag rviz 2>/dev/null
sleep 2

# Force local ROS
source /opt/ros/noetic/setup.bash
unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=127.0.0.1

# Start roscore and wait
roscore &
echo "Waiting for roscore..."
until rostopic list > /dev/null 2>&1; do sleep 1; done
echo "roscore ready"

rosparam set use_sim_time true

# Launch camera_info publisher (estimated Go1 fisheye intrinsics)
echo "Starting camera_info publisher..."
python3 "$SCRIPT_DIR/camera_info_publisher.py" &
CAMINFO_PID=$!
sleep 1

# Launch RTAB-Map
echo "Launching RTAB-Map (LiDAR ICP + visual loop closure)..."
roslaunch "$SCRIPT_DIR/go1_rtabmap.launch" \
    db_path:="$SCRIPT_DIR/output/rtabmap.db" \
    use_sim_time:=true &
RTABMAP_PID=$!
sleep 5

# Play bag
echo "Playing bag... (this will take ~$(rosbag info --yaml "$BAG" 2>/dev/null | grep duration | head -1 | awk '{printf "%.0f", $2}')s)"
rosbag play --clock "$BAG"

echo ""
echo "============================================"
echo "  Bag playback finished!"
echo "  Database: $SCRIPT_DIR/output/rtabmap.db"
echo "============================================"
echo ""
echo "Press Enter to stop RTAB-Map, or keep it open to inspect the map..."
read -r
kill $RTABMAP_PID $CAMINFO_PID 2>/dev/null
killall roscore rosmaster 2>/dev/null
