#!/bin/bash
# record.sh — Record all Go1 sensor data to a bag file
#
# Records the RELAYED topics (with corrected frame_ids) so downstream
# consumers get consistent data without needing to know about the relay.
#
# Usage:
#   ./record.sh              # records to bags/YYYY-MM-DD_HHMMSS.bag
#   ./record.sh myrun01      # records to bags/myrun01.bag

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAG_DIR="${SCRIPT_DIR}/../bags"
mkdir -p "$BAG_DIR"

# Session name
if [ -n "$1" ]; then
    SESSION="$1"
else
    SESSION=$(date "+%Y-%m-%d_%H%M%S")
fi

BAG_PATH="${BAG_DIR}/${SESSION}"

echo "=== Go1 Bag Recording ==="
echo "Output: ${BAG_PATH}.bag"
echo "Topics:"
echo "  /velodyne_points_odom   (3D LiDAR, frame=trunk)"
echo "  /scan_odom              (2D scan, frame=trunk)"
echo "  /odom_fixed             (odometry, frame=odom→trunk)"
echo "  /camera_face/left/image_raw   (front camera)"
echo "  /camera_left/left/image_raw   (left camera)"
echo "  /camera_right/left/image_raw  (right camera)"
echo "  /tf                     (transforms)"
echo "  /tf_static              (static transforms)"
echo ""
echo "Press Ctrl+C to stop recording."
echo ""

rosbag record -O "$BAG_PATH" \
    /velodyne_points_odom \
    /scan_odom \
    /odom_fixed \
    /camera_face/left/image_raw \
    /camera_left/left/image_raw \
    /camera_right/left/image_raw \
    /tf \
    /tf_static
