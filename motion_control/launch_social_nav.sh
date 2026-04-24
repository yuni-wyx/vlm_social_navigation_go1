#!/bin/bash
# launch_social_nav.sh — start the minimal social navigation controller
#
# Usage:
#   ./launch_social_nav.sh baseline
#   ./launch_social_nav.sh human_aware
#   ./launch_social_nav.sh human_aware _baseline_stop_dist:=0.7 _social_stop_dist:=1.4
#
# Assumes:
#   - streaming/launch.sh lidar and streaming/launch.sh relay are already running
#   - streaming/launch.sh cameras is running for human_aware mode
#   - sdk_udp_bridge.py is already running on NX1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-baseline}"
shift || true

PI_IP="${PI_IP:-192.168.123.161}"
LAPTOP_IP="${LAPTOP_IP:-192.168.50.86}"

export ROS_MASTER_URI="http://${PI_IP}:11311"
export ROS_IP="${LAPTOP_IP}"
source /opt/ros/noetic/setup.bash

echo "[social_nav] ROS_MASTER_URI=${ROS_MASTER_URI}"
echo "[social_nav] ROS_IP=${ROS_IP}"
echo "[social_nav] mode=${MODE}"

pkill -f "social_nav_controller.py" 2>/dev/null || true

python3 "${SCRIPT_DIR}/scripts/social_nav_controller.py" _mode:="${MODE}" "$@"
