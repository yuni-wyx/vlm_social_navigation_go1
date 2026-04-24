#!/bin/bash
# Live test of the LLM Subsumption Controller on the Go1
# Usage: bash test_live.sh
#
# Prerequisites:
#   - Robot is on and streaming (bash launch.sh on the NX)
#   - roscore running locally
#
# This script:
#   1. Sets up ROS env to point to the robot
#   2. Starts sensor_relay.py (relays scan_odom, odom_fixed)
#   3. Starts the controller in DRY RUN mode (prints only, no movement)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(dirname "$SCRIPT_DIR")"

source /opt/ros/noetic/setup.bash

echo "=== LLM Subsumption Controller — LIVE DRY RUN TEST ==="
echo ""
echo "This will connect to the robot and print controller decisions."
echo "NO movement commands will be sent (dry_run=true)."
echo ""

# Check if roscore is running
if ! rostopic list > /dev/null 2>&1; then
    echo "ERROR: roscore not running. Start it first: roscore"
    exit 1
fi

# Start sensor relay if not already running
if ! rostopic list 2>/dev/null | grep -q "/scan_odom"; then
    echo "[1/2] Starting sensor relay..."
    python3 "$WS_DIR/streaming/scripts/sensor_relay.py" &
    RELAY_PID=$!
    sleep 3
    echo "      Relay PID: $RELAY_PID"
else
    echo "[1/2] sensor_relay already running (scan_odom topic exists)"
fi

# Wait for topics
echo "[2/2] Waiting for /scan_odom topic..."
timeout 15 rostopic echo /scan_odom -n 1 > /dev/null 2>&1 && echo "      Got scan data!" || echo "      WARNING: No scan data after 15s"

echo ""
echo "=== Starting controller (dry run) ==="
echo "    Put objects in front of the robot to test P1 safety stop."
echo "    Watch for RED P1_SAFETY_STOP decisions."
echo "    Ctrl+C to stop."
echo ""

python3 "$SCRIPT_DIR/llm_subsumption_controller.py" _dry_run:=true
