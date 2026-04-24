#!/bin/bash
# launch.sh — Launch all Go1 sensor streaming from one script
#
# Prerequisites:
#   - Laptop connected to dog_go_zoom WiFi (static IP 192.168.123.67)
#   - sshpass installed on laptop
#   - ROS Noetic sourced
#
# Usage:
#   ./launch.sh              # launch everything (sensors + relay)
#   ./launch.sh lidar        # launch only lidar on NX1
#   ./launch.sh cameras      # launch cameras on Nano + NX2
#   ./launch.sh relay        # launch the sensor frame relay on laptop
#   ./launch.sh rviz         # launch rviz
#   ./launch.sh record       # record bag file (default name)
#   ./launch.sh record name  # record bag file with custom name
#   ./launch.sh stop         # kill all launched processes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# === Config ===
PI_IP="192.168.123.161"
NX1_IP="192.168.123.15"
NX2_IP="192.168.123.14"
NANO_IP="192.168.123.13"
SSH_PASS="123"
LAPTOP_IP="192.168.123.67"
ROS_MASTER="http://${PI_IP}:11311"

ssh_cmd() {
    sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$@"
}

scp_cmd() {
    sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$@"
}

setup_ros_env() {
    export ROS_MASTER_URI="$ROS_MASTER"
    export ROS_IP="$LAPTOP_IP"
    source /opt/ros/noetic/setup.bash
}

wait_for_boot() {
    echo "[boot] Waiting for robot to be reachable..."
    for ip in $PI_IP $NX1_IP; do
        while ! ping -c 1 -W 2 $ip > /dev/null 2>&1; do
            echo "[boot]   waiting for $ip..."
            sleep 2
        done
        echo "[boot]   $ip OK"
    done
    echo "[boot] Waiting 15s for autostart processes to settle..."
    sleep 15
}

kill_autostart_cameras() {
    local ip=$1
    local user=$2
    echo "[kill] Killing autostarted camera processes on $ip..."
    ssh_cmd ${user}@${ip} "pkill -9 -f 'point_cloud_node|example_point' 2>/dev/null; sleep 2; echo done" 2>/dev/null || true
}

fix_clocks() {
    # Use UTC to avoid timezone mismatches (Pi=HDT, Jetsons=MDT)
    local UTCNOW=$(date -u "+%Y-%m-%d %H:%M:%S")
    echo "[clock] Setting all clocks to UTC: $UTCNOW"
    for target in "pi@${PI_IP}" "unitree@${NX1_IP}" "unitree@${NX2_IP}" "unitree@${NANO_IP}"; do
        echo "[clock]   $target..."
        ssh_cmd -t $target "echo '${SSH_PASS}' | sudo -S date -u -s '$UTCNOW'" 2>/dev/null || echo "[clock]   WARN: failed for $target"
    done
    echo "[clock] Done."
}

launch_lidar() {
    echo "[lidar] Launching LiDAR on NX1 ($NX1_IP)..."
    ssh_cmd unitree@${NX1_IP} \
        "source /opt/ros/melodic/setup.bash; \
         source ~/UnitreeSLAM/catkin_lidar_slam_3d/devel/setup.bash; \
         export ROS_MASTER_URI=${ROS_MASTER}; \
         export ROS_IP=${NX1_IP}; \
         nohup roslaunch rslidar_sdk start_for_unitree_lidar_slam_3d.launch > /tmp/lidar.log 2>&1 &"
    echo "[lidar] Done. Topics: /velodyne_points, /scan"
}

launch_face_camera() {
    echo "[camera] Deploying face camera script to Head Nano ($NANO_IP)..."
    scp_cmd "${SCRIPT_DIR}/scripts/rgb_publisher.py" unitree@${NANO_IP}:/tmp/rgb_publisher.py 2>/dev/null

    kill_autostart_cameras ${NANO_IP} unitree

    echo "[camera] Launching face camera (video1) on Nano..."
    ssh_cmd unitree@${NANO_IP} \
        "source /opt/ros/melodic/setup.bash; \
         export ROS_MASTER_URI=${ROS_MASTER}; \
         export ROS_IP=${NANO_IP}; \
         nohup python /tmp/rgb_publisher.py camera_face 1 > /tmp/rgb_face.log 2>&1 &"
    echo "[camera] Done. Topic: /camera_face/left/image_raw"
}

launch_side_cameras() {
    echo "[camera] Deploying side camera script to NX2 ($NX2_IP)..."
    scp_cmd "${SCRIPT_DIR}/scripts/rgb_pub_ffmpeg.py" unitree@${NX2_IP}:/tmp/rgb_pub_ffmpeg.py 2>/dev/null

    kill_autostart_cameras ${NX2_IP} unitree

    echo "[camera] Launching right camera (video0) on NX2..."
    ssh_cmd unitree@${NX2_IP} \
        "source /opt/ros/melodic/setup.bash; \
         export ROS_MASTER_URI=${ROS_MASTER}; \
         export ROS_IP=${NX2_IP}; \
         nohup python /tmp/rgb_pub_ffmpeg.py camera_right 0 > /tmp/rgb_right.log 2>&1 &"

    echo "[camera] Launching left camera (video1) on NX2..."
    ssh_cmd unitree@${NX2_IP} \
        "source /opt/ros/melodic/setup.bash; \
         export ROS_MASTER_URI=${ROS_MASTER}; \
         export ROS_IP=${NX2_IP}; \
         nohup python /tmp/rgb_pub_ffmpeg.py camera_left 1 > /tmp/rgb_left.log 2>&1 &"
    echo "[camera] Done. Topics: /camera_left/left/image_raw, /camera_right/left/image_raw"
}

launch_relay() {
    echo "[relay] Launching sensor frame relay on laptop..."
    setup_ros_env
    pkill -f "sensor_relay.py" 2>/dev/null
    python3 "${SCRIPT_DIR}/scripts/sensor_relay.py" &
    echo "[relay] Done. Relayed topics: /velodyne_points_odom, /scan_odom, /odom_fixed"
}

launch_rviz() {
    echo "[rviz] Launching RViz..."
    setup_ros_env
    rosrun rviz rviz -d "${SCRIPT_DIR}/go1_rviz.rviz" &
    echo "[rviz] Done."
}

launch_record() {
    setup_ros_env
    bash "${SCRIPT_DIR}/scripts/record.sh" "${2:-}"
}

stop_all() {
    echo "[stop] Killing launched processes..."
    pkill -f "sensor_relay.py" 2>/dev/null || true
    pkill -f "rosrun rviz" 2>/dev/null || true
    pkill -f "rosbag record" 2>/dev/null || true
    ssh_cmd unitree@${NX1_IP} "pkill -f 'rslidar_sdk\|rgb_publisher' 2>/dev/null" || true
    ssh_cmd unitree@${NANO_IP} "pkill -f 'rgb_publisher' 2>/dev/null" || true
    ssh_cmd unitree@${NX2_IP} "pkill -f 'rgb_pub_ffmpeg' 2>/dev/null" || true
    echo "[stop] Done."
}

# === Main ===
case "${1:-all}" in
    lidar)
        fix_clocks
        launch_lidar
        ;;
    cameras)
        fix_clocks
        launch_face_camera
        launch_side_cameras
        ;;
    relay)
        launch_relay
        ;;
    rviz)
        launch_rviz
        ;;
    record)
        launch_record "$@"
        ;;
    stop)
        stop_all
        ;;
    all)
        wait_for_boot
        fix_clocks

        echo ""
        echo "=== Launching LiDAR ==="
        launch_lidar

        echo ""
        echo "=== Launching Cameras ==="
        launch_face_camera
        launch_side_cameras

        echo ""
        echo "=== Launching Relay ==="
        launch_relay

        echo ""
        echo "=== All Sensors Running ==="
        echo ""
        echo "Relayed topics (use these for rviz/recording):"
        echo "  /velodyne_points_odom   (~10 Hz, frame=trunk)"
        echo "  /scan_odom              (~10 Hz, frame=trunk)"
        echo "  /odom_fixed             (~200 Hz, frame=odom→trunk)"
        echo "  /camera_face/left/image_raw   (~3 Hz)"
        echo "  /camera_left/left/image_raw   (~5 Hz)"
        echo "  /camera_right/left/image_raw  (~5 Hz)"
        echo ""
        echo "Next steps:"
        echo "  ./launch.sh rviz              # visualize"
        echo "  ./launch.sh record myrun01    # record bag"
        echo "  ./launch.sh stop              # stop everything"
        ;;
    *)
        echo "Usage: $0 [all|lidar|cameras|relay|rviz|record [name]|stop]"
        ;;
esac
