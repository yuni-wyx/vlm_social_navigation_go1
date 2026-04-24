# Phase 1 — Raw Data Streaming (Go1 + ROS 1)

## Go1 Internal Architecture

This document explains the streaming setup and the ROS-side architecture.
Operational secrets, live credentials, and site-specific network details should
be kept in a separate private runbook rather than committed here.

```
┌─────────────────────────────────────────────────────────────┐
│                      Unitree Go1                            │
│                                                             │
│  ┌──────────────┐   192.168.123.x internal LAN (switch)     │
│  │ Raspberry Pi  │◄──────────────────────────────────────┐   │
│  │ .123.xxx      │  (runs roscore, publishes TF + odom)  │   │
│  └──────┬───────┘                                        │   │
│         │ eth (switch)                                   │   │
│  ┌──────┴───────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  Main NX     │  │    NX 2      │  │  Head Nano   │   │   │
│  │ (NX1/.xx)    │  │   (.xx)      │  │   (.xx)      │   │   │
│  │ service acct │  │ service acct │  │ service acct │   │   │
│  │              │  │              │  │              │   │   │
│  │ LiDAR driver │  │ Cam L/R      │  │ Face/Chin cam│   │   │
│  │ Cam 5 (rear) │  │ (3 & 4)      │  │ (1 & 2)     │   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│                                                             │
│  dedicated robot router plugged into the internal switch    │
│  → gives laptop direct access to the 192.168.123.x LAN     │
└─────────────────────────────────────────────────────────────┘
```

### Host Roles

| Computer | Role | Notes |
|----------|------|-------|
| Raspberry Pi | ROS master, TF, odometry | publishes core ROS graph |
| Main NX (NX1) | LiDAR and robot-side bridge | has OpenCV |
| NX 2 | Left/Right cameras | no OpenCV in the tested setup |
| Head Nano | Front / head camera | has OpenCV |
| Laptop | ROS client and relay host | runs the relay and recording scripts |

---

## Step 1 — Connect Laptop to Dog Network

### Recommended: Dedicated Router / Static IP Profile

A dedicated router can be connected to the Go1's internal Ethernet switch so
the laptop gets direct access to the robot-side computers.

```bash
# One-time: create a static IP connection profile for your local robot network
nmcli connection add type wifi con-name "go1_static" \
  ifname YOUR_WIFI_IFACE ssid YOUR_ROBOT_AP_SSID \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk YOUR_ROBOT_AP_PASSWORD \
  ipv4.method manual ipv4.addresses "YOUR_LAPTOP_IP/24" \
  ipv4.never-default yes

# Connect
nmcli connection up "go1_static"

# Verify connectivity to your robot-side hosts
ping -c 1 YOUR_ROS_MASTER_IP
ping -c 1 YOUR_NX1_IP
```

> In the tested setup, the dedicated router path was more reliable than direct
> robot Wi-Fi because it provided direct access to the internal hosts.

### Hostname Fix (one-time)

The ROS master hostname must resolve correctly on the laptop:

```bash
echo "YOUR_ROS_MASTER_IP YOUR_ROS_MASTER_HOSTNAME" | sudo tee -a /etc/hosts
```

Without this, ROS nodes that use `raspberrypi` as their hostname will fail to communicate.

---

## Step 2 — Set ROS Environment

On the laptop, set these in **every terminal**:

```bash
export ROS_MASTER_URI=http://YOUR_ROS_MASTER_IP:11311
export ROS_IP=YOUR_LAPTOP_IP
source /opt/ros/noetic/setup.bash

# Verify
rostopic list
```

---

## Step 3 — Fix Clocks on All Computers

> **Critical**: The Go1 has no RTC backup battery. After every power cycle, clocks revert to ~Sep 2025. This causes `TF_OLD_DATA` errors and breaks TF lookups across frames.

Fix using UTC to avoid timezone issues between computers:

```bash
# Compute UTC time on laptop, then set on each computer using your private
# hostnames / credentials from a non-committed runbook.
UTCNOW=$(date -u "+%Y-%m-%d %H:%M:%S")

ssh YOUR_PI_HOST "sudo date -u -s '$UTCNOW'"
ssh YOUR_NX1_HOST "sudo date -u -s '$UTCNOW'"
ssh YOUR_NX2_HOST "sudo date -u -s '$UTCNOW'"
ssh YOUR_HEAD_HOST "sudo date -u -s '$UTCNOW'"
```

> **Important**: The Pi is in HDT timezone, NX1/NX2 are in MDT. Using `date -u -s` sets UTC directly, avoiding mismatches where the same local time string produces different UNIX epoch values.

---

## Step 4 — Auto-Started Topics (No Launch Needed)

These are published automatically when the Go1 boots:

| Topic | Type | Rate | Source | Frame |
|-------|------|------|--------|-------|
| `/ros2udp/odom` | `nav_msgs/Odometry` | ~200 Hz | Pi (`ros2udp_motion_mode_adv`) | `uodom` → `base_link` |
| `/tf` | `tf2_msgs/TFMessage` | ~100 Hz | Pi | `base → odom → trunk → cameras` |
| `/camera_face/left/image_raw` | `sensor_msgs/Image` | ~3 Hz | Head Nano (autostarted) | — |

> **Odometry note**: Reference point is wherever the robot was at boot (position 0,0,0). Dead reckoning from leg kinematics — drifts over time.

---

## Step 5 — Launch LiDAR (on NX1)

The LiDAR is a **Robosense** using `rslidar_sdk`. It is **not autostarted**.

```bash
ssh YOUR_NX1_HOST \
  'source /opt/ros/melodic/setup.bash; \
   source ~/YOUR_LIDAR_WS/devel/setup.bash; \
   export ROS_MASTER_URI=http://YOUR_ROS_MASTER_IP:11311; \
   export ROS_IP=YOUR_NX1_IP; \
   nohup roslaunch rslidar_sdk start_for_unitree_lidar_slam_3d.launch > /tmp/lidar.log 2>&1 &'
```

**Published topics:**

| Topic | Type | Rate | Frame |
|-------|------|------|-------|
| `/velodyne_points` | `sensor_msgs/PointCloud2` | ~10 Hz | `velodyne` |
| `/scan` | `sensor_msgs/LaserScan` | ~10 Hz | `base_link` |

---

## Step 6 — Launch Side Cameras (on NX2)

NX2 does **not have OpenCV or cv_bridge**. We use a custom `ffmpeg`-based publisher (`scripts/rgb_pub_ffmpeg.py`) that reads V4L2 devices via `ffmpeg` subprocess.

```bash
# 1. Kill autostarted depth camera processes that hold /dev/video*
ssh YOUR_NX2_HOST \
  'pkill -9 -f "point_cloud_node|example_point" 2>/dev/null; sleep 2'

# 2. Deploy the ffmpeg publisher
scp scripts/rgb_pub_ffmpeg.py YOUR_NX2_HOST:/tmp/

# 3. Launch right (video0) and left (video1) cameras
ssh YOUR_NX2_HOST \
  'source /opt/ros/melodic/setup.bash; \
   export ROS_MASTER_URI=http://YOUR_ROS_MASTER_IP:11311; export ROS_IP=YOUR_NX2_IP; \
   nohup python /tmp/rgb_pub_ffmpeg.py camera_right 0 > /tmp/rgb_right.log 2>&1 & \
   nohup python /tmp/rgb_pub_ffmpeg.py camera_left 1 > /tmp/rgb_left.log 2>&1 &'
```

**Published topics:**

| Topic | Type | Rate |
|-------|------|------|
| `/camera_left/left/image_raw` | `sensor_msgs/Image` | ~5 Hz |
| `/camera_right/left/image_raw` | `sensor_msgs/Image` | ~5 Hz |

---

## Step 7 — Sensor Frame Relay (on Laptop)

### The TF Problem

The Go1 publishes **two disconnected TF trees**:

```
Tree 1 (Go1 built-in):  base → odom → trunk → camera_face, camera_left, ...
Tree 2 (lio_sam URDF):  uodom → base_link → chassis_link → velodyne
```

LiDAR uses `velodyne` (Tree 2), odometry uses `uodom`/`base_link` (Tree 2), but cameras and the main `odom` frame are in Tree 1. These trees are **not connected**, so rviz cannot transform LiDAR or odom data into the `odom` fixed frame.

### The Fix: `sensor_relay.py`

A relay node on the laptop republishes the data with corrected frame_ids:

```bash
python3 scripts/sensor_relay.py
```

| Original Topic | Relayed Topic | Frame Change |
|---------------|--------------|--------------|
| `/velodyne_points` (frame: `velodyne`) | `/velodyne_points_odom` | → `trunk` |
| `/scan` (frame: `base_link`) | `/scan_odom` | → `trunk` |
| `/ros2udp/odom` (frame: `uodom`, child: `base_link`) | `/odom_fixed` | → `odom`, child → `trunk` |

After this relay, rviz can set Fixed Frame to `odom` and display all sensors together.

---

## Step 8 — Visualize in RViz

```bash
rosrun rviz rviz -d go1_rviz.rviz
```

The included `go1_rviz.rviz` config has:
- **Fixed Frame**: `odom`
- **LiDAR**: `/velodyne_points_odom` (PointCloud2, rainbow by intensity)
- **Laser Scan**: `/scan_odom` (LaserScan, green flat squares)
- **Odometry**: `/odom_fixed` (arrow trail)
- **Front Camera**: `/camera_face/left/image_raw`
- **Left Camera**: `/camera_left/left/image_raw`
- **Right Camera**: `/camera_right/left/image_raw`
- **TF**: all frames displayed

> **Camera images are fisheye**: The Go1 cameras have wide-angle fisheye lenses. Raw images appear barrel-distorted. Undistortion requires camera intrinsic calibration parameters.

---

## Quick Start (All-in-One)

Use the launch script to automate clock fixes and sensor launches:

```bash
# Start everything after robot boot
./go1_launch.sh all

# Or launch components individually
./go1_launch.sh lidar
./go1_launch.sh cameras
./go1_launch.sh relay
./go1_launch.sh rviz
./go1_launch.sh stop
```

### Manual Quick Start

```bash
# 1. Connect to your robot-network profile
nmcli connection up "go1_static"

# 2. ROS env (every terminal)
export ROS_MASTER_URI=http://YOUR_ROS_MASTER_IP:11311
export ROS_IP=YOUR_LAPTOP_IP
source /opt/ros/noetic/setup.bash

# 3. Fix clocks (after every robot reboot)
UTCNOW=$(date -u "+%Y-%m-%d %H:%M:%S")
ssh YOUR_PI_HOST "sudo date -u -s '$UTCNOW'"
ssh YOUR_NX1_HOST "sudo date -u -s '$UTCNOW'"
ssh YOUR_NX2_HOST "sudo date -u -s '$UTCNOW'"

# 4. Launch LiDAR (NX1)
ssh YOUR_NX1_HOST \
  'source /opt/ros/melodic/setup.bash; \
   source ~/YOUR_LIDAR_WS/devel/setup.bash; \
   export ROS_MASTER_URI=http://YOUR_ROS_MASTER_IP:11311; export ROS_IP=YOUR_NX1_IP; \
   nohup roslaunch rslidar_sdk start_for_unitree_lidar_slam_3d.launch > /tmp/lidar.log 2>&1 &'

# 5. Launch side cameras (NX2) — kill autostart first
ssh YOUR_NX2_HOST \
  'pkill -9 -f "point_cloud_node|example_point" 2>/dev/null; sleep 2; \
   source /opt/ros/melodic/setup.bash; \
   export ROS_MASTER_URI=http://YOUR_ROS_MASTER_IP:11311; export ROS_IP=YOUR_NX2_IP; \
   nohup python /tmp/rgb_pub_ffmpeg.py camera_right 0 > /tmp/rgb_right.log 2>&1 & \
   nohup python /tmp/rgb_pub_ffmpeg.py camera_left 1 > /tmp/rgb_left.log 2>&1 &'

# 6. Sensor relay (laptop terminal, keep running)
python3 scripts/sensor_relay.py &

# 7. RViz
rosrun rviz rviz -d go1_rviz.rviz
```

---

## Complete Topic Summary

| Topic | Relayed | Rate | Source | Notes |
|-------|---------|------|--------|-------|
| `/ros2udp/odom` | `/odom_fixed` | ~200 Hz | Pi (auto) | Leg odometry, frame fixed by relay |
| `/velodyne_points` | `/velodyne_points_odom` | ~10 Hz | NX1 (manual) | 3D LiDAR cloud, frame fixed by relay |
| `/scan` | `/scan_odom` | ~10 Hz | NX1 (manual) | 2D laser scan slice, frame fixed by relay |
| `/camera_face/left/image_raw` | — | ~3 Hz | Nano (auto) | Front fisheye, no relay needed |
| `/camera_left/left/image_raw` | — | ~5 Hz | NX2 (manual) | Left side fisheye, ffmpeg publisher |
| `/camera_right/left/image_raw` | — | ~5 Hz | NX2 (manual) | Right side fisheye, ffmpeg publisher |
| `/tf` | — | ~100 Hz | Pi (auto) | Tree: base→odom→trunk→cameras |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `rostopic list` empty | Wrong `ROS_MASTER_URI` or WiFi disconnected | `ping YOUR_ROS_MASTER_IP`; check `ROS_MASTER_URI` |
| Topics visible, `echo` empty | On a partial network path instead of the internal robot LAN | Use your dedicated robot-network profile for full host access |
| `TF_OLD_DATA` | Clock wrong on robot computers | Fix clocks with `date -u -s` (see Step 3) |
| LiDAR visible in `trunk` but not `odom` | Pi clock still wrong (timestamps far off) | Fix the ROS-master clock specifically with `ssh YOUR_PI_HOST` + `date -u -s` |
| "No transform from uodom to odom" | Disconnected TF trees | Run `sensor_relay.py` — use relayed topics |
| SSH hangs to NX1/NX2 | Stale connections or router / LAN issue | Wait and retry; if needed, restart your dedicated robot-network path |
| Camera shows black / "no new messages" | Autostarted `point_cloud_node` holds device | Kill it: `pkill -9 -f point_cloud_node` then launch publisher |
| Side cameras not working | NX2 has no OpenCV | Use `rgb_pub_ffmpeg.py` (ffmpeg-based, no OpenCV dependency) |
| Camera images look spherical | Fisheye lens distortion | Normal — undistort with camera calibration if needed |
| Clock fix doesn't persist | No RTC battery on Go1 | Must re-fix clocks after every robot power cycle |
| Pi timezone HDT vs NX MDT | Same `date -s` string → different UNIX epochs | Always use `date -u -s` (set UTC directly) |

---

## File Layout

```
dog_ws/
├── go1_launch.sh           # Master launch script
├── go1_rviz.rviz           # RViz config (odom frame, all displays)
├── scripts/
│   ├── sensor_relay.py     # Frame relay (laptop, bridges TF trees)
│   ├── rgb_publisher.py    # OpenCV RGB publisher (Face cam on Nano)
│   └── rgb_pub_ffmpeg.py   # ffmpeg RGB publisher (Side cams on NX2)
└── docs/
    └── streaming.md        # This file
```
