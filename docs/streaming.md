# Phase 1 — Raw Data Streaming (Go1 + ROS 1)

## Go1 Internal Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Unitree Go1                            │
│                                                             │
│  ┌──────────────┐   192.168.123.x internal LAN (switch)     │
│  │ Raspberry Pi  │◄──────────────────────────────────────┐   │
│  │ .123.161      │  (runs roscore, publishes TF + odom)  │   │
│  │ pw: 123       │                                       │   │
│  └──────┬───────┘                                        │   │
│         │ eth (switch)                                   │   │
│  ┌──────┴───────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  Main NX     │  │    NX 2      │  │  Head Nano   │   │   │
│  │ (NX1/.15)    │  │   (.14)      │  │   (.13)      │   │   │
│  │ unitree@     │  │ unitree@     │  │ unitree@     │   │   │
│  │ pw: 123      │  │ pw: 123      │  │ pw: 123      │   │   │
│  │              │  │              │  │              │   │   │
│  │ LiDAR driver │  │ Cam L/R      │  │ Face/Chin cam│   │   │
│  │ Cam 5 (rear) │  │ (3 & 4)      │  │ (1 & 2)     │   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│                                                             │
│  router "dog_go_zoom" plugged into the internal switch      │
│  → gives laptop direct access to the 192.168.123.x LAN     │
└─────────────────────────────────────────────────────────────┘
```

### Key IPs

| Computer | IP | SSH User/Password | Notes |
|----------|----|--------------------|-------|
| Raspberry Pi | `192.168.123.161` | `pi` / `123` | roscore, TF, odom |
| Main NX (NX1) | `192.168.123.15` | `unitree` / `123` | LiDAR, has OpenCV |
| NX 2 | `192.168.123.14` | `unitree` / `123` | Left/Right cams, **no OpenCV** |
| Head Nano | `192.168.123.13` | `unitree` / `123` | Face cam, has OpenCV |
| Laptop | `192.168.123.67` | — | Static IP via `dog_go_zoom` |

---

## Step 1 — Connect Laptop to Dog Network

### Recommended: `dog_go_zoom` Router (Static IP)

A router named `dog_go_zoom` is physically connected to the Go1's internal Ethernet switch. This gives the laptop **direct access** to all internal computers.

```bash
# One-time: create static IP connection profile
nmcli connection add type wifi con-name "dog_go_zoom_static" \
  ifname wlx8c3bad120bdc ssid "dog_go_zoom" \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "eces111A" \
  ipv4.method manual ipv4.addresses "192.168.123.67/24" \
  ipv4.never-default yes

# Connect
nmcli connection up "dog_go_zoom_static"

# Verify: should be able to ping ALL internal IPs directly
ping -c 1 192.168.123.161  # Pi
ping -c 1 192.168.123.15   # NX1
```

> **Why not direct Go1 WiFi?** The Go1 broadcasts `Unitree_Go337822A` (pw: `00000000`), but that WiFi only bridges traffic to/from the Pi. You cannot directly SSH to NX1/NX2/Nano from it.

### Hostname Fix (one-time)

The Pi's hostname `raspberrypi` must resolve to the correct IP on the laptop:

```bash
echo "192.168.123.161 raspberrypi" | sudo tee -a /etc/hosts
```

Without this, ROS nodes that use `raspberrypi` as their hostname will fail to communicate.

---

## Step 2 — Set ROS Environment

On the laptop, set these in **every terminal**:

```bash
export ROS_MASTER_URI=http://192.168.123.161:11311
export ROS_IP=192.168.123.67
source /opt/ros/noetic/setup.bash

# Verify
rostopic list
```

---

## Step 3 — Fix Clocks on All Computers

> **Critical**: The Go1 has no RTC backup battery. After every power cycle, clocks revert to ~Sep 2025. This causes `TF_OLD_DATA` errors and breaks TF lookups across frames.

Fix using UTC to avoid timezone issues between computers:

```bash
# Compute UTC time on laptop, then set on each computer
UTCNOW=$(date -u "+%Y-%m-%d %H:%M:%S")

sshpass -p '123' ssh -t pi@192.168.123.161 "echo '123' | sudo -S date -u -s '$UTCNOW'"
sshpass -p '123' ssh -t unitree@192.168.123.15 "echo '123' | sudo -S date -u -s '$UTCNOW'"
sshpass -p '123' ssh -t unitree@192.168.123.14 "echo '123' | sudo -S date -u -s '$UTCNOW'"
sshpass -p '123' ssh -t unitree@192.168.123.13 "echo '123' | sudo -S date -u -s '$UTCNOW'"
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
sshpass -p '123' ssh unitree@192.168.123.15 \
  'source /opt/ros/melodic/setup.bash; \
   source ~/UnitreeSLAM/catkin_lidar_slam_3d/devel/setup.bash; \
   export ROS_MASTER_URI=http://192.168.123.161:11311; \
   export ROS_IP=192.168.123.15; \
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
sshpass -p '123' ssh unitree@192.168.123.14 \
  'pkill -9 -f "point_cloud_node|example_point" 2>/dev/null; sleep 2'

# 2. Deploy the ffmpeg publisher
sshpass -p '123' scp scripts/rgb_pub_ffmpeg.py unitree@192.168.123.14:/tmp/

# 3. Launch right (video0) and left (video1) cameras
sshpass -p '123' ssh unitree@192.168.123.14 \
  'source /opt/ros/melodic/setup.bash; \
   export ROS_MASTER_URI=http://192.168.123.161:11311; export ROS_IP=192.168.123.14; \
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
# 1. Connect to dog_go_zoom
nmcli connection up "dog_go_zoom_static"

# 2. ROS env (every terminal)
export ROS_MASTER_URI=http://192.168.123.161:11311
export ROS_IP=192.168.123.67
source /opt/ros/noetic/setup.bash

# 3. Fix clocks (after every robot reboot)
UTCNOW=$(date -u "+%Y-%m-%d %H:%M:%S")
sshpass -p '123' ssh -t pi@192.168.123.161 "echo '123' | sudo -S date -u -s '$UTCNOW'"
sshpass -p '123' ssh -t unitree@192.168.123.15 "echo '123' | sudo -S date -u -s '$UTCNOW'"
sshpass -p '123' ssh -t unitree@192.168.123.14 "echo '123' | sudo -S date -u -s '$UTCNOW'"

# 4. Launch LiDAR (NX1)
sshpass -p '123' ssh unitree@192.168.123.15 \
  'source /opt/ros/melodic/setup.bash; \
   source ~/UnitreeSLAM/catkin_lidar_slam_3d/devel/setup.bash; \
   export ROS_MASTER_URI=http://192.168.123.161:11311; export ROS_IP=192.168.123.15; \
   nohup roslaunch rslidar_sdk start_for_unitree_lidar_slam_3d.launch > /tmp/lidar.log 2>&1 &'

# 5. Launch side cameras (NX2) — kill autostart first
sshpass -p '123' ssh unitree@192.168.123.14 \
  'pkill -9 -f "point_cloud_node|example_point" 2>/dev/null; sleep 2; \
   source /opt/ros/melodic/setup.bash; \
   export ROS_MASTER_URI=http://192.168.123.161:11311; export ROS_IP=192.168.123.14; \
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
| `rostopic list` empty | Wrong `ROS_MASTER_URI` or WiFi disconnected | `ping 192.168.123.161`; check `ROS_MASTER_URI` |
| Topics visible, `echo` empty | On Go1 direct WiFi (not `dog_go_zoom`) | Use `dog_go_zoom` for full network access |
| `TF_OLD_DATA` | Clock wrong on robot computers | Fix clocks with `date -u -s` (see Step 3) |
| LiDAR visible in `trunk` but not `odom` | Pi clock still wrong (timestamps 6 months off) | Fix Pi clock specifically: `ssh pi@...` + `date -u -s` |
| "No transform from uodom to odom" | Disconnected TF trees | Run `sensor_relay.py` — use relayed topics |
| SSH hangs to NX1/NX2 | Stale connections or dog_go_zoom router issue | Wait/retry; power-cycle `dog_go_zoom` router if needed |
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
