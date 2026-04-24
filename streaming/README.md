# Go1 Sensor Streaming

Self-contained package for streaming and recording raw sensor data from the Unitree Go1 robot.

## Quick Start

```bash
# 1. Connect laptop to dog_go_zoom WiFi
nmcli connection up "dog_go_zoom_static"

# 2. Launch all sensors + relay
./launch.sh all

# 3. Visualize
./launch.sh rviz

# 4. Record a bag file
./launch.sh record myrun01

# 5. Stop everything
./launch.sh stop
```

## What You Get

| Relayed Topic | Type | Rate | Content |
|--------------|------|------|---------|
| `/velodyne_points_odom` | PointCloud2 | ~10 Hz | 3D LiDAR (frame=`trunk`) |
| `/scan_odom` | LaserScan | ~10 Hz | 2D laser scan (frame=`trunk`) |
| `/odom_fixed` | Odometry | ~200 Hz | Leg odometry (frame=`odom`→`trunk`) |
| `/camera_face/left/image_raw` | Image | ~3 Hz | Front fisheye camera |
| `/camera_left/left/image_raw` | Image | ~5 Hz | Left fisheye camera |
| `/camera_right/left/image_raw` | Image | ~5 Hz | Right fisheye camera |

> Use the `/..._odom` and `/odom_fixed` relayed topics (not the originals) — they have corrected frame_ids that work with rviz and downstream pipelines.

## Prerequisites

- **Hardware**: USB WiFi dongle, `dog_go_zoom` router plugged into Go1
- **Software**: ROS Noetic, `sshpass`, `rosbag`
- **One-time setup**:
  ```bash
  # Create static IP profile for dog_go_zoom
  nmcli connection add type wifi con-name "dog_go_zoom_static" \
    ifname wlx8c3bad120bdc ssid "dog_go_zoom" \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "eces111A" \
    ipv4.method manual ipv4.addresses "192.168.123.67/24" \
    ipv4.never-default yes

  # Fix hostname resolution
  echo "192.168.123.161 raspberrypi" | sudo tee -a /etc/hosts
  ```

## Recording Bags

```bash
./launch.sh record                # auto-named: bags/2026-03-11_151000.bag
./launch.sh record hallway_walk   # custom: bags/hallway_walk.bag
```

Bag files are saved to `bags/` and contain all relayed topics plus TF data.

## Replaying Bags

The Go1's onboard computers have independent clocks that drift apart. Raw bag files will show "message too old" errors in rviz. **Always fix timestamps before replaying:**

```bash
# 1. Fix timestamps (normalizes all clocks to laptop time)
python3 scripts/fix_bag_timestamps.py bags/myrun.bag bags/myrun_fixed.bag

# 2. Replay with local roscore + rviz (no robot needed)
bash scripts/replay.sh bags/myrun_fixed.bag
```

The replay script handles roscore, `use_sim_time`, rviz, and bag playback automatically. Ctrl+C to stop.

## Offline Social Navigation Extraction

For offline social-navigation evaluation you can also extract front-camera
frames and optional front-distance metadata from a fixed bag:

```bash
python3 scripts/extract_social_nav_data.py \
  bags/myrun_fixed.bag \
  /tmp/social_nav_eval
```

This writes:
- `images/` extracted PPM images
- `frames.jsonl` frame/timestamp/front-distance metadata
- `extraction_summary.json`

## Architecture

See [full documentation](../docs/streaming.md) for:
- Go1 internal computer layout (Pi, NX1, NX2, Nano)
- Why custom Python scripts are needed (no default RGB, broken TF tree)
- Clock synchronization (no RTC — must fix after every power cycle)
- Troubleshooting table

## Files

```
streaming/
├── launch.sh                # Master launch script
├── go1_rviz.rviz            # RViz config (odom frame, all displays)
├── bags/                    # Recorded bag files (gitignored)
└── scripts/
    ├── sensor_relay.py      # Bridges disconnected TF trees on laptop
    ├── rgb_publisher.py     # OpenCV camera publisher (runs on Head Nano)
    ├── rgb_pub_ffmpeg.py    # ffmpeg camera publisher (runs on NX2)
    ├── record.sh            # rosbag record wrapper
    ├── fix_bag_timestamps.py  # Normalizes multi-clock bag timestamps
    ├── extract_social_nav_data.py  # Offline frame + distance extraction
    └── replay.sh            # Standalone bag replay (roscore + rviz + play)
```

## Gotchas

- **`/tmp` cleared on reboot**: Camera scripts are SCP'd to `/tmp/` on the Go1 boards. After every power cycle, the launch script must re-deploy them.
- **NX2 ffmpeg stderr blocking**: The `rgb_pub_ffmpeg.py` script redirects ffmpeg stderr to `/dev/null` to prevent pipe buffer deadlocks.
- **Clock skew**: Always run `fix_bag_timestamps.py` before replaying bags. The Go1's computers have no RTC and clocks drift even after manual sync.
- **Low battery → network drops**: NX2 (.14) drops off the network when Go1 battery is low. Side cameras will silently stop publishing.
