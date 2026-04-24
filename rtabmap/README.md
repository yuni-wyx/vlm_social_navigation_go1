# RTAB-Map 3D Mapping for Go1

Run RTAB-Map on recorded Go1 bag data to build a 3D map using LiDAR point clouds and leg odometry.

## Quick Start

```bash
# Run mapping on the latest fixed bag
bash rtabmap/run.sh streaming/bags/go1_session_20260313_124847_fixed.bag
```

This launches roscore, RTAB-Map (with visualization), and plays the bag. When playback finishes, press Enter to close.

## What It Does

- Uses 3D LiDAR point cloud (`/velodyne_points_odom`) for ICP scan matching
- Uses leg odometry (`/odom_fixed`) as motion estimate
- Builds a 3D map stored in `output/rtabmap.db`
- No cameras used (uncalibrated fisheye — see below)

## Output

```
rtabmap/output/
└── rtabmap.db    # RTAB-Map database with 3D map + trajectory
```

Inspect the database with:
```bash
rtabmap-databaseViewer rtabmap/output/rtabmap.db
```

## Future: Camera Integration

The Go1's fisheye cameras can be calibrated using Unitree's Camera SDK:
```bash
# On the robot (NX2 or Nano):
cd ~/Unitree/autostart/imageai/UnitreecameraSDK
./bins/example_getCalibParamsFile
# Outputs calibration YAML with intrinsics + distortion coefficients
```

With calibration data, we could:
1. Publish `camera_info` alongside images
2. Undistort fisheye images using OpenCV
3. Enable visual loop closure in RTAB-Map for better mapping

## Files

```
rtabmap/
├── go1_rtabmap.launch   # RTAB-Map launch (LiDAR + odom, scan-only mode)
├── run.sh               # One-command mapping pipeline
├── README.md            # This file
└── output/              # Generated databases (gitignored)
```
