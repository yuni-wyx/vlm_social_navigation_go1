# Go1 Camera Undistortion

Fisheye → rectilinear (pinhole) rectification for the Unitree Go1's cameras. Produces standard images suitable for deep learning inference.

## Quick Start

```bash
# Batch export rectified images from a bag
python3 camera_calib/undistort.py --bag streaming/bags/go1_session_*_fixed.bag --output rectified/

# Custom FOV (default 120°)
python3 camera_calib/undistort.py --bag file.bag --fov 100 --output rectified/

# ROS node (during bag playback or live — publishes /camera_*/left/image_rect)
python3 camera_calib/undistort.py --ros
```

## Output

```
rectified/
├── camera_face/          # Undistorted front camera frames
├── camera_face_raw/      # Original fisheye frames (for comparison)
├── camera_left/           
├── camera_left_raw/       
├── camera_right/          
├── camera_right_raw/      
└── rectified_camera_info.yaml  # Pinhole intrinsics for rectified images
```

## Calibration

The Go1 uses OV9281 sensors with 222° FOV fisheye lenses (Kannala-Brandt model). Current parameters in `go1_fisheye.yaml` are **estimated** from specs.

For unit-specific calibration, SSH into the robot:
```bash
# Front/Chin cameras — Head Nano
ssh unitree@192.168.123.13
cd ~/Unitree/autostart/imageai/UnitreecameraSDK
./bin/example_getCalibParamsFile
# → outputs output_camCalibParams.yaml

# Side cameras — Body Nano
ssh unitree@192.168.123.14
# (same procedure)
```

## Files

```
camera_calib/
├── go1_fisheye.yaml    # Estimated Kannala-Brandt calibration
├── undistort.py        # Undistortion tool (ROS node + batch export)
└── README.md
```
