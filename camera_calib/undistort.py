#!/usr/bin/env python3
"""
Fisheye undistortion for Unitree Go1 cameras.

Reads raw fisheye images from a bag file, applies Kannala-Brandt
undistortion, and either:
  1. Publishes rectified images as ROS topics (live mode)
  2. Exports rectified images to disk (batch mode)

Usage:
  # ROS node mode (during bag playback):
  python3 undistort.py --ros

  # Batch export from bag:
  python3 undistort.py --bag path/to/fixed.bag --output ./rectified/

  # Custom config:
  python3 undistort.py --bag file.bag --config go1_fisheye.yaml --fov 100
"""
import argparse
import os
import sys
import yaml
import cv2
import numpy as np


def load_config(config_path):
    """Load camera calibration from YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    K = np.array(cfg['camera_matrix']['data']).reshape(3, 3)
    D = np.array(cfg['distortion_coefficients']['data']).reshape(4, 1)
    w = cfg['image_width']
    h = cfg['image_height']
    fov = cfg.get('output_fov_degrees', 120)
    out_w = cfg.get('output_width', 640)
    out_h = cfg.get('output_height', 480)

    return K, D, (w, h), fov, (out_w, out_h)


def compute_undistort_maps(K, D, input_size, output_size, fov_degrees=120):
    """
    Compute undistortion+rectification maps for fisheye → pinhole.

    The output is a standard rectilinear (pinhole) image with
    the specified FOV — suitable for deep learning models.
    """
    w_in, h_in = input_size
    w_out, h_out = output_size

    # Compute new camera matrix for the desired output FOV
    fov_rad = np.deg2rad(fov_degrees)
    f_new = w_out / (2.0 * np.tan(fov_rad / 2.0))

    K_new = np.array([
        [f_new, 0, w_out / 2.0],
        [0, f_new, h_out / 2.0],
        [0, 0, 1]
    ])

    # Compute undistortion maps using OpenCV fisheye model
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w_out, h_out), cv2.CV_16SC2
    )

    return map1, map2, K_new


def undistort_image(img, map1, map2):
    """Apply precomputed undistortion maps to an image."""
    return cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def batch_export(bag_path, config_path, output_dir, fov_override=None,
                 camera_topics=None):
    """Extract and undistort all camera frames from a bag file."""
    import rosbag
    from cv_bridge import CvBridge

    K, D, input_size, fov, output_size = load_config(config_path)
    if fov_override:
        fov = fov_override

    map1, map2, K_new = compute_undistort_maps(K, D, input_size,
                                                output_size, fov)
    bridge = CvBridge()

    if camera_topics is None:
        camera_topics = [
            '/camera_face/left/image_raw',
            '/camera_left/left/image_raw',
            '/camera_right/left/image_raw',
        ]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Config: K={K.diagonal()}, D={D.ravel()}")
    print(f"Output: {output_size[0]}x{output_size[1]} @ {fov}° FOV")
    print(f"Reading bag: {bag_path}")

    counts = {}
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=camera_topics):
            # Parse camera name from topic
            cam_name = topic.split('/')[1]  # e.g. camera_face
            cam_dir = os.path.join(output_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)

            # Also save raw for comparison
            raw_dir = os.path.join(output_dir, cam_name + '_raw')
            os.makedirs(raw_dir, exist_ok=True)

            # Convert ROS image to OpenCV
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                print(f"  Skip {topic}: {e}")
                continue

            # Undistort
            rectified = undistort_image(cv_img, map1, map2)

            # Save
            idx = counts.get(cam_name, 0)
            ts = f"{t.secs}_{t.nsecs:09d}"
            cv2.imwrite(os.path.join(raw_dir, f"{idx:05d}_{ts}.jpg"), cv_img)
            cv2.imwrite(os.path.join(cam_dir, f"{idx:05d}_{ts}.jpg"), rectified)
            counts[cam_name] = idx + 1

            if sum(counts.values()) % 50 == 0:
                print(f"  Exported {counts}")

    print(f"\nDone! {counts}")
    print(f"Raw images:       {output_dir}/<camera>_raw/")
    print(f"Rectified images: {output_dir}/<camera>/")

    # Save the new (rectified) camera matrix
    rect_info = {
        'image_width': output_size[0],
        'image_height': output_size[1],
        'fov_degrees': fov,
        'camera_matrix': {
            'rows': 3, 'cols': 3,
            'data': K_new.flatten().tolist()
        },
        'distortion_model': 'pinhole',
        'distortion_coefficients': {
            'rows': 1, 'cols': 5,
            'data': [0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
    rect_yaml = os.path.join(output_dir, 'rectified_camera_info.yaml')
    with open(rect_yaml, 'w') as f:
        yaml.dump(rect_info, f, default_flow_style=False)
    print(f"Rectified intrinsics: {rect_yaml}")


def ros_node(config_path, fov_override=None):
    """Run as a ROS node: subscribe to raw, publish undistorted."""
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge

    K, D, input_size, fov, output_size = load_config(config_path)
    if fov_override:
        fov = fov_override

    map1, map2, K_new = compute_undistort_maps(K, D, input_size,
                                                output_size, fov)
    bridge = CvBridge()

    rospy.init_node('fisheye_undistort', anonymous=True)

    # Publishers for each camera
    topics = {
        '/camera_face/left/image_raw': '/camera_face/left/image_rect',
        '/camera_left/left/image_raw': '/camera_left/left/image_rect',
        '/camera_right/left/image_raw': '/camera_right/left/image_rect',
    }
    pubs = {}
    info_pubs = {}
    for raw_topic, rect_topic in topics.items():
        pubs[raw_topic] = rospy.Publisher(rect_topic, Image, queue_size=1)
        info_topic = rect_topic.replace('image_rect', 'camera_info_rect')
        info_pubs[raw_topic] = rospy.Publisher(info_topic, CameraInfo,
                                               queue_size=1)

    def make_camera_info(header):
        info = CameraInfo()
        info.header = header
        info.width = output_size[0]
        info.height = output_size[1]
        info.distortion_model = "plumb_bob"
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.K = K_new.flatten().tolist()
        info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        info.P = [K_new[0, 0], 0, K_new[0, 2], 0,
                  0, K_new[1, 1], K_new[1, 2], 0,
                  0, 0, 1, 0]
        return info

    def callback(msg, raw_topic):
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        rectified = undistort_image(cv_img, map1, map2)
        rect_msg = bridge.cv2_to_imgmsg(rectified, encoding='bgr8')
        rect_msg.header = msg.header
        pubs[raw_topic].publish(rect_msg)
        info_pubs[raw_topic].publish(make_camera_info(msg.header))

    for raw_topic in topics:
        rospy.Subscriber(raw_topic, Image,
                         callback, callback_args=raw_topic, queue_size=1)

    rospy.loginfo(f"Fisheye undistort: {output_size[0]}x{output_size[1]} "
                  f"@ {fov}° FOV")
    rospy.spin()


def main():
    parser = argparse.ArgumentParser(
        description='Go1 fisheye camera undistortion')
    parser.add_argument('--config', default=None,
                        help='Calibration YAML (default: go1_fisheye.yaml)')
    parser.add_argument('--fov', type=float, default=None,
                        help='Override output FOV in degrees')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ros', action='store_true',
                      help='Run as ROS node')
    mode.add_argument('--bag', type=str,
                      help='Batch export from bag file')

    parser.add_argument('--output', default='./rectified',
                        help='Output directory for batch export')

    args = parser.parse_args()

    # Default config path relative to this script
    if args.config is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.config = os.path.join(script_dir, 'go1_fisheye.yaml')

    if args.ros:
        ros_node(args.config, args.fov)
    else:
        batch_export(args.bag, args.config, args.output, args.fov)


if __name__ == '__main__':
    main()
