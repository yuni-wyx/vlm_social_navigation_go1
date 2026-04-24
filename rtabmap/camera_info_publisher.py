#!/usr/bin/env python3
"""
Publish approximate camera_info for the Go1 front fisheye camera.

The Go1's cameras are 222° FOV fisheye with 1856x800 stereo frames
(928x800 per eye). Intrinsics are estimated from the known specs.

This node subscribes to the camera image topic and publishes a matching
camera_info message that RTAB-Map needs for visual loop closure with
gen_depth (LiDAR → depth projection).

Usage:
    rosrun ... camera_info_publisher.py
  or:
    python3 camera_info_publisher.py
"""
import rospy
from sensor_msgs.msg import Image, CameraInfo

# Estimated Go1 front camera intrinsics
# 222° FOV equidistant fisheye, 928x800 half-frame
# f = r_max / theta_max ≈ 612 / 1.937 ≈ 316
WIDTH = 928
HEIGHT = 800
FX = 316.0
FY = 316.0
CX = 464.0  # WIDTH / 2
CY = 400.0  # HEIGHT / 2


def make_camera_info(header):
    info = CameraInfo()
    info.header = header
    info.width = WIDTH
    info.height = HEIGHT
    info.distortion_model = "plumb_bob"
    # Zero distortion — we're feeding raw fisheye, RTAB-Map will
    # match features in the center region where distortion is low
    info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.K = [FX, 0.0, CX,
              0.0, FY, CY,
              0.0, 0.0, 1.0]
    info.R = [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0]
    info.P = [FX, 0.0, CX, 0.0,
              0.0, FY, CY, 0.0,
              0.0, 0.0, 1.0, 0.0]
    return info


class CameraInfoPublisher:
    def __init__(self):
        rospy.init_node('camera_info_publisher', anonymous=True)
        self.pub = rospy.Publisher(
            '/camera_face/left/camera_info', CameraInfo, queue_size=1)
        rospy.Subscriber(
            '/camera_face/left/image_raw', Image, self.image_cb, queue_size=1)
        rospy.loginfo("Publishing camera_info for /camera_face/left "
                      f"({WIDTH}x{HEIGHT}, f={FX})")

    def image_cb(self, msg):
        info = make_camera_info(msg.header)
        self.pub.publish(info)


if __name__ == '__main__':
    try:
        node = CameraInfoPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
