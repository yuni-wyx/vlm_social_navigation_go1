#!/usr/bin/env python
"""Minimal RGB camera publisher for Go1.
Opens /dev/video0 (stereo camera), splits left/right, publishes as ROS Image.
Requires: source /opt/ros/melodic/setup.bash before running.
Usage: python rgb_publisher.py [camera_name] [device_id]
"""
import sys
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def main():
    camera_name = sys.argv[1] if len(sys.argv) > 1 else "camera_rearDown"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    rospy.init_node('rgb_publisher', anonymous=True)
    bridge = CvBridge()

    vid = cv2.VideoCapture(device_id)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1856)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    pub_left = rospy.Publisher('/{}/left/image_raw'.format(camera_name), Image, queue_size=1)
    pub_right = rospy.Publisher('/{}/right/image_raw'.format(camera_name), Image, queue_size=1)

    rate = rospy.Rate(10)  # 10 Hz to save bandwidth over WiFi
    seq = 0

    rospy.loginfo("Publishing RGB from /dev/video{} as {}".format(device_id, camera_name))

    while not rospy.is_shutdown():
        ret, frame = vid.read()
        if not ret:
            rospy.logwarn("Frame capture failed, retrying...")
            continue

        # Stereo frame: left is right half, right is left half (Unitree convention)
        left = frame[0:800, 928:1856]
        right = frame[0:800, 0:928]

        now = rospy.get_rostime()

        msg_l = bridge.cv2_to_imgmsg(left, "bgr8")
        msg_l.header.stamp = now
        msg_l.header.frame_id = camera_name
        msg_l.header.seq = seq

        msg_r = bridge.cv2_to_imgmsg(right, "bgr8")
        msg_r.header.stamp = now
        msg_r.header.frame_id = camera_name
        msg_r.header.seq = seq

        pub_left.publish(msg_l)
        pub_right.publish(msg_r)

        seq += 1
        rate.sleep()

    vid.release()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
