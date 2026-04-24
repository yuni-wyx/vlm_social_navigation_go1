#!/usr/bin/env python3
"""
Sensor Frame Relay for Unitree Go1

The Go1's internal ROS nodes publish TF and sensor data using two disconnected
frame trees:
  Tree 1 (Go1 built-in):  base -> odom -> trunk -> camera_*
  Tree 2 (lio_sam URDF):  uodom -> base_link -> chassis_link -> velodyne

LiDAR, /scan, and odometry use frames from Tree 2 (velodyne, base_link, uodom),
which are not connected to the odom frame in Tree 1. This relay republishes
those topics with corrected frame_ids so everything lives in Tree 1.

Relayed topics:
  /velodyne_points  ->  /velodyne_points_odom   (frame: trunk)
  /scan             ->  /scan_odom              (frame: trunk)
  /ros2udp/odom     ->  /odom_fixed             (frame: odom, child: trunk)

Usage:
  rosrun --prefix 'python3' my_pkg sensor_relay.py
  # or simply:
  python3 sensor_relay.py
"""

import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import Odometry

FRAME_MAP = {
    "uodom": "odom",
    "base_link": "trunk",
    "velodyne": "trunk",
    "rslidar": "trunk",
}


def fix_frame(frame):
    return FRAME_MAP.get(frame, frame)


def main():
    rospy.init_node("sensor_frame_relay", anonymous=True)

    pc_pub = rospy.Publisher("/velodyne_points_odom", PointCloud2, queue_size=1)
    scan_pub = rospy.Publisher("/scan_odom", LaserScan, queue_size=1)
    odom_pub = rospy.Publisher("/odom_fixed", Odometry, queue_size=1)

    def pc_cb(msg):
        msg.header.frame_id = fix_frame(msg.header.frame_id)
        pc_pub.publish(msg)

    def scan_cb(msg):
        if rospy.is_shutdown():
            return
        try:
            scan_pub.publish(msg)
        except rospy.ROSException:
            pass

    def odom_cb(msg):
        msg.header.frame_id = fix_frame(msg.header.frame_id)
        msg.child_frame_id = fix_frame(msg.child_frame_id)
        odom_pub.publish(msg)

    rospy.Subscriber("/velodyne_points", PointCloud2, pc_cb)
    rospy.Subscriber("/scan", LaserScan, scan_cb)
    rospy.Subscriber("/ros2udp/odom", Odometry, odom_cb)

    rospy.loginfo("Sensor relay running. Mapping frames: %s", FRAME_MAP)
    rospy.spin()


if __name__ == "__main__":
    main()
