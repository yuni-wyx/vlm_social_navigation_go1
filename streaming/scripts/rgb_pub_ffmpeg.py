#!/usr/bin/env python
"""RGB camera publisher using ffmpeg - NO OpenCV needed.
Uses ffmpeg subprocess for V4L2 capture + numpy for image handling.
Usage: python rgb_pub_ffmpeg.py <camera_name> <device_id>
"""
import sys
import subprocess
import numpy as np
import rospy
from sensor_msgs.msg import Image

def main():
    camera_name = sys.argv[1] if len(sys.argv) > 1 else "camera_left"
    device_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dev_path = "/dev/video{}".format(device_id)

    rospy.init_node('rgb_publisher_' + camera_name, anonymous=True)

    WIDTH = 1856
    HEIGHT = 800
    FPS = 5
    FRAME_SIZE = WIDTH * HEIGHT * 3  # BGR24

    pub_left = rospy.Publisher('/{}/left/image_raw'.format(camera_name), Image, queue_size=1)
    
    rospy.loginfo("Launching ffmpeg capture from {} as {}".format(dev_path, camera_name))

    # ffmpeg: capture from v4l2, output raw BGR frames to stdout
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'v4l2',
        '-framerate', str(FPS),
        '-video_size', '{}x{}'.format(WIDTH, HEIGHT),
        '-i', dev_path,
        '-pix_fmt', 'bgr24',
        '-f', 'rawvideo',
        '-an',
        '-'
    ]

    DEVNULL = open('/dev/null', 'w')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=DEVNULL, bufsize=FRAME_SIZE*2)
    seq = 0

    rate = rospy.Rate(FPS)

    while not rospy.is_shutdown():
        raw = proc.stdout.read(FRAME_SIZE)
        if len(raw) < FRAME_SIZE:
            rospy.logwarn("Short read, ffmpeg may have exited")
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        # Left image is right half of stereo pair (Unitree convention)
        left = frame[:, 928:1856, :]
        left_contiguous = np.ascontiguousarray(left)

        msg = Image()
        msg.height = left_contiguous.shape[0]
        msg.width = left_contiguous.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = left_contiguous.shape[1] * 3
        msg.data = left_contiguous.tobytes()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = camera_name
        msg.header.seq = seq

        pub_left.publish(msg)
        seq += 1

    proc.terminate()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
