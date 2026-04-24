#!/usr/bin/env python3
"""
Fix timestamp skew in Go1 recorded bags.

The Go1 has multiple onboard computers whose clocks drift apart.
This causes rviz to reject sensor data during bag replay with
'message removed because it is too old' errors.

This script reads a bag and rewrites ALL header.stamp fields
to use the bag's receive time (rosbag record time), which comes
from a single clock (the recording laptop). This makes TF lookups
work during replay.

Usage:
    python3 fix_bag_timestamps.py input.bag output.bag
"""

import sys
import rosbag


def fix_bag(inpath, outpath):
    print(f"Reading {inpath} ...")
    with rosbag.Bag(inpath, 'r') as inbag:
        info = inbag.get_type_and_topic_info()
        total = inbag.get_message_count()
        print(f"  {total} messages across {len(info.topics)} topics")

        with rosbag.Bag(outpath, 'w') as outbag:
            for i, (topic, msg, t) in enumerate(inbag.read_messages()):
                # t = bag receive time (single laptop clock)
                # msg.header.stamp = sensor clock (may be skewed)
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    msg.header.stamp = t
                # TF messages have an array of transforms
                if hasattr(msg, 'transforms'):
                    for tf in msg.transforms:
                        tf.header.stamp = t
                outbag.write(topic, msg, t)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{total} ...")

    print(f"Done! Written to {outpath}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.bag output.bag")
        sys.exit(1)
    fix_bag(sys.argv[1], sys.argv[2])
