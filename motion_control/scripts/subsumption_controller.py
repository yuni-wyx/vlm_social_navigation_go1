#!/usr/bin/env python3
"""
Live Subsumption Controller for Unitree Go1
============================================
Reads /scan_odom (LaserScan) from the robot's rosmaster and sends velocity
commands via UDP to the SDK bridge running on NX1.

Architecture:
    Laptop (this script)  ──UDP──>  NX1 (sdk_udp_bridge.py)  ──SDK──>  MCU

Behaviors (priority order):
    P1: Safety Stop   — Hard stop if front (±10°) is < 1.5m. If stuck >8 frames,
                         rotates in place to scan for open direction.
    P2: Wall Avoid    — Avoids walls closer than 0.7m (AVOID threshold).
                         Slows to 70% speed while correcting.
    P3: Center        — Centers in hallway by equalizing R and L distances.
                         Gentle proportional gain (0.10) to avoid oscillation.
    P4: Forward       — Default behavior: drive forward at 0.2 m/s.

Key design choices:
    - Front sector narrowed to ±10° to avoid wall diagonals triggering false P1.
    - LaserScan is median-filtered (kernel=5) to suppress transient noise.
    - P2 centering only engages when both walls are < 3.0m (in a hallway).
    - SDK bridge has 500ms auto-stop timeout for safety if UDP is lost.
    - All corrections are gentle (max vz=0.12) to avoid over-rotation.

Usage:
    # On NX1 first:
    python3.7 sdk_udp_bridge.py

    # On laptop:
    export ROS_MASTER_URI=http://192.168.123.161:11311
    export ROS_IP=192.168.123.67
    python3 subsumption_controller.py
"""

import rospy
import numpy as np
import math
import struct
import socket
import sys
import signal
from sensor_msgs.msg import LaserScan
from scipy.ndimage import median_filter

# ─── Configuration ───────────────────────────────────────────────────────────

# NX1 SDK bridge address
NX1_IP = "192.168.123.15"
NX1_PORT = 9900

# Safety
SAFETY_DIST = 1.5       # P1: hard stop if front < this (meters)
SAFETY_DEBOUNCE = 3     # P1: frames before triggering stop
STUCK_SCAN_THRESH = 8   # P1: frames stuck before rotating to find open dir

# Wall avoidance
AVOID_DIST = 0.7        # P2: steer away if wall < this (meters)
AVOID_VZ = 0.12         # P2: angular velocity for avoidance
AVOID_SPEED_MULT = 0.7  # P2: slow forward during avoidance

# Hallway centering
CENTER_GAIN = 0.10      # P3: proportional gain for centering
CENTER_VZ_MAX = 0.10    # P3: max angular velocity for centering
CENTER_DEADBAND = 0.1   # P3: ignore imbalance smaller than this
CENTER_MAX_WALL = 3.0   # P3: only center if both walls within this

# Forward speed
FORWARD_VX = 0.2        # P4: default forward velocity (m/s)

# Scan sectors (degrees from front)
FRONT_HALF = 10         # Front sector: ±10° (narrow to avoid wall diagonals)
SIDE_START = 75         # Side sectors: 75°-105° for left/right walls
SIDE_END = 105

# Logging
LOG_FILE = "/tmp/ctrl.log"
LOG_EVERY = 5           # Log every N frames

# ─── UDP Communication ──────────────────────────────────────────────────────

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
NX_ADDR = (NX1_IP, NX1_PORT)


def send_velocity(vx, vy, wz):
    """Send velocity command to NX1 SDK bridge."""
    sock.sendto(struct.pack("fff", vx, vy, wz), NX_ADDR)


def send_stop():
    """Send multiple stop commands for reliability."""
    for _ in range(5):
        sock.sendto(b"STOP", NX_ADDR)


# ─── Logging ─────────────────────────────────────────────────────────────────

def log(s):
    with open(LOG_FILE, "a") as f:
        f.write(s + "\n")


# ─── Signal Handling ─────────────────────────────────────────────────────────

def shutdown(*args):
    log("STOP")
    send_stop()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)


# ─── Scan Processing ────────────────────────────────────────────────────────

def get_sector(ranges, angle_min, angle_inc, n, deg_start, deg_end):
    """Get range values for a sector defined by degree bounds."""
    def deg_to_idx(d):
        return max(0, min(n - 1, int((math.radians(d) - angle_min) / angle_inc)))
    i0, i1 = deg_to_idx(deg_start), deg_to_idx(deg_end)
    return ranges[min(i0, i1):max(i0, i1) + 1]


# ─── Main Control Loop ──────────────────────────────────────────────────────

def main():
    rospy.init_node("subsumption_controller", anonymous=True)
    rospy.on_shutdown(send_stop)

    open(LOG_FILE, "w").close()
    log(f"Subsumption Controller Started")
    log(f"  Safety={SAFETY_DIST}m, Avoid={AVOID_DIST}m, CenterGain={CENTER_GAIN}")
    log(f"  NX1={NX1_IP}:{NX1_PORT}")

    threat_count = 0
    stuck_count = 0

    for i in range(5000):  # ~8 minutes at 10Hz
        try:
            msg = rospy.wait_for_message("/scan_odom", LaserScan, timeout=2.0)
        except Exception:
            log(f"[{i}] NO SCAN")
            send_velocity(0, 0, 0)
            continue

        # Sanitize and filter
        r = np.array(msg.ranges, dtype=np.float32)
        r = np.where(np.isfinite(r), r, 10.0)
        r = median_filter(r, size=5).astype(np.float32)
        n = len(r)

        # Extract sectors
        front = get_sector(r, msg.angle_min, msg.angle_increment, n,
                           -FRONT_HALF, FRONT_HALF)
        right = get_sector(r, msg.angle_min, msg.angle_increment, n,
                           -SIDE_END, -SIDE_START)
        left = get_sector(r, msg.angle_min, msg.angle_increment, n,
                          SIDE_START, SIDE_END)

        fm = float(np.min(front))
        rm = float(np.median(right))
        lm = float(np.median(left))

        vx, vz = FORWARD_VX, 0.0
        tag = "FWD"

        # ─── P1: Safety Stop ─────────────────────────────────────────────
        if fm < SAFETY_DIST:
            threat_count += 1
        else:
            threat_count = 0

        if threat_count >= SAFETY_DEBOUNCE:
            stuck_count += 1
            if stuck_count > STUCK_SCAN_THRESH:
                # Rotate in place to find open direction
                vx = 0.0
                vz = 0.15
                tag = f"SCAN F={fm:.2f}"
            else:
                vx = 0.0
                vz = 0.0
                tag = f"P1 F={fm:.2f}"
        else:
            stuck_count = 0

            # ─── P2: Wall Avoidance ──────────────────────────────────────
            if rm < AVOID_DIST:
                vz = AVOID_VZ
                vx = FORWARD_VX * AVOID_SPEED_MULT
                tag = f"AVOID_R {rm:.2f}"
            elif lm < AVOID_DIST:
                vz = -AVOID_VZ
                vx = FORWARD_VX * AVOID_SPEED_MULT
                tag = f"AVOID_L {lm:.2f}"

            # ─── P3: Hallway Centering ───────────────────────────────────
            elif rm < CENTER_MAX_WALL and lm < CENTER_MAX_WALL:
                imbalance = lm - rm  # positive = more space on left
                if abs(imbalance) > CENTER_DEADBAND:
                    vz = max(-CENTER_VZ_MAX,
                             min(CENTER_VZ_MAX, imbalance * CENTER_GAIN))
                    tag = f"CTR R={rm:.1f} L={lm:.1f}"
                else:
                    tag = f"FWD (centered)"

        # Send command
        send_velocity(vx, 0, vz)

        # Log
        if i % LOG_EVERY == 0:
            log(f"[{i:4d}] F={fm:5.2f} R={rm:5.2f} L={lm:5.2f} "
                f"vx={vx:+.2f} vz={vz:+.2f} {tag}")

    log("DONE")
    send_stop()


if __name__ == "__main__":
    main()
