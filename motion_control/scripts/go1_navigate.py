#!/usr/bin/env python3.7
"""
Simple Go1 Navigation via Unitree SDK (no ROS required)
=========================================================
Runs on NX1. Reads commands from /tmp/nav_cmd.txt and executes
them directly via the Unitree SDK. Useful for open-loop control
(rotation, forward bursts) without needing the full subsumption
controller.

Commands:
    ROTATE_180  — Rotate ~180° in place (0.3 rad/s for π/0.3 seconds)
    FORWARD     — Creep forward at 0.15 m/s for 3 seconds
    FORWARD_N   — Creep forward for N seconds (e.g. FORWARD_5)
    STOP        — Immediate stop

Prerequisites on NX1:
    1. Route to MCU: sudo ip route add 192.168.12.0/24 dev eth0
    2. Unitree SDK at: /home/unitree/Development/joint-control/unitree_legged_sdk/lib/python/arm64

Usage:
    # On NX1:
    python3.7 go1_navigate.py &
    echo ROTATE_180 > /tmp/nav_cmd.txt
    echo FORWARD > /tmp/nav_cmd.txt
"""
import sys
import time
import math
import os
import signal

sys.path.append('/home/unitree/Development/joint-control/unitree_legged_sdk/lib/python/arm64')
import robot_interface as sdk

HIGHLEVEL = 0xee
udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.12.1", 8082)
cmd = sdk.HighCmd()
state = sdk.HighState()
udp.InitCmdData(cmd)


def send(vx, vy, wz):
    cmd.mode = 2
    cmd.gaitType = 1
    cmd.velocity = [vx, vy]
    cmd.yawSpeed = wz
    udp.SetSend(cmd)
    udp.Send()


def stop():
    for _ in range(10):
        cmd.mode = 0
        cmd.gaitType = 0
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.05)


def shutdown(*args):
    print("STOPPING!")
    stop()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

CMD_FILE = "/tmp/nav_cmd.txt"
print("Go1 Navigator ready. Watching " + CMD_FILE)
print("Write commands: ROTATE_180 | FORWARD | STOP")

while True:
    try:
        if os.path.exists(CMD_FILE):
            with open(CMD_FILE) as f:
                action = f.read().strip()
            os.remove(CMD_FILE)

            if action == "STOP":
                print("STOPPING")
                stop()

            elif action == "ROTATE_180":
                print("ROTATING 180 degrees...")
                duration = math.pi / 0.3
                t0 = time.time()
                while time.time() - t0 < duration:
                    send(0, 0, 0.3)
                    time.sleep(0.05)
                stop()
                print("ROTATION COMPLETE")

            elif action == "FORWARD":
                print("CREEPING FORWARD (0.15 m/s, 3s bursts)")
                for i in range(60):  # 3 seconds max
                    if os.path.exists(CMD_FILE):
                        break
                    send(0.15, 0, 0)
                    time.sleep(0.05)
                stop()
                print("FORWARD DONE (3s burst)")

            elif action.startswith("FORWARD_"):
                secs = float(action.split("_")[1])
                print(f"FORWARD {secs}s at 0.15 m/s")
                steps = int(secs / 0.05)
                for i in range(steps):
                    if os.path.exists(CMD_FILE):
                        with open(CMD_FILE) as f:
                            if f.read().strip() == "STOP":
                                break
                    send(0.15, 0, 0)
                    time.sleep(0.05)
                stop()
                print("FORWARD DONE")
            else:
                print(f"Unknown: {action}")

        time.sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")
        stop()
