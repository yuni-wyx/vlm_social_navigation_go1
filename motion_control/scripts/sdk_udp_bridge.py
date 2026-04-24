#!/usr/bin/env python3.7
"""
UDP Receiver → Unitree SDK Bridge
===================================
Runs on NX1 (192.168.123.15). Receives (vx, vy, wz) velocity commands
as 3 packed floats via UDP and forwards them to the Go1 MCU via the
Unitree SDK.

Network topology:
    Laptop (192.168.123.67)  ──UDP:9900──>  NX1 (192.168.123.15)  ──SDK:8080──>  MCU (192.168.12.1)

Prerequisites on NX1:
    1. Route to MCU: sudo ip route add 192.168.12.0/24 dev eth0
    2. Unitree SDK at: /home/unitree/Development/joint-control/unitree_legged_sdk/lib/python/arm64'

Safety:
    - 500ms timeout auto-stop if no commands received
    - SIGINT/SIGTERM handlers send stop before exit
    - Stop command = mode 0 (idle), sent 5x for reliability

Protocol:
    - 12-byte packets: struct.pack('fff', vx, vy, wz)
    - 4-byte b'STOP' packet: immediate stop

Usage:
    # On NX1:
    sudo ip route add 192.168.12.0/24 dev eth0
    python3.7 sdk_udp_bridge.py
"""
import sys
import struct
import socket
import signal
import time

sys.path.append('/home/unitree/Development/joint-control/unitree_legged_sdk/lib/python/arm64')
import robot_interface as sdk

HIGHLEVEL = 0xee
udp_sdk = sdk.UDP(HIGHLEVEL, 8080, "192.168.12.1", 8082)
cmd = sdk.HighCmd()
state = sdk.HighState()
udp_sdk.InitCmdData(cmd)

# UDP receiver
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 9900))
sock.settimeout(0.5)  # 500ms timeout → auto-stop if no commands

print("SDK Bridge ready on UDP :9900")
print("Expecting 12-byte packets: (vx, vy, wz) as 3 floats")


def stop():
    for _ in range(5):
        cmd.mode = 0
        cmd.gaitType = 0
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0
        udp_sdk.SetSend(cmd)
        udp_sdk.Send()
        time.sleep(0.02)


def shutdown(*args):
    print("STOPPING")
    stop()
    sock.close()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

last_cmd_time = time.time()
active = False

while True:
    try:
        data, addr = sock.recvfrom(64)
        if len(data) == 12:
            vx, vy, wz = struct.unpack('fff', data)
            cmd.mode = 2        # Walk mode
            cmd.gaitType = 1    # Trot gait
            cmd.velocity = [vx, vy]
            cmd.yawSpeed = wz
            udp_sdk.SetSend(cmd)
            udp_sdk.Send()
            last_cmd_time = time.time()
            if not active:
                print(f"Moving: vx={vx:.2f} wz={wz:.2f}")
                active = True
        elif data == b'STOP':
            stop()
            active = False
            print("STOPPED")
    except socket.timeout:
        # No command in 500ms → safety stop
        if active:
            stop()
            active = False
            print("TIMEOUT - stopped")
    except Exception as e:
        print(f"Error: {e}")
        stop()
