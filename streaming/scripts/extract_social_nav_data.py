#!/usr/bin/env python3
"""
Extract front-camera frames and optional front-distance metadata from a rosbag.

This is an offline dataset-prep helper for social navigation evaluation.
It keeps the output simple:
  - images/                extracted PPM frames
  - frames.jsonl           one record per saved frame
  - extraction_summary.json
"""

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path


def import_rosbag():
    """Import the ROS rosbag module without being shadowed by the repo's rosbag/ data dir."""
    repo_root = Path(__file__).resolve().parents[2]
    original_sys_path = list(sys.path)
    filtered_sys_path = []
    for entry in original_sys_path:
        resolved = Path(entry or os.getcwd()).resolve()
        # The benchmark assets live in <repo>/rosbag/, which would otherwise shadow
        # the ROS Python package named "rosbag". Temporarily remove repo-local import
        # roots while importing the real ROS module.
        if resolved == repo_root:
            continue
        filtered_sys_path.append(entry)

    try:
        sys.path[:] = filtered_sys_path
        rosbag = importlib.import_module("rosbag")  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on ROS environment
        raise SystemExit(
            "rosbag is not available. Run this script in a ROS Python environment "
            "(for example after sourcing ROS Noetic setup.bash)."
        ) from exc
    finally:
        sys.path[:] = original_sys_path

    if not hasattr(rosbag, "Bag"):  # pragma: no cover - defensive guard
        raise SystemExit(
            "Imported a module named 'rosbag' but it does not provide rosbag.Bag. "
            "This usually means a local path is shadowing the ROS rosbag package, or "
            "the ROS Python environment has not been sourced correctly."
        )
    return rosbag


def get_sector(ranges, angle_min, angle_inc, n, deg_start, deg_end):
    def deg_to_idx(deg):
        angle = math.radians(deg)
        raw = int((angle - angle_min) / angle_inc)
        return max(0, min(n - 1, raw))

    i0 = deg_to_idx(deg_start)
    i1 = deg_to_idx(deg_end)
    lo, hi = min(i0, i1), max(i0, i1)
    return ranges[lo:hi + 1]


def compute_front_distance(scan_msg, front_sector_half_angle_deg, percentile, max_range_fallback):
    sanitized = []
    for value in scan_msg.ranges:
        if value is None or not math.isfinite(value) or value <= 0.05:
            sanitized.append(max_range_fallback)
        else:
            sanitized.append(float(value))

    if not sanitized:
        return None

    front = get_sector(
        sanitized,
        scan_msg.angle_min,
        scan_msg.angle_increment,
        len(sanitized),
        -front_sector_half_angle_deg,
        front_sector_half_angle_deg,
    )
    front_valid = [value for value in front if math.isfinite(value) and value > 0.05]
    if not front_valid:
        return None

    front_valid.sort()
    clamped_percentile = max(0.0, min(100.0, percentile))
    rank = (len(front_valid) - 1) * (clamped_percentile / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(front_valid[low])
    frac = rank - low
    return float(front_valid[low] * (1.0 - frac) + front_valid[high] * frac)


def image_bytes_from_ros_msg(msg):
    encoding = (getattr(msg, "encoding", "") or "").lower()
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    data = bytes(msg.data)
    row_bytes = []

    if encoding == "rgb8":
        channels = 3
        for row_idx in range(height):
            start = row_idx * step
            row_bytes.append(data[start:start + width * channels])
        return b"".join(row_bytes)

    if encoding == "bgr8":
        channels = 3
        for row_idx in range(height):
            start = row_idx * step
            row = data[start:start + width * channels]
            rgb = bytearray()
            for idx in range(0, len(row), 3):
                blue, green, red = row[idx:idx + 3]
                rgb.extend((red, green, blue))
            row_bytes.append(bytes(rgb))
        return b"".join(row_bytes)

    if encoding == "rgba8":
        channels = 4
        for row_idx in range(height):
            start = row_idx * step
            row = data[start:start + width * channels]
            rgb = bytearray()
            for idx in range(0, len(row), 4):
                red, green, blue, _alpha = row[idx:idx + 4]
                rgb.extend((red, green, blue))
            row_bytes.append(bytes(rgb))
        return b"".join(row_bytes)

    if encoding == "bgra8":
        channels = 4
        for row_idx in range(height):
            start = row_idx * step
            row = data[start:start + width * channels]
            rgb = bytearray()
            for idx in range(0, len(row), 4):
                blue, green, red, _alpha = row[idx:idx + 4]
                rgb.extend((red, green, blue))
            row_bytes.append(bytes(rgb))
        return b"".join(row_bytes)

    if encoding == "mono8":
        for row_idx in range(height):
            start = row_idx * step
            row = data[start:start + width]
            rgb = bytearray()
            for value in row:
                rgb.extend((value, value, value))
            row_bytes.append(bytes(rgb))
        return b"".join(row_bytes)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def write_ppm_image(msg, output_path):
    width = int(msg.width)
    height = int(msg.height)
    rgb_data = image_bytes_from_ros_msg(msg)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with open(output_path, "wb") as handle:
        handle.write(header)
        handle.write(rgb_data)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def extract_dataset(args):
    rosbag = import_rosbag()

    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    latest_scan = None
    saved_frames = 0
    seen_images = 0
    seen_scans = 0

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, msg, bag_time in bag.read_messages(topics=[args.image_topic, args.scan_topic]):
            timestamp = float(bag_time.to_sec())
            if topic == args.scan_topic:
                seen_scans += 1
                front_dist = compute_front_distance(
                    msg,
                    front_sector_half_angle_deg=args.front_sector_half_angle_deg,
                    percentile=args.front_distance_percentile,
                    max_range_fallback=args.max_range_fallback,
                )
                latest_scan = {
                    "timestamp": timestamp,
                    "front_dist": front_dist,
                }
                continue

            if topic != args.image_topic:
                continue

            seen_images += 1
            if args.image_every_n > 1 and ((seen_images - 1) % args.image_every_n != 0):
                continue
            if args.max_frames and saved_frames >= args.max_frames:
                break

            try:
                frame_name = f"frame_{saved_frames:06d}.ppm"
                image_path = images_dir / frame_name
                write_ppm_image(msg, image_path)
            except Exception as exc:
                print(f"Skipping image at {timestamp:.3f}: {exc}")
                continue

            scan_age = None
            front_dist = None
            scan_timestamp = None
            if latest_scan is not None:
                scan_age = timestamp - latest_scan["timestamp"]
                scan_timestamp = latest_scan["timestamp"]
                if scan_age <= args.max_scan_age_sec:
                    front_dist = latest_scan["front_dist"]

            row = {
                "frame_id": saved_frames,
                "timestamp": timestamp,
                "image_path": os.path.join("images", frame_name),
                "image_topic": args.image_topic,
                "scan_topic": args.scan_topic,
                "front_dist": front_dist,
                "scan_timestamp": scan_timestamp,
                "scan_age_sec": scan_age,
            }
            frame_rows.append(row)
            saved_frames += 1

    frames_path = output_dir / "frames.jsonl"
    write_jsonl(frames_path, frame_rows)

    summary = {
        "bag": str(Path(args.bag).resolve()),
        "output_dir": str(output_dir),
        "image_topic": args.image_topic,
        "scan_topic": args.scan_topic,
        "saved_frames": saved_frames,
        "seen_images": seen_images,
        "seen_scans": seen_scans,
        "frames_jsonl": str(frames_path),
    }
    with open(output_dir / "extraction_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="Input rosbag path")
    parser.add_argument("output_dir", help="Output directory for extracted data")
    parser.add_argument("--image-topic", default="/camera_face/left/image_raw")
    parser.add_argument("--scan-topic", default="/scan_odom")
    parser.add_argument("--image-every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-scan-age-sec", type=float, default=0.5)
    parser.add_argument("--front-sector-half-angle-deg", type=float, default=10.0)
    parser.add_argument("--front-distance-percentile", type=float, default=20.0)
    parser.add_argument("--max-range-fallback", type=float, default=10.0)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    extract_dataset(args)


if __name__ == "__main__":
    main()
