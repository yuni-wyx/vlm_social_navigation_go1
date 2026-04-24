#!/usr/bin/env python3
"""
Prepare extracted frame datasets for the social navigation scenario manifest.

This script is meant to run once in a ROS environment. It converts each bag in
the manifest into a standardized extracted dataset layout:

    extracted_social_nav/<bag_id>/
      images/
      frames.jsonl
      extraction_summary.json

Later benchmark runs can use --input-mode extracted and avoid ROS entirely.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import social_nav_eval  # noqa: E402


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "motion_control" / "eval" / "scenario_manifest.json"),
        help="Scenario manifest JSON path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "extracted_social_nav"),
        help="Root directory for extracted datasets.",
    )
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

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scenarios = social_nav_eval.load_scenario_manifest(args.manifest)
    extract_module = social_nav_eval.load_extract_module()

    prepared = []
    for scenario in scenarios:
        bag_id = scenario["bag_id"]
        extracted_dir = output_root / bag_id
        extract_args = SimpleNamespace(
            bag=str(social_nav_eval.resolve_scenario_bag_path(scenario)),
            output_dir=str(extracted_dir),
            image_topic=args.image_topic,
            scan_topic=args.scan_topic,
            image_every_n=args.image_every_n,
            max_frames=args.max_frames,
            max_scan_age_sec=args.max_scan_age_sec,
            front_sector_half_angle_deg=args.front_sector_half_angle_deg,
            front_distance_percentile=args.front_distance_percentile,
            max_range_fallback=args.max_range_fallback,
        )
        extract_module.extract_dataset(extract_args)
        social_nav_eval.validate_extracted_dir(extracted_dir, bag_id)
        prepared.append(
            {
                "bag_id": bag_id,
                "scenario_name": scenario["scenario_name"],
                "bag_path": str(social_nav_eval.resolve_scenario_bag_path(scenario)),
                "extracted_dir": str(extracted_dir),
            }
        )

    summary_path = output_root / "extraction_manifest.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"output_root": str(output_root), "scenarios": prepared}, handle, indent=2, sort_keys=True)

    print(json.dumps({"output_root": str(output_root), "num_scenarios": len(prepared), "summary": str(summary_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
