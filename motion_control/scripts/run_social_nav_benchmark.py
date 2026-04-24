#!/usr/bin/env python3
"""
Reproducible benchmark runner for the 13 rosbag social navigation scenarios.

This is a thin convenience wrapper over social_nav_eval.py's manifest mode.
It keeps the existing single-bag workflow untouched while standardizing
batch outputs under:

    benchmark_runs/<run_name>/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import social_nav_eval  # noqa: E402


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="",
        help="Benchmark run directory name under benchmark_runs/. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "motion_control" / "eval" / "scenario_manifest.json"),
        help="Scenario manifest JSON path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "benchmark_runs"),
        help="Root directory under which run_name directories are created.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["rosbag", "extracted"],
        default="rosbag",
        help=(
            "Use rosbag files directly or consume pre-extracted datasets under "
            "extracted_social_nav/<bag_id>/ style directories."
        ),
    )
    parser.add_argument(
        "--extracted-root",
        default="",
        help=(
            "Optional root directory containing extracted datasets under <root>/<bag_id>/. "
            "Used only in extracted mode when manifest entries do not define extracted_dir."
        ),
    )
    parser.add_argument("--image-topic", default="/camera_face/left/image_raw")
    parser.add_argument("--scan-topic", default="/scan_odom")
    parser.add_argument("--image-every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-scan-age-sec", type=float, default=0.5)
    parser.add_argument("--front-sector-half-angle-deg", type=float, default=10.0)
    parser.add_argument("--front-distance-percentile", type=float, default=20.0)
    parser.add_argument("--max-range-fallback", type=float, default=10.0)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--geometry-stop-threshold", type=float, default=1.0)
    parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--include-legacy-prompts",
        action="store_true",
        help="Include legacy/debug prompt methods in the benchmark run.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_name = args.run_name.strip() or datetime.now().strftime("social_nav_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_args = argparse.Namespace(
        manifest=args.manifest,
        output_dir=str(output_dir),
        input_mode=args.input_mode,
        extracted_root=args.extracted_root,
        image_topic=args.image_topic,
        scan_topic=args.scan_topic,
        image_every_n=args.image_every_n,
        max_frames=args.max_frames,
        max_scan_age_sec=args.max_scan_age_sec,
        front_sector_half_angle_deg=args.front_sector_half_angle_deg,
        front_distance_percentile=args.front_distance_percentile,
        max_range_fallback=args.max_range_fallback,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        geometry_stop_threshold=args.geometry_stop_threshold,
        request_timeout_sec=args.request_timeout_sec,
        include_legacy_prompts=args.include_legacy_prompts,
    )

    social_nav_eval.run_manifest(manifest_args)
    print("")
    print(f"Benchmark run complete: {output_dir}")
    print(f"Aggregate CSV:  {output_dir / 'aggregate_results.csv'}")
    print(f"Aggregate JSON: {output_dir / 'aggregate_results.json'}")
    print(f"Primary cases:  {output_dir / 'primary_cases_summary.csv'}")
    print(f"Review cases:   {output_dir / 'review_cases_summary.csv'}")


if __name__ == "__main__":
    main()
