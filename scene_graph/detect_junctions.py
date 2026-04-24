#!/usr/bin/env python3
"""
Detect corridor junctions from a 2D occupancy grid map.

Extracts the free-space skeleton (medial axis) and classifies
branch points as junction types:
  - Dead-end (1 branch)
  - L-junction / corner (2 branches, ~90°)
  - T-junction (3 branches)
  - X-junction / 4-way (4+ branches)

Input: Occupancy grid as PGM/PNG (from RTAB-Map or map_server)
Output: Junction list with positions, types, and visualization

Usage:
  python3 detect_junctions.py --map rtabmap/output/map.pgm
  python3 detect_junctions.py --map map.pgm --resolution 0.05 --output junctions/
"""
import argparse
import json
import os
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def load_map(map_path, yaml_path=None):
    """Load occupancy grid map, return binary free space."""
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read map: {map_path}")

    resolution = 0.05  # default 5cm/pixel
    origin = [0.0, 0.0]

    # Try to load map YAML for resolution/origin
    if yaml_path and os.path.exists(yaml_path):
        import yaml
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        resolution = meta.get('resolution', resolution)
        origin = meta.get('origin', origin)[:2]

    # Binary: free space = white (>250), occupied = dark (<50)
    # ROS convention: 0=free, 100=occupied, -1=unknown
    # As image: 254=free, 0=occupied, 205=unknown
    free_mask = img > 230  # free space

    return free_mask, resolution, origin, img


def extract_skeleton(free_mask, min_area=500):
    """Extract medial axis skeleton from free space."""
    # Clean up small regions
    cleaned = ndimage.binary_fill_holes(free_mask)
    labeled, n = ndimage.label(cleaned)
    for i in range(1, n + 1):
        if np.sum(labeled == i) < min_area:
            cleaned[labeled == i] = False

    # Skeletonize
    skeleton = skeletonize(cleaned)
    return skeleton.astype(np.uint8)


def find_junctions(skeleton):
    """
    Find branch points on skeleton and classify junction type.

    A pixel is a junction if it has 3+ skeleton neighbors (8-connected).
    Type is determined by the number of branches radiating out.
    """
    # Count neighbors using convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)

    # Junction = skeleton pixel with 3+ neighbors
    junction_mask = (skeleton > 0) & (neighbor_count >= 3)

    # End points (dead ends) = skeleton pixel with exactly 1 neighbor
    endpoint_mask = (skeleton > 0) & (neighbor_count == 1)

    # Cluster nearby junction pixels (they often form small blobs)
    junction_coords = np.argwhere(junction_mask)
    endpoint_coords = np.argwhere(endpoint_mask)

    junctions = []

    if len(junction_coords) > 0:
        # Cluster nearby junction pixels by dilating and finding components
        junc_img = junction_mask.astype(np.uint8) * 255
        dilated = cv2.dilate(junc_img, np.ones((7, 7), np.uint8))
        n_labels, labels = cv2.connectedComponents(dilated)

        for label_id in range(1, n_labels):
            cluster = junction_coords[
                labels[(junction_coords[:, 0],
                        junction_coords[:, 1])] == label_id
            ]
            if len(cluster) == 0:
                continue

            center_y = int(cluster[:, 0].mean())
            center_x = int(cluster[:, 1].mean())

            # Count distinct branches by examining skeleton in a radius
            branches = count_branches(skeleton, center_y, center_x, radius=15)

            jtype = classify_junction(branches)
            junctions.append({
                "pixel": [center_x, center_y],
                "branches": branches,
                "type": jtype,
            })

    # Add dead ends
    for ey, ex in endpoint_coords:
        junctions.append({
            "pixel": [int(ex), int(ey)],
            "branches": 1,
            "type": "dead_end",
        })

    return junctions


def count_branches(skeleton, cy, cx, radius=15):
    """Count distinct branches radiating from a junction point."""
    h, w = skeleton.shape

    # Create a ring mask around the junction center
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius + 1)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius + 1)

    patch = skeleton[y_min:y_max, x_min:x_max].copy()

    # Clear center circle to break connections
    py, px = cy - y_min, cx - x_min
    cv2.circle(patch, (px, py), radius // 3, 0, -1)

    # Count connected components in the remaining patch
    n_labels, _ = cv2.connectedComponents(patch)
    branches = n_labels - 1  # subtract background

    return max(branches, 1)


def classify_junction(n_branches):
    """Classify junction type from branch count."""
    if n_branches <= 1:
        return "dead_end"
    elif n_branches == 2:
        return "L_junction"  # corner / bend
    elif n_branches == 3:
        return "T_junction"
    elif n_branches >= 4:
        return "X_junction"
    return "unknown"


def to_world_coords(pixel_x, pixel_y, resolution, origin):
    """Convert pixel coordinates to world (meters)."""
    wx = pixel_x * resolution + origin[0]
    wy = pixel_y * resolution + origin[1]
    return [round(wx, 3), round(wy, 3)]


def visualize(map_img, skeleton, junctions, output_path):
    """Draw skeleton and junctions on the map."""
    vis = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)

    # Draw skeleton in blue
    vis[skeleton > 0] = [255, 100, 0]

    # Draw junctions
    colors = {
        "dead_end": (128, 128, 128),
        "L_junction": (0, 255, 255),   # yellow
        "T_junction": (0, 165, 255),   # orange
        "X_junction": (0, 0, 255),     # red
    }

    for j in junctions:
        x, y = j["pixel"]
        color = colors.get(j["type"], (255, 255, 255))
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.circle(vis, (x, y), 8, (0, 0, 0), 1)
        cv2.putText(vis, j["type"].replace("_", " "),
                   (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, color, 1)

    cv2.imwrite(output_path, vis)
    return vis


def detect_junctions_from_map(map_path, yaml_path=None, output_dir=None,
                              resolution=None):
    """Main pipeline: load map → skeleton → junctions."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(map_path), 'junctions')
    os.makedirs(output_dir, exist_ok=True)

    # Load
    free_mask, res, origin, map_img = load_map(map_path, yaml_path)
    if resolution:
        res = resolution
    print(f"Map: {map_img.shape[1]}x{map_img.shape[0]}, "
          f"resolution={res}m/px")

    # Skeleton
    skeleton = extract_skeleton(free_mask)
    print(f"Skeleton: {skeleton.sum()} pixels")

    # Junctions
    junctions = find_junctions(skeleton)

    # Convert to world coordinates
    for j in junctions:
        j["world"] = to_world_coords(
            j["pixel"][0], j["pixel"][1], res, origin)

    print(f"\nJunctions found: {len(junctions)}")
    for j in junctions:
        print(f"  {j['type']}: pixel={j['pixel']}, "
              f"world={j['world']}, branches={j['branches']}")

    # Save
    vis_path = os.path.join(output_dir, "junctions_map.jpg")
    visualize(map_img, skeleton, junctions, vis_path)
    print(f"Visualization: {vis_path}")

    junc_file = os.path.join(output_dir, "junctions.json")
    with open(junc_file, 'w') as f:
        json.dump(junctions, f, indent=2)
    print(f"Data: {junc_file}")

    return junctions, skeleton


def main():
    parser = argparse.ArgumentParser(
        description='Detect corridor junctions from occupancy grid')
    parser.add_argument('--map', required=True,
                        help='Path to occupancy grid (PGM/PNG)')
    parser.add_argument('--yaml', default=None,
                        help='Map YAML file (for resolution/origin)')
    parser.add_argument('--resolution', type=float, default=None,
                        help='Override map resolution (m/pixel)')
    parser.add_argument('--output', default=None,
                        help='Output directory')

    args = parser.parse_args()
    detect_junctions_from_map(args.map, args.yaml, args.output,
                              args.resolution)


if __name__ == '__main__':
    main()
