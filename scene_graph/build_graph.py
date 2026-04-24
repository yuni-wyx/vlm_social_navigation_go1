#!/usr/bin/env python3
"""
Build a scene graph from YOLO World detections and robot odometry.

Reads per-frame detections, clusters them spatially using the robot's
odometry trajectory, and produces a multi-layer graph:
  - Place nodes (trajectory waypoints)
  - Object nodes (detected semantics, clustered across frames)
  - Edges (place↔object proximity, place↔place sequential)

Usage:
  python3 build_graph.py \
    --detections scene_graph/detections/face/detections.json \
    --bag streaming/bags/go1_session_*_fixed.bag \
    --output scene_graph/output/
"""
import argparse
import json
import os
import cv2
import numpy as np
import networkx as nx


def load_odometry_from_bag(bag_path, odom_topic="/odom_fixed"):
    """Extract timestamped robot poses from bag."""
    import rosbag

    poses = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[odom_topic]):
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            # Convert quaternion to yaw
            import math
            siny = 2.0 * (o.w * o.z + o.x * o.y)
            cosy = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
            yaw = math.atan2(siny, cosy)

            poses.append({
                "time": t.to_sec(),
                "x": p.x,
                "y": p.y,
                "z": p.z,
                "yaw": yaw,
            })

    return poses


def load_image_timestamps(detections_json):
    """Extract timestamps from detection image filenames."""
    with open(detections_json) as f:
        dets = json.load(f)

    frames = []
    for fname, frame_dets in dets.items():
        # Filename format: 00123_1773427780_123456789.jpg
        parts = fname.replace('.jpg', '').split('_')
        if len(parts) >= 3:
            secs = int(parts[1])
            nsecs = int(parts[2])
            t = secs + nsecs * 1e-9
        else:
            t = 0.0

        frames.append({
            "filename": fname,
            "time": t,
            "detections": frame_dets,
        })

    return sorted(frames, key=lambda x: x["time"])


def find_closest_pose(poses, timestamp):
    """Find the pose closest to the given timestamp."""
    if not poses:
        return None
    times = [p["time"] for p in poses]
    idx = np.argmin(np.abs(np.array(times) - timestamp))
    return poses[idx]


def cluster_detections(frames_with_poses, max_dist=0.5):
    """
    Cluster detections that appear across frames into unique objects.

    Same class detected from nearby robot positions = same object.
    """
    raw_objects = []  # (class, x, y, confidence, frame_idx)

    for i, frame in enumerate(frames_with_poses):
        if frame["pose"] is None:
            continue
        px, py = frame["pose"]["x"], frame["pose"]["y"]
        yaw = frame["pose"]["yaw"]

        for det in frame["detections"]:
            # Approximate object world position:
            # place it ~1.5m in front of robot at detection angle
            cx = det["center"][0]
            img_w = 640  # rectified width
            angle_offset = (cx - img_w / 2) / img_w * np.deg2rad(120)
            obj_dist = 1.5  # assumed distance

            obj_x = px + obj_dist * np.cos(yaw + angle_offset)
            obj_y = py + obj_dist * np.sin(yaw + angle_offset)

            raw_objects.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "world_x": obj_x,
                "world_y": obj_y,
                "frame_idx": i,
            })

    # Simple spatial clustering: merge same-class objects within max_dist
    clusters = []
    used = set()

    for i, obj in enumerate(raw_objects):
        if i in used:
            continue
        cluster = [obj]
        used.add(i)

        for j, other in enumerate(raw_objects):
            if j in used or other["class"] != obj["class"]:
                continue
            dist = np.sqrt((obj["world_x"] - other["world_x"])**2 +
                          (obj["world_y"] - other["world_y"])**2)
            if dist < max_dist:
                cluster.append(other)
                used.add(j)

        # Merge cluster into single object
        avg_x = np.mean([c["world_x"] for c in cluster])
        avg_y = np.mean([c["world_y"] for c in cluster])
        max_conf = max(c["confidence"] for c in cluster)

        clusters.append({
            "class": obj["class"],
            "world_x": round(float(avg_x), 3),
            "world_y": round(float(avg_y), 3),
            "confidence": round(float(max_conf), 3),
            "n_observations": len(cluster),
        })

    return clusters


def build_scene_graph(poses, objects, waypoint_interval=0.3):
    """
    Build a NetworkX scene graph.

    Layers:
      - Place nodes (trajectory sampled every waypoint_interval meters)
      - Object nodes (clustered semantic detections)
    """
    G = nx.Graph()

    # --- Place nodes (subsample trajectory) ---
    place_nodes = []
    last_x, last_y = None, None

    for i, p in enumerate(poses):
        if last_x is not None:
            dist = np.sqrt((p["x"] - last_x)**2 + (p["y"] - last_y)**2)
            if dist < waypoint_interval:
                continue

        node_id = f"place_{len(place_nodes)}"
        G.add_node(node_id,
                   layer="place",
                   x=round(p["x"], 3),
                   y=round(p["y"], 3),
                   yaw=round(p["yaw"], 3),
                   label=f"P{len(place_nodes)}")
        place_nodes.append(node_id)
        last_x, last_y = p["x"], p["y"]

    # Sequential edges between places
    for i in range(len(place_nodes) - 1):
        n1, n2 = place_nodes[i], place_nodes[i + 1]
        dist = np.sqrt(
            (G.nodes[n1]["x"] - G.nodes[n2]["x"])**2 +
            (G.nodes[n1]["y"] - G.nodes[n2]["y"])**2
        )
        G.add_edge(n1, n2, relation="sequential", distance=round(dist, 3))

    # --- Object nodes ---
    for i, obj in enumerate(objects):
        node_id = f"obj_{i}_{obj['class'].replace(' ', '_')}"
        G.add_node(node_id,
                   layer="object",
                   x=obj["world_x"],
                   y=obj["world_y"],
                   label=obj["class"],
                   confidence=obj["confidence"],
                   observations=obj["n_observations"])

        # Link to nearest place node
        min_dist = float('inf')
        nearest = None
        for pn in place_nodes:
            d = np.sqrt(
                (G.nodes[pn]["x"] - obj["world_x"])**2 +
                (G.nodes[pn]["y"] - obj["world_y"])**2
            )
            if d < min_dist:
                min_dist = d
                nearest = pn

        if nearest and min_dist < 5.0:
            G.add_edge(node_id, nearest,
                       relation="near", distance=round(min_dist, 3))

    return G


def visualize_graph(G, output_path, map_bounds=None):
    """Render scene graph as both a map plot and a topology diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: Spatial map view ---
    ax1.set_title("Scene Graph — Spatial View", fontsize=14, fontweight='bold')

    # Place nodes trajectory
    place_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == "place"]
    obj_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == "object"]

    px = [G.nodes[n]["x"] for n in place_nodes]
    py = [G.nodes[n]["y"] for n in place_nodes]
    ax1.plot(px, py, 'b-', alpha=0.3, linewidth=1, label="trajectory")
    ax1.scatter(px, py, c='dodgerblue', s=15, zorder=3, alpha=0.6)

    # Object nodes by class with colors
    class_colors = {}
    cmap = plt.cm.Set1
    unique_classes = list(set(G.nodes[n]["label"] for n in obj_nodes))
    for i, cls in enumerate(unique_classes):
        class_colors[cls] = cmap(i / max(len(unique_classes), 1))

    for n in obj_nodes:
        d = G.nodes[n]
        color = class_colors.get(d["label"], "red")
        size = 60 + d.get("observations", 1) * 20
        ax1.scatter(d["x"], d["y"], c=[color], s=size,
                   edgecolors='black', linewidths=0.5, zorder=5)
        ax1.annotate(d["label"], (d["x"], d["y"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                             facecolor='white', alpha=0.8))

    # Draw edges
    for u, v, edata in G.edges(data=True):
        if edata.get("relation") == "near":
            ax1.plot([G.nodes[u]["x"], G.nodes[v]["x"]],
                    [G.nodes[u]["y"], G.nodes[v]["y"]],
                    'r--', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # Legend
    patches = [mpatches.Patch(color=class_colors[c], label=c) for c in unique_classes]
    patches.insert(0, mpatches.Patch(color='dodgerblue', label='trajectory'))
    ax1.legend(handles=patches, loc='upper left', fontsize=8)

    # --- Right: Topology graph ---
    ax2.set_title("Scene Graph — Topology", fontsize=14, fontweight='bold')

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_colors = []
    node_sizes = []
    labels = {}

    for n in G.nodes():
        d = G.nodes[n]
        if d.get("layer") == "place":
            node_colors.append("dodgerblue")
            node_sizes.append(30)
            labels[n] = ""
        else:
            node_colors.append(class_colors.get(d["label"], "red"))
            node_sizes.append(150)
            labels[n] = d["label"]

    edge_colors = ['gray' if G.edges[e].get("relation") == "sequential"
                   else 'red' for e in G.edges()]
    edge_styles = ['solid' if G.edges[e].get("relation") == "sequential"
                   else 'dashed' for e in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors,
                           node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color=edge_colors,
                           alpha=0.3, width=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax2, labels=labels, font_size=6)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Build scene graph')
    parser.add_argument('--detections', required=True,
                        help='detections.json from detect_objects.py')
    parser.add_argument('--bag', required=True,
                        help='Fixed bag file (for odometry)')
    parser.add_argument('--output', default='scene_graph/output/',
                        help='Output directory')
    parser.add_argument('--junctions', default=None,
                        help='junctions.json from detect_junctions.py')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load odometry
    print("Loading odometry from bag...")
    import rospy  # noqa: ensure ROS env
    poses = load_odometry_from_bag(args.bag)
    print(f"  {len(poses)} poses loaded")

    # Load detections with timestamps
    print("Loading detections...")
    frames = load_image_timestamps(args.detections)
    print(f"  {len(frames)} frames, "
          f"{sum(len(f['detections']) for f in frames)} total detections")

    # Match each frame to closest pose
    for frame in frames:
        frame["pose"] = find_closest_pose(poses, frame["time"])

    # Cluster detections into unique objects
    print("Clustering detections...")
    objects = cluster_detections(frames)
    print(f"  {len(objects)} unique objects found:")
    for obj in objects:
        print(f"    {obj['class']}: ({obj['world_x']:.1f}, {obj['world_y']:.1f}), "
              f"conf={obj['confidence']}, seen {obj['n_observations']}x")

    # Load junctions if available
    junction_nodes = []
    if args.junctions and os.path.exists(args.junctions):
        with open(args.junctions) as f:
            junction_nodes = json.load(f)
        print(f"  {len(junction_nodes)} junctions loaded")

    # Build graph
    print("Building scene graph...")
    G = build_scene_graph(poses, objects)

    # Add junction nodes if available
    for i, j in enumerate(junction_nodes):
        node_id = f"junction_{i}_{j['type']}"
        G.add_node(node_id,
                   layer="junction",
                   x=j.get("world", [0, 0])[0],
                   y=j.get("world", [0, 0])[1],
                   label=j["type"].replace("_", " "),
                   branches=j["branches"])

    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Stats
    layers = {}
    for n, d in G.nodes(data=True):
        layer = d.get("layer", "unknown")
        layers[layer] = layers.get(layer, 0) + 1
    print(f"  Layers: {layers}")

    # Visualize
    vis_path = os.path.join(args.output, "scene_graph.png")
    visualize_graph(G, vis_path)

    # Save graph data
    graph_data = {
        "nodes": [],
        "edges": [],
        "stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "layers": layers,
        }
    }
    for n, d in G.nodes(data=True):
        graph_data["nodes"].append({"id": n, **d})
    for u, v, d in G.edges(data=True):
        graph_data["edges"].append({"source": u, "target": v, **d})

    graph_file = os.path.join(args.output, "scene_graph.json")
    with open(graph_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    print(f"Graph data: {graph_file}")


if __name__ == '__main__':
    main()
