# Scene Graph — Semantics + Junctions

Build a multi-layer scene graph from Go1 sensor data, combining YOLO World open-vocab object detection with skeleton-based junction classification.

## Quick Start

```bash
# 1. Detect objects in rectified camera images
python3 scene_graph/detect_objects.py \
    --input camera_calib/rectified/camera_face/ \
    --output scene_graph/detections/

# 2. Detect junctions from occupancy grid (after RTAB-Map)
python3 scene_graph/detect_junctions.py \
    --map rtabmap/output/map.pgm

# 3. (Coming) Build unified scene graph
python3 scene_graph/build_graph.py \
    --detections scene_graph/detections/detections.json \
    --junctions scene_graph/junctions/junctions.json
```

## Object Detection (YOLO World)

Uses open-vocabulary detection — specify what to find via text:
```bash
# Custom classes
python3 scene_graph/detect_objects.py --input images/ \
    --classes "door,sink,fire extinguisher,exit sign,elevator"
```

Default vocabulary: door, sink, toilet, fire extinguisher, exit sign, elevator, stairs, whiteboard, window, vending machine, water fountain, trash can, chair, table, monitor, hallway sign, fire alarm.

## Junction Detection

Classifies corridor junctions from 2D occupancy grids:
- **Dead-end** (1 branch)
- **L-junction** / corner (2 branches)
- **T-junction** (3 branches)
- **X-junction** / 4-way (4+ branches)

Uses morphological skeletonization + branch point analysis — no ML needed.

## Files

```
scene_graph/
├── detect_objects.py      # YOLO World open-vocab detector
├── detect_junctions.py    # Skeleton junction classifier
├── build_graph.py         # (Coming) Fuse into NetworkX graph
├── visualize.py           # (Coming) Render on map
├── detections/            # YOLO output (gitignored)
└── junctions/             # Junction output (gitignored)
```
