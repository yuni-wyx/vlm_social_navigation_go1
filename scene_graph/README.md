# Scene Graph Utilities — Semantics + Junctions

This folder currently contains scene-graph preparation utilities, not a
finished end-to-end scene-graph pipeline.
What is implemented here today is:

- open-vocabulary object detection from Go1 imagery
- junction classification from occupancy-grid structure

What is not yet implemented as a completed workflow is the final fusion step
that would turn those outputs into one unified multi-layer scene graph.

## Quick Start

```bash
# 1. Detect objects in rectified camera images
python3 scene_graph/detect_objects.py \
    --input camera_calib/rectified/camera_face/ \
    --output scene_graph/detections/

# 2. Detect junctions from occupancy grid (after RTAB-Map)
python3 scene_graph/detect_junctions.py \
    --map rtabmap/output/map.pgm

# 3. (Planned, not fully implemented) Build unified scene graph
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

## Current Files

```
scene_graph/
├── detect_objects.py      # YOLO World open-vocab detector
├── detect_junctions.py    # Skeleton junction classifier
├── build_graph.py         # Partial / planned graph-fusion entry point
├── visualize.py           # Planned rendering helper
├── detections/            # YOLO output (gitignored)
└── junctions/             # Junction output (gitignored)
```
