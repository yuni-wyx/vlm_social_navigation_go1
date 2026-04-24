#!/usr/bin/env python3
"""
Detect semantic objects in Go1 camera frames using YOLO World.

YOLO World is an open-vocabulary detector — you specify what to
look for via text prompts (no retraining needed).

Usage:
  # Run on rectified images from the undistortion pipeline
  python3 detect_objects.py --input camera_calib/rectified/camera_face/ \
                            --output scene_graph/detections/

  # Custom vocabulary
  python3 detect_objects.py --input images/ --output dets/ \
    --classes "door,sink,fire extinguisher,exit sign,elevator,stairs,whiteboard"
"""
import argparse
import json
import os
import cv2
from pathlib import Path


DEFAULT_CLASSES = [
    "door", "doorway", "sink", "toilet", "fire extinguisher",
    "exit sign", "elevator", "stairs", "whiteboard", "window",
    "vending machine", "water fountain", "trash can", "chair",
    "table", "monitor", "hallway sign", "fire alarm",
]


def detect_objects(input_dir, output_dir, classes=None, confidence=0.15,
                   model_size="yolov8x-worldv2"):
    """Run YOLO World on a directory of images."""
    from ultralytics import YOLO

    if classes is None:
        classes = DEFAULT_CLASSES

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading YOLO World model: {model_size}")
    model = YOLO(model_size)
    model.set_classes(classes)
    print(f"Classes: {classes}")
    print(f"Confidence threshold: {confidence}")

    image_files = sorted([
        f for f in Path(input_dir).glob("*")
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    ])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    all_detections = {}
    total_objects = 0

    for i, img_path in enumerate(image_files):
        results = model.predict(
            str(img_path),
            conf=confidence,
            verbose=False,
        )

        frame_dets = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                frame_dets.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "center": [round((x1+x2)/2), round((y1+y2)/2)],
                })

        all_detections[img_path.name] = frame_dets
        total_objects += len(frame_dets)

        # Save annotated image
        if frame_dets:
            img = cv2.imread(str(img_path))
            for det in frame_dets:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(Path(output_dir) / f"det_{img_path.name}"), img)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(image_files)}] {total_objects} objects so far")

    # Save detections JSON
    det_file = Path(output_dir) / "detections.json"
    with open(det_file, 'w') as f:
        json.dump(all_detections, f, indent=2)

    # Summary
    class_counts = {}
    for dets in all_detections.values():
        for d in dets:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    print(f"\nDone! {total_objects} objects in {len(image_files)} frames")
    print(f"Detections: {det_file}")
    print(f"\nClass summary:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    return all_detections


def main():
    parser = argparse.ArgumentParser(description='YOLO World object detection')
    parser.add_argument('--input', required=True,
                        help='Directory of rectified images')
    parser.add_argument('--output', default='scene_graph/detections/',
                        help='Output directory for detections')
    parser.add_argument('--classes', default=None,
                        help='Comma-separated list of classes to detect')
    parser.add_argument('--confidence', type=float, default=0.15,
                        help='Detection confidence threshold')
    parser.add_argument('--model', default='yolov8x-worldv2',
                        help='YOLO World model variant')

    args = parser.parse_args()

    classes = args.classes.split(',') if args.classes else None
    detect_objects(args.input, args.output, classes,
                   args.confidence, args.model)


if __name__ == '__main__':
    main()
