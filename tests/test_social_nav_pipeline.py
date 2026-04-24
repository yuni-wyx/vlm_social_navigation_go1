from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


social_nav_eval = load_module(
    "social_nav_eval",
    "motion_control/scripts/social_nav_eval.py",
)
social_nav_policy = load_module(
    "social_nav_policy",
    "social_nav_policy.py",
)


def test_normalize_action_aliases():
    assert social_nav_eval.normalize_action("go_left") == "LEFT"
    assert social_nav_eval.normalize_action(" proceed ") == "FORWARD"
    assert social_nav_eval.normalize_action("ambiguous") == "REVIEW"
    assert social_nav_eval.normalize_action("nonsense") is None


def test_predict_vlm_uses_mocked_model_outputs(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.ppm"
    image_path.write_bytes(b"fake-image-bytes")

    sample = {
        "sample_id": "sequence_000001",
        "input_type": "sequence",
        "timestamp": 1.0,
        "image_paths": [str(image_path)],
        "source_frames_jsonl": str(tmp_path / "frames.jsonl"),
    }
    method_spec = social_nav_eval.METHOD_REGISTRY["qwen_sequence_image_navigation"]

    cases = [
        ("crossing_rightward", "LEFT"),
        ("crossing_leftward", "RIGHT"),
        ("receding", "FORWARD"),
        ("ambiguous", "REVIEW"),
        ("blocked", "STOP"),
    ]

    for motion, expected_action in cases:
        def fake_call(*_args, **_kwargs):
            return {
                "ok": True,
                "response_json": {
                    "motion": motion,
                    "recommended_action": expected_action,
                },
                "raw_text": json.dumps(
                    {"motion": motion, "recommended_action": expected_action}
                ),
                "latency_sec": 0.01,
            }

        monkeypatch.setattr(social_nav_eval, "call_navigation_wrapper", fake_call)
        result = social_nav_eval.predict_vlm(
            sample=sample,
            method_spec=method_spec,
            wrapper_url="http://example.invalid/analyze_navigation",
            timeout_sec=1.0,
            samples_path=str(tmp_path / "samples.jsonl"),
        )
        assert result["success"] is True
        assert result["predicted_action"] == expected_action
        assert result["response_json"]["motion"] == motion


def test_build_samples_caps_sequence_from_ten_frames_to_five():
    frames = [
        {
            "timestamp": float(index),
            "image_path": f"images/frame_{index:06d}.ppm",
            "front_dist": 2.0,
        }
        for index in range(10)
    ]

    rows = social_nav_eval.build_samples_from_frames(
        frames=frames,
        input_type="sequence",
        sequence_length=10,
        sequence_sampling_mode="capped",
        max_frames_per_sequence=5,
    )

    assert len(rows) == 1
    sample = rows[0]
    assert sample["original_num_images"] == 10
    assert sample["num_images"] == 5
    assert sample["frame_indices"] == [0, 2, 4, 7, 9]
    assert sample["sequence_sampling_mode"] == "capped"


def test_bag_level_majority_vote_summary():
    reporting_rows = [
        {
            "bag_id": "bag_05",
            "scenario_name": "right_to_left_crossing",
            "scenario_type": "crossing_person",
            "primary_case": True,
            "expected_action": "RIGHT",
            "method": "internvl_sequence_image_navigation",
            "prediction": "RIGHT",
        },
        {
            "bag_id": "bag_05",
            "scenario_name": "right_to_left_crossing",
            "scenario_type": "crossing_person",
            "primary_case": True,
            "expected_action": "RIGHT",
            "method": "qwen_sequence_image_navigation",
            "prediction": "RIGHT",
        },
        {
            "bag_id": "bag_05",
            "scenario_name": "right_to_left_crossing",
            "scenario_type": "crossing_person",
            "primary_case": True,
            "expected_action": "RIGHT",
            "method": "internvl_single_image_navigation",
            "prediction": "STOP",
        },
    ]

    summary = social_nav_eval.summarize_scenario_decisions(reporting_rows)[0]
    assert summary["consensus_action"] == "RIGHT"
    assert summary["consensus_vote_count"] == 2
    assert summary["consensus_matches_expected"] is True
    assert summary["disagreement"] is True


def test_unsafe_forward_rate(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    labels_path = tmp_path / "labels.csv"

    prediction_rows = [
        {
            "sample_id": "s1",
            "method": "qwen_sequence_image_navigation",
            "method_category": "primary",
            "model_name": "Qwen",
            "supports_action_evaluation": True,
            "predicted_action": "FORWARD",
            "success": True,
            "failure_reason": "",
            "latency_sec": 0.1,
            "front_dist": 0.4,
            "prompt_name": "sequence_image_navigation",
        },
        {
            "sample_id": "s2",
            "method": "qwen_sequence_image_navigation",
            "method_category": "primary",
            "model_name": "Qwen",
            "supports_action_evaluation": True,
            "predicted_action": "STOP",
            "success": True,
            "failure_reason": "",
            "latency_sec": 0.1,
            "front_dist": 0.3,
            "prompt_name": "sequence_image_navigation",
        },
        {
            "sample_id": "s3",
            "method": "qwen_sequence_image_navigation",
            "method_category": "primary",
            "model_name": "Qwen",
            "supports_action_evaluation": True,
            "predicted_action": "FORWARD",
            "success": True,
            "failure_reason": "",
            "latency_sec": 0.1,
            "front_dist": 1.2,
            "prompt_name": "sequence_image_navigation",
        },
    ]
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row) + "\n")

    with open(labels_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "ground_truth_action",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "s1", "ground_truth_action": "STOP"},
                {"sample_id": "s2", "ground_truth_action": "STOP"},
                {"sample_id": "s3", "ground_truth_action": "FORWARD"},
            ]
        )

    labels = social_nav_eval.load_labels_csv(labels_path)
    summary = social_nav_eval.summarize_prediction_file(
        predictions_path,
        labels,
        safety_distance_threshold=0.5,
    )

    assert summary["unsafe_forward_count"] == 1
    assert summary["close_obstacle_count"] == 2
    assert summary["unsafe_forward_rate"] == 0.5
    assert summary["action_accuracy"] == 2 / 3


def test_realtime_projection_keeps_lateral_as_advisory():
    action, response_json, note = social_nav_policy.project_realtime_action(
        "LEFT",
        {"recommended_avoidance_side": ""},
        allow_lateral=False,
    )

    assert action == "STOP"
    assert response_json["recommended_avoidance_side"] == "left"
    assert "advisory" in note.lower()


def test_realtime_projection_can_allow_explicit_lateral_execution():
    action, response_json, note = social_nav_policy.project_realtime_action(
        "RIGHT",
        {"recommended_avoidance_side": "right"},
        allow_lateral=True,
    )

    assert action == "RIGHT"
    assert response_json["recommended_avoidance_side"] == "right"
    assert note is None
