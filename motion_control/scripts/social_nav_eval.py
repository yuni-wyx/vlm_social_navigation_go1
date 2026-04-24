#!/usr/bin/env python3
"""
Offline evaluation pipeline for Go1 VLM-based interaction-aware navigation.

Subcommands:
  build-samples        Create single-image or sequence samples from frames.jsonl
  write-label-template Generate a CSV label template from samples
  predict              Run one method or compatibility alias on samples
  run-benchmark        Run the default primary benchmark suite, optionally adding legacy prompts
  run-manifest         Run the manifest-driven rosbag or extracted-data benchmark workflow
  evaluate             Compare one or more prediction files against labels
"""

import argparse
import base64
import csv
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from social_nav_eval_prompts import ACTIONS  # noqa: E402

ACTION_ALIASES = {
    # Uncertainty / deferral synonyms -> REVIEW
    "UNKNOWN": "REVIEW",
    "UNSURE": "REVIEW",
    "UNCERTAIN": "REVIEW",
    "IDK": "REVIEW",
    "I_DONT_KNOW": "REVIEW",
    "I DON'T KNOW": "REVIEW",
    "CANNOT_TELL": "REVIEW",
    "CAN'T TELL": "REVIEW",
    "HOLD": "REVIEW",
    "WAIT": "REVIEW",
    "DEFER": "REVIEW",
    "NOT_SURE": "REVIEW",
    "AMBIGUOUS": "REVIEW",
    # Lateral-avoidance synonyms -> LEFT / RIGHT
    "GO_LEFT": "LEFT",
    "MOVE_LEFT": "LEFT",
    "STEP_LEFT": "LEFT",
    "SIDESTEP_LEFT": "LEFT",
    "AVOID_LEFT": "LEFT",
    "BYPASS_LEFT": "LEFT",
    "TURN_LEFT": "LEFT",
    "GO_RIGHT": "RIGHT",
    "MOVE_RIGHT": "RIGHT",
    "STEP_RIGHT": "RIGHT",
    "SIDESTEP_RIGHT": "RIGHT",
    "AVOID_RIGHT": "RIGHT",
    "BYPASS_RIGHT": "RIGHT",
    "TURN_RIGHT": "RIGHT",
    # Basic synonyms for other actions
    "HALT": "STOP",
    "YIELD": "STOP",
    "CONTINUE": "FORWARD",
    "GO": "FORWARD",
    "PROCEED": "FORWARD",
}


DEFAULT_INTERNVL_WRAPPER_BASE_URL = os.getenv(
    "SOCIAL_NAV_EVAL_INTERNVL_WRAPPER_URL",
    "http://localhost:8100",
)
DEFAULT_QWEN_WRAPPER_BASE_URL = os.getenv(
    "SOCIAL_NAV_EVAL_QWEN_WRAPPER_URL",
    "http://localhost:8101",
)
DEFAULT_SCENARIO_MANIFEST_PATH = (
    REPO_ROOT / "motion_control" / "eval" / "scenario_manifest.json"
)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    category: str
    default_enabled: bool
    kind: str
    input_type: str
    model_name: str
    prompt_name: str
    endpoint_path: str
    wrapper_group: str
    supports_action_evaluation: bool
    description: str


METHOD_REGISTRY = {
    "geometry": MethodSpec(
        method_id="geometry",
        category="primary",
        default_enabled=True,
        kind="geometry",
        input_type="single",
        model_name="rule_based_geometry",
        prompt_name="",
        endpoint_path="",
        wrapper_group="",
        supports_action_evaluation=True,
        description="Rule-based baseline using front distance only.",
    ),
    "internvl_single_image_navigation": MethodSpec(
        method_id="internvl_single_image_navigation",
        category="primary",
        default_enabled=True,
        kind="vlm",
        input_type="single",
        model_name="InternVL",
        prompt_name="single_image_navigation",
        endpoint_path="/analyze_navigation",
        wrapper_group="internvl",
        supports_action_evaluation=True,
        description="InternVL with the primary single-image navigation prompt.",
    ),
    "internvl_sequence_image_navigation": MethodSpec(
        method_id="internvl_sequence_image_navigation",
        category="primary",
        default_enabled=True,
        kind="vlm",
        input_type="sequence",
        model_name="InternVL",
        prompt_name="sequence_image_navigation",
        endpoint_path="/analyze_navigation",
        wrapper_group="internvl",
        supports_action_evaluation=True,
        description="InternVL with the primary sequence-image navigation prompt.",
    ),
    "qwen_single_image_navigation": MethodSpec(
        method_id="qwen_single_image_navigation",
        category="primary",
        default_enabled=True,
        kind="vlm",
        input_type="single",
        model_name="Qwen",
        prompt_name="single_image_navigation",
        endpoint_path="/analyze_navigation",
        wrapper_group="qwen",
        supports_action_evaluation=True,
        description="Qwen with the primary single-image navigation prompt.",
    ),
    "qwen_sequence_image_navigation": MethodSpec(
        method_id="qwen_sequence_image_navigation",
        category="primary",
        default_enabled=True,
        kind="vlm",
        input_type="sequence",
        model_name="Qwen",
        prompt_name="sequence_image_navigation",
        endpoint_path="/analyze_navigation",
        wrapper_group="qwen",
        supports_action_evaluation=True,
        description="Qwen with the primary sequence-image navigation prompt.",
    ),
    "internvl_legacy_analyze": MethodSpec(
        method_id="internvl_legacy_analyze",
        category="legacy",
        default_enabled=False,
        kind="vlm",
        input_type="single",
        model_name="InternVL",
        prompt_name="legacy_analyze",
        endpoint_path="/analyze",
        wrapper_group="internvl",
        supports_action_evaluation=False,
        description=(
            "Legacy InternVL person-detection prompt kept only for debugging/ablation. "
            "Excluded from the default benchmark because no-person false positives were unstable."
        ),
    ),
    "internvl_structured_localization": MethodSpec(
        method_id="internvl_structured_localization",
        category="legacy",
        default_enabled=False,
        kind="vlm",
        input_type="single",
        model_name="InternVL",
        prompt_name="structured_localization",
        endpoint_path="/analyze_navigation",
        wrapper_group="internvl",
        supports_action_evaluation=False,
        description=(
            "Legacy InternVL structured-localization prompt kept only for debugging/ablation. "
            "Excluded from the default benchmark because no-person false positives were unstable."
        ),
    ),
    "qwen_legacy_analyze": MethodSpec(
        method_id="qwen_legacy_analyze",
        category="legacy",
        default_enabled=False,
        kind="vlm",
        input_type="single",
        model_name="Qwen",
        prompt_name="legacy_analyze",
        endpoint_path="/analyze",
        wrapper_group="qwen",
        supports_action_evaluation=False,
        description="Legacy Qwen person-detection prompt kept only for debugging/ablation.",
    ),
    "qwen_structured_localization": MethodSpec(
        method_id="qwen_structured_localization",
        category="legacy",
        default_enabled=False,
        kind="vlm",
        input_type="single",
        model_name="Qwen",
        prompt_name="structured_localization",
        endpoint_path="/analyze_navigation",
        wrapper_group="qwen",
        supports_action_evaluation=False,
        description="Legacy Qwen structured-localization prompt kept only for debugging/ablation.",
    ),
}

COMPATIBILITY_METHOD_ALIASES = {
    "single_vlm": "compat_single_vlm",
    "sequence_vlm": "compat_sequence_vlm",
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_action(value):
    if value is None:
        return None
    action = str(value).strip().upper()
    action = ACTION_ALIASES.get(action, action)
    return action if action in ACTIONS else None


def bool_from_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def format_rate(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def resolve_image_path(base_path, image_path):
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return base_path.parent / candidate


def default_wrapper_base_url(wrapper_group):
    if wrapper_group == "internvl":
        return DEFAULT_INTERNVL_WRAPPER_BASE_URL
    if wrapper_group == "qwen":
        return DEFAULT_QWEN_WRAPPER_BASE_URL
    return ""


def join_url(base_url, endpoint_path):
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def method_ids_for_run(include_legacy_prompts):
    methods = [
        spec.method_id
        for spec in METHOD_REGISTRY.values()
        if spec.default_enabled or include_legacy_prompts
    ]
    return sorted(
        methods,
        key=lambda method_id: (
            0 if METHOD_REGISTRY[method_id].category == "primary" else 1,
            method_id,
        ),
    )


def load_extract_module():
    module_path = REPO_ROOT / "streaming" / "scripts" / "extract_social_nav_data.py"
    spec = importlib.util.spec_from_file_location("extract_social_nav_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_scenario_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("scenario_manifest_invalid: scenarios must be a list")
    return scenarios


def resolve_manifest_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_scenario_bag_path(scenario):
    return resolve_manifest_path(scenario["bag_path"])


def resolve_scenario_extracted_dir(scenario, extracted_root=""):
    extracted_dir_value = scenario.get("extracted_dir")
    if extracted_dir_value:
        return resolve_manifest_path(extracted_dir_value)
    if extracted_root:
        return Path(extracted_root).resolve() / scenario["bag_id"]
    raise ValueError(
        f"scenario_missing_extracted_dir: {scenario['bag_id']} does not define extracted_dir and "
        "--extracted-root was not provided"
    )


def validate_extracted_dir(extracted_dir, bag_id):
    extracted_dir = Path(extracted_dir).resolve()
    frames_jsonl = extracted_dir / "frames.jsonl"
    images_dir = extracted_dir / "images"
    summary_json = extracted_dir / "extraction_summary.json"
    missing = [
        str(path.name)
        for path in (frames_jsonl, images_dir, summary_json)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"extracted_dataset_incomplete: {bag_id} is missing {', '.join(missing)} in {extracted_dir}"
        )
    return {
        "frames_jsonl": frames_jsonl,
        "images_dir": images_dir,
        "summary_json": summary_json,
        "root_dir": extracted_dir,
    }


SEQUENCE_SAMPLING_MODES = ("none", "uniform", "capped", "tail")


def subsample_sequence_indices(indices, mode="none", rate=0, max_frames=0):
    """Reduce a sequence of frame indices according to the selected policy.

    Modes:
      - "none"    : return indices unchanged (default, backward-compatible).
      - "uniform" : keep every ``rate``-th index (rate>=2 required to shrink).
      - "capped"  : keep at most ``max_frames`` evenly spaced indices.
      - "tail"    : keep the last ``max_frames`` indices (preserves late-entry
                    signal; drops older frames that can wash out late cues).

    The first and last indices in the input are always preserved when a
    reduction happens so that the temporal anchors of the window remain stable.
    """
    if not indices or mode in (None, "", "none"):
        return list(indices)
    if mode == "uniform":
        step = max(1, int(rate) or 1)
        if step <= 1:
            return list(indices)
        kept = [idx for pos, idx in enumerate(indices) if pos % step == 0]
        if indices[-1] not in kept:
            kept.append(indices[-1])
        return kept
    if mode == "capped":
        cap = max(1, int(max_frames) or len(indices))
        if cap >= len(indices):
            return list(indices)
        if cap == 1:
            return [indices[-1]]
        # Evenly spaced positions, inclusive of first and last.
        last = len(indices) - 1
        positions = [round(i * last / (cap - 1)) for i in range(cap)]
        # Deduplicate while preserving order.
        seen = set()
        kept = []
        for pos in positions:
            if pos not in seen:
                seen.add(pos)
                kept.append(indices[pos])
        return kept
    if mode == "tail":
        cap = max(1, int(max_frames) or len(indices))
        if cap >= len(indices):
            return list(indices)
        return list(indices[-cap:])
    raise ValueError(f"unknown sequence_sampling_mode: {mode}")


def build_samples_from_frames(
    frames,
    input_type,
    sequence_length=3,
    sequence_stride=1,
    sequence_sampling_mode="none",
    subsample_rate=0,
    max_frames_per_sequence=0,
):
    """Build offline samples from frames.jsonl rows.

    Sequence samples may be optionally subsampled via
    ``sequence_sampling_mode`` + ``subsample_rate`` / ``max_frames_per_sequence``
    to control how many images are actually sent to the VLM per call. Defaults
    preserve legacy behaviour (no subsampling).
    """
    rows = []
    sample_idx = 0

    if input_type == "single":
        indices = [[idx] for idx in range(len(frames))]
    else:
        indices = []
        step = max(1, sequence_stride)
        for start in range(0, len(frames) - sequence_length + 1, step):
            indices.append(list(range(start, start + sequence_length)))

    apply_subsample = input_type == "sequence" and sequence_sampling_mode not in (None, "", "none")

    for frame_indices in indices:
        original_indices = list(frame_indices)
        if apply_subsample:
            frame_indices = subsample_sequence_indices(
                original_indices,
                mode=sequence_sampling_mode,
                rate=subsample_rate,
                max_frames=max_frames_per_sequence,
            )
        selected = [frames[idx] for idx in frame_indices]
        front_values = [row["front_dist"] for row in selected if row.get("front_dist") is not None]
        sample = {
            "sample_id": f"{input_type}_{sample_idx:06d}",
            "input_type": input_type,
            "frame_indices": frame_indices,
            "timestamps": [row["timestamp"] for row in selected],
            "timestamp": selected[-1]["timestamp"],
            "image_paths": [row["image_path"] for row in selected],
            "front_dist": selected[-1].get("front_dist"),
            "front_dist_min": min(front_values) if front_values else None,
            "original_frame_indices": original_indices,
            "original_num_images": len(original_indices),
            "num_images": len(frame_indices),
            "sequence_sampling_mode": sequence_sampling_mode if apply_subsample else "none",
        }
        rows.append(sample)
        sample_idx += 1
    return rows


def build_samples(args):
    frames = read_jsonl(args.frames_jsonl)
    rows = build_samples_from_frames(
        frames=frames,
        input_type=args.input_type,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        sequence_sampling_mode=getattr(args, "sequence_sampling_mode", "none"),
        subsample_rate=getattr(args, "subsample_rate", 0),
        max_frames_per_sequence=getattr(args, "max_frames_per_sequence", 0),
    )
    for row in rows:
        row["source_frames_jsonl"] = str(Path(args.frames_jsonl).resolve())

    sequence_length = args.sequence_length if args.input_type == "sequence" else 1

    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "input_type": args.input_type,
                "sequence_length": sequence_length,
                "num_samples": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


def write_label_template(args):
    samples = read_jsonl(args.samples_jsonl)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "timestamp",
                "frame_indices",
                "input_type",
                "ground_truth_action",
                "human_motion",
                "notes",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "timestamp": sample["timestamp"],
                    "frame_indices": ",".join(str(item) for item in sample["frame_indices"]),
                    "input_type": sample["input_type"],
                    "ground_truth_action": "",
                    "human_motion": "",
                    "notes": "",
                }
            )

    print(f"Wrote label template: {output_path}")


def encode_image_base64(path):
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def call_wrapper_analyze(wrapper_url, image_path, timeout_sec):
    payload = {
        "image_base64": encode_image_base64(image_path),
    }
    response = requests.post(wrapper_url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def call_navigation_wrapper(wrapper_url, prompt_name, image_paths, timeout_sec):
    payload = {
        "prompt_name": prompt_name,
        "images_base64": [encode_image_base64(path) for path in image_paths],
    }
    response = requests.post(wrapper_url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def predict_geometry(sample, stop_threshold):
    front_dist = sample.get("front_dist")
    if front_dist is None:
        return {
            "success": False,
            "predicted_action": None,
            "failure_reason": "missing_front_dist",
            "response_json": None,
            "raw_response": "",
            "latency_sec": 0.0,
        }
    action = "STOP" if float(front_dist) < stop_threshold else "FORWARD"
    return {
        "success": True,
        "predicted_action": action,
        "failure_reason": "",
        "response_json": {
            "recommended_action": action,
            "front_dist": front_dist,
            "threshold": stop_threshold,
        },
        "raw_response": "",
        "latency_sec": 0.0,
    }


def predict_vlm(sample, method_spec, wrapper_url, timeout_sec, samples_path):
    base_path = Path(sample.get("source_frames_jsonl") or samples_path).resolve()
    sample_images = [resolve_image_path(base_path, item) for item in sample["image_paths"]]
    if method_spec.input_type == "single":
        sample_images = [sample_images[-1]]

    prompt_name = method_spec.prompt_name

    start = time.time()
    try:
        if method_spec.endpoint_path == "/analyze":
            response = call_wrapper_analyze(
                wrapper_url=wrapper_url,
                image_path=sample_images[-1],
                timeout_sec=timeout_sec,
            )
        else:
            response = call_navigation_wrapper(
                wrapper_url=wrapper_url,
                prompt_name=prompt_name,
                image_paths=sample_images,
                timeout_sec=timeout_sec,
            )
        latency = time.time() - start
    except requests.RequestException as exc:
        latency = time.time() - start
        return {
            "success": False,
            "predicted_action": None,
            "failure_reason": f"http_error: {exc}",
            "response_json": None,
            "raw_response": "",
            "latency_sec": latency,
            "prompt_name": prompt_name,
        }

    response_json = response.get("response_json")
    if response_json is None and method_spec.endpoint_path == "/analyze":
        response_json = {
            "person_detected": response.get("person_detected"),
            "person_in_front": response.get("person_in_front"),
        }
    predicted_action = None
    if method_spec.supports_action_evaluation and isinstance(response_json, dict):
        predicted_action = normalize_action(response_json.get("recommended_action"))

    if not response.get("ok", False):
        return {
            "success": False,
            "predicted_action": predicted_action,
            "failure_reason": response.get("error", "wrapper_error"),
            "response_json": response_json,
            "raw_response": response.get("raw_text", ""),
            "latency_sec": latency,
            "prompt_name": prompt_name,
        }

    if method_spec.supports_action_evaluation and predicted_action is None:
        return {
            "success": False,
            "predicted_action": None,
            "failure_reason": "missing_or_invalid_recommended_action",
            "response_json": response_json,
            "raw_response": response.get("raw_text", ""),
            "latency_sec": latency,
            "prompt_name": prompt_name,
        }

    return {
        "success": True,
        "predicted_action": predicted_action,
        "failure_reason": "",
        "response_json": response_json,
        "raw_response": response.get("raw_text", ""),
        "latency_sec": float(response.get("latency_sec", latency)),
        "prompt_name": prompt_name,
    }


def resolve_method_and_wrapper(args):
    if args.method in METHOD_REGISTRY:
        method_spec = METHOD_REGISTRY[args.method]
        wrapper_base_url = args.wrapper_base_url or default_wrapper_base_url(method_spec.wrapper_group)
        wrapper_url = (
            join_url(wrapper_base_url, method_spec.endpoint_path) if method_spec.kind == "vlm" else ""
        )
        return method_spec, wrapper_url

    if args.method == "geometry":
        return METHOD_REGISTRY["geometry"], ""

    if args.method == "single_vlm":
        return (
            MethodSpec(
                method_id="single_vlm",
                category="legacy",
                default_enabled=False,
                kind="vlm",
                input_type="single",
                model_name="custom_wrapper",
                prompt_name="single_image_navigation",
                endpoint_path="/analyze_navigation",
                wrapper_group="",
                supports_action_evaluation=True,
                description="Backward-compatible alias for single-image navigation.",
            ),
            args.wrapper_url,
        )

    if args.method == "sequence_vlm":
        return (
            MethodSpec(
                method_id="sequence_vlm",
                category="legacy",
                default_enabled=False,
                kind="vlm",
                input_type="sequence",
                model_name="custom_wrapper",
                prompt_name="sequence_image_navigation",
                endpoint_path="/analyze_navigation",
                wrapper_group="",
                supports_action_evaluation=True,
                description="Backward-compatible alias for sequence-image navigation.",
            ),
            args.wrapper_url,
        )

    raise ValueError(f"unknown_method: {args.method}")


def _load_prediction_checkpoint(checkpoint_path):
    rows = []
    if not checkpoint_path:
        return rows
    path = Path(checkpoint_path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                # Trailing partial line from a prior kill; discard and stop.
                break
    return rows


def run_predictions(samples, samples_jsonl, method_spec, wrapper_url, geometry_stop_threshold, request_timeout_sec, checkpoint_path=None):
    existing_rows = _load_prediction_checkpoint(checkpoint_path)
    done_ids = {row.get("sample_id") for row in existing_rows}
    rows = list(existing_rows)

    out_fh = None
    if checkpoint_path is not None:
        # Rewrite only the valid rows to drop any partially-written trailing line.
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            for row in existing_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        out_fh = open(checkpoint_path, "a", encoding="utf-8")

    try:
        for sample in samples:
            if sample["sample_id"] in done_ids:
                continue
            if method_spec.kind == "geometry":
                result = predict_geometry(sample, geometry_stop_threshold)
                prompt_name = ""
            else:
                result = predict_vlm(
                    sample=sample,
                    method_spec=method_spec,
                    wrapper_url=wrapper_url,
                    timeout_sec=request_timeout_sec,
                    samples_path=samples_jsonl,
                )
                prompt_name = result.get("prompt_name", "")

            row = {
                "sample_id": sample["sample_id"],
                "method": method_spec.method_id,
                "method_category": method_spec.category,
                "model_name": method_spec.model_name,
                "default_enabled": method_spec.default_enabled,
                "supports_action_evaluation": method_spec.supports_action_evaluation,
                "input_type": sample["input_type"],
                "timestamp": sample["timestamp"],
                "frame_indices": sample["frame_indices"],
                "num_images": len(sample["image_paths"]),
                "front_dist": sample.get("front_dist"),
                "front_dist_min": sample.get("front_dist_min"),
                "predicted_action": result["predicted_action"],
                "success": result["success"],
                "failure_reason": result["failure_reason"],
                "latency_sec": result["latency_sec"],
                "prompt_name": prompt_name,
                "response_json": result["response_json"],
                "raw_response": result["raw_response"],
            }
            rows.append(row)
            if out_fh is not None:
                out_fh.write(json.dumps(row, sort_keys=True) + "\n")
                out_fh.flush()
    finally:
        if out_fh is not None:
            out_fh.close()
    return rows


def predict(args):
    samples = read_jsonl(args.samples_jsonl)
    method_spec, wrapper_url = resolve_method_and_wrapper(args)
    rows = run_predictions(
        samples=samples,
        samples_jsonl=args.samples_jsonl,
        method_spec=method_spec,
        wrapper_url=wrapper_url,
        geometry_stop_threshold=args.geometry_stop_threshold,
        request_timeout_sec=args.request_timeout_sec,
    )

    write_jsonl(args.output, rows)
    success_count = sum(1 for row in rows if row["success"])
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "method": method_spec.method_id,
                "num_predictions": len(rows),
                "success_count": success_count,
                "failure_count": len(rows) - success_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_benchmark(args):
    single_samples = read_jsonl(args.single_samples_jsonl)
    sequence_samples = read_jsonl(args.sequence_samples_jsonl)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for method_id in method_ids_for_run(args.include_legacy_prompts):
        method_spec = METHOD_REGISTRY[method_id]
        samples = single_samples if method_spec.input_type == "single" else sequence_samples
        samples_jsonl = (
            args.single_samples_jsonl if method_spec.input_type == "single" else args.sequence_samples_jsonl
        )
        wrapper_url = ""
        if method_spec.kind == "vlm":
            wrapper_base_url = default_wrapper_base_url(method_spec.wrapper_group)
            wrapper_url = join_url(wrapper_base_url, method_spec.endpoint_path)

        rows = run_predictions(
            samples=samples,
            samples_jsonl=samples_jsonl,
            method_spec=method_spec,
            wrapper_url=wrapper_url,
            geometry_stop_threshold=args.geometry_stop_threshold,
            request_timeout_sec=args.request_timeout_sec,
        )
        output_path = output_dir / f"{method_id}.jsonl"
        write_jsonl(output_path, rows)
        manifest_rows.append(
            {
                "method_id": method_id,
                "category": method_spec.category,
                "prediction_file": str(output_path),
                "num_predictions": len(rows),
            }
        )

    with open(output_dir / "benchmark_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"methods": manifest_rows}, handle, indent=2, sort_keys=True)

    print(json.dumps({"output_dir": str(output_dir), "methods": manifest_rows}, indent=2, sort_keys=True))


def summarize_bag_predictions(rows, expected_action):
    successful_rows = [row for row in rows if row.get("success")]
    failure_reasons = Counter(
        row.get("failure_reason") for row in rows if not row.get("success") and row.get("failure_reason")
    )
    predicted_actions = [
        normalize_action(row.get("predicted_action"))
        for row in successful_rows
        if normalize_action(row.get("predicted_action")) is not None
    ]
    action_counts = Counter(predicted_actions)
    predicted_action_summary = action_counts.most_common(1)[0][0] if action_counts else ""
    correct_action = (
        predicted_action_summary == normalize_action(expected_action)
        if predicted_action_summary and normalize_action(expected_action)
        else None
    )
    latencies = [float(row["latency_sec"]) for row in rows if row.get("latency_sec") is not None]
    unavailable_reason = ""
    if rows and not successful_rows and failure_reasons == Counter({"missing_front_dist": len(rows)}):
        unavailable_reason = (
            "No front distance was available for this scenario. "
            "The rosbag/extracted dataset does not contain the LiDAR scan topic used by the geometry baseline."
        )
    # Per-action fractions (over successful predictions) for LEFT/RIGHT/REVIEW visibility.
    n_ok = len(successful_rows) or 1
    rate = lambda action: action_counts.get(action, 0) / n_ok if successful_rows else 0.0
    # Sequence-sample subsample bookkeeping (0 if missing / single-image methods).
    orig_images = [int(row["original_num_images"]) for row in rows if row.get("original_num_images")]
    actual_images = [int(row["num_images"]) for row in rows if row.get("num_images") is not None]
    sampling_modes = {row.get("sequence_sampling_mode", "") for row in rows if row.get("sequence_sampling_mode")}
    return {
        "predicted_action_summary": predicted_action_summary,
        "correct_action": correct_action,
        "prediction_coverage": (len(successful_rows) / len(rows)) if rows else 0.0,
        "avg_inference_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
        "num_predictions": len(rows),
        "num_successful_predictions": len(successful_rows),
        "failure_reason_counts": dict(failure_reasons),
        "unavailable_reason": unavailable_reason,
        "action_counts": dict(action_counts),
        "stop_rate": rate("STOP"),
        "forward_rate": rate("FORWARD"),
        "left_rate": rate("LEFT"),
        "right_rate": rate("RIGHT"),
        "review_rate": rate("REVIEW"),
        "avg_original_num_images": (sum(orig_images) / len(orig_images)) if orig_images else None,
        "avg_num_images": (sum(actual_images) / len(actual_images)) if actual_images else None,
        "sequence_sampling_modes": sorted(m for m in sampling_modes if m),
    }


def write_aggregate_csv(path, rows):
    fieldnames = [
        "bag_id",
        "scenario_name",
        "scenario_type",
        "primary_case",
        "expected_action",
        "method",
        "method_category",
        "model_name",
        "prediction",
        "correct_action",
        "prediction_coverage",
        "avg_inference_latency_sec",
        "stop_rate",
        "forward_rate",
        "left_rate",
        "right_rate",
        "review_rate",
        "avg_original_num_images",
        "avg_num_images",
        "sequence_sampling_modes",
        "unavailable_reason",
        "notes",
        "artifact_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_manifest(args):
    scenarios = load_scenario_manifest(args.manifest)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_module = load_extract_module() if args.input_mode == "rosbag" else None
    aggregate_rows = []
    method_ids = method_ids_for_run(args.include_legacy_prompts)

    for scenario in scenarios:
        bag_id = scenario["bag_id"]
        scenario_name = scenario["scenario_name"]
        scenario_dir = output_dir / bag_id
        predictions_dir = scenario_dir / "predictions"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        if args.input_mode == "rosbag":
            extracted_dir = scenario_dir / "extracted"
            extract_args = SimpleNamespace(
                bag=str(resolve_scenario_bag_path(scenario)),
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
            assert extract_module is not None
            extract_module.extract_dataset(extract_args)
            extracted_info = validate_extracted_dir(extracted_dir, bag_id)
        else:
            extracted_info = validate_extracted_dir(
                resolve_scenario_extracted_dir(scenario, args.extracted_root),
                bag_id,
            )

        frames = read_jsonl(extracted_info["frames_jsonl"])
        single_samples = build_samples_from_frames(
            frames=frames,
            input_type="single",
            sequence_length=1,
            sequence_stride=args.sequence_stride,
        )
        sequence_samples = build_samples_from_frames(
            frames=frames,
            input_type="sequence",
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            sequence_sampling_mode=getattr(args, "sequence_sampling_mode", "none"),
            subsample_rate=getattr(args, "subsample_rate", 0),
            max_frames_per_sequence=getattr(args, "max_frames_per_sequence", 0),
        )
        for row in single_samples:
            row["source_frames_jsonl"] = str(extracted_info["frames_jsonl"])
        for row in sequence_samples:
            row["source_frames_jsonl"] = str(extracted_info["frames_jsonl"])

        single_samples_path = scenario_dir / "single_samples.jsonl"
        sequence_samples_path = scenario_dir / "sequence_samples.jsonl"
        write_jsonl(single_samples_path, single_samples)
        write_jsonl(sequence_samples_path, sequence_samples)

        label_rows = []
        for sample_collection in (single_samples, sequence_samples):
            for sample in sample_collection:
                label_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "timestamp": sample["timestamp"],
                        "frame_indices": ",".join(str(item) for item in sample["frame_indices"]),
                        "input_type": sample["input_type"],
                        "ground_truth_action": scenario["expected_action"],
                        "human_motion": scenario["human_motion"],
                        "notes": scenario["notes"],
                    }
                )
        labels_path = scenario_dir / "labels.csv"
        with open(labels_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "timestamp",
                    "frame_indices",
                    "input_type",
                    "ground_truth_action",
                    "human_motion",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(label_rows)

        per_scenario_manifest = {
            "bag_id": bag_id,
            "scenario_name": scenario_name,
            "input_mode": args.input_mode,
            "bag_path": scenario.get("bag_path", ""),
            "extracted_dir": str(extracted_info["root_dir"]),
            "labels_path": str(labels_path),
            "single_samples_path": str(single_samples_path),
            "sequence_samples_path": str(sequence_samples_path),
        }
        with open(scenario_dir / "scenario_run.json", "w", encoding="utf-8") as handle:
            json.dump(per_scenario_manifest, handle, indent=2, sort_keys=True)

        # Group methods by wrapper backend (geometry / internvl / qwen) so that
        # methods sharing a GPU run serially within a group while independent
        # groups run in parallel threads. I/O-bound (HTTP) so threading is fine.
        method_groups = {}
        for method_id in method_ids:
            spec = METHOD_REGISTRY[method_id]
            key = spec.wrapper_group or spec.kind
            method_groups.setdefault(key, []).append(method_id)

        def run_method_group(group_method_ids):
            group_rows = []
            for method_id in group_method_ids:
                method_spec = METHOD_REGISTRY[method_id]
                samples = single_samples if method_spec.input_type == "single" else sequence_samples
                samples_jsonl = single_samples_path if method_spec.input_type == "single" else sequence_samples_path
                wrapper_url = ""
                if method_spec.kind == "vlm":
                    wrapper_base_url = default_wrapper_base_url(method_spec.wrapper_group)
                    wrapper_url = join_url(wrapper_base_url, method_spec.endpoint_path)

                prediction_path = predictions_dir / f"{method_id}.jsonl"
                rows = run_predictions(
                    samples=samples,
                    samples_jsonl=str(samples_jsonl),
                    method_spec=method_spec,
                    wrapper_url=wrapper_url,
                    geometry_stop_threshold=args.geometry_stop_threshold,
                    request_timeout_sec=args.request_timeout_sec,
                    checkpoint_path=prediction_path,
                )

                bag_summary = summarize_bag_predictions(rows, scenario["expected_action"])
                group_rows.append(
                    {
                        "bag_id": bag_id,
                        "scenario_name": scenario_name,
                        "scenario_type": scenario["scenario_type"],
                        "primary_case": scenario["primary_case"],
                        "expected_action": scenario["expected_action"],
                        "method": method_id,
                        "method_category": method_spec.category,
                        "model_name": method_spec.model_name,
                        "prediction": bag_summary["predicted_action_summary"],
                        "correct_action": bag_summary["correct_action"],
                        "prediction_coverage": bag_summary["prediction_coverage"],
                        "avg_inference_latency_sec": bag_summary["avg_inference_latency_sec"],
                        "unavailable_reason": bag_summary["unavailable_reason"],
                        "stop_rate": bag_summary["stop_rate"],
                        "forward_rate": bag_summary["forward_rate"],
                        "left_rate": bag_summary["left_rate"],
                        "right_rate": bag_summary["right_rate"],
                        "review_rate": bag_summary["review_rate"],
                        "avg_original_num_images": bag_summary["avg_original_num_images"],
                        "avg_num_images": bag_summary["avg_num_images"],
                        "sequence_sampling_modes": ",".join(bag_summary["sequence_sampling_modes"]),
                        "notes": scenario["notes"],
                        "artifact_path": str(prediction_path),
                    }
                )
            return group_rows

        with ThreadPoolExecutor(max_workers=max(1, len(method_groups))) as executor:
            futures = [executor.submit(run_method_group, ids) for ids in method_groups.values()]
            for future in futures:
                aggregate_rows.extend(future.result())

    reporting_rows = (
        aggregate_rows
        if args.include_legacy_prompts
        else [row for row in aggregate_rows if row["method_category"] == "primary"]
    )
    primary_case_rows = [row for row in reporting_rows if bool(row["primary_case"])]
    review_case_rows = [row for row in reporting_rows if not bool(row["primary_case"])]
    with open(output_dir / "aggregate_results.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "include_legacy_prompts": args.include_legacy_prompts,
                "rows": reporting_rows,
                "primary_case_rows": primary_case_rows,
                "review_case_rows": review_case_rows,
                "primary_method_rows": [row for row in aggregate_rows if row["method_category"] == "primary"],
                "legacy_method_rows": [row for row in aggregate_rows if row["method_category"] != "primary"],
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    write_aggregate_csv(output_dir / "aggregate_results.csv", reporting_rows)
    write_aggregate_csv(output_dir / "primary_cases_summary.csv", primary_case_rows)
    write_aggregate_csv(output_dir / "review_cases_summary.csv", review_case_rows)
    write_scenario_analysis(output_dir, reporting_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "input_mode": args.input_mode,
                "num_scenarios": len(scenarios),
                "num_result_rows": len(reporting_rows),
                "manifest": str(Path(args.manifest).resolve()),
                "aggregate_results_csv": str(output_dir / "aggregate_results.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_labels_csv(path):
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[row["sample_id"]] = row
    return labels


def compute_direction_metrics(rows):
    """Evaluate the crossing-direction rule for internal consistency.

    A prediction is considered for rule-consistency scoring only when:
      - motion contains ``crossing`` (leftward / rightward / generic),
      - response_json provides ``crossing_direction`` as leftward or rightward,
      - predicted_action is LEFT or RIGHT.

    Expected mapping:
      crossing_rightward -> LEFT
      crossing_leftward  -> RIGHT

    Also reports:
      - direction_activation_rate : fraction of crossing samples where the
        predicted action is LEFT or RIGHT (i.e. the rule ``fired`` rather than
        collapsing to STOP/FORWARD/REVIEW).
      - crossing_detection_rate   : fraction of ALL samples where motion
        contains ``crossing``.

    Operates on prediction rows as written by ``run_predictions``; expects
    ``response_json`` and ``predicted_action`` fields. Ground truth is not
    used — this metric measures internal rule consistency only.
    """
    n_all = 0
    n_crossing = 0
    n_valid = 0
    n_correct = 0
    n_activated = 0
    for row in rows:
        if not row.get("success"):
            continue
        n_all += 1
        rj = row.get("response_json") or {}
        motion = str(rj.get("motion") or "").lower()
        if "crossing" not in motion:
            continue
        n_crossing += 1
        pred = normalize_action(row.get("predicted_action"))
        if pred in ("LEFT", "RIGHT"):
            n_activated += 1
        # Derive direction from either the dedicated field or the motion label.
        cross_dir = str(rj.get("crossing_direction") or "").lower()
        if cross_dir not in ("leftward", "rightward"):
            if motion == "crossing_leftward":
                cross_dir = "leftward"
            elif motion == "crossing_rightward":
                cross_dir = "rightward"
        if cross_dir not in ("leftward", "rightward"):
            continue
        if pred not in ("LEFT", "RIGHT"):
            continue
        n_valid += 1
        expected = "RIGHT" if cross_dir == "leftward" else "LEFT"
        if pred == expected:
            n_correct += 1
    return {
        "n_all_successful": n_all,
        "n_crossing": n_crossing,
        "n_direction_activated": n_activated,
        "n_direction_valid": n_valid,
        "n_direction_correct": n_correct,
        "direction_rule_consistency": (n_correct / n_valid) if n_valid else None,
        "direction_activation_rate": (n_activated / n_crossing) if n_crossing else None,
        "crossing_detection_rate": (n_crossing / n_all) if n_all else None,
    }


def summarize_prediction_file(predictions_path, labels, safety_distance_threshold):
    rows = read_jsonl(predictions_path)
    method_name = rows[0]["method"] if rows else Path(predictions_path).stem
    method_category = rows[0].get("method_category", "primary") if rows else "primary"
    model_name = rows[0].get("model_name", "") if rows else ""
    supports_action_evaluation = (
        bool(rows[0].get("supports_action_evaluation", True)) if rows else True
    )
    labeled_rows = []
    successful_rows = []
    correct_rows = []
    confusion = {gt: {pred: 0 for pred in ACTIONS} for gt in ACTIONS}
    predicted_counts = Counter()
    ground_truth_counts = Counter()
    unsafe_forward_count = 0
    close_obstacle_count = 0
    gt_stop_count = 0
    gt_forward_count = 0
    unsafe_forward_gt_count = 0
    unnecessary_stop_count = 0
    review_prediction_count = 0
    failure_reasons = Counter()

    for row in rows:
        label = labels.get(row["sample_id"])
        if label is None:
            continue
        gt_action = normalize_action(label.get("ground_truth_action"))
        if supports_action_evaluation and gt_action is None:
            continue
        labeled_rows.append(row)
        if not row.get("success"):
            if row.get("failure_reason"):
                failure_reasons[row["failure_reason"]] += 1
            continue
        successful_rows.append(row)
        if not supports_action_evaluation:
            continue
        pred_action = normalize_action(row.get("predicted_action"))
        if pred_action is None:
            continue
        ground_truth_counts[gt_action] += 1
        predicted_counts[pred_action] += 1
        confusion[gt_action][pred_action] += 1
        if gt_action == pred_action:
            correct_rows.append(row)
        if pred_action == "REVIEW":
            review_prediction_count += 1
        if gt_action == "STOP":
            gt_stop_count += 1
            if pred_action == "FORWARD":
                unsafe_forward_gt_count += 1
        if gt_action == "FORWARD":
            gt_forward_count += 1
            if pred_action == "STOP":
                unnecessary_stop_count += 1

        front_dist = row.get("front_dist")
        if safety_distance_threshold is not None and front_dist is not None:
            if float(front_dist) < safety_distance_threshold:
                close_obstacle_count += 1
                if pred_action == "FORWARD":
                    unsafe_forward_count += 1

    latencies = [float(row["latency_sec"]) for row in rows if row.get("latency_sec") is not None]
    unavailable_reason = ""
    if rows and not successful_rows and failure_reasons == Counter({"missing_front_dist": len(labeled_rows)}):
        unavailable_reason = (
            "Geometry baseline unavailable for this dataset because no LiDAR/front-distance topic was recorded."
        )
    summary = {
        "method": method_name,
        "category": method_category,
        "model_name": model_name,
        "supports_action_evaluation": supports_action_evaluation,
        "prediction_file": str(Path(predictions_path).resolve()),
        "num_predictions": len(rows),
        "num_labeled_samples": len(labeled_rows),
        "num_successful_predictions": len(successful_rows),
        "failure_count": len([row for row in rows if not row.get("success")]),
        "prediction_coverage": (len(successful_rows) / len(rows)) if rows else 0.0,
        "action_accuracy": (
            (len(correct_rows) / len(successful_rows)) if (successful_rows and supports_action_evaluation) else None
        ),
        "avg_inference_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
        "predicted_action_counts": dict(predicted_counts),
        "ground_truth_action_counts": dict(ground_truth_counts),
        "confusion_matrix": confusion,
        "prompt_names": sorted({row.get("prompt_name", "") for row in rows if row.get("prompt_name")}),
        "failure_reason_counts": dict(failure_reasons),
        "unavailable_reason": unavailable_reason,
        "unsafe_forward_gt_count": unsafe_forward_gt_count,
        "unsafe_forward_gt_rate": (unsafe_forward_gt_count / gt_stop_count) if gt_stop_count else None,
        "unnecessary_stop_count": unnecessary_stop_count,
        "unnecessary_stop_rate": (unnecessary_stop_count / gt_forward_count) if gt_forward_count else None,
        "review_prediction_count": review_prediction_count,
        "review_prediction_rate": (
            review_prediction_count / len(successful_rows) if successful_rows else None
        ),
        **compute_direction_metrics(rows),
        "stop_rate": (
            predicted_counts.get("STOP", 0) / len(successful_rows) if successful_rows else None
        ),
        "forward_rate": (
            predicted_counts.get("FORWARD", 0) / len(successful_rows) if successful_rows else None
        ),
        "left_rate": (
            predicted_counts.get("LEFT", 0) / len(successful_rows) if successful_rows else None
        ),
        "right_rate": (
            predicted_counts.get("RIGHT", 0) / len(successful_rows) if successful_rows else None
        ),
    }
    if safety_distance_threshold is not None:
        summary["safety_distance_threshold"] = safety_distance_threshold
        summary["close_obstacle_count"] = close_obstacle_count
        summary["unsafe_forward_count"] = unsafe_forward_count
        summary["unsafe_forward_rate"] = (
            unsafe_forward_count / close_obstacle_count if close_obstacle_count else None
        )
    return summary


def write_summary_csv(path, summaries):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "category",
                "model_name",
                "num_predictions",
                "num_labeled_samples",
                "num_successful_predictions",
                "failure_count",
                "prediction_coverage",
                "action_accuracy",
                "avg_inference_latency_sec",
                "unsafe_forward_count",
                "unsafe_forward_rate",
                "unsafe_forward_gt_count",
                "unsafe_forward_gt_rate",
                "unnecessary_stop_count",
                "unnecessary_stop_rate",
                "review_prediction_count",
                "review_prediction_rate",
                "stop_rate",
                "forward_rate",
                "left_rate",
                "right_rate",
                "unavailable_reason",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "method": summary["method"],
                    "category": summary["category"],
                    "model_name": summary["model_name"],
                    "num_predictions": summary["num_predictions"],
                    "num_labeled_samples": summary["num_labeled_samples"],
                    "num_successful_predictions": summary["num_successful_predictions"],
                    "failure_count": summary["failure_count"],
                    "prediction_coverage": summary["prediction_coverage"],
                    "action_accuracy": summary["action_accuracy"],
                    "avg_inference_latency_sec": summary["avg_inference_latency_sec"],
                    "unsafe_forward_count": summary.get("unsafe_forward_count"),
                    "unsafe_forward_rate": summary.get("unsafe_forward_rate"),
                    "unsafe_forward_gt_count": summary.get("unsafe_forward_gt_count"),
                    "unsafe_forward_gt_rate": summary.get("unsafe_forward_gt_rate"),
                    "unnecessary_stop_count": summary.get("unnecessary_stop_count"),
                    "unnecessary_stop_rate": summary.get("unnecessary_stop_rate"),
                    "review_prediction_count": summary.get("review_prediction_count"),
                    "review_prediction_rate": summary.get("review_prediction_rate"),
                    "stop_rate": summary.get("stop_rate"),
                    "forward_rate": summary.get("forward_rate"),
                    "left_rate": summary.get("left_rate"),
                    "right_rate": summary.get("right_rate"),
                    "unavailable_reason": summary.get("unavailable_reason", ""),
                }
            )


def print_terminal_summary(summaries):
    print("")
    print("Method Comparison")
    print(
        f"{'method':36s} {'acc':>8s} {'cov':>6s} "
        f"{'stop':>6s} {'fwd':>6s} {'left':>6s} {'right':>6s} {'review':>7s} "
        f"{'unsafe_fwd':>11s} {'unnec_stop':>11s} {'latency_s':>10s}"
    )
    for summary in summaries:
        latency = summary["avg_inference_latency_sec"]
        latency_text = f"{latency:.3f}" if latency is not None else "n/a"
        acc = summary["action_accuracy"]
        acc_text = f"{acc:.3f}" if acc is not None else "n/a"
        print(
            f"{summary['method']:36s} "
            f"{acc_text:>8s} "
            f"{summary['prediction_coverage']:.3f} "
            f"{format_rate(summary.get('stop_rate')):>6s} "
            f"{format_rate(summary.get('forward_rate')):>6s} "
            f"{format_rate(summary.get('left_rate')):>6s} "
            f"{format_rate(summary.get('right_rate')):>6s} "
            f"{format_rate(summary.get('review_prediction_rate')):>7s} "
            f"{format_rate(summary.get('unsafe_forward_gt_rate')):>11s} "
            f"{format_rate(summary.get('unnecessary_stop_rate')):>11s} "
            f"{latency_text:>10s}"
        )


def summarize_scenario_decisions(reporting_rows):
    grouped = {}
    for row in reporting_rows:
        grouped.setdefault(row["bag_id"], []).append(row)

    summaries = []
    for bag_id, rows in sorted(grouped.items()):
        predictions_by_method = {}
        predicted_actions = []
        for row in rows:
            prediction = normalize_action(row.get("prediction"))
            predictions_by_method[row["method"]] = prediction or ""
            if prediction:
                predicted_actions.append(prediction)
        action_counts = Counter(predicted_actions)
        consensus_action = action_counts.most_common(1)[0][0] if action_counts else ""
        consensus_vote_count = action_counts.most_common(1)[0][1] if action_counts else 0
        expected_action = normalize_action(rows[0].get("expected_action"))
        primary_case = bool_from_value(rows[0].get("primary_case"))
        summaries.append(
            {
                "bag_id": bag_id,
                "scenario_name": rows[0]["scenario_name"],
                "scenario_type": rows[0]["scenario_type"],
                "primary_case": primary_case,
                "expected_action": rows[0]["expected_action"],
                "consensus_action": consensus_action,
                "consensus_vote_count": consensus_vote_count,
                "num_methods": len(rows),
                "num_nonempty_predictions": len(predicted_actions),
                "num_distinct_predictions": len(action_counts),
                "disagreement": len(action_counts) > 1,
                "consensus_matches_expected": (
                    consensus_action == expected_action if consensus_action and expected_action else None
                ),
                "review_case_without_review": (
                    (not primary_case) and bool(action_counts) and ("REVIEW" not in action_counts)
                ),
                "predictions_by_method": predictions_by_method,
            }
        )
    return summaries


def write_scenario_analysis(output_dir, reporting_rows):
    scenario_rows = summarize_scenario_decisions(reporting_rows)
    with open(output_dir / "scenario_analysis.json", "w", encoding="utf-8") as handle:
        json.dump({"rows": scenario_rows}, handle, indent=2, sort_keys=True)

    method_names = sorted({row["method"] for row in reporting_rows})
    fieldnames = [
        "bag_id",
        "scenario_name",
        "scenario_type",
        "primary_case",
        "expected_action",
        "consensus_action",
        "consensus_vote_count",
        "num_methods",
        "num_nonempty_predictions",
        "num_distinct_predictions",
        "disagreement",
        "consensus_matches_expected",
        "review_case_without_review",
    ] + method_names
    with open(output_dir / "scenario_analysis.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scenario_rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key not in row["predictions_by_method"]},
                    **row["predictions_by_method"],
                }
            )


def evaluate(args):
    labels = load_labels_csv(args.labels_csv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = [
        summarize_prediction_file(path, labels, args.safety_distance_threshold)
        for path in args.predictions_jsonl
    ]

    primary_summaries = [summary for summary in all_summaries if summary["category"] == "primary"]
    legacy_summaries = [summary for summary in all_summaries if summary["category"] != "primary"]
    reporting_summaries = all_summaries if args.include_legacy_prompts else primary_summaries

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "reporting_methods": reporting_summaries,
                "primary_methods": primary_summaries,
                "legacy_methods": legacy_summaries,
                "include_legacy_prompts": args.include_legacy_prompts,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    write_summary_csv(output_dir / "summary.csv", reporting_summaries)
    if legacy_summaries:
        write_summary_csv(output_dir / "summary_primary.csv", primary_summaries)
        write_summary_csv(output_dir / "summary_all.csv", all_summaries)
    print_terminal_summary(reporting_summaries)
    print("")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Summary CSV:  {output_dir / 'summary.csv'}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-samples")
    build_parser.add_argument("frames_jsonl")
    build_parser.add_argument("output")
    build_parser.add_argument("--input-type", choices=["single", "sequence"], required=True)
    build_parser.add_argument("--sequence-length", type=int, default=3)
    build_parser.add_argument("--sequence-stride", type=int, default=1)
    build_parser.add_argument(
        "--sequence-sampling-mode",
        choices=list(SEQUENCE_SAMPLING_MODES),
        default="none",
        help=(
            "Subsample frames inside each sequence window. "
            "'uniform' keeps every --subsample-rate frame, "
            "'capped' keeps --max-frames-per-sequence evenly spaced frames, "
            "'tail' keeps the last --max-frames-per-sequence frames "
            "(preserves late-entry signal). Default 'none' keeps legacy behavior."
        ),
    )
    build_parser.add_argument("--subsample-rate", type=int, default=0,
        help="Used with --sequence-sampling-mode=uniform; keep every Nth frame.")
    build_parser.add_argument("--max-frames-per-sequence", type=int, default=0,
        help="Used with --sequence-sampling-mode=capped/tail; cap frames per sample.")
    build_parser.set_defaults(func=build_samples)

    labels_parser = subparsers.add_parser("write-label-template")
    labels_parser.add_argument("samples_jsonl")
    labels_parser.add_argument("output")
    labels_parser.set_defaults(func=write_label_template)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("samples_jsonl")
    predict_parser.add_argument("output")
    predict_parser.add_argument(
        "--method",
        choices=sorted(list(METHOD_REGISTRY.keys()) + ["single_vlm", "sequence_vlm"]),
        required=True,
    )
    predict_parser.add_argument("--geometry-stop-threshold", type=float, default=1.0)
    predict_parser.add_argument("--wrapper-url", default="http://localhost:8000/analyze_navigation")
    predict_parser.add_argument("--wrapper-base-url", default="")
    predict_parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    predict_parser.set_defaults(func=predict)

    benchmark_parser = subparsers.add_parser("run-benchmark")
    benchmark_parser.add_argument("--single-samples-jsonl", required=True)
    benchmark_parser.add_argument("--sequence-samples-jsonl", required=True)
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--geometry-stop-threshold", type=float, default=1.0)
    benchmark_parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    benchmark_parser.add_argument(
        "--include-legacy-prompts",
        action="store_true",
        help=(
            "Include legacy/debug-only prompts in the generated prediction suite. "
            "By default the formal benchmark excludes InternVL legacy prompts because "
            "they produced unstable no-person false positives."
        ),
    )
    benchmark_parser.set_defaults(func=run_benchmark)

    manifest_parser = subparsers.add_parser("run-manifest")
    manifest_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_SCENARIO_MANIFEST_PATH),
        help="Scenario manifest JSON describing rosbag or extracted-data benchmark cases.",
    )
    manifest_parser.add_argument("--output-dir", required=True)
    manifest_parser.add_argument(
        "--input-mode",
        choices=["rosbag", "extracted"],
        default="rosbag",
        help=(
            "Read directly from rosbag files or from pre-extracted frame datasets. "
            "Use extracted mode to run the benchmark without ROS installed."
        ),
    )
    manifest_parser.add_argument(
        "--extracted-root",
        default="",
        help=(
            "Optional root directory containing extracted datasets under <root>/<bag_id>/. "
            "Used only in extracted mode when a scenario does not define extracted_dir."
        ),
    )
    manifest_parser.add_argument("--image-topic", default="/camera_face/left/image_raw")
    manifest_parser.add_argument("--scan-topic", default="/scan_odom")
    manifest_parser.add_argument("--image-every-n", type=int, default=1)
    manifest_parser.add_argument("--max-frames", type=int, default=0)
    manifest_parser.add_argument("--max-scan-age-sec", type=float, default=0.5)
    manifest_parser.add_argument("--front-sector-half-angle-deg", type=float, default=10.0)
    manifest_parser.add_argument("--front-distance-percentile", type=float, default=20.0)
    manifest_parser.add_argument("--max-range-fallback", type=float, default=10.0)
    manifest_parser.add_argument("--sequence-length", type=int, default=5)
    manifest_parser.add_argument("--sequence-stride", type=int, default=1)
    manifest_parser.add_argument(
        "--sequence-sampling-mode",
        choices=list(SEQUENCE_SAMPLING_MODES),
        default="none",
        help=(
            "Subsample frames inside each sequence window before sending to the VLM. "
            "'uniform' keeps every --subsample-rate frame, "
            "'capped' keeps --max-frames-per-sequence evenly spaced frames, "
            "'tail' keeps the last --max-frames-per-sequence frames "
            "(preserves late-entry signal; recommended when interaction often "
            "occurs near the end of the clip). Default 'none' preserves "
            "legacy behavior (all frames in window sent)."
        ),
    )
    manifest_parser.add_argument("--subsample-rate", type=int, default=0,
        help="Used with --sequence-sampling-mode=uniform; keep every Nth frame.")
    manifest_parser.add_argument("--max-frames-per-sequence", type=int, default=0,
        help="Used with --sequence-sampling-mode=capped/tail; cap frames per sample.")
    manifest_parser.add_argument("--geometry-stop-threshold", type=float, default=1.0)
    manifest_parser.add_argument("--request-timeout-sec", type=float, default=120.0)
    manifest_parser.add_argument(
        "--include-legacy-prompts",
        action="store_true",
        help=(
            "Include legacy/debug-only prompt methods in the manifest batch run. "
            "Default reporting stays focused on the formal primary benchmark suite."
        ),
    )
    manifest_parser.set_defaults(func=run_manifest)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--labels-csv", required=True)
    eval_parser.add_argument("--predictions-jsonl", nargs="+", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--safety-distance-threshold", type=float, default=None)
    eval_parser.add_argument(
        "--include-legacy-prompts",
        action="store_true",
        help=(
            "Show legacy/debug prompt methods in the reporting table. "
            "Default reporting stays focused on the primary benchmark suite."
        ),
    )
    eval_parser.set_defaults(func=evaluate)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
