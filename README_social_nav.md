# Social Navigation Implementation Notes

This document focuses on the code paths used for the final project's social
navigation evaluation and real-time safety story. It is intentionally narrow so
the rubric-facing implementation is easy to inspect.

## What This Adds

The project bottleneck was that binary `STOP` / `FORWARD` decisions were too
coarse for human-aware navigation, while offline VLM outputs like `LEFT` and
`RIGHT` could not be sent directly to the Go1 without a safety layer. The repo
now addresses that in two parts:

1. Offline evaluation supports the richer action space
   `STOP / FORWARD / LEFT / RIGHT / REVIEW`.
2. Real-time controllers treat `LEFT` and `RIGHT` as advisory hints by default
   unless an operator explicitly enables lateral execution.

Relevant files:

- [streaming/scripts/extract_social_nav_data.py](</Users/yuni/CSCI7000 Final Code/streaming/scripts/extract_social_nav_data.py:1>)
- [motion_control/scripts/social_nav_eval.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/social_nav_eval.py:1>)
- [social_nav_eval_prompts.py](</Users/yuni/CSCI7000 Final Code/social_nav_eval_prompts.py:1>)
- [vlm_wrapper.py](</Users/yuni/CSCI7000 Final Code/vlm_wrapper.py:1>)
- [motion_control/scripts/social_nav_controller.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/social_nav_controller.py:1>)
- [motion_control/scripts/vlm_only_controller.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/vlm_only_controller.py:1>)
- [motion_control/scripts/vlm_minimal_controller.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/vlm_minimal_controller.py:1>)
- [benchmark_runs/final_policy_v2_crossing](</Users/yuni/CSCI7000 Final Code/benchmark_runs/final_policy_v2_crossing>)

## Setup

For offline evaluation and tests, a plain Python environment is enough:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest requests fastapi pillow
```

Optional components:

- ROS is only needed for live extraction or live robot control.
- Real VLM backends are only needed when running wrapper-backed prediction jobs.

To run both wrappers locally on separate ports:

```bash
scripts/run_wrapper_internvl.sh
scripts/run_wrapper_qwen.sh
```

Expected models:

- `Qwen3-VL-30B`
- `InternVL-3.5-14B`

## Offline Evaluation

The offline pipeline is:

1. Extract frames from a rosbag with
   [streaming/scripts/extract_social_nav_data.py](</Users/yuni/CSCI7000 Final Code/streaming/scripts/extract_social_nav_data.py:1>).
2. Build single-frame and sequence samples with
   [motion_control/scripts/social_nav_eval.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/social_nav_eval.py:1>).
3. Run predictions through the wrapper or geometry baseline.
4. Evaluate with sample-level metrics and bag-level summaries.

Sequence sampling for the final crossing-focused policy uses:

```bash
python3 motion_control/scripts/social_nav_eval.py run-manifest \
  --manifest motion_control/eval/scenario_manifest_directional_avoidance.json \
  --input-mode extracted \
  --output-dir benchmark_runs/final_policy_v2_crossing \
  --sequence-length 10 \
  --sequence-sampling-mode capped \
  --max-frames-per-sequence 5
```

Meaning:

- Each candidate sequence window spans `10` ordered frames.
- At inference time, each 10-frame window is reduced to at most `5` evenly
  spaced frames before sending images to the VLM.

## Action Space

The evaluator and prompts use the action space:

- `STOP`
- `FORWARD`
- `LEFT`
- `RIGHT`
- `REVIEW`

Intended semantics:

- `LEFT` / `RIGHT` are valid offline labels and predictions.
- `REVIEW` explicitly captures uncertainty instead of forcing a wrong hard
  action.

## Metrics

The evaluator reports two rubric-friendly metrics directly:

- `action_accuracy`
  Sample-level accuracy over labeled successful predictions.
- `unsafe_forward_rate`
  Fraction of close-obstacle samples where the model still predicted
  `FORWARD`.

It also reports:

- action distribution (`stop_rate`, `forward_rate`, `left_rate`, `right_rate`)
- `review_prediction_rate`
- bag-level majority summaries via `scenario_analysis.json`

For the final crossing-focused benchmark artifacts, see:

- [benchmark_runs/final_policy_v2_crossing/aggregate_results.json](</Users/yuni/CSCI7000 Final Code/benchmark_runs/final_policy_v2_crossing/aggregate_results.json:1>)
- [benchmark_runs/final_policy_v2_crossing/scenario_analysis.json](</Users/yuni/CSCI7000 Final Code/benchmark_runs/final_policy_v2_crossing/scenario_analysis.json:1>)

## Real-Time Safety Projection

Offline VLM outputs like `LEFT` and `RIGHT` are not automatically safe to
execute on the Go1. A language model can suggest a side, but it does not
guarantee free space, kinematic feasibility, or collision clearance.

Because of that, the real-time controllers now use a safety projection:

- If the VLM returns `LEFT` or `RIGHT`, the controller keeps the side hint in
  `recommended_avoidance_side`.
- The executable action is projected to `STOP` by default.
- Lateral execution is only allowed if the operator explicitly sets
  `fsm_allow_vlm_lateral=true`.

This keeps the offline benchmark expressive while keeping the live controller
conservative on hardware.

## Tests

The pytest suite is intentionally pure Python and does not require ROS or real
VLM calls. It covers:

- action normalization
- mocked action outputs for crossing, receding, ambiguous, and blocked cases
- bag-level majority vote summaries
- unsafe-forward rate
- sequence sampling from 10 frames down to 5
- real-time safety projection of `LEFT` / `RIGHT`

Run:

```bash
pytest -q
```
