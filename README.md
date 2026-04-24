# VLM Social Navigation on Unitree Go1

This repository contains my final project on using pretrained
vision-language models to make structured social navigation decisions for a
Unitree Go1 robot.

The core question from the proposal is:

> Can a VLM support socially aware navigation decisions on Unitree Go1, and can
> that decision layer be integrated into a practical robot system?

The final implementation answers that question with an offline benchmark and a
conservative real-time controller interface:

- offline, the VLM predicts social navigation actions from front-camera image
  sequences
- online, the robot uses those predictions only through a safety projection
  layer rather than executing raw VLM outputs directly

## Project Summary

Traditional robot navigation handles geometry well, but it does not naturally
reason about human intent. In hallway scenes, a robot may need to distinguish:

- a person blocking the corridor
- a person crossing from left to right
- a person moving away from the robot
- an ambiguous late-entry case where the safest answer is to defer

To make those distinctions explicit, this project formulates navigation as a
discrete action prediction problem:

`{STOP, FORWARD, LEFT, RIGHT, REVIEW}`

Compared with a binary `STOP/FORWARD` framing, this richer action space lets
the model express lateral avoidance and uncertainty instead of collapsing
everything into overconfident halt-or-go decisions.

## What Was Built

The implemented system has four main parts:

1. A rosbag extraction pipeline for front-camera social navigation data.
2. An offline benchmark for single-image and sequence-based VLM evaluation.
3. Prompt-policy designs for Qwen and InternVL with explicit temporal social
   reasoning.
4. A real-time controller path for Go1 where VLM decisions are safety-projected
   before they can affect hardware.

Key files:

- [streaming/scripts/extract_social_nav_data.py](</Users/yuni/CSCI7000 Final Code/streaming/scripts/extract_social_nav_data.py:1>)
- [motion_control/scripts/social_nav_eval.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/social_nav_eval.py:1>)
- [social_nav_eval_prompts.py](</Users/yuni/CSCI7000 Final Code/social_nav_eval_prompts.py:1>)
- [vlm_wrapper.py](</Users/yuni/CSCI7000 Final Code/vlm_wrapper.py:1>)
- [motion_control/scripts/social_nav_controller.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/social_nav_controller.py:1>)
- [motion_control/scripts/vlm_only_controller.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/vlm_only_controller.py:1>)
- [motion_control/scripts/vlm_minimal_controller.py](</Users/yuni/CSCI7000 Final Code/motion_control/scripts/vlm_minimal_controller.py:1>)
- [README_social_nav.md](</Users/yuni/CSCI7000 Final Code/README_social_nav.md:1>)

## System Architecture

The project follows the same modular idea described in the proposal:

`Image sequence -> VLM decision -> structured action -> controller -> robot`

The important design choice is separation between:

- semantic decision-making
- low-level motion execution

That separation improves interpretability and safety. The VLM does not command
the Go1 directly. Instead, it outputs structured fields such as motion class,
crossing direction, lateral preference, uncertainty reason, and recommended
action.

## Models

The two primary VLM backends are:

- `Qwen3-VL-30B`
- `InternVL-3.5-14B`

They are accessed through the wrapper in [vlm_wrapper.py](</Users/yuni/CSCI7000 Final Code/vlm_wrapper.py:1>)
using OpenAI-compatible HTTP endpoints.

Helper scripts for local wrapper launch:

- [scripts/run_wrapper_qwen.sh](</Users/yuni/CSCI7000 Final Code/scripts/run_wrapper_qwen.sh:1>)
- [scripts/run_wrapper_internvl.sh](</Users/yuni/CSCI7000 Final Code/scripts/run_wrapper_internvl.sh:1>)

## Action Space

The final policy uses:

- `STOP`
- `FORWARD`
- `LEFT`
- `RIGHT`
- `REVIEW`

Intended meaning:

- `STOP`: forward motion is unsafe
- `FORWARD`: the corridor is clear enough to continue
- `LEFT` / `RIGHT`: an offline directional avoidance preference
- `REVIEW`: the scene is ambiguous and should not be forced into a confident
  action

## Prompt and Policy Contribution

The main contribution in the final report is not model training, but prompt and
policy design.

The final sequence prompt adds:

- temporal motion categories
- directional crossing logic
- a receding-person override
- an uncertainty fallback via `REVIEW`

This directly targets the bottleneck identified in the report:

- binary `STOP/FORWARD` navigation is not expressive enough for human-aware
  social navigation
- offline `LEFT/RIGHT` decisions cannot be sent to Go1 directly without safety
  projection

## Offline Evaluation

The benchmark runs on extracted rosbag data and compares:

- geometry baseline
- single-image VLM policies
- sequence-based VLM policies

The final crossing-focused benchmark artifacts are stored in:

- [benchmark_runs/final_policy_v2_crossing](</Users/yuni/CSCI7000 Final Code/benchmark_runs/final_policy_v2_crossing>)

Sequence evaluation settings used in the final policy:

- sequence length: `10`
- maximum frames passed to the VLM per sample: `5`
- sequence sampling mode: `capped`

Example command:

```bash
python3 motion_control/scripts/social_nav_eval.py run-manifest \
  --manifest motion_control/eval/scenario_manifest_directional_avoidance.json \
  --input-mode extracted \
  --output-dir benchmark_runs/final_policy_v2_crossing \
  --sequence-length 10 \
  --sequence-sampling-mode capped \
  --max-frames-per-sequence 5
```

## Metrics

The offline evaluator reports metrics used in the final report, including:

- sample-level action accuracy
- unsafe-forward rate
- review rate
- action distribution
- bag-level majority-vote summaries
- crossing-direction consistency diagnostics

Important outputs:

- [benchmark_runs/final_policy_v2_crossing/aggregate_results.json](</Users/yuni/CSCI7000 Final Code/benchmark_runs/final_policy_v2_crossing/aggregate_results.json:1>)
- [benchmark_runs/final_policy_v2_crossing/scenario_analysis.json](</Users/yuni/CSCI7000 Final Code/benchmark_runs/final_policy_v2_crossing/scenario_analysis.json:1>)

## Real-Time Safety Projection

This is the most important systems constraint from the final report.

Offline `LEFT` and `RIGHT` predictions are useful for analysis, but they are
not automatically safe executable robot commands. A VLM can suggest a side, but
it does not verify free space or collision clearance on hardware.

For that reason, the live controller treats lateral actions as advisory by
default:

- `LEFT` / `RIGHT` are preserved as hints in logs and structured output
- the executed command is projected to `STOP` unless the operator explicitly
  enables lateral execution

This keeps the benchmark expressive while keeping the deployed Go1 control path
conservative.

## Repository Layout

```text
streaming/          rosbag extraction and sensor utilities
motion_control/     offline evaluator and live controller scripts
benchmark_runs/     saved benchmark artifacts referenced in the final report
rtabmap/            mapping-related scripts and export helpers
scene_graph/        scene graph utilities
tests/              pytest coverage for social navigation logic
```

## Setup

For offline evaluation and tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest requests fastapi pillow
```

ROS is only required for live robot-side workflows such as bag extraction from
ROS topics or real-time controller execution.

## Tests

The social navigation tests are pure Python and do not require ROS or real VLM
calls.

Run:

```bash
python3 -m pytest -q
```

The tests cover:

- action normalization
- mocked social navigation action outputs
- bag-level majority vote summaries
- unsafe-forward rate
- sequence subsampling from 10 frames to 5
- real-time projection that prevents raw `LEFT/RIGHT` from becoming executable
  commands directly

## Current Scope and Limits

This repository demonstrates that prompt-level policy design can induce more
structured social decision behavior from pretrained VLMs in offline evaluation.

It does not yet claim a fully validated high-frequency closed-loop social
navigation controller for Go1. The current strongest claim is:

- VLMs can support structured social navigation decisions offline
- those decisions can be integrated into a robot system conservatively through a
  safety-projected controller interface

## Related Documents

- [final_proposal.mdx](/Users/yuni/vla-foundations/content/course/assignments/capstone/final_report/final_proposal.mdx)
- [final_report.mdx](</Users/yuni/CSCI7000 Final Code/final_report.mdx:1>)
- [README_social_nav.md](</Users/yuni/CSCI7000 Final Code/README_social_nav.md:1>)
