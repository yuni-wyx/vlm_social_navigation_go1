# `where-am-I` repository — guided tour

This document is an end-to-end map of the project for anyone joining it cold.
It explains what the code is for, how the pieces fit together, how the VLM
social-navigation benchmark was built and evolved, where all the artefacts
live, and how to reproduce any of the saved runs.

---

## 1. What this project is

A Unitree Go1 quadruped research stack with two related concerns:

1. **Live-robot infrastructure** — ROS-based sensor relay, camera publishers,
   SDK bridge for the Go1, teleop / motion control scaffolding, RTAB-Map
   mapping, scene graph construction, camera calibration.
2. **Offline social-navigation decision benchmarking** — a pipeline that turns
   13 recorded rosbags of front-camera footage into a reproducible VLM-driven
   action-prediction benchmark. This is where most of the recent work has
   happened.

The primary research question driving the benchmark is:

> Can a pretrained VLM be converted, via prompt-level architectural changes,
> into a structured **five-action** social-navigation policy
> `{STOP, FORWARD, LEFT, RIGHT, REVIEW}` that reasons about human motion
> temporally and expresses genuine uncertainty?

The headline technical artefact of the project is
`social_nav_eval_prompts.py` + `motion_control/scripts/social_nav_eval.py`
plus a chain of saved benchmark runs that show how the policy evolved from a
binary STOP/FORWARD collapse to a five-action directional policy with
validated direction-rule consistency.

---

## 2. Top-level directory map

```
where-am-I/
├── social_nav_eval_prompts.py       # Prompt templates + ACTIONS + PROMPT_CONFIG
├── vlm_wrapper.py                   # FastAPI wrapper that fronts vLLM backends
├── test_vlm_wrapper.sh              # Convenience script to exercise wrapper
│
├── motion_control/
│   ├── eval/
│   │   ├── scenario_manifest.json                            # 13-bag original manifest
│   │   ├── scenario_manifest_directional_avoidance.json      # directional variant
│   │   ├── scenario_manifest_targeted_4bags.json             # 4-bag activation probe
│   │   └── sample_labels_template.csv
│   └── scripts/
│       ├── social_nav_eval.py                 # Benchmark + evaluator + metrics
│       ├── run_social_nav_benchmark.py        # Reproducible batch runner
│       ├── prepare_social_nav_extracted.py    # Extract all 13 bags once
│       ├── social_nav_controller.py           # Controller-side code
│       ├── subsumption_controller.py          # Live subsumption controller
│       ├── go1_navigate.py                    # High-level navigation
│       └── sdk_udp_bridge.py                  # Go1 SDK bridge
│
├── streaming/
│   ├── scripts/
│   │   ├── extract_social_nav_data.py    # Rosbag → images/, frames.jsonl, summary
│   │   ├── fix_bag_timestamps.py
│   │   ├── rgb_pub_ffmpeg.py / rgb_publisher.py
│   │   ├── record.sh / replay.sh
│   │   └── sensor_relay.py
│   └── bags/                              # Live-recording area (gitignored)
│
├── rosbag/                          # 13 recorded Go1 bags (1.bag … 13.bag)
│
├── extracted_social_nav/
│   └── bag_<id>/
│       ├── images/                  # PPM frames
│       ├── frames.jsonl             # subsampled row per frame
│       ├── frames.jsonl.full        # full-resolution backup
│       └── extraction_summary.json
│
├── benchmark_runs/                  # saved benchmark runs, see §5
│   ├── go1_social_nav_apr18_extracted/              # first successful end-to-end
│   ├── go1_social_nav_apr18_extracted_scores/       # scored apr18 run
│   ├── go1_social_nav_rerun_parallel/               # parallel-lanes verification
│   ├── final_policy_v1/                             # tail=3 policy run (no crossing)
│   ├── final_policy_v2_crossing/                    # new-prompt crossing-aware run
│   ├── final_policy_directional_labels_v1/          # directional-labels re-eval
│   ├── targeted_directional_activation_test/        # 4-bag fresh activation probe
│   └── prompt_experiments/
│       ├── results/                 # A/B + receding-fix + crossing probe JSON
│       ├── scripts/                 # the experiment scripts that produced them
│       └── prompt_snapshots/        # prompt text at specific moments
│
├── rtabmap/                         # RTAB-Map offline mapping outputs / scripts
├── scene_graph/                     # Scene-graph extraction from mapped scene
├── camera_calib/                    # Stereo calibration + rectification pipeline
├── llm_controller/                  # LLM-driven high-level controller experiments
├── docs/                            # Project docs
├── scripts/                         # Miscellaneous scripts
│
├── final_report.mdx                 # Capstone write-up (this project)
├── report_eval.md                   # Detailed evaluation-run report
└── README.md                        # Project-level README
```

---

## 3. The social-navigation pipeline end-to-end

### 3.1 From bags to extracted datasets

`streaming/scripts/extract_social_nav_data.py` reads one rosbag and writes a
standardized per-bag directory structure:

```
extracted_social_nav/<bag_id>/
├── images/                 # PPM frames (one file per saved frame)
├── frames.jsonl            # per-frame record: image_path, timestamp, scan_age, front_dist
└── extraction_summary.json
```

`motion_control/scripts/prepare_social_nav_extracted.py` iterates the manifest
and calls the extractor once per bag so all 13 bags are prepared in a single
pass. Extraction requires a ROS-capable Python environment (for `rosbag`). The
repo hard-codes the image topic `/camera_face/left/image_raw` by default; the
actual bags use `/camera/image_raw`, so `--image-topic /camera/image_raw` is
required on first extraction.

The repo keeps every 5th frame (`frames.jsonl`) as the default subsample,
with the full frame list preserved at `frames.jsonl.full`. That 1/5 subsample
exists because running all ~6,400 raw frames through the VLMs was projected at
~28 hours, whereas 1,290 subsampled frames run in ~3 hours.

### 3.2 Scenario manifests

`motion_control/eval/scenario_manifest.json` is the canonical 13-bag
specification. Each scenario carries:

- `bag_id`, `bag_path`, `extracted_dir`
- `scenario_name`, `scenario_type`, `notes`
- `expected_action` — one of `STOP`/`FORWARD`/`REVIEW` (conservative rubric)
- `human_presence`, `human_position`, `human_motion`
- `primary_case` — whether the bag contributes to headline accuracy

Variant manifests:

| manifest | purpose |
|---|---|
| `scenario_manifest.json` | conservative safety labels (original) |
| `scenario_manifest_directional_avoidance.json` | relabels bag_03/04/05/07 to LEFT/RIGHT; preserves the original `expected_action` in a new `original_expected_action` field, adds `directional_avoidance_rationale` per bag |
| `scenario_manifest_targeted_4bags.json` | subset (bag_03, 04, 05, 07) used for the capability-vs-activation probe |

### 3.3 The wrapper and the backends

`vlm_wrapper.py` is a thin FastAPI service that:

- Exposes `GET /health` and `POST /analyze_navigation`.
- Looks up `prompt_name` in `PROMPT_CONFIG` (imported from
  `social_nav_eval_prompts.py`) — this is the critical line that decides
  which prompt text is sent to the backend.
- Decodes base64 images, resizes to max width 512, re-encodes as JPEG
  (quality 60).
- Builds an OpenAI-compatible chat payload and POSTs it to the vLLM backend
  configured via `VLM_BASE_URL` / `VLM_MODEL`.
- Parses the first JSON object from the model response and returns it in a
  structured response envelope.

Two wrapper instances run in the deployed setup, each fronting a different
model:

- `10.157.141.10:8100` → `OpenGVLab/InternVL3_5-14B-HF` on backend
  `10.157.141.181:8000`.
- `10.157.141.10:8101` → `Qwen3-VL-30B` on backend `10.157.141.181:8001`.

**Critical operational note:** the wrapper uses its own in-process copy of
`PROMPT_CONFIG`, so local edits to `social_nav_eval_prompts.py` do not
propagate to the wrapper host until that file is re-deployed and the wrapper
is restarted. A session in this project lost several hours to that subtlety
before the wrapper was redeployed with the new five-action prompt.

### 3.4 The benchmark runner

`motion_control/scripts/social_nav_eval.py` is the backbone. Subcommands:

| subcommand | purpose |
|---|---|
| `build-samples` | Build single-image or sequence samples from `frames.jsonl` |
| `write-label-template` | Emit a CSV label template from samples |
| `predict` | Run one method / compatibility alias on samples |
| `run-benchmark` | Run the default primary benchmark suite |
| `run-manifest` | End-to-end manifest run (primary workflow) |
| `evaluate` | Compare prediction files against labels |

Key classes / functions to know:

- `ACTION_ALIASES` — normalizes model-emitted variants into the canonical
  five actions (handles `GO_LEFT`, `SIDESTEP_RIGHT`, `WAIT`, `HOLD`, etc.).
- `normalize_action(value)` — the gate that rejects off-schema outputs.
- `MethodSpec` + `METHOD_REGISTRY` — declares each method
  (id, kind, input_type, wrapper_group, endpoint_path, prompt_name).
- `subsample_sequence_indices(indices, mode, rate, max_frames)` — supports
  the sampling modes `none`, `uniform`, `capped`, `tail`. Used to control
  how many frames per sequence sample actually reach the model.
- `build_samples_from_frames(…)` — produces sample rows with deterministic
  `sample_id`s like `sequence_000123`. These IDs are the key that makes
  checkpoint-resume and label-only reruns possible.
- `run_predictions(samples, …, checkpoint_path)` — streams per-sample rows
  to disk, skips rows with `sample_id` already present in the checkpoint,
  and flushes per write. This is what makes the runner safe against wrapper
  outages.
- `run_manifest(args)` — the function driven by the `run-manifest`
  subcommand. Iterates scenarios, builds samples, groups methods by wrapper
  backend (`geometry` / `internvl` / `qwen`), and runs the groups in
  parallel threads using `ThreadPoolExecutor`. Per-bag sync at the end of
  each bag.
- `compute_direction_metrics(rows)` — the custom metric added to evaluate
  **internal consistency** of the crossing-direction rule (does the model's
  `recommended_action` match the opposite side of its own
  `crossing_direction`?). Returns `direction_rule_consistency`,
  `direction_activation_rate`, `crossing_detection_rate`.
- `summarize_prediction_file(path, labels, threshold)` — per-method summary
  with STOP/FORWARD/LEFT/RIGHT/REVIEW rates, unsafe-forward rates,
  unnecessary-stop rates, REVIEW prediction rate, and the direction metrics.
- `write_aggregate_csv`, `write_scenario_analysis`, `print_terminal_summary`
  — the final outputs.

Two architectural properties matter for operations:

1. **Parallelism by wrapper group.** InternVL (:8100) and Qwen (:8101) are
   independent GPUs, so the runner dispatches their methods concurrently.
   Within a group methods run serially because they share a GPU.
   `geometry` runs in a third lane with negligible cost.
2. **Streaming checkpoints.** Every successful prediction is flushed to disk
   before the next is issued. A kill or an outage costs at most one sample.
   On restart, the runner reads existing rows, drops any truncated trailing
   line, and continues.

### 3.5 Prompts — the heart of the delta

`social_nav_eval_prompts.py` is short but load-bearing. Three prompt templates:

- `PROMPT_SINGLE_IMAGE` — one-image decision with `person_detected`,
  `person_position`, `path_blocked`, `safer_lateral_side`,
  `uncertainty_reason`, `recommended_action`.
- `PROMPT_SEQUENCE_IMAGES` — the long one. Lays out:
  - A temporal motion taxonomy: `approaching`, `receding`, `crossing_leftward`,
    `crossing_rightward`, `crossing`, `entering_late`, `stationary`, `none`.
  - A **RECEDING CASE RULE** block that overrides default-STOP when the
    latest frame shows a clear path.
  - A **CROSSING DIRECTION RULE** block mapping
    `crossing_leftward → RIGHT`, `crossing_rightward → LEFT`, and
    `crossing + unknown direction → REVIEW`.
  - Six ordered decision rules with priorities
    REVIEW → FORWARD → receding-FORWARD → crossing → approaching/blocking →
    STOP.
  - A structured JSON response schema that includes `motion`,
    `crossing_direction`, `path_blocked_latest_frame`, `safer_lateral_side`,
    `recommended_avoidance_side`, `risk_level`, `uncertainty_reason`,
    `recommended_action`.
- `PROMPT_STRUCTURED_LOCALIZATION` — legacy localization probe (kept for
  compatibility in `METHOD_REGISTRY` but excluded from the primary run).

Two earlier snapshots of this file are preserved as text at
`benchmark_runs/prompt_experiments/prompt_snapshots/`, which lets future
analyses trace exactly when each rule was introduced.

---

## 4. Methods and metrics

Five methods are evaluated in the primary benchmark:

| method | kind | input | wrapper | prompt_name | notes |
|---|---|---|---|---|---|
| `geometry` | rule | last frame's `front_dist` | — | — | STOP if `front_dist < --geometry-stop-threshold`. Needs `/scan_odom` which this bag set lacks, so the baseline is unavailable. |
| `internvl_single_image_navigation` | VLM | 1 image | internvl | `single_image_navigation` | Single-frame decision |
| `internvl_sequence_image_navigation` | VLM | sequence | internvl | `sequence_image_navigation` | Sequence decision |
| `qwen_single_image_navigation` | VLM | 1 image | qwen | `single_image_navigation` | |
| `qwen_sequence_image_navigation` | VLM | sequence | qwen | `sequence_image_navigation` | |

Behavioural metrics (computed per method, surfaced in `summary.csv` and in
`aggregate_results.csv`):

- `stop_rate`, `forward_rate`, `left_rate`, `right_rate`, `review_rate`
- `prediction_coverage` (fraction of rows with `success=True`)
- `avg_inference_latency_sec`

Safety metrics:

- `unsafe_forward_gt_rate` — fraction of STOP-labelled frames where the model
  said FORWARD.
- `unnecessary_stop_rate` — fraction of FORWARD-labelled frames where the
  model said STOP.
- `review_prediction_rate` — fraction of all successful predictions where
  action is REVIEW.

Direction metrics (the custom `compute_direction_metrics`):

- `direction_rule_consistency` — internal: of all crossing frames where the
  model committed both a direction AND a lateral action, what fraction chose
  the opposite side.
- `direction_activation_rate` — of crossing-tagged frames, what fraction
  actually emitted a lateral action.
- `crossing_detection_rate` — of all successful predictions, what fraction
  were tagged as `crossing_*`.

Scenario analysis (`scenario_analysis.csv/json`): per-bag consensus action
across methods, disagreement flag, whether the consensus matches the expected
action, and whether a review case got a REVIEW prediction from any method.

---

## 5. Inventory of saved benchmark runs

Each run sits under `benchmark_runs/<run-name>/` with the same structure:

```
<run-name>/
├── aggregate_results.csv      # one row per (bag × method)
├── aggregate_results.json
├── primary_cases_summary.csv  # subset where primary_case=True
├── review_cases_summary.csv   # subset where primary_case=False
├── scenario_analysis.csv      # per-bag consensus + methods-by-bag table
├── scenario_analysis.json
└── bag_<id>/
    ├── labels.csv
    ├── scenario_run.json
    ├── single_samples.jsonl
    ├── sequence_samples.jsonl
    └── predictions/
        ├── geometry.jsonl
        ├── internvl_sequence_image_navigation.jsonl
        ├── internvl_single_image_navigation.jsonl
        ├── qwen_sequence_image_navigation.jsonl
        └── qwen_single_image_navigation.jsonl
```

### Chronological lineage

| run | date | purpose | config |
|---|---|---|---|
| `go1_social_nav_apr18_extracted` | Apr 18 | first working end-to-end; image-topic fix | seq_len=5, no subsample |
| `go1_social_nav_apr18_extracted_scores` | Apr 18–19 | scored run against real backends | seq_len=5, tail=3 |
| `go1_social_nav_rerun_parallel` | Apr 19 | verify parallel wrapper-group lanes | seq_len=5, none |
| `final_policy_v1` | Apr 19 | tail-subsample policy (no crossing rule yet) | seq_len=5, tail=3 |
| `final_policy_v2_crossing` | Apr 20–21 | primary new-prompt run (receding + crossing rules, wrapper redeployed) | seq_len=10, capped=5 |
| `final_policy_directional_labels_v1` | Apr 21 | label-only reuse of v2 predictions against directional manifest | seq_len=10, capped=5 |
| `targeted_directional_activation_test` | Apr 21–22 | fresh 4-bag rerun to diagnose capability vs activation | seq_len=10, capped=5 |

### Prompt experiments

`benchmark_runs/prompt_experiments/` captures the evolution of the prompt as
independent probe runs that do **not** follow the full-benchmark structure:

- `results/prompt_ab_results.json` — BEFORE vs AFTER prompt A/B on
  bags 05/08/09/10, two models × three configs (BEFORE, AFTER_prompt,
  AFTER_tail).
- `results/bag10_receding_fix_results.json` — canonical four-config
  validation of the receding-fix rule on bag_10.
- `results/bag05_direction_predictions.json` — 18-window scan of bag_05 used
  to compute the first `direction_rule_consistency` values.
- `scripts/*.py` — the four scripts that produced those JSONs (direct
  backend calls, bypassing the wrapper so old-vs-new prompts can be
  compared on the same infra).
- `prompt_snapshots/current_new_seq_prompt.txt` — snapshot of the
  prompt after the receding fix but before the crossing rule.
- `prompt_snapshots/prompt_before_crossing.txt` — snapshot just before the
  crossing-direction rule was inserted.

---

## 6. Operational notes and gotchas

### 6.1 Network flakiness

The wrapper host `10.157.141.10` and the backend hosts `10.157.141.181:8000`
(InternVL) and `10.157.141.181:8001` (Qwen) went up and down several times
during the project. The runner handles this gracefully thanks to streaming
checkpoints, but the cycle is:

1. Monitor alarms on a burst of new network failures.
2. `kill -STOP <pid>` the runner so it stops burning calls.
3. Wait for infra recovery (`curl backend /v1/models` → 200 on both).
4. `kill <pid>` and scrub: the scrub script drops rows whose
   `failure_reason` matches `http_error | Connection refused |
   No route to host | Read timed out | Max retries exceeded |
   vlm_backend_request_failed`.
5. Re-launch the same command. Checkpoint resume recomputes only the
   scrubbed samples.

### 6.2 Wrapper prompt deployment

Every time `social_nav_eval_prompts.py` is edited, the wrapper host's copy of
the file must be synced and the wrapper restarted. Otherwise the backend is
called with the *old* prompt, and the schema of the model response will be
wrong. The smoke test that verifies the new schema is live:

```bash
python3 - <<'PY'
import base64,io,json,urllib.request
from PIL import Image
img=Image.open('extracted_social_nav/bag_05/images/frame_000120.ppm')
if img.width>512: img=img.resize((512,int(img.height*512/img.width)),Image.LANCZOS)
buf=io.BytesIO(); img.convert('RGB').save(buf,format='JPEG',quality=60)
b=base64.b64encode(buf.getvalue()).decode()
for p,n in [(8100,'IVL'),(8101,'Qwen')]:
    body=json.dumps({'prompt_name':'sequence_image_navigation','images_base64':[b]*5}).encode()
    r=json.loads(urllib.request.urlopen(urllib.request.Request(
        f'http://10.157.141.10:{p}/analyze_navigation',data=body,
        headers={'Content-Type':'application/json'}),timeout=60).read())
    rj=r.get('response_json') or {}
    print(f"{n}: has_crossing_direction={'crossing_direction' in rj} keys={list(rj.keys())}")
PY
```

If `crossing_direction` is not a key, the wrapper is still serving the old
prompt. Re-deploy before starting a new benchmark.

### 6.3 Sampling regime is a load-bearing knob

The `--sequence-sampling-mode` + `--max-frames-per-sequence` combination
determines how much temporal context reaches the model. Empirically:

- `--sequence-length 5 --sequence-sampling-mode tail --max-frames-per-sequence 3`
  is **too aggressive** on top of a 1/5 frame subsample — crossing is rarely
  detected, LEFT/RIGHT rarely fires.
- `--sequence-length 10 --sequence-sampling-mode capped --max-frames-per-sequence 5`
  is the sweet spot used by the final runs: wider temporal spread restores
  crossing detection while still keeping the per-request payload small
  enough that Qwen-sequence stays near ~5 s/call.

### 6.4 Labels can change without rerunning inference

Because sample IDs are deterministic and the runner's checkpoint logic keys
on them, a pure label change (e.g., the directional-avoidance variant) does
not require rerunning VLM inference. The recipe:

```bash
# seed a new run with predictions from an existing run
mkdir -p benchmark_runs/new_run
cp -r benchmark_runs/existing_run/bag_*/ benchmark_runs/new_run/

# rerun with the new manifest; all VLM calls are skipped via checkpoint
python3 motion_control/scripts/social_nav_eval.py run-manifest \
  --input-mode extracted \
  --manifest motion_control/eval/scenario_manifest_directional_avoidance.json \
  --output-dir benchmark_runs/new_run \
  --sequence-length 10 --sequence-sampling-mode capped --max-frames-per-sequence 5
```

This is how `final_policy_directional_labels_v1/` was built in ~40 seconds.

### 6.5 Geometry baseline unavailability

The 13 bags in the current dataset do NOT contain a `/scan_odom` topic.
`extract_social_nav_data.py` silently produces `front_dist: null` for every
frame. The geometry baseline requires `front_dist`, so all 1,290 geometry
predictions are `failure_reason: missing_front_dist`. It is not a bug of
the geometry path — it is a dataset limitation. If a bag with
`/scan_odom` is added, the geometry baseline will light up again.

---

## 7. How to reproduce the headline results

### 7.1 Extract the dataset (one time, requires ROS)

```bash
python3 motion_control/scripts/prepare_social_nav_extracted.py \
  --manifest motion_control/eval/scenario_manifest.json \
  --output-root extracted_social_nav \
  --image-topic /camera/image_raw
```

Optionally subsample to every 5th frame:

```bash
for d in extracted_social_nav/bag_*; do
  cp "$d/frames.jsonl" "$d/frames.jsonl.full"
  awk 'NR % 5 == 1' "$d/frames.jsonl.full" > "$d/frames.jsonl"
done
```

### 7.2 The primary crossing-aware run

```bash
tmux new -s bench
SOCIAL_NAV_EVAL_INTERNVL_WRAPPER_URL=http://10.157.141.10:8100 \
SOCIAL_NAV_EVAL_QWEN_WRAPPER_URL=http://10.157.141.10:8101 \
python3 motion_control/scripts/social_nav_eval.py run-manifest \
  --input-mode extracted \
  --manifest motion_control/eval/scenario_manifest.json \
  --output-dir benchmark_runs/final_policy_v2_crossing \
  --sequence-length 10 \
  --sequence-sampling-mode capped \
  --max-frames-per-sequence 5
```

Wall time: ~3–4 h given stable wrappers.

### 7.3 Directional-avoidance variant (label-only rerun)

```bash
SRC=benchmark_runs/final_policy_v2_crossing
DST=benchmark_runs/final_policy_directional_labels_v1
mkdir -p "$DST"
for bag in $(ls $SRC | grep ^bag_); do
  mkdir -p "$DST/$bag/predictions"
  cp "$SRC/$bag/predictions/"*.jsonl "$DST/$bag/predictions/"
done
# environment variables still required even though no VLM calls will be issued:
SOCIAL_NAV_EVAL_INTERNVL_WRAPPER_URL=http://10.157.141.10:8100 \
SOCIAL_NAV_EVAL_QWEN_WRAPPER_URL=http://10.157.141.10:8101 \
python3 motion_control/scripts/social_nav_eval.py run-manifest \
  --input-mode extracted \
  --manifest motion_control/eval/scenario_manifest_directional_avoidance.json \
  --output-dir benchmark_runs/final_policy_directional_labels_v1 \
  --sequence-length 10 --sequence-sampling-mode capped --max-frames-per-sequence 5
```

Wall time: ~40 s.

### 7.4 Targeted four-bag activation probe

```bash
python3 motion_control/scripts/social_nav_eval.py run-manifest \
  --input-mode extracted \
  --manifest motion_control/eval/scenario_manifest_targeted_4bags.json \
  --output-dir benchmark_runs/targeted_directional_activation_test \
  --sequence-length 10 --sequence-sampling-mode capped --max-frames-per-sequence 5
```

Wall time: ~25 min.

### 7.5 Quick direction-metric sanity check on an existing run

```bash
python3 - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0,'motion_control/scripts')
from social_nav_eval import compute_direction_metrics

root = Path('benchmark_runs/final_policy_v2_crossing')
for model in ['internvl','qwen']:
    rows=[]
    for bag in sorted(root.glob('bag_*')):
        for m in [f'{model}_sequence_image_navigation', f'{model}_single_image_navigation']:
            p = bag/'predictions'/f'{m}.jsonl'
            if p.exists():
                rows += [json.loads(l) for l in open(p)]
    dm = compute_direction_metrics(rows)
    print(f"{model}: consistency={dm['direction_rule_consistency']} "
          f"activation={dm['direction_activation_rate']} "
          f"detect={dm['crossing_detection_rate']}")
PY
```

---

## 8. Where to look when you need answers

| question | file |
|---|---|
| "What prompt is the benchmark sending?" | `social_nav_eval_prompts.py` |
| "How does a sample become a prediction?" | `motion_control/scripts/social_nav_eval.py` — `predict_vlm`, `call_navigation_wrapper`, `run_predictions` |
| "Why did my run die and how do I resume?" | `run_predictions` streaming checkpoint logic in `social_nav_eval.py` |
| "What's the five-action policy rulebook?" | `PROMPT_SEQUENCE_IMAGES` literal in `social_nav_eval_prompts.py` |
| "Which bags are primary vs review?" | `motion_control/eval/scenario_manifest.json` (`primary_case` field) |
| "How is `direction_rule_consistency` computed?" | `compute_direction_metrics()` in `social_nav_eval.py` |
| "What did the wrapper actually see?" | per-scenario `bag_<id>/predictions/*.jsonl`, `raw_response` field |
| "Why does my run show old-schema motion labels?" | wrapper host has not been redeployed with the current `social_nav_eval_prompts.py` |
| "What's the capstone narrative?" | `final_report.mdx` |
| "Frame counts, benchmark workflow details" | `report_eval.md` |

---

## 9. What's unfinished

- `LEFT` and `RIGHT` are schema actions only — they are not yet mapped to
  Go1 control primitives.
- Safety metrics `unsafe_forward_gt_rate` are still too high on the
  STOP-labelled primary bags to consider deployment.
- Crossing detection fires on only a small fraction of frames in the full
  benchmark; the rule itself is correct but perception under the current
  temporal sampling is the limiter.
- No end-to-end fine-tuning has been done. The loss formalisation in
  `final_report.mdx` is conceptual, not optimised.
- The bag set has no `/scan_odom`, so the geometry baseline is unavailable
  on this data. A new bag set with LiDAR would re-enable it without code
  change.

See `final_report.mdx` → "Limitations" and "What Is Needed for Real Go1
Online Deployment" for the full discussion.
